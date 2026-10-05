"""TEMPORARY evaluation: score online System One providers on the held-out test slice.

This is test-grade tooling, not part of the serving stack: it asks one or more
custom "System One"-compatible endpoints (e.g. TypeSafe relays) the same three
mirrored questions the web app sends (choice / score / noul) for every text of
a stratified sample from the canonical held-out test split, then reports the
full metric battery per provider and a side-by-side comparison table.

Everything is configured in training/config.json — providers (models) and run
settings — so a plain `python training/eval_online_systemone.py` works:

    {
      "run": {"n": 100, "threshold": 0.5, "concurrency": 4, "seed": 42,
              "out": "training/online_eval_results.json"},
      "providers": [
        {"name": "jev-relay", "baseUrl": "http://localhost:20128/v1/systemone",
         "apiKey": "sk-...", "model": "openrouter/typesafe/jev-1.13"},
        {"name": "relay-multi", "baseUrl": "http://...", "apiKey": "sk-...",
         "models": ["openrouter/typesafe/jev-1.13", "openrouter/typesafe/jev-1.12"]}
      ]
    }

A provider targets ONE endpoint with either a single "model" or a "models"
list (expanded into one comparison target per model); "enabled": false skips
an entry. A legacy bare-array config (eval_online_providers.json) is still
accepted, and every run setting can be overridden on the command line
(--provider "name=x;baseUrl=...;model=...;apiKey=..." adds/overrides targets).

Measured per provider (AI class = positive, threshold on the choice
probability, default 0.5):
- accuracy, precision / recall / F1 (both classes + macro)
- specificity, MCC, confusion matrix (tn/fp/fn/tp)
- ROC-AUC and PR-AUC of the ai_generated probability
- Brier score and 10-bin ECE (calibration)
- score-question MAE against the label mapped to the 5-level scale (human=0, AI=4)
- noul AUC (does "needs review" rank borderline cases sensibly?)
- mean latency, token usage, error count

The config file holds real API keys, so it (and the response cache and results
file) are gitignored.
"""

import argparse
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from split import RANDOM_SEED, load_split

DEFAULT_CONFIG = PROJECT_ROOT / "training" / "config.json"
DEFAULT_OUT = PROJECT_ROOT / "training" / "online_eval_results.json"
CACHE_PATH = PROJECT_ROOT / "training" / ".online_eval_cache.jsonl"

REQUEST_TIMEOUT_S = 60
RETRY_STATUSES = {429, 529}
RETRIES = 3

MAX_SCORE_LEVEL = 4  # 5-level machine-likeness scale: human=0, AI=4

# Same mirrored questions the web app's /api/typesafe/detect route sends, so
# online numbers are directly comparable with the local System One pipeline.
# Single source of truth in aitextd.constants: the pytorch-laya serving
# predictor asks the fine-tuned laya checkpoint the exact same wording.
from aitextd.constants import SYSTEM_ONE_DETECTION_QUESTIONS as QUESTIONS


def load_config(config_path: Path) -> tuple[dict, list[dict]]:
    """Read the config file; returns (run settings, providers).

    Accepted shapes:
    - {"run": {...}, "providers": [...]}   (preferred)
    - a bare array of providers            (legacy eval_online_providers.json)

    A provider entry targets ONE endpoint with either a single "model" or a
    "models" list; the latter is expanded into one target per model so the
    same relay credentials can compare several models without repetition.
    Disabled entries ("enabled": false) are skipped.
    """
    if not config_path.exists():
        return {}, []
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    run = {}
    entries = raw
    if isinstance(raw, dict):
        run = raw.get("run", {}) or {}
        entries = raw.get("providers", []) or []
    providers: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("enabled", True):
            continue
        base = {k: v for k, v in entry.items() if k not in ("models", "_note")}
        models = entry.get("models") or [entry.get("model")]
        for model in models:
            if not model:
                continue
            providers.append({**base, "model": model} if len(models) > 1 else base)
    return run, providers


def parse_provider_flag(spec: str) -> dict:
    provider = {}
    for part in spec.split(";"):
        if not part.strip():
            continue
        key, _, value = part.partition("=")
        provider[key.strip()] = value.strip()
    for required in ("baseUrl", "model"):
        if not provider.get(required):
            raise SystemExit(f"--provider needs a {required} field: {spec}")
    return provider


def merge_providers(config: list[dict], overrides: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for provider in config:
        name = (
            provider.get("name")
            or f"{provider.get('model', 'model')}@{provider.get('baseUrl', '')}"
        )
        merged[name] = {**merged.get(name, {}), **provider}
    for provider in overrides:
        name = provider.get("name") or f"provider-{len(merged)}"
        merged[name] = {**merged.get(name, {}), **provider}
    return [p for p in merged.values() if p.get("enabled", True)]


def endpoint_for(baseUrl: str) -> str:
    base = baseUrl.rstrip("/")
    return base if base.endswith("/v1/systemone") else f"{base}/v1/systemone"


# --- response cache ---------------------------------------------------------
# One JSON object per line, keyed by provider name + text hash, so reruns of
# this temporary script never re-pay for already-scored samples.
_cache_lock = threading.Lock()


def _cache_key(provider: dict, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{provider['name']}|{provider['model']}|{digest}"


def cache_load() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    entries = {}
    for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        entries[entry["key"]] = entry
    return entries


def cache_store(key: str, payload: dict) -> None:
    with _cache_lock, CACHE_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, **payload}, ensure_ascii=False) + "\n")


# --- provider calls ----------------------------------------------------------
def ask_provider(provider: dict, text: str) -> dict:
    """Call one provider once; returns a compact record or raises.

    Retries transient failures (429/529/network) with linear backoff.
    """
    body = json.dumps(
        {"state": {"text": text}, "model": provider["model"], "questions": QUESTIONS}
    ).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {provider.get('apiKey', '')}",
        "Content-Type": "application/json",
    }
    url = endpoint_for(provider["baseUrl"])

    last_error: Exception | None = None
    for attempt in range(RETRIES):
        started = time.perf_counter()
        try:
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUSES:
                raise RuntimeError(f"HTTP {exc.code}: {exc.reason}") from exc
            last_error = exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"retries exhausted: {last_error}")

    latency = time.perf_counter() - started
    answers = data.get("answers", {})

    choice = answers.get("source", {})
    probs: dict = choice.get("probabilities", {})
    ai_prob = float(probs.get("ai_generated", 0.0))

    score = answers.get("machine_likeness", {})
    score_value = float(score.get("score", 0.0))

    noul = float(answers.get("needs_review", {}).get("noul", 0.0))

    usage = data.get("usage") or {}
    return {
        "aiProbability": ai_prob,
        "choiceConfidence": float(choice.get("confidence", 0.0)),
        "scoreValue": score_value,
        "noulProbability": noul,
        "resolvedModel": data.get("model"),
        "inputTokens": int(usage.get("input_tokens", 0) or 0),
        "outputTokens": int(usage.get("output_tokens", 0) or 0),
        "cost": usage.get("cost"),
        "latencyS": round(latency, 3),
    }


def evaluate_texts(
    provider: dict, texts: list[str], cache: dict, concurrency: int
) -> list[dict | None]:
    results: list[dict | None] = [None] * len(texts)
    progress_lock = threading.Lock()
    done = 0

    def worker(i: int) -> None:
        nonlocal done
        key = _cache_key(provider, texts[i])
        if key in cache:
            results[i] = {**cache[key], "cached": True}
        else:
            try:
                record = ask_provider(provider, texts[i])
            except Exception as exc:
                results[i] = None
                print(f"  [{provider['name']}] sample {i} failed: {exc}")
                return
            payload = dict(record)
            with _cache_lock:
                cache[key] = payload
            cache_store(key, payload)
            results[i] = {**payload, "cached": False}
        with progress_lock:
            done += 1
            if done % 10 == 0 or done == len(texts):
                print(f"  [{provider['name']}] {done}/{len(texts)} scored")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pool.map(worker, range(len(texts)))
    return results


# --- metrics -----------------------------------------------------------------
def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob > lo) & (prob <= hi) if lo > 0 else (prob >= lo) & (prob <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - prob[mask].mean())
    return float(ece)


def compute_metrics(y_true: np.ndarray, records: list[dict | None], threshold: float) -> dict:
    valid = [r for r in records if r is not None]
    errors = len(records) - len(valid)
    ai_prob = np.array([r["aiProbability"] for r in valid])
    noul_prob = np.array([r["noulProbability"] for r in valid])
    score_value = np.array([r["scoreValue"] for r in valid])
    y = y_true[: len(records)][np.array([r is not None for r in records])]

    pred = (ai_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    mae_score = float(np.mean(np.abs(score_value - MAX_SCORE_LEVEL * y)))

    metrics = {
        "samples": len(y),
        "errors": errors,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "precisionHuman": float(precision[0]),
        "recallHuman": float(recall[0]),
        "f1Human": float(f1[0]),
        "precisionAI": float(precision[1]),
        "recallAI": float(recall[1]),
        "f1AI": float(f1[1]),
        "f1Macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "mcc": float(matthews_corrcoef(y, pred)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "rocAuc": float(roc_auc_score(y, ai_prob)) if len(set(y)) > 1 else None,
        "prAuc": float(average_precision_score(y, ai_prob)) if len(set(y)) > 1 else None,
        "brier": float(np.mean((ai_prob - y) ** 2)),
        "ece10": expected_calibration_error(y, ai_prob),
        "scoreMae": mae_score,
        "noulAuc": float(roc_auc_score(y, noul_prob)) if len(set(y)) > 1 else None,
        "meanLatencyS": float(np.mean([r["latencyS"] for r in valid])),
        "totalInputTokens": int(sum(r["inputTokens"] for r in valid)),
        "totalOutputTokens": int(sum(r["outputTokens"] for r in valid)),
        "totalCost": float(sum(r["cost"] for r in valid if r["cost"] is not None)) or None,
    }
    return metrics


TABLE_ROWS = [
    ("accuracy", "acc"),
    ("precisionAI", "prec(AI)"),
    ("recallAI", "rec(AI)"),
    ("f1AI", "f1(AI)"),
    ("precisionHuman", "prec(hu)"),
    ("recallHuman", "rec(hu)"),
    ("f1Macro", "f1(macro)"),
    ("rocAuc", "rocAuc"),
    ("prAuc", "prAuc"),
    ("ece10", "ece10"),
    ("scoreMae", "scoreMae"),
    ("noulAuc", "noulAuc"),
    ("mcc", "mcc"),
]


def print_comparison_table(per_provider: dict[str, dict]) -> None:
    headers = ["provider", *(label for _, label in TABLE_ROWS), "err"]
    widths = [max(len(h), 20 if i == 0 else 8) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print("\n" + line)
    print("-" * len(line))
    for name, metrics in sorted(per_provider.items(), key=lambda kv: -kv[1]["accuracy"]):
        cells = [name.ljust(widths[0])]
        for key, _ in TABLE_ROWS:
            value = metrics[key]
            cells.append(
                f"{value:.4f}".ljust(widths[len(cells)])
                if value is not None
                else "-".ljust(widths[len(cells)])
            )
        cells.append(str(metrics["errors"]).ljust(widths[-1]))
        print("  ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "n", nargs="?", type=int, default=None, help="sample size from the held-out test split"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help='inline provider: "name=x;baseUrl=http://...;model=...;apiKey=..." (repeatable)',
    )
    args = parser.parse_args()

    # Everything lives in config.json (providers + run settings); explicit CLI
    # arguments override it when present.
    run, config_providers = load_config(args.config)
    n = args.n if args.n is not None else int(run.get("n", 200))
    configured_out = Path(run.get("out", str(DEFAULT_OUT)))
    out = args.out or (
        configured_out if configured_out.is_absolute() else PROJECT_ROOT / configured_out
    )
    threshold = args.threshold if args.threshold is not None else float(run.get("threshold", 0.5))
    concurrency = (
        args.concurrency if args.concurrency is not None else int(run.get("concurrency", 4))
    )
    seed = args.seed if args.seed is not None else int(run.get("seed", RANDOM_SEED))

    providers = merge_providers(config_providers, [parse_provider_flag(s) for s in args.provider])
    if not providers:
        raise SystemExit(f"No providers configured. Edit {args.config} or pass --provider.")

    df, _, test_idx = load_split()
    y = df["aiGenerated"].to_numpy()
    rng = np.random.default_rng(seed)
    if n < len(test_idx):
        sample_idx = rng.choice(test_idx, size=n, replace=False)
    else:
        sample_idx = test_idx
    texts = [str(t) for t in df["text"].to_numpy()[sample_idx]]
    y_sample = y[sample_idx]
    print(
        f"Evaluating {len(providers)} provider(s) on {len(texts)} held-out samples "
        f"(threshold={threshold}, concurrency={concurrency})"
    )

    cache = cache_load()
    per_provider: dict[str, dict] = {}
    per_sample: dict[str, list] = {}
    for provider in providers:
        print(f"\n=== {provider['name']} ({provider['model']} @ {provider['baseUrl']}) ===")
        records = evaluate_texts(provider, texts, cache, concurrency)
        metrics = compute_metrics(y_sample, records, threshold)
        per_provider[provider["name"]] = metrics
        per_sample[provider["name"]] = records
        conf = metrics["confusion"]
        print(
            f"  acc={metrics['accuracy']:.4f}  prec(AI)={metrics['precisionAI']:.4f}  "
            f"rec(AI)={metrics['recallAI']:.4f}  f1(AI)={metrics['f1AI']:.4f}  "
            f"rocAuc={metrics['rocAuc'] if metrics['rocAuc'] is None else round(metrics['rocAuc'], 4)}  "
            f"errors={metrics['errors']}"
        )
        print(f"  confusion: tn={conf['tn']} fp={conf['fp']} fn={conf['fn']} tp={conf['tp']}")

    print_comparison_table(per_provider)

    out.write_text(
        json.dumps(
            {
                "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sampleSize": len(texts),
                "threshold": threshold,
                "seed": seed,
                "labels": y_sample.tolist(),
                "providers": {
                    p["name"]: {
                        "model": p["model"],
                        "baseUrl": p["baseUrl"],
                        "metrics": per_provider[p["name"]],
                    }
                    for p in providers
                },
                "perSample": per_sample,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nFull results (incl. per-sample probabilities) -> {out}")


if __name__ == "__main__":
    main()
