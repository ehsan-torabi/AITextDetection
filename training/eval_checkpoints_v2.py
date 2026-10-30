"""Evaluate the *existing* (v1) checkpoints on data they verifiably never saw.

Why this is separate from ``training/eval_suite.py``: the nine checkpoints in
``models/`` were trained on the v1 split of ``processedDataset.csv``. Any
comparison against the v2 splits therefore has to restrict itself to rows that
are provably outside the v1 training data, which is what ``extra.v1_split``
marks. This script scores exactly those rows, plus the newly collected human
domains, so the report can answer two questions honestly:

1. how do the deployed models do on held-out data from their own distribution
   (``v1_test``)?
2. how do they do when the human side comes from a domain they have never seen
   (news / encyclopedia / social media / legal-admin / translated subtitles)?

No threshold is tuned here on purpose: the v1 checkpoints ship no calibrated
threshold and the serving path uses a fixed 0.5, so the reported numbers are
ROC-AUC / PR-AUC / FPR@TPR95 (threshold-free) plus the raw 0.5 decision.
Per-domain proportions come with Wilson intervals because the per-domain
sample is capped (``--limit-per-domain``).

Run from the project root:
    python training/eval_checkpoints_v2.py --limit-per-domain 400
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from aitextd import dataset as ds
from aitextd import paths
from aitextd.features import STYLE_FEATURE_NAMES, compute_style_features
from aitextd.store import ModelStore
from aitextd.text import preprocess_persian_text

import eval_metrics as em  # type: ignore[import-not-found]

DATA = paths.DATA_DIR
DATASET = DATA / "dataset_v2.parquet"
SPLITS = DATA / "splits_v2.joblib"
OUT_JSON = PROJECT_ROOT / "reports" / "checkpoints_v2.json"
OUT_MD = PROJECT_ROOT / "reports" / "tables" / "checkpoints_v2.md"

NEW_DOMAINS = ("news", "encyclopedia", "social_media", "legal_admin", "translated_subtitles")
ENSEMBLE_ID = "ensemble"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: correct coverage for proportions near 0/1, which
    is exactly the regime of a false-positive rate."""
    if total == 0:
        return float("nan"), float("nan")
    phat = successes / total
    denominator = 1 + z**2 / total
    centre = phat + z**2 / (2 * total)
    spread = z * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2))
    return (centre - spread) / denominator, (centre + spread) / denominator


def build_subsets(df: pd.DataFrame, splits: dict, seed: int, limit_per_domain: int) -> dict[str, np.ndarray]:
    """Index sets to score: v1-comparable rows, each unseen human domain, the
    unseen AI rows, and machine-revised rows when they exist."""
    rng = np.random.default_rng(seed)
    subsets: dict[str, np.ndarray] = {}
    subsets["v1_test"] = splits["v1_test"]

    label = df["label"].to_numpy()
    domain = df["domain"].to_numpy()
    v1_split = df["extra"].map(lambda e: json.loads(e).get("v1_split", ""))

    # human rows from the new domains that the v1 checkpoints never saw
    for name in NEW_DOMAINS:
        mask = (label == ds.HUMAN) & (domain == name) & (v1_split != "train").to_numpy()
        positions = np.flatnonzero(mask)
        positions = positions[rng.choice(len(positions), size=min(limit_per_domain, len(positions)), replace=False)] if len(positions) > limit_per_domain else positions
        subsets[f"human__{name}"] = np.sort(positions)

    # a same-size AI reference slice (rows that are not in v1 train either)
    ai_positions = np.flatnonzero((label == ds.AI) & (v1_split != "train").to_numpy())
    if len(ai_positions) > limit_per_domain:
        ai_positions = np.sort(
            rng.choice(ai_positions, size=limit_per_domain, replace=False)
        )
    subsets["ai__reference"] = ai_positions

    revised = np.flatnonzero(label == ds.AI_REVISED)
    if revised.size:
        subsets["ai_revised__all"] = (
            revised
            if revised.size <= limit_per_domain
            else np.sort(rng.choice(revised, size=limit_per_domain, replace=False))
        )
    return subsets


def score_subsets(df: pd.DataFrame, subsets: dict[str, np.ndarray]) -> dict[str, dict[str, list[float]]]:
    """model_id -> subset -> calibrated-free probability per document."""
    store = ModelStore()
    store.init()
    vectorizer = store.vectorizer
    model_ids = [md["id"] for md in store.models_dict.values()]
    model_ids.append(ENSEMBLE_ID)

    scores: dict[str, dict[str, list[float]]] = {m: {} for m in model_ids}
    for subset_name, idx in subsets.items():
        texts = df.iloc[idx]["text"].tolist()
        cleaned = [preprocess_persian_text(t) for t in texts]
        style = [compute_style_features(t, c, vectorizer) for t, c in zip(texts, cleaned, strict=True)]
        started = time.time()
        for model_id in model_ids:
            values: list[float] = []
            for text, cleaned_text, style_features in zip(texts, cleaned, style, strict=True):
                if model_id == ENSEMBLE_ID:
                    _, _, votes = store._ensemble_predict(text, cleaned_text, style_features)
                    weight_total = sum(v.weight for v in votes) or 1.0
                    values.append(
                        float(sum(v.aiProbability * v.weight for v in votes) / weight_total)
                    )
                else:
                    result = store._predict_single(model_id, text, cleaned_text, style_features)
                    values.append(float(result["aiProbability"]) if isinstance(result, dict) else float(result))
            scores[model_id][subset_name] = values
            print(
                f"  {subset_name:26s} {model_id:16s} "
                f"{len(values):5d} docs ({time.time() - started:.0f}s)",
                flush=True,
            )
    return scores


def summarize(df: pd.DataFrame, subsets: dict[str, np.ndarray], scores: dict) -> dict:
    label = df["label"].to_numpy()
    out: dict = {}
    for model_id, per_subset in scores.items():
        out[model_id] = {}
        for subset_name, values in per_subset.items():
            idx = subsets[subset_name]
            y = (label[idx] != ds.HUMAN).astype(int)
            p = np.asarray(values, dtype=float)
            record: dict = {"n": int(len(y)), "n_human": int((y == 0).sum()), "n_machine": int((y == 1).sum())}
            if 0 < y.sum() < len(y):
                metrics = em.binary_metrics(y, p)
                record["metrics"] = {
                    k: v
                    for k, v in metrics.to_dict()["metrics"].items()
                    if k in ("accuracy", "macro_f1", "roc_auc", "pr_auc", "fpr_at_tpr95", "fpr", "recall_ai", "recall_human")
                }
                record["confusion"] = metrics.confusion
                record["ci95"] = em.bootstrap_ci(y, p, n_boot=300)
            else:
                mean_p = float(p.mean())
                record["probabilities"] = {
                    "mean": round(mean_p, 4),
                    "p90": round(float(np.quantile(p, 0.90)), 4),
                    "p99": round(float(np.quantile(p, 0.99)), 4),
                    "share_above_0.5": round(float((p >= 0.5).mean()), 4),
                }
                lo, hi = wilson_interval(int((p >= 0.5).sum()), len(p))
                record["share_above_0.5_ci95"] = [round(lo, 4), round(hi, 4)]
            out[model_id][subset_name] = record
    return out


def write_markdown(summary: dict) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Existing (v1) checkpoints on data they never saw",
        "",
        "Threshold-free metrics for every subset that has both classes; for the",
        "single-class human subsets the share of documents the deployed 0.5",
        "decision point flags as machine text is reported instead.",
        "",
        "| model | subset | n | ROC-AUC | PR-AUC | FPR@TPR95 | recall AI | FPR at 0.5 | flagged share (95% CI) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model_id, per_subset in sorted(summary.items()):
        for subset_name, record in per_subset.items():
            if "metrics" in record:
                metrics = record["metrics"]
                row = [
                    model_id,
                    subset_name,
                    str(record["n"]),
                    f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "—",
                    f"{metrics['pr_auc']:.4f}" if metrics["pr_auc"] is not None else "—",
                    f"{metrics['fpr_at_tpr95']:.4f}" if metrics["fpr_at_tpr95"] is not None else "—",
                    f"{metrics['recall_ai']:.4f}" if metrics["recall_ai"] is not None else "—",
                    f"{metrics['fpr']:.4f}" if metrics["fpr"] is not None else "—",
                    "—",
                ]
            else:
                share = record["probabilities"]["share_above_0.5"]
                lo, hi = record["share_above_0.5_ci95"]
                row = [
                    model_id,
                    subset_name,
                    str(record["n"]),
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    f"{share:.4f} [{lo:.4f}, {hi:.4f}]",
                ]
            lines.append("| " + " | ".join(row) + " |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit-per-domain", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(DATASET).reset_index(drop=True)
    splits = joblib.load(SPLITS)
    subsets = build_subsets(df, splits, args.seed, args.limit_per_domain)
    print(json.dumps({k: int(len(v)) for k, v in subsets.items()}, ensure_ascii=False, indent=1))

    started = time.time()
    scores = score_subsets(df, subsets)
    summary = summarize(df, subsets, scores)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_fingerprint": ds.dataset_fingerprint(df),
        "limit_per_domain": args.limit_per_domain,
        "seed": args.seed,
        "subset_rows": {k: int(len(v)) for k, v in subsets.items()},
        "style_feature_names": list(STYLE_FEATURE_NAMES),
        "seconds": round(time.time() - started, 1),
        "models": summary,
        "note": (
            "checkpoints are the v1 models in models/ (trained on the v1 train "
            "split); every subset here excludes v1 train rows, per-domain human "
            "subsets are capped and sampled, and no threshold is tuned -- the "
            "0.5 column is the deployed decision rule"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    write_markdown(summary)
    print(f"-> {OUT_JSON}")
    print(f"-> {OUT_MD}")


if __name__ == "__main__":
    main()
