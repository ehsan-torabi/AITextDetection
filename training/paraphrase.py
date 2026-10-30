"""Paraphrase / revision module: training-time augmentation *and* attack set.

Review section 6 requires paraphrase to be used in both roles, with the method,
model and prompt recorded per sample, and an attack test set whose degradation
relative to the clean condition is reported.

What runs on this host (offline, no LLM endpoint available):

- ``backtranslation``        fa -> en -> fa round trip (Helsinki-NLP opus-mt)
- ``lexical_substitution``   masked-LM substitution with ParsBERT's MLM head
- ``sentence_perturbation``  sentence-order and sentence-boundary changes
- ``punct_spacing_noise``    half-space / punctuation / digit-form noise

What is implemented but requires credentials (recorded as pending):

- ``llm_paraphrase`` with five prompt templates (preserve meaning, more
  formal, simpler, student-voice, preserve all facts) over an OpenAI-compatible
  endpoint.  ``--llm-base-url`` + ``--llm-api-key``; skipped otherwise.

Two modes:

    --mode revision   sample human rows -> machine-revised rows (label
                      ai_revised, revision_type = method). These become the
                      third class and the human-vs-revised test set.
    --mode attack     take the AI test rows reserved by splits_v2.py and
                      rewrite them; evaluation then measures how much recall
                      drops versus the clean condition.

Artifacts (all with provenance columns method/model/prompt_id):
    data/ai_revised.parquet
    data/paraphrase_attack.parquet
    data/paraphrase_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from netutil import force_ipv4  # type: ignore[import-not-found]

from aitextd import dataset as ds
from aitextd import paths
from aitextd.features import PUNCTUATION

force_ipv4()  # this host has no IPv6 route; see training/netutil.py

DATASET = paths.DATA_DIR / "dataset_v2.parquet"
SPLITS = paths.DATA_DIR / "splits_v2.joblib"
OUT_REVISED = paths.DATA_DIR / "ai_revised.parquet"
OUT_ATTACK = paths.DATA_DIR / "paraphrase_attack.parquet"
CHECKPOINT_REVISED = paths.DATA_DIR / "ai_revised.partial.parquet"
MANIFEST = paths.DATA_DIR / "paraphrase_manifest.json"

BT_MODEL_FORWARD = "Helsinki-NLP/opus-mt-mul-en"
BT_MODEL_BACK = "Helsinki-NLP/opus-mt-en-mul"
MLM_MODEL = "HooshvareLab/bert-fa-base-uncased"

DEFAULT_LLM_TEMPLATES = {
    "preserve_meaning": "متن زیر را بازنویسی کن و معنا را کامل حفظ کن:\n\n{text}",
    "more_formal": "متن زیر را با لحن رسمی‌تر بازنویسی کن:\n\n{text}",
    "simpler": "متن زیر را ساده‌تر و روان‌تر بازنویسی کن:\n\n{text}",
    "student_voice": "متن زیر را طوری بازنویسی کن که یک دانشجو نوشته باشد:\n\n{text}",
    "preserve_facts": "متن زیر را بازنویسی کن؛ همهٔ واقعیت‌ها را حفظ کن:\n\n{text}",
}

_SENT_SPLIT = re.compile(r"(?<=[.!?؟…])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


# --- offline rewriters ------------------------------------------------------


class BackTranslator:
    """fa -> en -> fa round trip. Pivot quality is documented as medium: the
    multilingual opus-mt checkpoints are the only Persian-capable MT models
    reachable offline here, and revision quality is not the object of study --
    the *detector's* sensitivity to it is."""

    def __init__(self, batch_size: int = 24, max_words: int = 300) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.batch_size = batch_size
        self.max_words = max_words
        self.tok_fwd = AutoTokenizer.from_pretrained(BT_MODEL_FORWARD)
        self.fwd = AutoModelForSeq2SeqLM.from_pretrained(BT_MODEL_FORWARD)
        self.tok_back = AutoTokenizer.from_pretrained(BT_MODEL_BACK)
        self.back = AutoModelForSeq2SeqLM.from_pretrained(BT_MODEL_BACK)
        self.fwd.eval()
        self.back.eval()
        self.model_id = f"{BT_MODEL_FORWARD}->{BT_MODEL_BACK}"

    def _run(self, model, tokenizer, sentences: list[str]) -> list[str]:
        """Greedy decoding over length-sorted mini-batches.

        Sorting by length keeps padding waste low, and the per-batch token
        budget stops a rare non-terminating generation from paying for the
        full 256-token ceiling on every sentence.
        """
        import torch

        out: list[str] = [""] * len(sentences)
        order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                index = order[start : start + self.batch_size]
                batch = [sentences[i] for i in index]
                enc = tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=256
                )
                budget = int(enc["input_ids"].shape[1] * 2.2) + 24
                generated = model.generate(
                    **enc, max_new_tokens=min(256, max(48, budget)), num_beams=1
                )
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for i, text in zip(index, decoded):
                    out[i] = text
        return out

    def rewrite(self, text: str) -> str:
        words = text.split()
        truncated = len(words) > self.max_words
        body = " ".join(words[: self.max_words])
        sentences = split_sentences(body) or [body]
        english = self._run(self.fwd, self.tok_fwd, sentences)
        back = self._run(self.back, self.tok_back, english)
        result = " ".join(b.strip() for b in back if b.strip())
        return result + (" …" if truncated else "")


class MaskedLmSubstituter:
    """Replace a share of content tokens by sampled masked-LM predictions.

    Token selection skips very short tokens and punctuation; substitution uses
    the model's top-k with sampling at low temperature, which keeps the text
    fluent while changing wording -- the classic lexical paraphrase attack.
    """

    def __init__(self, rate: float = 0.12, top_k: int = 8, seed: int = 42) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.rate = rate
        self.top_k = top_k
        self.rng = np.random.default_rng(seed)
        self.model_id = MLM_MODEL
        self.tok = AutoTokenizer.from_pretrained(
            str(paths.PARSBERT_TOKENIZER_DIR)
            if paths.PARSBERT_TOKENIZER_DIR.is_dir()
            else MLM_MODEL
        )
        self.model = AutoModelForMaskedLM.from_pretrained(MLM_MODEL)
        self.model.eval()

    def rewrite(self, text: str) -> str:
        import torch

        words = text.split()
        if len(words) < 8:
            return text
        candidates = [
            i
            for i, w in enumerate(words)
            if len(w) >= 4 and not all(ch in PUNCTUATION for ch in w)
        ]
        if not candidates:
            return text
        n_replace = max(1, int(self.rate * len(candidates)))
        chosen = sorted(self.rng.choice(candidates, size=n_replace, replace=False))
        marked = [w if i not in chosen else self.tok.mask_token for i, w in enumerate(words)]
        enc = self.tok(
            " ".join(marked), return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
        out = list(words)
        for pos, word_index in enumerate(chosen):
            mask_positions = (
                (enc["input_ids"][0] == self.tok.mask_token_id).nonzero().flatten()
            )
            if pos >= len(mask_positions):
                break
            token_logits = logits[mask_positions[pos]]
            top = torch.topk(token_logits, self.top_k)
            probs = torch.softmax(top.values, dim=-1).numpy()
            pick = int(self.rng.choice(len(probs), p=probs / probs.sum()))
            replacement = self.tok.decode([int(top.indices[pick])]).strip()
            if replacement:
                out[word_index] = replacement
        return " ".join(out)


def sentence_perturbation(text: str, rng: np.random.Generator, shuffle_rate: float = 0.35) -> str:
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return text
    head, rest = sentences[0], sentences[1:]
    order = rng.permutation(len(rest))
    keep = max(1, int(len(rest) * shuffle_rate))
    moved = set(order[:keep])
    reordered = [sentences[i + 1] for i in range(len(rest)) if i in moved] + [
        sentences[i + 1] for i in range(len(rest)) if i not in moved
    ]
    return " ".join([head, *reordered])


def punct_spacing_noise(text: str, rng: np.random.Generator, rate: float = 0.25) -> str:
    """Randomise the *typographic* layer only: ZWNJ insertion/removal,
    comma/kasma spacing, digit form. No words change, so this isolates the
    detector's reliance on orthography."""
    out = text
    if rng.random() < rate:
        out = out.replace(ds.ZWNJ, " ")
    if rng.random() < rate:
        out = out.replace(ds.ZWNJ, "")
    if rng.random() < rate:
        out = re.sub(r"\s*،\s*", "، ", out)
    if rng.random() < rate:
        persian_digits = {ord(str(d)): ord(str(d).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))) for d in range(10)}
        out = out.translate(persian_digits)
    if rng.random() < rate:
        out = re.sub(r"\s+([.!?؟])", r"\1", out)
    return out


class LlmParaphraser:
    """OpenAI-compatible chat client; the review expects prompt templates and
    per-sample prompt recording. Disabled unless a base URL and key are given."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.7):
        import urllib.request

        self._urlopen = urllib.request.urlopen
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def rewrite(self, text: str, prompt_id: str) -> str:
        import urllib.request

        template = DEFAULT_LLM_TEMPLATES[prompt_id]
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": template.format(text=text)}],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with self._urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()


# --- drivers ----------------------------------------------------------------


def _row_record(source: pd.Series, text: str, method: str, model: str, prompt_id: str) -> dict:
    return {
        "doc_id": ds.stable_doc_id(text),
        "text": text,
        "label": ds.AI_REVISED,
        "label_binary": 1,
        "domain": source["domain"],
        "source": f"revision:{method}",
        "license": "derived-from-source",
        "origin": source["origin"],
        "generator": model,
        "provider": "local" if "Helsinki" in model or "bert" in model.lower() else "llm",
        "prompt_id": prompt_id,
        "prompt_style": source["prompt_style"],
        "topic_id": source["topic_id"],
        "revision_type": method,
        "paraphrase_method": method,
        "formality": ds.formality_label(text),
        "length_words": len(text.split()),
        "length_bucket": ds.length_bucket(len(text.split())),
        "author_group": source["author_group"],
        "non_native_flag": int(source["non_native_flag"]),
        "pii_scrubbed": int(source["pii_scrubbed"]),
        "split": "",
        "extra": json.dumps({"source_doc_id": source["doc_id"]}, ensure_ascii=False),
    }


def _attack_record(source: pd.Series, text: str, method: str, model: str, prompt_id: str) -> dict:
    record = _row_record(source, text, method, model, prompt_id)
    record["label"] = ds.AI  # the *original* label is what the attack must preserve
    record["label_binary"] = 1
    record["source"] = f"attack:{method}"
    record["extra"] = json.dumps(
        {"source_doc_id": source["doc_id"], "original_label": source["label"]},
        ensure_ascii=False,
    )
    return record


def run_revision(
    df: pd.DataFrame,
    methods: list[str],
    per_domain: int,
    seed: int,
    llm=None,
    bt_per_domain: int | None = None,
    min_words: int = 60,
    max_words: int = 300,
) -> pd.DataFrame:
    """Revise *human* sources into the ``ai_revised`` class.

    Sources are drawn from a *per-domain* length window: the 10th-90th
    percentile of each domain's own length distribution, clipped to
    ``[min_words, max_words]``.  A single global window would have dropped
    social-media posts entirely (they are an order of magnitude shorter than
    blog posts), while the upper clip keeps the expensive rewriters -- and the
    truncation artefact that would otherwise let a classifier separate the
    revised class by ``min(words, cap)`` alone -- under control.

    The back-translation round trip dominates the cost, so ``bt_per_domain`` can
    cap how many sources per domain get it while every remaining method still
    runs on the full sample.  Rows are checkpointed per finished domain and the
    run resumes from the checkpoint.
    """
    rng = np.random.default_rng(seed)
    candidates = df[df["label"] == ds.HUMAN].copy()
    candidates["_words"] = candidates["text"].str.split().str.len()
    quantiles = candidates.groupby("domain")["_words"].quantile([0.10, 0.90]).unstack()
    lo = quantiles[0.10].clip(lower=min_words)
    hi = quantiles[0.90].clip(upper=max_words)
    candidates["_lo"] = candidates["domain"].map(lo)
    candidates["_hi"] = candidates["domain"].map(hi)
    candidates = candidates[candidates["_words"].between(candidates["_lo"], candidates["_hi"])]
    frames: list[dict] = []
    if CHECKPOINT_REVISED.exists():
        partial = pd.read_parquet(CHECKPOINT_REVISED)
        frames = [dict(record) for record in partial.to_dict("records")]
        done = set(partial["domain"].unique())
        candidates = candidates[~candidates["domain"].isin(done)]
        print(
            f"[revision] resume: {len(frames)} rows on disk from {sorted(done)}; "
            f"{len(candidates)} candidate sources left",
            flush=True,
        )
    rewriters: dict[str, object] = {}
    for method in methods:
        if method == "backtranslation":
            rewriters[method] = BackTranslator()
        elif method == "lexical_substitution":
            rewriters[method] = MaskedLmSubstituter(seed=seed)

    for domain, group in candidates.groupby("domain"):
        if domain == "translated_subtitles":
            continue  # already translated; revising it again would double-count
        take = min(per_domain, len(group))
        chosen = group.sample(take, random_state=seed)
        bt_cap = len(chosen) if bt_per_domain is None else min(bt_per_domain, len(chosen))
        started = time.time()
        for position, (_, source) in enumerate(chosen.iterrows()):
            for method in methods:
                if method == "backtranslation" and position >= bt_cap:
                    continue
                began = time.time()
                if method == "backtranslation":
                    text = rewriters[method].rewrite(source["text"])
                    model = rewriters[method].model_id
                elif method == "lexical_substitution":
                    text = rewriters[method].rewrite(source["text"])
                    model = rewriters[method].model_id
                elif method == "sentence_perturbation":
                    text = sentence_perturbation(source["text"], rng)
                    model = "rule-based"
                elif method == "punct_spacing_noise":
                    text = punct_spacing_noise(source["text"], rng)
                    model = "rule-based"
                elif method == "llm_paraphrase":
                    if llm is None:
                        continue
                    prompt_id = str(rng.choice(list(DEFAULT_LLM_TEMPLATES)))
                    text = llm.rewrite(source["text"], prompt_id)
                    model = llm.model
                else:
                    raise ValueError(f"unknown method {method!r}")
                if not text or text.strip() == source["text"].strip():
                    continue
                record = _row_record(source, text, method, model, prompt_id="" if method != "llm_paraphrase" else prompt_id)
                record["extra"] = json.dumps(
                    {
                        "source_doc_id": source["doc_id"],
                        "source_words": int(source["_words"]),
                        "seconds": round(time.time() - began, 2),
                    },
                    ensure_ascii=False,
                )
                frames.append(record)
        if frames:
            pd.DataFrame(frames).to_parquet(CHECKPOINT_REVISED, index=False)
        print(
            f"[revision] {domain}: {take} sources x {len(methods)} methods "
            f"(bt {bt_cap}) in {time.time() - started:.0f}s; total rows {len(frames)}",
            flush=True,
        )
    return pd.DataFrame(frames)


def run_attack(df: pd.DataFrame, methods: list[str], sample_limit: int, seed: int, llm=None) -> pd.DataFrame:
    import joblib

    splits = joblib.load(SPLITS)
    attack_idx = splits["paraphrase_attention__test"]
    if len(attack_idx) == 0:
        raise SystemExit("no reserved attack rows; run training/splits_v2.py first")
    rows = df.iloc[attack_idx].head(sample_limit)
    rng = np.random.default_rng(seed)
    frames: list[dict] = []
    rewriters: dict[str, object] = {}
    if "backtranslation" in methods:
        rewriters["backtranslation"] = BackTranslator()
    if "lexical_substitution" in methods:
        rewriters["lexical_substitution"] = MaskedLmSubstituter(seed=seed)
    for i, (_, source) in enumerate(rows.iterrows(), start=1):
        for method in methods:
            if method == "backtranslation":
                text = rewriters[method].rewrite(source["text"])
                model = rewriters[method].model_id
            elif method == "lexical_substitution":
                text = rewriters[method].rewrite(source["text"])
                model = rewriters[method].model_id
            elif method == "sentence_perturbation":
                text = sentence_perturbation(source["text"], rng)
                model = "rule-based"
            elif method == "punct_spacing_noise":
                text = punct_spacing_noise(source["text"], rng)
                model = "rule-based"
            elif method == "llm_paraphrase":
                if llm is None:
                    continue
                prompt_id = str(rng.choice(list(DEFAULT_LLM_TEMPLATES)))
                text = llm.rewrite(source["text"], prompt_id)
                model = llm.model
            else:
                raise ValueError(f"unknown method {method!r}")
            if not text:
                continue
            frames.append(_attack_record(source, text, method, model, ""))
        if i % 25 == 0:
            print(f"[attack] {i}/{len(rows)} documents rewritten", flush=True)
    return pd.DataFrame(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["revision", "attack"], required=True)
    ap.add_argument(
        "--methods",
        default="backtranslation,lexical_substitution,sentence_perturbation,punct_spacing_noise",
    )
    ap.add_argument("--per-domain", type=int, default=200, help="revision sources per human domain")
    ap.add_argument(
        "--bt-per-domain",
        type=int,
        default=0,
        help="cap on sources per domain that get the (expensive) round-trip MT; 0 = no cap",
    )
    ap.add_argument("--min-words", type=int, default=8, help="lower clip of the per-domain length window")
    ap.add_argument("--max-words", type=int, default=300, help="upper clip of the per-domain length window")
    ap.add_argument("--fresh", action="store_true", help="ignore the revision checkpoint and start over")
    ap.add_argument("--attack-limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--llm-base-url", default="")
    ap.add_argument("--llm-api-key", default="")
    ap.add_argument("--llm-model", default="")
    args = ap.parse_args()

    df = pd.read_parquet(DATASET)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    llm = None
    if args.llm_base_url and args.llm_api_key and args.llm_model:
        llm = LlmParaphraser(args.llm_base_url, args.llm_api_key, args.llm_model)
        if "llm_paraphrase" not in methods:
            methods.append("llm_paraphrase")

    started = time.time()
    if args.mode == "revision":
        if args.fresh and CHECKPOINT_REVISED.exists():
            CHECKPOINT_REVISED.unlink()
        out = run_revision(
            df,
            methods,
            args.per_domain,
            args.seed,
            llm,
            bt_per_domain=args.bt_per_domain or None,
            min_words=args.min_words,
            max_words=args.max_words,
        )
        target = OUT_REVISED
    else:
        out = run_attack(df, methods, args.attack_limit, args.seed, llm)
        target = OUT_ATTACK

    if out.empty:
        raise SystemExit("no rewritten rows were produced")
    out = out.reset_index(drop=True)
    out.to_parquet(target, index=False)
    if args.mode == "revision" and CHECKPOINT_REVISED.exists():
        CHECKPOINT_REVISED.unlink()

    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest[args.mode] = {
        "rows": int(len(out)),
        "methods": out["revision_type"].value_counts().to_dict(),
        "models": out["generator"].value_counts().to_dict(),
        "domains": out["domain"].value_counts().to_dict(),
        "seconds": round(time.time() - started, 1),
        "llm_used": bool(llm),
        "llm_pending_note": (
            "LLM-based paraphrase (five prompt templates) is implemented but was "
            "not run: no OpenAI-compatible endpoint was reachable from this host."
            if llm is None
            else "LLM paraphrase executed."
        ),
        "file": target.name,
        "file_sha256": ds.sha256_file(target),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if args.mode == "revision":
        manifest[args.mode]["source_words_window"] = {
            "rule": "per-domain p10-p90, clipped",
            "clip": [args.min_words, args.max_words],
        }
        manifest[args.mode]["bt_per_domain"] = args.bt_per_domain or None
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest[args.mode], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
