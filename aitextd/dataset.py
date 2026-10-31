"""Canonical dataset schema for the rigorous (v2) evaluation layer.

The v1 pipeline uses a two-column frame (`text`, `aiGenerated`). That is not
enough to satisfy the review-driven requirements: every sample must carry its
label, domain, source, generator, prompt, revision type, formality, length,
author group and split, so that group-aware splitting, cross-domain and
cross-generator evaluation, and honest error analysis are all possible.

This module is the single place where those columns, their allowed values and
the Persian-specific text statistics (half-space / diacritic / informality)
are defined. Both the dataset builders under training/ and the evaluation
suite import from here, so the schema cannot drift between them.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# --- labels -----------------------------------------------------------------

HUMAN = "human"
AI = "ai"
AI_REVISED = "ai_revised"
LABELS = (HUMAN, AI, AI_REVISED)

# Canonical (publication) domains. `ai_multi` is the AI corpus itself: it is
# generated across many topics rather than belonging to one human domain.
DOMAINS = (
    "blog",
    "news",
    "encyclopedia",
    "social_media",
    "legal_admin",
    "translated_subtitles",
    "ai_multi",
)

# Persian-original vs translated-into-Persian: required to be labelled
# separately so false positives on translated text can be measured.
ORIGINS = ("persian", "translated")

# How an ai_revised row was produced. "none" = not a revised row.
REVISION_TYPES = (
    "none",
    "backtranslation",
    "lexical_substitution",
    "sentence_perturbation",
    "punct_spacing_noise",
    "llm_paraphrase",
    "llm_summarize",
    "llm_expand",
    "llm_formalize",
    "llm_humanize",
    "human_edit",
)

FORMALITY = ("formal", "neutral", "informal")
LENGTH_BUCKETS = ("short", "medium", "long")

# --- canonical columns ------------------------------------------------------

COLUMNS = [
    "doc_id",  # stable content hash (blake2b-16 of the raw text)
    "text",  # raw text (never mutated in place)
    "label",  # human | ai | ai_revised
    "label_binary",  # 1 if label != human else 0 (secondary formulation)
    "domain",  # one of DOMAINS
    "source",  # corpus identifier (raw file / HF dataset id)
    "license",  # distribution license of the originating corpus
    "origin",  # persian | translated
    "generator",  # LLM id, or "human", or the rewriter id for ai_revised
    "provider",  # API provider of the generator ("" for human text)
    "prompt_id",  # prompt/style identifier for AI rows ("", "none")
    "prompt_style",  # human-readable style bucket where known
    "topic_id",  # topic string (AI) or cluster id (human)
    "revision_type",  # REVISION_TYPES
    "paraphrase_method",  # method id for attack/augmentation rows
    "formality",  # formal | neutral | informal (marker-based estimate)
    "length_words",  # whitespace word count
    "length_bucket",  # short | medium | long
    "author_group",  # human cohort (blogger / journalist / ...)
    "non_native_flag",  # 1 when the source is known to be non-native prose
    "pii_scrubbed",  # 1 when a PII scrubber ran on the raw text
    "split",  # assigned by training/splits_v2.py
    "extra",  # JSON blob for source-specific leftovers
]

REQUIRED = [c for c in COLUMNS if c not in ("split", "extra")]

# --- Persian text statistics ------------------------------------------------

ZWNJ = "\u200c"  # نیم‌فاصله
_DIACRITICS = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0653\u0654\u0655\u0670"
_DIACRITIC_RE = re.compile(f"[{_DIACRITICS}]")
_DIGIT_RE = re.compile(r"[0-9\u06f0-\u06f9\u0660-\u0669]")
_WORD_RE = re.compile(r"[^\s\u200c]+")

# Small, auditable marker sets. Deliberately colloquial/spelling-level: the
# goal is register, not topic. Matching happens on a ZWNJ-stripped copy of a
# normalised string (see _marker_text), so «می‌شود» and «میشود» both count.
FORMAL_MARKERS = (
    "میباشد",
    "میگردد",
    "گردید",
    "لذا",
    "بدینوسیله",
    "خواهشمند",
    "حسب",
    "وفق",
    "مدنظر",
    "مسبوق",
    "بعنوان",
    "درخصوص",
    "مقتضی",
    "فوق‌الذکر",
    "مع‌الوصف",
    "علیهذا",
)
INFORMAL_STRONG_MARKERS = (
    "میشه",
    "نمیشه",
    "چیه",
    "واسه",
    "چیکار",
    "داداش",
    "بابا",
    "چطوره",
    "اینطوری",
    "همینجوری",
    "بچه‌ها",
    "دیگه",
)
INFORMAL_WEAK_MARKERS = ("یه", "الان", "بعدا")
INFORMAL_MARKERS = INFORMAL_STRONG_MARKERS + INFORMAL_WEAK_MARKERS


def _marker_text(text: str) -> str:
    """Lower-noise view for marker counting: ZWNJ dropped, Arabic letters
    unified, whitespace collapsed. Keeps word boundaries intact."""
    text = text.replace(ZWNJ, "")
    text = text.replace("ي", "ی").replace("ك", "ک")
    return re.sub(r"\s+", " ", text)


def stable_doc_id(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def normalize_for_dedup(text: str) -> str:
    """Aggressive normalisation used only for duplicate detection, never for
    modelling: it removes ZWNJ/punctuation/digits/spacing differences so that
    two copies of the same text that differ only in typing noise collide."""
    text = _marker_text(text)
    text = _DIACRITIC_RE.sub("", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"[\d\u06f0-\u06f9\u0660-\u0669]+", "0", text)
    return re.sub(r"\s+", " ", text).strip()


def simhash64(text: str, shingle_words: int = 4) -> int:
    """64-bit SimHash over word shingles, for near-duplicate detection.

    Pure-Python and deterministic; adequate for the tens of thousands of rows
    this project handles (it is O(tokens) per document).
    """
    words = normalize_for_dedup(text).split()
    if len(words) < shingle_words:
        shingles = [" ".join(words)] if words else []
    else:
        shingles = [
            " ".join(words[i : i + shingle_words]) for i in range(len(words) - shingle_words + 1)
        ]
    if not shingles:
        return 0
    vector = [0] * 64
    for shingle in shingles:
        h = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if (h >> bit) & 1 else -1
    out = 0
    for bit in range(64):
        if vector[bit] > 0:
            out |= 1 << bit
    return out


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def zwnj_rate(text: str) -> float:
    """Half-space characters per word token -- a Persian typing-style signal."""
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    return text.count(ZWNJ) / len(words)


def diacritic_rate(text: str) -> float:
    """Diacritics per character. Diacritics are stylistic (recitation, formal
    orthography), so they are counted rather than stripped everywhere."""
    if not text:
        return 0.0
    return len(_DIACRITIC_RE.findall(text)) / len(text)


def digit_rate(text: str) -> float:
    if not text:
        return 0.0
    return len(_DIGIT_RE.findall(text)) / len(text)


def marker_rates(text: str) -> tuple[float, float, float]:
    """(formal_rate, informal_strong_rate, informal_weak_rate), per word."""
    marked = _marker_text(text)
    n_words = max(len(marked.split()), 1)
    formal = sum(marked.count(m) for m in FORMAL_MARKERS) / n_words
    strong = sum(marked.count(m) for m in INFORMAL_STRONG_MARKERS) / n_words
    weak = sum(marked.count(m) for m in INFORMAL_WEAK_MARKERS) / n_words
    return formal, strong, weak


def formality_label(text: str) -> str:
    """Marker-based formality estimate. Explicitly a weak label: colloquial
    spelling and formal vocabulary are the only signals used, so it is used
    for slicing error analysis, never as ground truth. Thresholds are set so
    that a *strong* colloquial marker («واسه», «چیه», ...) is required before
    a document is called informal; «یه»/«الان» alone are too common to carry
    the decision (they are counted in the weak bucket instead)."""
    formal, strong, weak = marker_rates(text)
    if strong >= 0.004 or (strong > 0 and strong > 2 * formal):
        return "informal"
    if formal >= 0.004 and formal > strong:
        return "formal"
    if weak == 0 and formal > 0:
        return "formal"
    return "neutral"


def text_statistics(text: str) -> dict:
    words = _WORD_RE.findall(text)
    sents = [s for s in re.split(r"[.!?؟…\n]+", text) if s.strip()]
    sent_lengths = [len(_WORD_RE.findall(s)) for s in sents]
    hapax = sum(1 for _, c in Counter(_WORD_RE.findall(_marker_text(text))).items() if c == 1)
    formal, strong, weak = marker_rates(text)
    return {
        "length_chars": len(text),
        "length_words": len(words),
        "n_sentences": len(sents),
        "mean_sentence_words": (sum(sent_lengths) / len(sent_lengths)) if sent_lengths else 0.0,
        "p90_sentence_words": (
            float(pd.Series(sent_lengths).quantile(0.9)) if sent_lengths else 0.0
        ),
        "zwnj_rate": zwnj_rate(text),
        "diacritic_rate": diacritic_rate(text),
        "digit_rate": digit_rate(text),
        "formal_marker_rate": formal,
        "informal_marker_rate": strong + weak,
        "informal_strong_marker_rate": strong,
        "informal_weak_marker_rate": weak,
        "hapax_ratio": (hapax / len(words)) if words else 0.0,
        "avg_word_len": (sum(len(w) for w in words) / len(words) if words else 0.0),
    }


def length_bucket(n_words: int, short_max: int = 100, long_min: int = 500) -> str:
    if n_words <= short_max:
        return "short"
    if n_words >= long_min:
        return "long"
    return "medium"


def extended_style_features(text: str) -> dict:
    """The Persian-specific style statistics added by the v2 layer to the ten
    served features in aitextd.features (which stay untouched so the trained
    serving checkpoints keep working):

    zwnj_rate, diacritic_rate, digit_rate, formal_marker_rate,
    informal_marker_rate, hapax_ratio, avg_word_len, p90_sentence_words,
    mean_sentence_words.

    POS sequences and dependency depth are *not* computable here: this host
    has no offline Persian POS tagger / dependency parser, and the report
    records that gap rather than silently substituting a proxy.
    """
    stats = text_statistics(text)
    return {
        key: stats[key]
        for key in (
            "zwnj_rate",
            "diacritic_rate",
            "digit_rate",
            "formal_marker_rate",
            "informal_marker_rate",
            "informal_strong_marker_rate",
            "hapax_ratio",
            "avg_word_len",
            "p90_sentence_words",
            "mean_sentence_words",
        )
    }


# --- validation -------------------------------------------------------------


class SchemaError(ValueError):
    """Raised when a dataset frame violates the canonical schema."""


def validate_frame(df: pd.DataFrame, require_split: bool = False) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if require_split and "split" not in df.columns:
        missing.append("split")
    if missing:
        raise SchemaError(f"missing columns: {missing}")

    def _check(column: str, allowed) -> None:
        bad = sorted(set(df[column].dropna().unique()) - set(allowed))
        if bad:
            raise SchemaError(f"column {column!r} has illegal values: {bad[:8]}")

    _check("label", LABELS)
    _check("domain", DOMAINS)
    _check("origin", ORIGINS)
    _check("revision_type", REVISION_TYPES)
    _check("formality", FORMALITY)
    _check("length_bucket", LENGTH_BUCKETS)

    if df["doc_id"].duplicated().any():
        dupes = df.loc[df["doc_id"].duplicated(), "doc_id"].head(5).tolist()
        raise SchemaError(f"duplicate doc_id values: {dupes}")
    if (df["label_binary"].astype(int) != (df["label"] != HUMAN).astype(int)).any():
        raise SchemaError("label_binary disagrees with label")
    return df


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Content fingerprint of the labelled data (ignores split assignment so
    the same data can be re-split reproducibly with a different seed)."""
    h = hashlib.sha256()
    for doc_id, label, domain in sorted(zip(df["doc_id"], df["label"], df["domain"], strict=True)):
        h.update(f"{doc_id}|{label}|{domain}\n".encode())
    return h.hexdigest()
