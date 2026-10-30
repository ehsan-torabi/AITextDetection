"""Build the metadata-complete, de-duplicated v2 dataset (review section 2).

Outputs
-------
data/dataset_v2.csv / .parquet   canonical frame (aitextd.dataset.COLUMNS)
data/dataset_v2.manifest.json    counts, hashes, dedup statistics, provenance

What it does, in order:

1. Loads the AI corpus with its full metadata (model, provider, prompt style,
   topic, sampling parameters) -- nothing about an AI row is invented.
2. Loads every human domain collected by training/collect_human_domains.py.
3. Exact de-duplication by content hash, then near-duplicate removal by 64-bit
   SimHash (Hamming <= 3, banded lookup). Cross-label near-duplicates (a human
   row that mirrors an AI row) are dropped from the human side and *counted*,
   because such pairs would leak across a train/test boundary later.
4. Derives the modelling-only columns: length bucket, marker-based formality,
   and a topic group -- the raw topic string for AI rows, a TF-IDF k-means
   cluster for human rows. Group-aware splitting needs these and they must be
   stable, so they are materialised here rather than recomputed per run.
5. Records v1 membership (`extra.v1_split`) by matching doc_id against the
   processedDataset.csv + split_indices.joblib pair, so the already-trained
   checkpoints can be evaluated on data they verifiably never saw.

Run from the project root:
    python training/build_dataset_v2.py [--kmeans-clusters 200]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
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

HUMAN_DIR = paths.DATA_DIR / "human"
DATASET = paths.DATA_DIR / "dataset_v2.parquet"
REVISED = paths.DATA_DIR / "ai_revised.parquet"
OUT_CSV = paths.DATA_DIR / "dataset_v2.csv"
OUT_PARQUET = paths.DATA_DIR / "dataset_v2.parquet"
MANIFEST = paths.DATA_DIR / "dataset_v2.manifest.json"

MIN_AI_WORDS = 20  # rows shorter than this are prompt echoes, not documents
NEAR_DUP_HAMMING = 3


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


# --- loading ----------------------------------------------------------------


def load_ai_rows() -> pd.DataFrame:
    raw = pd.read_csv(paths.RAW_AI_CSV, encoding="utf-8-sig")
    raw = raw.rename(columns={"generated_text": "text", "model": "generator"})
    raw["text"] = raw["text"].map(_collapse)
    raw = raw[raw["text"].map(lambda t: len(t.split())) >= MIN_AI_WORDS].copy()
    extra_cols = [
        c for c in ("temperature", "top_p", "char_count", "word_count") if c in raw.columns
    ]
    out = pd.DataFrame(
        {
            "doc_id": raw["text"].map(ds.stable_doc_id),
            "text": raw["text"],
            "label": ds.AI,
            "label_binary": 1,
            "domain": "ai_multi",
            "source": "persian_ai_generated_final.csv",
            "license": "apache-2.0 (released dataset)",
            "origin": "persian",
            "generator": raw["generator"].fillna("unknown").astype(str),
            "provider": raw.get("provider", pd.Series([""] * len(raw))).fillna("").astype(str),
            "prompt_id": raw.get("prompt_style", pd.Series([""] * len(raw))).fillna("").astype(str),
            "prompt_style": raw.get("prompt_style", pd.Series([""] * len(raw))).fillna("").astype(str),
            "topic_id": raw.get("topic", pd.Series([""] * len(raw))).fillna("").astype(str),
            "revision_type": "none",
            "paraphrase_method": "",
            "author_group": "llm",
            "non_native_flag": 0,
            "pii_scrubbed": 0,
            "extra": raw[extra_cols].apply(
                lambda row: json.dumps(row.to_dict(), ensure_ascii=False), axis=1
            ),
        }
    )
    return out


def load_revised_rows() -> tuple[pd.DataFrame | None, dict]:
    """Machine-revised human documents (the ai_revised class), if the
    paraphrase module has produced them.

    They are *derived* data: the text is rewritten from a human source document,
    so the row carries the source's domain and origin, the rewriter as
    ``generator``, and the method as ``revision_type``. Provenance is preserved
    in ``extra.source_doc_id``.
    """
    if not REVISED.exists():
        return None, {"present": False}
    frame = pd.read_parquet(REVISED)
    frame["text"] = frame["text"].map(_collapse)
    keep = [c for c in ds.COLUMNS if c in frame.columns]
    frame = frame[keep]
    provenance = {
        "present": True,
        "file": REVISED.name,
        "rows": int(len(frame)),
        "methods": frame["revision_type"].value_counts().to_dict(),
        "rewriters": frame["generator"].value_counts().to_dict(),
        "domains": frame["domain"].value_counts().to_dict(),
    }
    return frame, provenance


def load_human_rows() -> tuple[pd.DataFrame, dict[str, dict]]:
    frames = []
    provenance = {}
    for csv_path in sorted(HUMAN_DIR.glob("*.csv")):
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        frame["text"] = frame["text"].map(_collapse)
        frames.append(frame)
        provenance[csv_path.stem] = {"rows": int(len(frame)), "file": csv_path.name}
    if not frames:
        raise SystemExit(
            f"no human domain CSVs under {HUMAN_DIR}; run training/collect_human_domains.py first"
        )
    return pd.concat(frames, ignore_index=True), provenance


def v1_membership() -> dict[str, str]:
    """doc_id -> v1 split, for data the trained checkpoints already saw."""
    split = joblib.load(paths.SPLIT_INDICES_JOBLIB)
    processed = pd.read_csv(paths.PROCESSED_CSV)
    mapping = {}
    for name, idx in (("train", split["train_idx"]), ("test", split["test_idx"])):
        for text in processed.iloc[idx]["text"]:
            mapping[ds.stable_doc_id(_collapse(text))] = name
    return mapping


# --- de-duplication ---------------------------------------------------------


def _revision_sources(frame: pd.DataFrame) -> list[str]:
    """``extra.source_doc_id`` for every machine-revised row ('' elsewhere).

    The splits later make a revision inherit its source document's partition, so
    a revision is never a leak; the value is also what keeps the de-duplication
    pass from deleting a revision merely because it resembles the document it
    was rewritten from.
    """
    if "extra" not in frame.columns:
        return [""] * len(frame)
    out: list[str] = []
    for label, extra in zip(frame["label"], frame["extra"], strict=True):
        source = ""
        if label == ds.AI_REVISED and isinstance(extra, str) and extra:
            try:
                source = json.loads(extra).get("source_doc_id") or ""
            except json.JSONDecodeError:
                source = ""
        out.append(source)
    return out


def _near_dup_mask(texts: list[str], sources: list[str] | None = None) -> np.ndarray:
    """True where a row is a near-duplicate of an earlier row (same order).
    Banded SimHash: only compares hashes sharing a 16-bit band.

    When ``sources`` is given, two rows carrying the same non-empty source id
    are never treated as duplicates of each other: they are deliberate
    rewritings of one document (the same words reordered, or the same text with
    ZWNJ and punctuation noise), not redundant copies.
    """
    bands = 4
    seen: dict[int, list[int]] = {}
    dup = np.zeros(len(texts), dtype=bool)
    fingerprints: list[int] = []
    for i, text in enumerate(texts):
        h = ds.simhash64(text)
        fingerprints.append(h)
        candidates: list[int] = []
        for b in range(bands):
            candidates.extend(seen.get((b, (h >> (16 * b)) & 0xFFFF), []))
        hit = False
        for j in candidates:
            if sources is not None and sources[i] and sources[i] == sources[j]:
                continue
            if ds.hamming64(h, fingerprints[j]) <= NEAR_DUP_HAMMING:
                hit = True
                break
        if hit:
            dup[i] = True
            continue
        for b in range(bands):
            seen.setdefault((b, (h >> (16 * b)) & 0xFFFF), []).append(i)
    return dup


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict = {"input_rows": int(len(frame))}

    before = len(frame)
    frame = frame.drop_duplicates(subset="doc_id", keep="first").reset_index(drop=True)
    stats["exact_duplicates_removed"] = int(before - len(frame))

    # near-duplicates inside each (label, domain) block
    sources = _revision_sources(frame)
    removed_near = 0
    keep_mask = np.ones(len(frame), dtype=bool)
    for (label, domain), group in frame.groupby(["label", "domain"], sort=False):
        positions = group.index.to_numpy()
        group_sources = [sources[i] for i in positions] if label == ds.AI_REVISED else None
        dup_local = _near_dup_mask(group["text"].tolist(), group_sources)
        removed_near += int(dup_local.sum())
        keep_mask[positions[dup_local]] = False
    frame = frame[keep_mask].reset_index(drop=True)
    stats["near_duplicates_removed"] = removed_near

    # cross-label near-duplicates: a human row mirroring an AI row would leak
    ai = frame[frame["label"] == ds.AI]
    if len(ai):
        ai_hashes = ai["text"].map(ds.simhash64).tolist()
        buckets: dict[tuple[int, int], list[int]] = {}
        for i, h in enumerate(ai_hashes):
            for b in range(4):
                buckets.setdefault((b, (h >> (16 * b)) & 0xFFFF), []).append(i)
        drop_human = []
        for idx, row in frame[frame["label"] == ds.HUMAN].iterrows():
            h = ds.simhash64(row["text"])
            hit = False
            for b in range(4):
                for j in buckets.get((b, (h >> (16 * b)) & 0xFFFF), []):
                    if ds.hamming64(h, ai_hashes[j]) <= NEAR_DUP_HAMMING:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                drop_human.append(idx)
        frame = frame.drop(index=drop_human).reset_index(drop=True)
        stats["cross_label_near_duplicates_removed"] = len(drop_human)
    else:
        stats["cross_label_near_duplicates_removed"] = 0

    # cross-label near-duplicates involving machine-revised rows: the *derived*
    # row goes, never the primary one, and never when the primary row is the
    # document the revision was made from (that resemblance is the point of the
    # revised class, and the two always share a partition).
    revised = frame[frame["label"] == ds.AI_REVISED]
    if len(revised):
        primary = frame[frame["label"] != ds.AI_REVISED]
        primary_hashes = primary["text"].map(ds.simhash64).tolist()
        primary_ids = primary["doc_id"].tolist()
        buckets: dict[tuple[int, int], list[int]] = {}
        for i, h in enumerate(primary_hashes):
            for b in range(4):
                buckets.setdefault((b, (h >> (16 * b)) & 0xFFFF), []).append(i)
        sources = _revision_sources(frame)
        drop_revised = []
        for idx, row in revised.iterrows():
            source = sources[idx]
            h = ds.simhash64(row["text"])
            hit = False
            for b in range(4):
                for j in buckets.get((b, (h >> (16 * b)) & 0xFFFF), []):
                    if primary_ids[j] == source:
                        continue
                    if ds.hamming64(h, primary_hashes[j]) <= NEAR_DUP_HAMMING:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                drop_revised.append(idx)
        frame = frame.drop(index=drop_revised).reset_index(drop=True)
        stats["revised_cross_label_duplicates_removed"] = len(drop_revised)
    else:
        stats["revised_cross_label_duplicates_removed"] = 0
    kept = frame[frame["label"] == ds.AI_REVISED]
    if len(kept):
        per_source = Counter(_revision_sources(kept))
        stats["revised_rows_kept"] = int(len(kept))
        stats["revised_sources_kept"] = int(len(per_source))
        stats["revised_kept_per_source_mean"] = round(len(kept) / max(len(per_source), 1), 2)
        stats["revised_methods_kept"] = kept["revision_type"].value_counts().to_dict()
    stats["output_rows"] = int(len(frame))
    return frame, stats


# --- derived columns --------------------------------------------------------


def assign_human_topic_groups(frame: pd.DataFrame, k: int, seed: int = 42) -> pd.DataFrame:
    """Cluster human rows by TF-IDF topic *within each domain* so group-aware
    splitting can hold whole topics out. AI rows keep their raw topic string
    as the group. Clustering per domain (rather than globally) keeps group ids
    unique per block -- a global cluster id would appear in several domains and
    a per-block split could then place the same group in both train and test.

    ``ai_revised`` rows inherit the topic group of the document they were
    rewritten from: a revision is not an independent sample, and giving it a
    fresh group would let the source and its revision land in different splits.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    for domain in sorted(frame.loc[frame["label"] == ds.HUMAN, "domain"].unique()):
        mask = (frame["label"] == ds.HUMAN) & (frame["domain"] == domain)
        texts = frame.loc[mask, "text"]
        if len(texts) < 40:
            frame.loc[mask, "topic_id"] = [f"{domain}_cluster_0"] * len(texts)
            continue
        k_domain = int(min(k, max(2, len(texts) // 20)))
        vectorizer = TfidfVectorizer(max_features=40000, min_df=3, ngram_range=(1, 1))
        matrix = vectorizer.fit_transform(texts)
        km = MiniBatchKMeans(n_clusters=k_domain, random_state=seed, n_init=3, batch_size=1024)
        labels = km.fit_predict(matrix)
        frame.loc[mask, "topic_id"] = [f"{domain}_cluster_{c}" for c in labels]

    revised_mask = frame["label"] == ds.AI_REVISED
    if revised_mask.any():
        doc_to_topic = dict(zip(frame["doc_id"], frame["topic_id"], strict=True))
        inherited = 0
        for position in frame.index[revised_mask]:
            source_id = None
            try:
                source_id = json.loads(frame.at[position, "extra"] or "{}").get("source_doc_id")
            except (TypeError, json.JSONDecodeError):
                source_id = None
            topic = doc_to_topic.get(source_id)
            if topic:
                frame.at[position, "topic_id"] = topic
                inherited += 1
        missing = int(revised_mask.sum()) - inherited
    else:
        inherited = missing = 0
    frame.attrs["revised_topic_inheritance"] = {
        "revised_rows": int(revised_mask.sum()),
        "topic_inherited": inherited,
        "source_missing": missing,
    }
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kmeans-clusters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ai = load_ai_rows()
    human, human_provenance = load_human_rows()
    revised, revised_provenance = load_revised_rows()
    frame = pd.concat([ai, human, *( [revised] if revised is not None else [] )], ignore_index=True)

    frame, dedup_stats = deduplicate(frame)
    frame = assign_human_topic_groups(frame, args.kmeans_clusters, args.seed)

    frame["length_words"] = frame["text"].map(lambda t: len(t.split()))
    frame["length_bucket"] = frame["length_words"].map(ds.length_bucket)
    frame["formality"] = frame["text"].map(ds.formality_label)
    frame["split"] = ""

    v1 = v1_membership()
    frame["extra"] = [
        json.dumps(
            {
                **json.loads(extra or "{}"),
                **({"v1_split": v1[doc_id]} if doc_id in v1 else {}),
            },
            ensure_ascii=False,
        )
        for doc_id, extra in zip(frame["doc_id"], frame["extra"], strict=True)
    ]

    frame = frame[[c for c in ds.COLUMNS if c in frame.columns]]
    frame = frame.sort_values(["label", "domain", "doc_id"]).reset_index(drop=True)
    ds.validate_frame(frame)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    frame.to_parquet(OUT_PARQUET, index=False)

    by_label_domain = (
        frame.groupby(["label", "domain"]).size().unstack(fill_value=0).to_dict()
    )
    generators = (
        frame.loc[frame["label"] != ds.HUMAN, "generator"].value_counts().to_dict()
    )
    v1_split_counts = Counter(
        json.loads(extra).get("v1_split", "not_in_v1") for extra in frame["extra"]
    )
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(frame)),
        "label_counts": frame["label"].value_counts().to_dict(),
        "label_domain_counts": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in by_label_domain.items()},
        "origin_counts": frame["origin"].value_counts().to_dict(),
        "formality_counts": frame["formality"].value_counts().to_dict(),
        "length_bucket_counts": frame["length_bucket"].value_counts().to_dict(),
        "n_generators": int(len(generators)),
        "generators": generators,
        "prompt_style_counts": (
            frame.loc[frame["label"] == ds.AI, "prompt_style"].value_counts().to_dict()
        ),
        "topic_group_counts": {
            "ai_unique_topics": int(
                frame.loc[frame["label"] == ds.AI, "topic_id"].nunique()
            ),
            "human_clusters": int(
                frame.loc[frame["label"] == ds.HUMAN, "topic_id"].nunique()
            ),
        },
        "v1_membership": dict(v1_split_counts),
        "dedup": dedup_stats,
        "human_provenance": human_provenance,
        "revised_provenance": revised_provenance,
        "revised_topic_inheritance": frame.attrs.get("revised_topic_inheritance", {}),
        "files": {
            OUT_CSV.name: ds.sha256_file(OUT_CSV),
            OUT_PARQUET.name: ds.sha256_file(OUT_PARQUET),
        },
        "dataset_fingerprint": ds.dataset_fingerprint(frame),
        "builder": "training/build_dataset_v2.py",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "label_counts", "dedup")}, ensure_ascii=False, indent=2))
    print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
