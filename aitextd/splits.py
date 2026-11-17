"""Canonical split and the leak-free evaluation sets, in one place.

Two baselines, both views of dataset_v2.parquet:

- canonical_frames(): the 70/10/20 doc_id split written by
  notebooks/01_data_preparation.ipynb (StratifiedGroupKFold, stratified on
  label, group-disjoint on topic_id). Use it to train and evaluate NEW models.

- v1_linked_dataset() / deployed_heldout_frame(): the same rows annotated
  with the v1 corpus split (processedDataset.csv) the DEPLOYED checkpoints
  were trained on. Deployed models must only be scored on rows whose
  v1_split == "test" -- their v1_split == "train" rows are verbatim training
  documents, so scoring them would inflate every metric.

doc_id is a stable content hash of the whitespace-collapsed text, so the v1
link survives row reordering and the near-duplicate rows inside v1 collapse
onto one id (kept as "train", the conservative side).
"""

from __future__ import annotations

import hashlib
import json

import joblib
import pandas as pd

from aitextd import paths

SPLIT_JSON = paths.DATA_DIR / "train_val_test_split.json"

_v1_membership: pd.DataFrame | None = None


def dataset() -> pd.DataFrame:
    """The labelled v2 corpus in stable parquet order."""
    return pd.read_parquet(paths.DATA_DIR / "dataset_v2.parquet").reset_index(drop=True)


def canonical_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, val, test) frames of the canonical 70/10/20 split."""
    splits = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    df = dataset()
    return tuple(df[df["doc_id"].isin(splits[name])].reset_index(drop=True) for name in ("train", "val", "test"))


def _stable_doc_id(text) -> str:
    """Same content hash the v2 builders used: blake2b of collapsed text."""
    collapsed = " ".join(str(text).split())
    return hashlib.blake2b(collapsed.encode("utf-8"), digest_size=16).hexdigest()


def v1_linked_dataset() -> pd.DataFrame:
    """dataset() plus v1_split ('train' / 'test' / '' when unseen) and v1_row
    (row index into processedDataset.csv and tokenized_texts.joblib, -1 when
    unseen). Cached: hashing the 2.2 GB pipeline's rows runs once per process."""
    global _v1_membership
    if _v1_membership is None:
        processed = pd.read_csv(paths.PROCESSED_CSV, usecols=["text"])
        indices = joblib.load(paths.DATA_DIR / "split_indices.joblib")
        rows = [
            (_stable_doc_id(processed.iat[int(row), 0]), name, int(row))
            for name in ("train", "test")
            for row in indices[f"{name}_idx"]
        ]
        _v1_membership = pd.DataFrame(rows, columns=["doc_id", "v1_split", "v1_row"]).drop_duplicates(
            "doc_id"
        )
    linked = dataset().merge(_v1_membership, on="doc_id", how="left")
    linked["v1_split"] = linked["v1_split"].fillna("")
    linked["v1_row"] = linked["v1_row"].fillna(-1).astype(int)
    return linked


def deployed_heldout_frame() -> pd.DataFrame:
    """Leak-free evaluation frame for the deployed (v1-trained) checkpoints:
    every v2 row whose content sits in the v1 held-out test split."""
    linked = v1_linked_dataset()
    return linked[linked["v1_split"] == "test"].reset_index(drop=True)


def v1_tokens() -> list[list[str]]:
    """Tokenized v1 documents (cached), aligned with processedDataset rows --
    index it with v1_row."""
    return joblib.load(paths.DATA_DIR / "tokenized_texts.joblib")
