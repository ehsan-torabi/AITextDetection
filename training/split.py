"""Shared train/test split used by every model trained after the original SVM.

Keeping one canonical split lets all newly trained models (classic ML and
deep learning alike) be compared on the exact same held-out test set.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from aitextd import paths

RANDOM_SEED = 42
TEST_SIZE = 0.2

DATASET_PATH = paths.PROCESSED_CSV
X_FEATURES_PATH = paths.TFIDF_X_JOBLIB
SPLIT_PATH = paths.SPLIT_INDICES_JOBLIB


def build_or_load_split():
    df = pd.read_csv(DATASET_PATH)
    y = df["aiGenerated"].to_numpy()

    idx = df.index.to_numpy()
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    joblib.dump({"train_idx": train_idx, "test_idx": test_idx}, SPLIT_PATH)
    return df, train_idx, test_idx


def load_split():
    df = pd.read_csv(DATASET_PATH)
    split = joblib.load(SPLIT_PATH)
    return df, split["train_idx"], split["test_idx"]


def load_tfidf_split():
    """Return (X_train, X_test, y_train, y_test, df, train_idx, test_idx)."""
    df, train_idx, test_idx = load_split()
    X = joblib.load(X_FEATURES_PATH)
    y = df["aiGenerated"].to_numpy()
    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        df,
        train_idx,
        test_idx,
    )


if __name__ == "__main__":
    df, train_idx, test_idx = build_or_load_split()
    print(f"Total samples: {len(df)}")
    print(f"Train: {len(train_idx)}  Test: {len(test_idx)}")
