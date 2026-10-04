"""Shared train/test split used by every model trained after the original SVM.

Keeping one canonical split lets all newly trained models (classic ML and
deep learning alike) be compared on the exact same held-out test set.
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TEST_SIZE = 0.2

DATASET_PATH = "processedDataset.csv"
X_FEATURES_PATH = "tfidf_vectorizer_X.joblib"
SPLIT_PATH = "training/split_indices.joblib"


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
