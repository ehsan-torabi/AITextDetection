"""Build the style-feature matrix for the whole dataset, and refresh the
training feature matrix (TF-IDF columns + style columns) that split.py and
train_ml_models.py load from data/tfidf_vectorizer_X.joblib.

Column order comes from aitextd.features.STYLE_FEATURE_NAMES -- the exact
list serving uses -- so train/serve column order cannot drift.

Run from the project root (after training/preprocess_cache.py):
    python training/build_style_features.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from tqdm import tqdm

from aitextd import paths
from aitextd.features import (
    STYLE_FEATURE_NAMES,
    STYLE_FEATURES_VERSION,
    compute_style_features,
)


def build_style_features(texts, cleaned_texts, vectorizer, cache_path=paths.STYLE_FEATURES_JOBLIB):
    """(n_docs, len(STYLE_FEATURE_NAMES)) float matrix, cached to disk.

    Single-process on purpose: compute_style_features shares one fitted
    vectorizer, and pickling it per worker costs more than the parallelism
    saves at this dataset size. The cache is only trusted when its version
    and shape match.
    ponytail: joblib workers with a forked vectorizer past ~100k rows.
    """
    n = len(texts)
    width = len(STYLE_FEATURE_NAMES)
    try:
        payload = joblib.load(cache_path)
        matrix = payload["matrix"]
        if payload["version"] == STYLE_FEATURES_VERSION and matrix.shape == (n, width):
            return matrix
    except (FileNotFoundError, KeyError, TypeError, IndexError):
        pass
    rows = []
    for text, cleaned in tqdm(
        zip(texts, cleaned_texts, strict=True), total=n, desc="style features"
    ):
        feats = compute_style_features(str(text), cleaned, vectorizer)
        rows.append([feats[name] for name in STYLE_FEATURE_NAMES])
    matrix = np.asarray(rows, dtype=np.float64)
    joblib.dump({"version": STYLE_FEATURES_VERSION, "matrix": matrix}, cache_path)
    return matrix


def main():
    df = pd.read_csv(paths.PROCESSED_CSV)
    tokenized = joblib.load(paths.TOKENIZED_TEXTS_JOBLIB)
    assert len(tokenized) == len(df), (
        "tokenized cache out of date; rerun training/preprocess_cache.py"
    )
    cleaned_texts = [" ".join(tokens) for tokens in tokenized]

    vectorizer = joblib.load(paths.TFIDF_VECTORIZER_JOBLIB)
    style = build_style_features(df["text"].astype(str).tolist(), cleaned_texts, vectorizer)
    tfidf = vectorizer.transform(tqdm(cleaned_texts, desc="tfidf"))
    X = hstack([tfidf, csr_matrix(style)]).tocsr()
    joblib.dump(X, paths.TFIDF_X_JOBLIB)

    print(f"Wrote style matrix {style.shape} -> {paths.STYLE_FEATURES_JOBLIB}")
    print(f"Wrote feature matrix {X.shape} -> {paths.TFIDF_X_JOBLIB}")


if __name__ == "__main__":
    main()
