"""Cache lemmatized/tokenized text for the whole dataset so deep-learning
training scripts don't need to re-run the (slow) shekar preprocessing
pipeline every time.

Run from the project root: python training/preprocess_cache.py
"""

import multiprocessing
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from aitextd import paths
from aitextd.text import tokenize_persian_text

DATASET_PATH = paths.PROCESSED_CSV
CACHE_PATH = paths.TOKENIZED_TEXTS_JOBLIB


def build_or_load_cache():
    try:
        return joblib.load(CACHE_PATH)
    except FileNotFoundError:
        pass

    df = pd.read_csv(DATASET_PATH)
    num_cores = multiprocessing.cpu_count()
    print(f"Tokenizing {len(df)} documents on {num_cores} cores...")
    tokenized = Parallel(n_jobs=num_cores)(
        delayed(tokenize_persian_text)(text) for text in tqdm(df["text"])
    )
    joblib.dump(tokenized, CACHE_PATH)
    return tokenized


if __name__ == "__main__":
    tokenized = build_or_load_cache()
    print(f"Cached {len(tokenized)} tokenized documents to {CACHE_PATH}")
    print("Example:", tokenized[0][:15])
