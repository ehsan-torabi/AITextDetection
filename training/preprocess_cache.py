"""Cache lemmatized/tokenized text for the whole dataset so deep-learning
training scripts don't need to re-run the (slow) shekar preprocessing
pipeline every time.

Run from the project root: python training/preprocess_cache.py
"""

import multiprocessing
import re

import joblib
import pandas as pd
from joblib import Parallel, delayed
from shekar import Lemmatizer, Normalizer, WordTokenizer
from shekar.preprocessing import EmojiRemover, NonPersianRemover, PunctuationRemover
from tqdm import tqdm

DATASET_PATH = "processedDataset.csv"
CACHE_PATH = "training/tokenized_texts.joblib"

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = (
            EmojiRemover()
            | PunctuationRemover()
            | NonPersianRemover()
            | Normalizer()
            | Lemmatizer()
            | WordTokenizer()
        )
    return _pipeline


def tokenize(text: str) -> list[str]:
    text = re.sub(r"\n", " ", str(text))
    return list(_get_pipeline()(text))


def build_or_load_cache():
    try:
        return joblib.load(CACHE_PATH)
    except FileNotFoundError:
        pass

    df = pd.read_csv(DATASET_PATH)
    num_cores = multiprocessing.cpu_count()
    print(f"Tokenizing {len(df)} documents on {num_cores} cores...")
    tokenized = Parallel(n_jobs=num_cores)(delayed(tokenize)(text) for text in tqdm(df["text"]))
    joblib.dump(tokenized, CACHE_PATH)
    return tokenized


if __name__ == "__main__":
    tokenized = build_or_load_cache()
    print(f"Cached {len(tokenized)} tokenized documents to {CACHE_PATH}")
    print("Example:", tokenized[0][:15])
