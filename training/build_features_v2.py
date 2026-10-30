"""Persian style-feature matrix for the v2 evaluation layer.

Builds one row per dataset_v2 document with:

- the ten served stylometric features (aitextd.features) so the v2 experiments
  measure the same feature block that is deployed -- `coherence` is computed
  with a TF-IDF vectorizer fitted on the text, as in training/build_style_features.py;
- the v2 additions required by the review: ZWNJ rate, diacritic rate, digit
  rate, formal / informal (strong + weak) marker rates, hapax ratio, average
  word length, p90 / mean sentence length (aitextd.dataset.extended_style_features).

Deliberately NOT here: POS-sequence and dependency-depth features. There is no
offline Persian POS tagger or dependency parser on this host; the report lists
that as a gap instead of substituting a proxy that would not mean the same
thing. Perplexity/mean log-likelihood is a separate signal (training/eval_perplexity.py).

Output: data/style_features_v2.parquet (+ manifest row in the dataset manifest)
Run from the project root:  python training/build_features_v2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from aitextd import dataset as ds
from aitextd import paths
from aitextd.features import STYLE_FEATURE_NAMES, compute_style_features

DATASET = paths.DATA_DIR / "dataset_v2.parquet"
OUT = paths.DATA_DIR / "style_features_v2.parquet"

EXTENDED_NAMES = [
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
]


def main() -> None:
    started = time.time()
    df = pd.read_parquet(DATASET)
    texts = df["text"].tolist()

    # A small TF-IDF space, used only to compute the coherence feature.
    print("fitting coherence vectorizer ...", flush=True)
    vectorizer = TfidfVectorizer(max_features=30000, min_df=5, ngram_range=(1, 1))
    vectorizer.fit(texts)

    rows = []
    for i, text in enumerate(texts, start=1):
        base = compute_style_features(text, text, vectorizer)
        extended = ds.extended_style_features(text)
        rows.append({**base, **extended})
        if i % 2000 == 0:
            print(f"  {i}/{len(texts)} documents ({time.time() - started:.0f}s)", flush=True)

    features = pd.DataFrame(rows, columns=[*STYLE_FEATURE_NAMES, *EXTENDED_NAMES])
    features.insert(0, "doc_id", df["doc_id"].to_numpy())
    features.to_parquet(OUT, index=False)

    manifest_path = paths.DATA_DIR / "dataset_v2.manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["style_features_v2"] = {
            "file": OUT.name,
            "rows": int(len(features)),
            "columns": list(features.columns[1:]),
            "sha256": ds.sha256_file(OUT),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"wrote {OUT} with {len(features)} rows, {features.shape[1] - 1} features "
          f"({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
