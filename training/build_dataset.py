"""Build the balanced processedDataset.csv from the two raw source files.

This used to live in the legacy main.ipynb (the original one-off SVM
experiment); it is preserved here as the canonical dataset builder. It:

1. reads persian_ai_generated_final.csv (AI-generated Persian text) and
   blogs.csv (human-written Persian blogs),
2. subsamples the human rows to match the AI row count (a balanced set),
3. maps them to (text, aiGenerated) rows, shuffles, and writes
   data/processedDataset.csv.

Run from the project root: python training/build_dataset.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from aitextd import paths


def main():
    ai_df = pd.read_csv(paths.RAW_AI_CSV, encoding="utf-8-sig")
    human_df = pd.read_csv(paths.RAW_HUMAN_CSV, encoding="utf-8-sig")
    print(f"AI rows: {len(ai_df)}  human rows: {len(human_df)}")

    human_df = human_df.sample(len(ai_df), random_state=42, ignore_index=True)

    cols = ("text", "aiGenerated")
    ai_part = pd.DataFrame(columns=cols)
    ai_part["text"] = ai_df["generated_text"]
    ai_part["aiGenerated"] = 1
    human_part = pd.DataFrame(columns=cols)
    human_part["text"] = human_df["text"]
    human_part["aiGenerated"] = 0

    final = pd.concat([ai_part, human_part]).sample(frac=1, ignore_index=True)
    assert final["text"].isna().sum() == 0, "unexpected NaN texts after the join"

    paths.PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(paths.PROCESSED_CSV, encoding="utf-8-sig", index=False)
    print(f"Wrote {len(final)} rows to {paths.PROCESSED_CSV}")


if __name__ == "__main__":
    main()
