"""Build the JEV-style ("System One") training data for the decision-head model.

JEV models don't generate answers; they answer *typed questions* about a
state in a single forward pass. Following the JEV question types (and the
open-source Valen implementation of the same idea), each dataset row becomes
a JSONL record with three typed questions about the text:

- ``source``        (choice) -> probability distribution over candidates
                               ["human_written", "ai_generated"]
- ``ai_level``      (score)  -> distribution over 5 ordinal levels
                               (definitely_human .. definitely_ai); at
                               inference the answer is the probability-weighted
                               value, which can fall between levels
- ``needs_review``  (noul)   -> 0..1 probability that a human must review

The dataset only carries a binary label, so soft targets are distilled from
the already-trained TF-IDF + Logistic Regression serving pipeline (the exact
preprocessing/style-feature code in utils.py):

- choice target: ground-truth one-hot blended with the teacher probability
- score target:  discretized Gaussian over the 5 levels centered at
                 4 * choice_target (teaches "between levels" answers)
- noul target:   1 where the teacher itself is uncertain (probability inside
                 NOUL_BAND) -- the only rows where a human reviewer genuinely
                 adds information

Run from the project root: python training/prepare_jev_data.py
Outputs: training/jev_data/jev_train.jsonl and jev_test.jsonl
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import utils  # noqa: E402  (serving pipeline: exact same preprocessing + style features)
from split import RANDOM_SEED, load_split  # noqa: E402

TOKENIZED_CACHE_PATH = PROJECT_ROOT / "training" / "tokenized_texts.joblib"
TEACHER_PATH = PROJECT_ROOT / "logregModel.joblib"
VECTORIZER_PATH = PROJECT_ROOT / "tfidf_vectorizer.joblib"
OUT_DIR = PROJECT_ROOT / "training" / "jev_data"

TRAIN_SUBSET = 6000  # CPU budget: mirrors training/train_gpt2_classifier.py
TEST_SUBSET = 1500

# JEV typed-question schema. Keep in sync with training/train_jev.py and the
# answer rendering in utils.py.
CHOICE_CANDIDATES = ["human_written", "ai_generated"]
CHOICE_CANDIDATES_FA = ["نوشته انسان", "تولیدشده توسط هوش مصنوعی"]
SCORE_LEVELS = [
    "definitely_human",
    "probably_human",
    "mixed_signals",
    "probably_ai",
    "definitely_ai",
]
SCORE_LEVELS_FA = [
    "قطعاً انسان",
    "احتمالاً انسان",
    "سیگنال‌های ترکیبی",
    "احتمالاً هوش مصنوعی",
    "قطعاً هوش مصنوعی",
]

QUESTIONS = {
    "source": {
        "type": "choice",
        "candidates": CHOICE_CANDIDATES,
        "candidatesFa": CHOICE_CANDIDATES_FA,
    },
    "ai_level": {
        "type": "score",
        "levels": SCORE_LEVELS,
        "levelsFa": SCORE_LEVELS_FA,
    },
    "needs_review": {"type": "noul"},
}

LABEL_WEIGHT = 0.8  # ground-truth one-hot weight in the choice target
TEACHER_WEIGHT = 0.2  # distillation weight from the teacher's probability
CHOICE_CLIP = (0.02, 0.98)
SCORE_SIGMA = 0.9  # spread of the score target around its center level
NOUL_BAND = (0.15, 0.85)  # teacher uncertainty band -> needs human review


def subsample(indices, labels, n, seed):
    if n >= len(indices):
        return indices
    sub, _ = train_test_split(
        indices, train_size=n, random_state=seed, stratify=labels[indices]
    )
    return sub


def cleaned_texts(texts, indices):
    """Serving-pipeline-preprocessed text for the given rows.

    The per-row token cache (training/preprocess_cache.py) holds exactly the
    output of the same shekar pipeline, cache[i] <-> df row i, so prefer it;
    fall back to running the pipeline when the cache is missing or stale."""
    try:
        cache = joblib.load(TOKENIZED_CACHE_PATH)
        if len(cache) == len(texts):
            return [" ".join(cache[i]) for i in indices]
        print(
            f"token cache length {len(cache)} != dataset length {len(texts)}; "
            "falling back to the shekar pipeline"
        )
    except FileNotFoundError:
        print("token cache not found; running the shekar pipeline (slow)")
    return Parallel(n_jobs=-1)(
        delayed(utils.preprocess_persian_text)(str(texts[i])) for i in indices
    )


# joblib/loky re-imports this module in each worker, so per-process globals
# start empty and the vectorizer is loaded lazily on the first document.
_WORKER_STATE = {}


def _style_worker(raw_text, cleaned_text):
    if "vectorizer" not in _WORKER_STATE:
        _WORKER_STATE["vectorizer"] = joblib.load(VECTORIZER_PATH)
        utils.vectorizer = _WORKER_STATE["vectorizer"]
    return utils.compute_style_features(raw_text, cleaned_text)


def choice_target(label, teacher_prob):
    """(target distribution over candidates, blended ai-probability)."""
    t_ai = LABEL_WEIGHT * float(label) + TEACHER_WEIGHT * float(teacher_prob)
    t_ai = min(max(t_ai, CHOICE_CLIP[0]), CHOICE_CLIP[1])
    return {"human_written": round(1.0 - t_ai, 4), "ai_generated": round(t_ai, 4)}, t_ai


def score_target(t_ai):
    """Discretized Gaussian over the ordinal levels, centered at 4 * t_ai."""
    center = (len(SCORE_LEVELS) - 1) * t_ai
    levels = np.arange(len(SCORE_LEVELS), dtype=float)
    probs = np.exp(-((levels - center) ** 2) / (2.0 * SCORE_SIGMA**2))
    probs /= probs.sum()
    return [round(float(p), 4) for p in probs]


def noul_target(teacher_prob):
    return 1.0 if NOUL_BAND[0] <= teacher_prob <= NOUL_BAND[1] else 0.0


def build_records(df, indices, split_name):
    texts = df["text"].to_numpy()
    y = df["aiGenerated"].to_numpy()

    cleaned = cleaned_texts(texts, indices)
    style_rows = Parallel(n_jobs=-1)(
        delayed(_style_worker)(str(texts[i]), cleaned[k])
        for k, i in enumerate(indices)
    )
    style_matrix = np.vstack(
        [[row[name] for name in utils.STYLE_FEATURE_NAMES] for row in style_rows]
    )

    vectorizer = joblib.load(VECTORIZER_PATH)
    teacher = joblib.load(TEACHER_PATH)
    features = hstack([vectorizer.transform(cleaned), style_matrix]).tocsr()
    p_teacher = teacher.predict_proba(features)[:, 1]

    records = []
    for k, i in enumerate(indices):
        source_t, t_ai = choice_target(y[i], p_teacher[k])
        records.append(
            {
                "id": f"{split_name}-{i:06d}",
                "split": split_name,
                "label": int(y[i]),
                "state": {"text": str(texts[i])},
                "questions": QUESTIONS,
                "targets": {
                    "source": source_t,
                    "ai_level": score_target(t_ai),
                    "needs_review": noul_target(p_teacher[k]),
                },
                "teacherAiProbability": round(float(p_teacher[k]), 4),
            }
        )
    return records


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(name, records):
    labels = [r["label"] for r in records]
    noul_pos = sum(r["targets"]["needs_review"] for r in records)
    ai_target = [
        r["targets"]["source"]["ai_generated"] for r in records if r["label"] == 1
    ]
    human_target = [
        r["targets"]["source"]["ai_generated"] for r in records if r["label"] == 0
    ]
    print(
        f"{name}: {len(records)} records | ai={sum(labels)} human={len(labels) - sum(labels)} "
        f"| needs_review targets={int(noul_pos)} | mean ai-target ai/human: "
        f"{np.mean(ai_target):.3f}/{np.mean(human_target):.3f}"
    )


def main():
    df, train_idx, test_idx = load_split()
    y = df["aiGenerated"].to_numpy()
    train_idx = subsample(train_idx, y, TRAIN_SUBSET, RANDOM_SEED)
    test_idx = subsample(test_idx, y, TEST_SUBSET, RANDOM_SEED)
    print(f"Building JEV records for {len(train_idx)} train / {len(test_idx)} test rows")

    train_records = build_records(df, train_idx, "train")
    test_records = build_records(df, test_idx, "test")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "jev_train.jsonl", train_records)
    write_jsonl(OUT_DIR / "jev_test.jsonl", test_records)

    summarize("train", train_records)
    summarize("test", test_records)
    print(f"Wrote {OUT_DIR / 'jev_train.jsonl'} and {OUT_DIR / 'jev_test.jsonl'}")
    print("Example record (text truncated):")
    example = dict(train_records[0])
    example["state"] = {"text": str(example["state"]["text"])[:120] + "..."}
    print(json.dumps(example, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
