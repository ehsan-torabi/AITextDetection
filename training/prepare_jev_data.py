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
the deployed fine-tuned GPT-2 classifier (gpt2ClassifierModel.pt on the
gpt2-fa.mdl backbone) -- the teacher the ensemble already weights highest.
It is run through the exact serving path in utils.py: same normalization
(_normalize_for_gpt2), tokenizer, char-budget guard and seq_len.

- choice target: ground-truth one-hot blended with the teacher probability
- score target:  discretized Gaussian over the 5 levels centered at
                 4 * choice_target (teaches "between levels" answers)
- noul target:   1 where the teacher itself is uncertain (probability inside
                 NOUL_BAND) -- the only rows where a human reviewer genuinely
                 adds information

Records store the raw text in ``state.text``; training/train_jev.py applies
the same normalization to it that both the teacher and the serving path use.

Run from the project root: python training/prepare_jev_data.py
Outputs: training/jev_data/jev_train.jsonl and jev_test.jsonl
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import utils  # noqa: E402  (serving pipeline: teacher runs the exact serving path)
from split import RANDOM_SEED, load_split  # noqa: E402

TEACHER_CHECKPOINT_PATH = PROJECT_ROOT / "gpt2ClassifierModel.pt"
OUT_DIR = PROJECT_ROOT / "training" / "jev_data"

TRAIN_SUBSET = 6000  # CPU budget: mirrors training/train_gpt2_classifier.py
TEST_SUBSET = 1500
TEACHER_BATCH_SIZE = 32
torch.set_num_threads(12)

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
    sub, _ = train_test_split(indices, train_size=n, random_state=seed, stratify=labels[indices])
    return sub


def batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def build_teacher():
    """The deployed fine-tuned GPT-2 classifier, rebuilt exactly as
    utils._load_gpt2_model does: base backbone from the local pickle, then the
    fine-tuned weights on top from the serving checkpoint."""
    backbone = utils._load_gpt2_backbone()
    checkpoint = torch.load(TEACHER_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = utils.GPT2Classifier(backbone, hidden_size=config["hidden_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


def teacher_probabilities(model, config, raw_texts):
    """P(AI) from the deployed GPT-2 classifier for each raw text, via the
    exact serving path of utils._predict_gpt2: same normalization, tokenizer,
    char-budget guard and seq_len, so the distilled targets match what the
    teacher actually outputs in production."""
    tokenizer = utils._get_gpt2_tokenizer()
    seq_len = config["seq_len"]
    char_budget = seq_len * 8  # same slow-tokenizer guard as the serving path

    n_batches = (len(raw_texts) + TEACHER_BATCH_SIZE - 1) // TEACHER_BATCH_SIZE
    probs = []
    start = time.time()
    with torch.no_grad():
        for step, batch in enumerate(batches(raw_texts, TEACHER_BATCH_SIZE), start=1):
            enc = tokenizer(
                [utils._normalize_for_gpt2(str(t))[:char_budget] for t in batch],
                return_tensors="pt",
                truncation=True,
                max_length=seq_len,
                padding="max_length",
            )
            logits = model(enc["input_ids"], enc["attention_mask"])
            probs.append(torch.sigmoid(logits).numpy())
            if step % 20 == 0 or step == n_batches:
                elapsed = time.time() - start
                rate = elapsed / step
                remaining = rate * (n_batches - step)
                print(
                    f"teacher batch {step}/{n_batches} elapsed={elapsed:.0f}s eta={remaining:.0f}s",
                    flush=True,
                )
    return np.concatenate(probs)


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


def build_records(df, indices, split_name, teacher, teacher_config):
    texts = df["text"].to_numpy()
    y = df["aiGenerated"].to_numpy()

    p_teacher = teacher_probabilities(teacher, teacher_config, texts[indices])

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
                "teacher": "gpt2_transformer",
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
    ai_target = [r["targets"]["source"]["ai_generated"] for r in records if r["label"] == 1]
    human_target = [r["targets"]["source"]["ai_generated"] for r in records if r["label"] == 0]
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

    print("Loading the fine-tuned GPT-2 teacher (gpt2-fa.mdl + gpt2ClassifierModel.pt)...")
    teacher, teacher_config = build_teacher()

    train_records = build_records(df, train_idx, "train", teacher, teacher_config)
    test_records = build_records(df, test_idx, "test", teacher, teacher_config)

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
