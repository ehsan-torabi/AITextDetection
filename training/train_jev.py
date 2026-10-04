"""Train the JEV-style "System One" decision model on the Persian dataset.

Architecture (following JEV / the open-source Valen implementation): a GPT-2
backbone whose pooled representation feeds three small *decision heads*, one
per typed question, all scored in a single forward pass instead of
autoregressive answer generation:

- choice head -> probability distribution over ["human_written", "ai_generated"]
- score head  -> distribution over 5 ordinal machine-likeness levels; the
                 answer is the probability-weighted value (can fall between levels)
- noul head   -> probability for the yes/no "needs human review" proposition

Training data comes from training/jev_data/jev_{train,test}.jsonl (built by
training/prepare_jev_data.py) with soft targets distilled from the TF-IDF +
Logistic Regression serving pipeline. Loss = weighted sum of soft
cross-entropy on the two distributions + binary cross-entropy on the noul head.

Same CPU budget as training/train_gpt2_classifier.py: only the last 4 of 12
transformer blocks + the final LayerNorm are unfrozen, SEQ_LEN=64, one epoch
over a 6000-sample stratified subsample.

Run from the project root: python training/train_jev.py
"""

import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoTokenizer

from split import RANDOM_SEED

MODELS_JSON_PATH = "models.json"
MODEL_PATH = "jevModel.pt"
LM_PATH = "gpt2-fa.mdl"
TOKENIZER_DIR = "gpt2_tokenizer"
TOKENIZER_PICKLE = "gpt2-fa-tokenizer.mdl"
DATA_DIR = "training/jev_data"

SEQ_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
LR_HEAD = 1e-3
LR_BACKBONE = 2e-5
UNFROZEN_BLOCKS = 4  # fine-tune only the last N of 12 transformer blocks
DROPOUT = 0.1

CHOICE_WEIGHT = 1.0
SCORE_WEIGHT = 0.5
NOUL_WEIGHT = 0.5

# Keep in sync with training/prepare_jev_data.py (which writes these into the
# JSONL schema) and with the Persian labels in utils.py.
CHOICE_CANDIDATES = ["human_written", "ai_generated"]
SCORE_LEVELS = [
    "definitely_human",
    "probably_human",
    "mixed_signals",
    "probably_ai",
    "definitely_ai",
]

# Decision policy thresholds served with the checkpoint (three tiers, as in
# the JEV guide: proceed / gather more info / reject).
PROCEED_THRESHOLD = 0.8
NEEDS_REVIEW_THRESHOLD = 0.6

torch.manual_seed(RANDOM_SEED)


class JevDecisionModel(nn.Module):
    """GPT2 backbone (causal, pooled on the last real token) + typed decision
    heads. Must match the copy in utils.py exactly so the saved state_dict
    loads at serving time."""

    def __init__(self, backbone, hidden_size, n_choice, n_score_levels):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(DROPOUT)
        self.choice_head = nn.Linear(hidden_size, n_choice)
        self.score_head = nn.Linear(hidden_size, n_score_levels)
        self.noul_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        last_idx = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0)), last_idx]
        pooled = self.dropout(pooled)
        return (
            self.choice_head(pooled),
            self.score_head(pooled),
            self.noul_head(pooled).squeeze(-1),
        )


def load_tokenizer():
    """Prefer the offline fast (Rust-backed) tokenizer export; same fallback
    order as utils._load_gpt2_backbone."""
    if os.path.isdir(TOKENIZER_DIR):
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    else:
        with open(TOKENIZER_PICKLE, "rb") as f:
            tokenizer = pickle.load(f)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"
    return tokenizer


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def target_arrays(records):
    choice = np.array(
        [[r["targets"]["source"][c] for c in CHOICE_CANDIDATES] for r in records],
        dtype=np.float32,
    )
    score = np.array(
        [r["targets"]["ai_level"] for r in records], dtype=np.float32
    )
    noul = np.array(
        [r["targets"]["needs_review"] for r in records], dtype=np.float32
    )
    labels = np.array([r["label"] for r in records])
    return choice, score, noul, labels


def batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def soft_cross_entropy(logits, target_probs):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def expected_calibration_error(probs, labels, n_bins=15):
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def main():
    print("Loading JEV training data...")
    train_records = load_jsonl(f"{DATA_DIR}/jev_train.jsonl")
    test_records = load_jsonl(f"{DATA_DIR}/jev_test.jsonl")
    if not train_records or not test_records:
        raise SystemExit(
            "No JEV records found -- run training/prepare_jev_data.py first."
        )
    print(f"{len(train_records)} train / {len(test_records)} test records")

    choice_t, score_t, noul_t, y_train = target_arrays(train_records)
    choice_t_test, score_t_test, noul_t_test, y_test = target_arrays(test_records)
    texts = [r["state"]["text"] for r in train_records]
    texts_test = [r["state"]["text"] for r in test_records]

    print("Loading local Persian GPT-2 checkpoint...")
    with open(LM_PATH, "rb") as f:
        lm = pickle.load(f)
    tokenizer = load_tokenizer()

    backbone = lm.transformer
    hidden_size = backbone.config.n_embd
    n_layer = backbone.config.n_layer

    for p in backbone.parameters():
        p.requires_grad = False
    for block in backbone.h[n_layer - UNFROZEN_BLOCKS :]:
        for p in block.parameters():
            p.requires_grad = True
    for p in backbone.ln_f.parameters():
        p.requires_grad = True

    device = torch.device("cpu")
    torch.set_num_threads(12)
    model = JevDecisionModel(
        backbone, hidden_size, n_choice=len(CHOICE_CANDIDATES), n_score_levels=len(SCORE_LEVELS)
    ).to(device)

    trainable_backbone = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_backbone, "lr": LR_BACKBONE},
            {"params": model.choice_head.parameters(), "lr": LR_HEAD},
            {"params": model.score_head.parameters(), "lr": LR_HEAD},
            {"params": model.noul_head.parameters(), "lr": LR_HEAD},
        ]
    )

    # The slow pure-Python tokenizer BPE-encodes the whole string before
    # truncation, so pre-truncate at the character level (generous upper bound
    # for SEQ_LEN tokens in Persian) -- same trick as train_gpt2_classifier.py.
    CHAR_BUDGET = SEQ_LEN * 8

    def encode_batch(batch_texts):
        batch = [str(t)[:CHAR_BUDGET] for t in batch_texts]
        return tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=SEQ_LEN,
            padding="max_length",
        )

    def loss_from_batch(enc, idx_batch):
        choice_logits, score_logits, noul_logit = model(
            enc["input_ids"], enc["attention_mask"]
        )
        choice_loss = soft_cross_entropy(
            choice_logits, torch.from_numpy(choice_t[idx_batch])
        )
        score_loss = soft_cross_entropy(
            score_logits, torch.from_numpy(score_t[idx_batch])
        )
        noul_loss = nn.functional.binary_cross_entropy_with_logits(
            noul_logit, torch.from_numpy(noul_t[idx_batch])
        )
        return (
            CHOICE_WEIGHT * choice_loss
            + SCORE_WEIGHT * score_loss
            + NOUL_WEIGHT * noul_loss
        )

    n_batches = (len(train_records) + BATCH_SIZE - 1) // BATCH_SIZE
    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng = np.random.default_rng(RANDOM_SEED + epoch)
        order = np.arange(len(train_records))
        rng.shuffle(order)
        epoch_start = time.time()
        total_loss = 0.0
        for step, idx_batch in enumerate(batches(order, BATCH_SIZE), start=1):
            enc = encode_batch([texts[i] for i in idx_batch])
            optimizer.zero_grad()
            loss = loss_from_batch(enc, idx_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx_batch)
            if step % 20 == 0 or step == n_batches:
                elapsed = time.time() - epoch_start
                rate = elapsed / step
                remaining = (n_batches - step) * rate
                print(
                    f"epoch {epoch} batch {step}/{n_batches} "
                    f"loss={loss.item():.4f} elapsed={elapsed:.0f}s "
                    f"eta={remaining:.0f}s",
                    flush=True,
                )
        print(
            f"Epoch {epoch} done. avg_loss={total_loss / len(train_records):.4f} "
            f"time={time.time() - epoch_start:.0f}s"
        )

    print("Evaluating typed answers on the held-out subsample...")
    model.eval()
    choice_probs_all, score_probs_all, noul_probs_all = [], [], []
    eval_start = time.time()
    with torch.no_grad():
        for idx_batch in batches(np.arange(len(test_records)), 32):
            enc = encode_batch([texts_test[i] for i in idx_batch])
            choice_logits, score_logits, noul_logit = model(
                enc["input_ids"], enc["attention_mask"]
            )
            choice_probs_all.append(torch.softmax(choice_logits, dim=-1).numpy())
            score_probs_all.append(torch.softmax(score_logits, dim=-1).numpy())
            noul_probs_all.append(torch.sigmoid(noul_logit).numpy())
    print(f"Eval forward passes done in {time.time() - eval_start:.0f}s")

    choice_probs = np.concatenate(choice_probs_all)
    score_probs = np.concatenate(score_probs_all)
    noul_probs = np.concatenate(noul_probs_all)

    preds = choice_probs.argmax(axis=1)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1Score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        # JEV per-type metrics: score answer error (probability-weighted value
        # vs target value on the 0..4 scale), noul ranking quality, and
        # calibration of the choice distribution.
        "scoreMae": round(
            float(
                mean_absolute_error(
                    score_t_test @ np.arange(len(SCORE_LEVELS)),
                    score_probs @ np.arange(len(SCORE_LEVELS)),
                )
            ),
            4,
        ),
        "noulAuc": None,
        "choiceEce": round(expected_calibration_error(choice_probs, y_test), 4),
    }
    # Both noul classes must be present for AUC; on a near-separable dataset
    # the uncertain band can be nearly empty.
    if len(np.unique(noul_t_test)) > 1:
        metrics["noulAuc"] = round(float(roc_auc_score(noul_t_test, noul_probs)), 4)
        metrics["noulAveragePrecision"] = round(
            float(average_precision_score(noul_t_test, noul_probs)), 4
        )
    print("Final metrics:", metrics)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "seq_len": SEQ_LEN,
                "hidden_size": hidden_size,
                "n_choice": len(CHOICE_CANDIDATES),
                "n_score_levels": len(SCORE_LEVELS),
                "choice_candidates": CHOICE_CANDIDATES,
                "score_levels": SCORE_LEVELS,
                "thresholds": {
                    "proceed": PROCEED_THRESHOLD,
                    "needsReview": NEEDS_REVIEW_THRESHOLD,
                },
            },
            "metrics": {k: v for k, v in metrics.items() if v is not None},
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")

    metrics_for_json = {k: v for k, v in metrics.items() if v is not None}
    entry = {
        "id": "jev_system_one",
        "name": "JEV System One (Decision Head)",
        "nameFa": "مدل تصمیم سیستم-یک (JEV)",
        "description": (
            "JEV-style 'System One' decision model: the Persian GPT-2 backbone "
            "(HooshvareLab/gpt2-fa, last 4 blocks fine-tuned) pooled into three "
            "typed decision heads answered in a single forward pass -- a choice "
            "question (human vs AI, as a probability distribution), a score "
            "question (machine-likeness on a 5-level scale, answered as a "
            "probability-weighted value between levels) and a noul question "
            "(probability that human review is needed). Trained on a stratified "
            "subsample with soft targets distilled from the TF-IDF + Logistic "
            "Regression pipeline; excluded from the ensemble because its choice "
            "head overlaps the GPT-2 classifier."
        ),
        "descriptionFa": (
            "مدل تصمیم «سیستم-یک» به سبک JEV: بک‌بون GPT-2 فارسی "
            "(HooshvareLab/gpt2-fa با فاین‌تیون ۴ بلوک آخر) که بازنمایی تجمیع‌شده "
            "آن به سه سر تصمیم نوع‌بندی‌شده می‌رود و همه در یک گذر رو به جلو "
            "پاسخ داده می‌شوند -- سؤال انتخاب (انسان در برابر هوش مصنوعی به شکل "
            "توزیع احتمال)، سؤال امتیاز (ماشین‌مانندبودن روی طیف ۵ سطحی با مقدار "
            "وزن‌دار احتمالاتی که می‌تواند بین سطوح بیفتد) و سؤانول (احتمال نیاز "
            "به بازبینی انسانی). آموزش روی نمونه طبقه‌بندی‌شده‌ای از داده با "
            "هدف‌های نرم تقطیرشده از خط لوله TF-IDF + رگرسیون لجستیک؛ به دلیل "
            "هم‌پوشانی سر انتخاب با دسته‌بند GPT-2، از ترکیب مدل‌ها حذف شده است."
        ),
        "algorithm": "GPT2 + typed decision heads (choice/score/noul)",
        "type": "dl",
        "version": "1.0",
        "filename": MODEL_PATH,
        "runtime": "pytorch-jev",
        "ensembleWeight": 0.0,
        "metrics": metrics_for_json,
        "confusionMatrix": {
            "trueNegative": int(tn),
            "falsePositive": int(fp),
            "falseNegative": int(fn),
            "truePositive": int(tp),
        },
        "training": {
            "totalSamples": len(train_records) + len(test_records),
            "trainSamples": len(train_records),
            "testSamples": len(test_records),
            "featureCount": hidden_size,
            "testSplitRatio": len(test_records)
            / (len(train_records) + len(test_records)),
            "randomSeed": RANDOM_SEED,
            "dataset": "processedDataset.csv (JEV JSONL, teacher-distilled soft targets)",
        },
    }

    with open(MODELS_JSON_PATH, encoding="utf-8") as f:
        existing = json.load(f)
    existing = [m for m in existing if m["id"] != entry["id"]]
    existing.append(entry)
    with open(MODELS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print("models.json updated.")


if __name__ == "__main__":
    main()
