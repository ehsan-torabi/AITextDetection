"""Fine-tune the locally available Persian GPT-2 (HooshvareLab/gpt2-fa,
pickled at gpt2-fa.mdl / gpt2-fa-tokenizer.mdl) as a sequence classifier.

Only the last few transformer blocks + a new classification head are
unfrozen: on this CPU-only machine the forward pass through all 12 blocks
dominates runtime regardless of how many blocks are trainable, so partial
unfreezing buys real fine-tuning at almost no extra cost over pure feature
extraction (benchmarked at ~0.4-0.5s/example for a 12-layer/768-dim GPT-2 at
seq_len=64 on this machine).

Full fine-tuning over the entire ~21k-sample dataset would take multiple
hours on CPU, so this script trains on a stratified subsample. That's a
deliberate speed/accuracy tradeoff, documented in the saved model metadata.

Run from the project root: python training/train_gpt2_classifier.py
"""

import json
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from aitextd import paths
from aitextd.architectures import GPT2Classifier
from split import RANDOM_SEED, load_split

MODELS_JSON_PATH = paths.MODELS_JSON
MODEL_FILENAME = "gpt2ClassifierModel.pt"
MODEL_PATH = paths.model_path(MODEL_FILENAME)
LM_PATH = paths.GPT2_LM_MDL
TOKENIZER_PATH = paths.GPT2_TOKENIZER_MDL

SEQ_LEN = 64
BATCH_SIZE = 16
EPOCHS = 1
LR_HEAD = 1e-3
LR_BACKBONE = 2e-5
UNFROZEN_BLOCKS = 4  # fine-tune only the last N of 12 transformer blocks

TRAIN_SUBSET = 6000
TEST_SUBSET = 1500

torch.manual_seed(RANDOM_SEED)


def subsample(indices, labels, n, seed):
    if n >= len(indices):
        return indices
    sub, _ = train_test_split(indices, train_size=n, random_state=seed, stratify=labels[indices])
    return sub


def batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main():
    print("Loading local Persian GPT-2 checkpoint...")
    with open(LM_PATH, "rb") as f:
        lm = pickle.load(f)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    tokenizer.pad_token = "<pad>"

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

    df, train_idx, test_idx = load_split()
    y = df["aiGenerated"].to_numpy()
    texts = df["text"].to_numpy()

    train_idx = subsample(train_idx, y, TRAIN_SUBSET, RANDOM_SEED)
    test_idx = subsample(test_idx, y, TEST_SUBSET, RANDOM_SEED)
    print(f"Fine-tuning on {len(train_idx)} train / {len(test_idx)} test samples")

    device = torch.device("cpu")
    torch.set_num_threads(12)
    model = GPT2Classifier(backbone, hidden_size).to(device)

    trainable = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable, "lr": LR_BACKBONE},
            {"params": model.head.parameters(), "lr": LR_HEAD},
        ]
    )
    criterion = nn.BCEWithLogitsLoss()

    # The pickled tokenizer is the slow, pure-Python GPT2Tokenizer: it BPE-encodes
    # the entire string before truncation, so a full multi-paragraph article costs
    # far more than a 64-token sequence should. Pre-truncate at the character
    # level (generous upper bound for SEQ_LEN tokens in Persian) to keep encoding
    # fast regardless of source document length.
    CHAR_BUDGET = SEQ_LEN * 8

    def encode_batch(idx_batch):
        batch_texts = [str(t)[:CHAR_BUDGET] for t in texts[idx_batch]]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=SEQ_LEN,
            padding="max_length",
        )
        labels = torch.tensor(y[idx_batch], dtype=torch.float32)
        return enc["input_ids"], enc["attention_mask"], labels

    n_batches_per_epoch = (len(train_idx) + BATCH_SIZE - 1) // BATCH_SIZE
    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng = np.random.default_rng(RANDOM_SEED + epoch)
        shuffled = train_idx.copy()
        rng.shuffle(shuffled)
        epoch_start = time.time()
        total_loss = 0.0
        for i, idx_batch in enumerate(batches(shuffled, BATCH_SIZE), start=1):
            input_ids, attention_mask, labels = encode_batch(idx_batch)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx_batch)
            if i % 20 == 0 or i == n_batches_per_epoch:
                elapsed = time.time() - epoch_start
                rate = elapsed / i
                remaining = (n_batches_per_epoch - i) * rate
                print(
                    f"epoch {epoch} batch {i}/{n_batches_per_epoch} "
                    f"loss={loss.item():.4f} elapsed={elapsed:.0f}s "
                    f"eta={remaining:.0f}s",
                    flush=True,
                )
        print(
            f"Epoch {epoch} done. avg_loss={total_loss / len(train_idx):.4f} "
            f"time={time.time() - epoch_start:.0f}s"
        )

    print("Evaluating on held-out subsample...")
    model.eval()
    all_probs = []
    eval_start = time.time()
    with torch.no_grad():
        for idx_batch in batches(test_idx, 32):
            input_ids, attention_mask, _ = encode_batch(idx_batch)
            probs = torch.sigmoid(model(input_ids, attention_mask))
            all_probs.append(probs.numpy())
    all_probs = np.concatenate(all_probs)
    print(f"Eval done in {time.time() - eval_start:.0f}s")

    y_test = y[test_idx]
    preds = (all_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds)), 4),
        "recall": round(float(recall_score(y_test, preds)), 4),
        "f1Score": round(float(f1_score(y_test, preds)), 4),
    }
    confusion = {
        "trueNegative": int(tn),
        "falsePositive": int(fp),
        "falseNegative": int(fn),
        "truePositive": int(tp),
    }
    print("Final metrics:", metrics)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "seq_len": SEQ_LEN,
                "hidden_size": hidden_size,
                "n_layer": n_layer,
                "vocab_size": backbone.config.vocab_size,
                "n_head": backbone.config.n_head,
            },
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")

    entry = {
        "id": "gpt2_transformer",
        "name": "GPT-2 Transformer (fine-tuned)",
        "nameFa": "ترنسفورمر GPT-2 (فاین‌تیون‌شده)",
        "description": (
            "HooshvareLab/gpt2-fa (a 12-layer, 124M-parameter Persian GPT-2) "
            "with its last 4 transformer blocks fine-tuned for sequence "
            "classification. Trained on a stratified subsample of the "
            "dataset (CPU-only fine-tuning budget) rather than the full "
            "training set used by the other models."
        ),
        "descriptionFa": (
            "مدل GPT-2 فارسی (HooshvareLab/gpt2-fa) با ۱۲ لایه و ۱۲۴ میلیون "
            "پارامتر که ۴ بلوک ترنسفورمر پایانی آن برای دسته‌بندی متن "
            "فاین‌تیون شده است. به دلیل محدودیت آموزش روی CPU، این مدل روی "
            "نمونه‌ای طبقه‌بندی‌شده از کل داده آموزش دیده، نه کل مجموعه "
            "آموزشی مانند سایر مدل‌ها."
        ),
        "algorithm": "GPT2ForSequenceClassification (partial fine-tune)",
        "type": "dl",
        "version": "1.0",
        "filename": MODEL_FILENAME,
        "runtime": "pytorch-gpt2",
        "metrics": metrics,
        "confusionMatrix": confusion,
        "training": {
            "totalSamples": len(train_idx) + len(test_idx),
            "trainSamples": len(train_idx),
            "testSamples": len(test_idx),
            "featureCount": hidden_size,
            "testSplitRatio": len(test_idx) / (len(train_idx) + len(test_idx)),
            "randomSeed": RANDOM_SEED,
            "dataset": "persian_ai_generated_final.csv + blogs.csv (subsampled)",
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
