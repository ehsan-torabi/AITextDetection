"""Fine-tune ParsBERT (HooshvareLab/bert-fa-base-uncased) as a sequence classifier.

ParsBERT is a bidirectional BERT encoder pretrained on Persian, so unlike the
causal GPT-2 in train_gpt2_classifier.py its pooled representation is the [CLS]
token rather than the last real token. Same CPU budget as the other transformer
trainers: only the last 4 of 12 encoder layers + the pooler are unfrozen,
SEQ_LEN=64, one epoch over the same stratified 6000/1500 subsample, so its test
metrics are directly comparable to the GPT-2 row of the results table.

Unlike GPT-2 this backbone is rebuilt at serving time from the config stored in
the checkpoint (utils._load_parsbert_model), so no pretrained-weight pickle is
needed for the server to start offline -- but the tokenizer is exported to
parsbert_tokenizer/ for exactly that reason.

Preprocessing calls utils._normalize_for_gpt2 rather than reimplementing it:
the light (non-lemmatized) normalization must be byte-for-byte identical at
train and serve time or the predictions go silently wrong.

Run from the project root: python training/train_parsbert.py
"""

import json
import sys
import time
from pathlib import Path

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
from transformers import AutoModel, AutoTokenizer, BertConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import utils  # noqa: E402  (serving pipeline: shared to guarantee train/serve parity)
from split import RANDOM_SEED, load_split  # noqa: E402

MODELS_JSON_PATH = "models.json"
MODEL_PATH = "parsbertModel.pt"
BACKBONE_NAME = "HooshvareLab/bert-fa-base-uncased"
TOKENIZER_DIR = "parsbert_tokenizer"

SEQ_LEN = 64
BATCH_SIZE = 16
EPOCHS = 1  # CPU budget, same as the GPT-2 trainer; raise on Colab/GPU
LR_HEAD = 1e-3
LR_BACKBONE = 2e-5
WEIGHT_DECAY = 0.01
UNFROZEN_LAYERS = 4  # fine-tune only the last N of 12 encoder layers

# Same subsample size/seed as train_gpt2_classifier.py so both models are
# scored on the exact same held-out documents.
TRAIN_SUBSET = 6000
TEST_SUBSET = 1500

torch.manual_seed(RANDOM_SEED)


class ParsBertClassifier(nn.Module):
    """ParsBERT backbone (bidirectional, so pooled on [CLS]) + linear head.
    Must match the copy in utils.py exactly so the saved state_dict loads."""

    def __init__(self, backbone, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_repr = hidden[:, 0]
        return self.head(cls_repr).squeeze(-1)


def subsample(indices, labels, n, seed):
    if n >= len(indices):
        return indices
    sub, _ = train_test_split(indices, train_size=n, random_state=seed, stratify=labels[indices])
    return sub


def batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def assert_serving_contract(model, backbone_config, hidden_size):
    """Rebuild the checkpoint the way utils._load_parsbert_model does and check
    the state_dict keys line up. A silent key/shape mismatch here would only
    surface as a load error on the first request against the live server."""
    rebuilt = ParsBertClassifier(
        AutoModel.from_config(BertConfig.from_dict(backbone_config)), hidden_size
    )
    drift = set(model.state_dict()) ^ set(rebuilt.state_dict())
    assert not drift, f"state_dict does not match the serving class: {drift}"


def main():
    print("Loading local Persian ParsBERT checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME)
    backbone = AutoModel.from_pretrained(BACKBONE_NAME)
    hidden_size = backbone.config.hidden_size
    n_layers = backbone.config.num_hidden_layers

    for p in backbone.parameters():
        p.requires_grad = False
    for layer in backbone.encoder.layer[n_layers - UNFROZEN_LAYERS :]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in backbone.pooler.parameters():
        p.requires_grad = True

    df, train_idx, test_idx = load_split()
    y = df["aiGenerated"].to_numpy()

    train_idx = subsample(train_idx, y, TRAIN_SUBSET, RANDOM_SEED)
    test_idx = subsample(test_idx, y, TEST_SUBSET, RANDOM_SEED)
    print(f"Fine-tuning on {len(train_idx)} train / {len(test_idx)} test samples")

    # Light normalization (same call the server makes for this model), only for
    # the rows this run actually touches.
    normalize_start = time.time()
    needed = np.concatenate([train_idx, test_idx])
    raw = df["text"].to_numpy()[needed]
    texts = {int(i): utils._normalize_for_gpt2(str(t)) for i, t in zip(needed, raw, strict=True)}
    print(f"Normalized {len(texts)} texts in {time.time() - normalize_start:.0f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(12)
    model = ParsBertClassifier(backbone, hidden_size).to(device)

    trainable_backbone = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_backbone, "lr": LR_BACKBONE},
            {"params": model.head.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    def encode_batch(idx_batch):
        enc = tokenizer(
            [texts[i] for i in idx_batch],
            return_tensors="pt",
            truncation=True,
            max_length=SEQ_LEN,
            padding="max_length",
        )
        labels = torch.tensor(y[idx_batch], dtype=torch.float32, device=device)
        return enc["input_ids"].to(device), enc["attention_mask"].to(device), labels

    n_batches_per_epoch = (len(train_idx) + BATCH_SIZE - 1) // BATCH_SIZE
    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng = np.random.default_rng(RANDOM_SEED + epoch)
        shuffled = train_idx.copy()
        rng.shuffle(shuffled)
        epoch_start = time.time()
        total_loss = 0.0
        for step, idx_batch in enumerate(batches(shuffled, BATCH_SIZE), start=1):
            input_ids, attention_mask, labels = encode_batch(idx_batch)
            optimizer.zero_grad()
            loss = criterion(model(input_ids, attention_mask), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx_batch)
            if step % 20 == 0 or step == n_batches_per_epoch:
                elapsed = time.time() - epoch_start
                rate = elapsed / step
                remaining = (n_batches_per_epoch - step) * rate
                print(
                    f"epoch {epoch} batch {step}/{n_batches_per_epoch} "
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
            all_probs.append(torch.sigmoid(model(input_ids, attention_mask)).cpu().numpy())
    all_probs = np.concatenate(all_probs)
    print(f"Eval done in {time.time() - eval_start:.0f}s")

    y_test = y[test_idx]
    preds = (all_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1Score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
    }
    confusion = {
        "trueNegative": int(tn),
        "falsePositive": int(fp),
        "falseNegative": int(fn),
        "truePositive": int(tp),
    }
    print("Final metrics:", metrics)

    backbone_config = backbone.config.to_dict()
    assert_serving_contract(model, backbone_config, hidden_size)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "seq_len": SEQ_LEN,
                "hidden_size": hidden_size,
                "n_layer": n_layers,
                # Serving rebuilds the encoder from this alone (no pretrained
                # download needed), then overwrites it with state_dict.
                "backbone_config": backbone_config,
            },
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")

    # Export the tokenizer so the server never needs Hub access at startup.
    tokenizer.save_pretrained(TOKENIZER_DIR)
    print(f"Tokenizer exported to {TOKENIZER_DIR}/")

    entry = {
        "id": "parsbert",
        "name": "ParsBERT (fine-tuned)",
        "nameFa": "پارس‌برت (فاین‌تیون‌شده)",
        "description": (
            "ParsBERT (HooshvareLab/bert-fa-base-uncased), a bidirectional BERT "
            "encoder pretrained on Persian, with its last 4 encoder layers "
            "fine-tuned for sequence classification on the [CLS] token. Trained "
            "on the same stratified subsample as the GPT-2 classifier (CPU-only "
            "fine-tuning budget), so the two are directly comparable."
        ),
        "descriptionFa": (
            "پارس‌برت (HooshvareLab/bert-fa-base-uncased)، یک رمزگذار BERT "
            "دوسویه که روی زبان فارسی از پیش‌آموزش دیده و ۴ لایه پایانی آن برای "
            "دسته‌بندی متن روی توکن [CLS] فاین‌تیون شده است. آموزش روی همان "
            "نمونهٔ طبقه‌بندی‌شدهٔ دسته‌بند GPT-2 انجام شده است."
        ),
        "algorithm": "ParsBERT (BERT encoder, partial fine-tune)",
        "type": "dl",
        "version": "1.0",
        "filename": MODEL_PATH,
        "runtime": "pytorch-parsbert",
        # Untuned default: raise it once real held-out numbers exist.
        "ensembleWeight": 1.0,
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
