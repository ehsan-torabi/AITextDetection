"""Train a small BiLSTM classifier from scratch (own embeddings, no
pretrained weights) on the lemmatized/tokenized Persian text.

Run from the project root: python training/train_lstm.py
"""

import json
import time
from collections import Counter

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
from torch.utils.data import DataLoader, Dataset

from preprocess_cache import build_or_load_cache
from split import RANDOM_SEED, TEST_SIZE, load_split

MODELS_JSON_PATH = "models.json"
MODEL_PATH = "lstmModel.pt"

MAX_VOCAB_SIZE = 30_000
MIN_FREQ = 2
MAX_LEN = 200
EMBED_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
PAD_IDX = 0
UNK_IDX = 1

torch.manual_seed(RANDOM_SEED)


class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        # h_n: (num_directions, batch, hidden_dim) -> concat last layer's fwd/bwd
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        h = self.dropout(h)
        return self.fc(h).squeeze(-1)


def build_vocab(train_tokens):
    counter = Counter()
    for tokens in train_tokens:
        counter.update(tokens)
    vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
    for word, freq in counter.most_common(MAX_VOCAB_SIZE):
        if freq < MIN_FREQ:
            break
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def encode(tokens, vocab, max_len=MAX_LEN):
    ids = [vocab.get(t, UNK_IDX) for t in tokens[:max_len]]
    if len(ids) < max_len:
        ids = ids + [PAD_IDX] * (max_len - len(ids))
    return ids


def main():
    tokenized = build_or_load_cache()
    df, train_idx, test_idx = load_split()
    y = df["aiGenerated"].to_numpy()

    train_tokens = [tokenized[i] for i in train_idx]
    test_tokens = [tokenized[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print("Building vocabulary from training split...")
    vocab = build_vocab(train_tokens)
    print(f"Vocab size: {len(vocab)}")

    X_train = torch.tensor(
        [encode(t, vocab) for t in train_tokens], dtype=torch.long
    )
    X_test = torch.tensor([encode(t, vocab) for t in test_tokens], dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TextDataset(X_train, y_train_t), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TextDataset(X_test, y_test_t), batch_size=256, shuffle=False
    )

    device = torch.device("cpu")
    torch.set_num_threads(6)  # leave headroom for the concurrent GPT-2 job
    model = BiLSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    n_batches_per_epoch = (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE
    best_f1 = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for i, (xb, yb) in enumerate(train_loader, start=1):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            if i % 50 == 0 or i == n_batches_per_epoch:
                elapsed = time.time() - t0
                print(
                    f"  epoch {epoch} batch {i}/{n_batches_per_epoch} "
                    f"loss={loss.item():.4f} elapsed={elapsed:.0f}s",
                    flush=True,
                )

        model.eval()
        all_preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                logits = model(xb.to(device))
                probs = torch.sigmoid(logits)
                all_preds.append(probs.numpy())
        all_preds = np.concatenate(all_preds)
        preds_binary = (all_preds >= 0.5).astype(int)
        f1 = f1_score(y_test, preds_binary)
        acc = accuracy_score(y_test, preds_binary)
        print(
            f"Epoch {epoch}/{EPOCHS}  loss={total_loss / len(X_train):.4f}  "
            f"val_acc={acc:.4f}  val_f1={f1:.4f}  ({time.time() - t0:.1f}s)"
        )
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_preds = []
        for xb, _ in test_loader:
            probs = torch.sigmoid(model(xb.to(device)))
            all_preds.append(probs.numpy())
    all_preds = np.concatenate(all_preds)
    preds_binary = (all_preds >= 0.5).astype(int)

    cm = confusion_matrix(y_test, preds_binary)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds_binary)), 4),
        "precision": round(float(precision_score(y_test, preds_binary)), 4),
        "recall": round(float(recall_score(y_test, preds_binary)), 4),
        "f1Score": round(float(f1_score(y_test, preds_binary)), 4),
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
            "vocab": vocab,
            "config": {
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "max_len": MAX_LEN,
                "pad_idx": PAD_IDX,
                "unk_idx": UNK_IDX,
            },
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")

    entry = {
        "id": "bilstm",
        "name": "BiLSTM",
        "nameFa": "شبکه دوسویه LSTM",
        "description": "A bidirectional LSTM neural network with its own learned word embeddings, trained from scratch on lemmatized Persian text to distinguish AI-generated content from human writing.",
        "descriptionFa": "شبکه عصبی LSTM دوسویه با embedding آموخته‌شده اختصاصی که از ابتدا روی متن فارسی نرمال‌شده و ریشه‌یابی‌شده آموزش دیده تا متن تولیدشده توسط هوش مصنوعی را از نوشته انسان جدا کند.",
        "algorithm": "BiLSTM",
        "type": "dl",
        "version": "1.0",
        "filename": MODEL_PATH,
        "runtime": "pytorch-lstm",
        "metrics": metrics,
        "confusionMatrix": confusion,
        "training": {
            "totalSamples": len(train_idx) + len(test_idx),
            "trainSamples": len(train_idx),
            "testSamples": len(test_idx),
            "featureCount": len(vocab),
            "testSplitRatio": TEST_SIZE,
            "randomSeed": RANDOM_SEED,
            "dataset": "persian_ai_generated_final.csv + blogs.csv",
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
