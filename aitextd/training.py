"""Shared torch helpers for the training notebooks: vocabulary building,
encoding, batched inference and the BCE training loop with best-F1
checkpointing.

Kept in its own module (and out of aitextd/__init__) so importing the serving
stack never pulls torch.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from aitextd import metrics

PAD_IDX, UNK_IDX = 0, 1


def build_vocab(token_lists, max_size: int = 30_000, min_freq: int = 2) -> dict:
    """<pad>=0 / <unk>=1, then words by descending frequency (>= min_freq)."""
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
    for word, freq in counter.most_common(max_size):
        if freq < min_freq:
            break
        vocab.setdefault(word, len(vocab))
    return vocab


def encode(tokens, vocab, max_len: int) -> list[int]:
    ids = [vocab.get(t, UNK_IDX) for t in tokens[:max_len]]
    return ids + [PAD_IDX] * (max_len - len(ids))


def encode_tokens(token_lists, vocab, max_len: int) -> torch.Tensor:
    return torch.tensor([encode(tokens, vocab, max_len) for tokens in token_lists], dtype=torch.long)


def predict_probs(model, batch, batch_size: int = 256) -> np.ndarray:
    """Batched sigmoid probabilities from a binary-logit model."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(batch), batch_size):
            out.append(torch.sigmoid(model(batch[start : start + batch_size])).numpy())
    return np.concatenate(out)


def fit_classifier(
    model, X_train, y_train, X_eval, y_eval, epochs: int, batch_size: int = 64, lr: float = 1e-3
):
    """BCEWithLogitsLoss + Adam; per-epoch eval F1, best weights restored.
    Returns (model, history) with one {epoch, loss, metric} record per epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    y_train_t = torch.as_tensor(np.asarray(y_train), dtype=torch.float32)
    history, best_f1, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train))
        total = 0.0
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_train[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        f1 = metrics.evaluate(y_eval, predict_probs(model, X_eval))["f1"]
        history.append({"epoch": epoch, "loss": total / len(perm), "metric": f1})
        print(f"epoch {epoch}/{epochs}  loss={total / len(perm):.4f}  eval_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def save_checkpoint(path: Path, model, **payload) -> None:
    """state_dict + vocab/config payload -- the checkpoint layout every
    aitextd.loaders loader rebuilds from at serving time."""
    torch.save({"state_dict": model.state_dict(), **payload}, path)
    print("saved", path)
