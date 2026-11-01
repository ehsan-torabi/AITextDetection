"""Deep-learning model architectures.

These must match training/train_*.py (and the Colab notebook) exactly so the
saved state_dicts load correctly at serving time. None of these models
generate text: every head emits a probability distribution or a scalar that
the serving layer turns into typed answers.
"""

import torch
import torch.nn as nn

# Bound CPU thread usage before any checkpoint is loaded (was a module-level
# side effect of the old monolithic utils.py).
torch.set_num_threads(4)


class MLPClassifier(nn.Module):
    """Plain feed-forward network over a reduced TF-IDF + style feature vector
    (the same joint layout the sklearn-tfidf models consume, chi2-reduced to
    keep the input layer dense-trainable)."""

    def __init__(self, input_dim, hidden_dims=(512, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TinyTransformerClassifier(nn.Module):
    """A small from-scratch Transformer encoder over the shared word vocabulary
    (no pretraining): learned positional embeddings, `n_layers` self-attention
    blocks with `n_heads`, masked mean-pooling over real tokens, linear head."""

    def __init__(
        self, vocab_size, embed_dim, n_layers, n_heads, ff_dim, max_len, pad_idx=0, dropout=0.3
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.max_len = max_len
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        emb = self.embedding(x) + self.pos_embedding(positions)
        mask = x != self.pad_idx
        hidden = self.encoder(emb, src_key_padding_mask=~mask)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(
            min=1
        )
        return self.fc(self.dropout(pooled)).squeeze(-1)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        h = self.dropout(h)
        return self.fc(h).squeeze(-1)


class TextCNNClassifier(nn.Module):
    """TextCNN: parallel 1D-convolution n-gram detectors over learned word
    embeddings, ReLU, then a global max-pool per kernel size, concatenated
    into one linear head with a single logit.

    Variable lengths are handled with a length mask: a convolution window
    counts only if it covers at least one real (non-pad) token, so windows of
    pure padding are excluded from pooling. Sequences shorter than the largest
    kernel size must still be padded before the forward pass (the serving
    predictor always pads to the checkpoint's max_len, so this only matters
    for ad-hoc callers). Must match the training cells of
    notebooks/03_deep_learning_models.ipynb exactly so the saved state_dict
    loads correctly at serving time."""

    def __init__(
        self, vocab_size, embed_dim, n_filters, kernel_sizes=(3, 4, 5), pad_idx=0, dropout=0.3
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            nn.Conv1d(embed_dim, n_filters, kernel_size) for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), 1)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)  # (B, E, T)
        real = x.ne(self.pad_idx).float()  # (B, T), 1 on real tokens
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(emb))  # (B, n_filters, T - k + 1)
            # A window is valid iff it covers at least one real token; since
            # ReLU output is non-negative, zeroing invalid windows is exactly
            # "exclude them from the max" (no -inf/NaN edge cases).
            window_ones = torch.ones(1, 1, conv.kernel_size[0], device=x.device)
            valid = torch.nn.functional.conv1d(real.unsqueeze(1), window_ones) > 0
            pooled.append((h * valid).max(dim=2).values)
        h = self.dropout(torch.cat(pooled, dim=1))  # (B, n_filters * len(kernel_sizes))
        return self.fc(h).squeeze(-1)


class AttentionBiLSTMClassifier(nn.Module):
    """BiLSTM encoder with additive attention pooling: a learned score vector
    produces one attention score per hidden state, softmaxed with the pad
    positions masked out so the pooled summary is built only from real tokens,
    then a linear head with a single logit. Must match the training cells of
    notebooks/03_deep_learning_models.ipynb exactly so the saved state_dict
    loads correctly at serving time."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx=0, dropout=0.3):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.score = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        mask = x.ne(self.pad_idx)  # (B, T), True on real tokens
        h, _ = self.lstm(self.embedding(x))  # (B, T, 2H)
        scores = self.score(h).squeeze(-1)  # (B, T)
        # finfo.min (not -inf) so an all-pad row softmaxes to a uniform
        # distribution instead of NaN; on mixed rows it underflows to exactly 0.
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)  # (B, T)
        pooled = (h * weights.unsqueeze(-1)).sum(dim=1)  # (B, 2H)
        return self.fc(self.dropout(pooled)).squeeze(-1)


class GPT2Classifier(nn.Module):
    """GPT2 backbone (causal, so pooled on the last real token) + linear head."""

    def __init__(self, backbone, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_idx = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0)), last_idx]
        return self.head(pooled).squeeze(-1)


class ParsBERTClassifier(nn.Module):
    """ParsBERT backbone (bidirectional, so pooled on the [CLS] token) + linear head."""

    def __init__(self, backbone, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_repr = hidden[:, 0]
        return self.head(cls_repr).squeeze(-1)


class SystemOneDecisionModel(nn.Module):
    """System One decision model: a pretrained-transformer backbone whose
    pooled representation feeds three typed decision heads -- choice
    (distribution over candidate answers), score (distribution over an ordinal
    scale, answered as a probability-weighted value) and noul (yes/no
    probability). Like every System One model it produces no text: all three
    typed answers come from a single forward pass. Current checkpoints pool
    the bidirectional ParsBERT backbone on [CLS]; legacy checkpoints pooled a
    causal GPT-2 backbone on the last real token, so pooling is selected by
    the pool_on_cls flag the loader sets from the checkpoint format. Must
    match training/train_system_one.py exactly so the saved state_dict loads
    correctly."""

    def __init__(self, backbone, hidden_size, n_choice, n_score_levels, pool_on_cls):
        super().__init__()
        self.backbone = backbone
        self.pool_on_cls = pool_on_cls
        self.dropout = nn.Dropout(0.1)
        self.choice_head = nn.Linear(hidden_size, n_choice)
        self.score_head = nn.Linear(hidden_size, n_score_levels)
        self.noul_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pool_on_cls:
            # [CLS] is always token position 0 for a BERT-style encoder: every
            # position already attends to the whole sequence, so unlike the
            # causal GPT-2 branch below there is no attention-mask index
            # arithmetic needed to find "the token that saw everything".
            pooled = hidden[:, 0]
        else:
            last_idx = attention_mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0)), last_idx]
        pooled = self.dropout(pooled)
        return (
            self.choice_head(pooled),
            self.score_head(pooled),
            self.noul_head(pooled).squeeze(-1),
        )
