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
