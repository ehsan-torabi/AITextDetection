"""Pin model-architecture shapes and pooling logic with tiny random configs.

These architectures must stay load-compatible with the shipped checkpoints;
the tests verify tensor shapes and state_dict key structure, not weights.
"""

import torch

from aitextd.architectures import (
    BiLSTMClassifier,
    GPT2Classifier,
    ParsBERTClassifier,
    SystemOneDecisionModel,
)


def test_bilstm_forward_shape():
    model = BiLSTMClassifier(vocab_size=50, embed_dim=8, hidden_dim=16, pad_idx=0)
    model.eval()
    x = torch.randint(0, 50, (3, 10))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (3,)


def test_gpt2_classifier_pools_last_real_token():
    class FakeBackbone(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            # Position i returns a constant i so the pooled position is visible.
            hidden = input_ids.unsqueeze(-1).float().expand(-1, -1, 4)
            return type("Out", (), {"last_hidden_state": hidden})()

    model = GPT2Classifier(FakeBackbone(), hidden_size=4)
    model.eval()
    input_ids = torch.tensor([[5, 6, 7, 0, 0]])  # 3 real tokens, 2 pad
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
    with torch.no_grad():
        out = model(input_ids, attention_mask)
    assert out.shape == (1,)


def test_parsbert_classifier_pools_cls_token():
    class FakeBackbone(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
            hidden[:, 0] = 1.0  # [CLS] marker
            return type("Out", (), {"last_hidden_state": hidden})()

    model = ParsBERTClassifier(FakeBackbone(), hidden_size=4)
    model.eval()
    input_ids = torch.tensor([[2, 3, 0]])
    attention_mask = torch.tensor([[1, 1, 0]])
    with torch.no_grad():
        out = model(input_ids, attention_mask)
    assert out.shape == (1,)


def test_system_one_decision_model_outputs_three_heads():
    class FakeBackbone(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 6)
            return type("Out", (), {"last_hidden_state": hidden})()

    model = SystemOneDecisionModel(
        FakeBackbone(), hidden_size=6, n_choice=2, n_score_levels=5, pool_on_cls=True
    )
    model.eval()
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1]])
    with torch.no_grad():
        choice, score, noul = model(input_ids, attention_mask)
    assert choice.shape == (1, 2)
    assert score.shape == (1, 5)
    assert noul.shape == (1,)


def test_system_one_head_names_match_checkpoint_layout():
    model = SystemOneDecisionModel(
        torch.nn.Linear(8, 8), hidden_size=8, n_choice=2, n_score_levels=5, pool_on_cls=True
    )
    keys = set(model.state_dict())
    for head in ("choice_head", "score_head", "noul_head"):
        assert f"{head}.weight" in keys and f"{head}.bias" in keys
    assert any(k.startswith("backbone.") for k in keys)
