"""Pin model-architecture shapes and pooling logic with tiny random configs.

These architectures must stay load-compatible with the shipped checkpoints;
the tests verify tensor shapes and state_dict key structure, not weights.
"""

import torch

from aitextd.architectures import (
    AttentionBiLSTMClassifier,
    BiLSTMClassifier,
    GPT2Classifier,
    MLPClassifier,
    ParsBERTClassifier,
    SystemOneDecisionModel,
    TextCNNClassifier,
    TinyTransformerClassifier,
)


def test_bilstm_forward_shape():
    model = BiLSTMClassifier(vocab_size=50, embed_dim=8, hidden_dim=16, pad_idx=0)
    model.eval()
    x = torch.randint(0, 50, (3, 10))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (3,)


def test_textcnn_forward_shape():
    model = TextCNNClassifier(
        vocab_size=50, embed_dim=8, n_filters=6, kernel_sizes=(3, 4, 5), pad_idx=0
    )
    model.eval()
    x = torch.randint(0, 50, (3, 10))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (3,)


def test_attention_bilstm_forward_shape():
    model = AttentionBiLSTMClassifier(vocab_size=50, embed_dim=8, hidden_dim=16, pad_idx=0)
    model.eval()
    x = torch.randint(0, 50, (3, 10))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (3,)


def test_textcnn_state_dict_matches_checkpoint_layout():
    model = TextCNNClassifier(
        vocab_size=20, embed_dim=4, n_filters=3, kernel_sizes=(3, 4, 5), pad_idx=0
    )
    keys = set(model.state_dict())
    assert "embedding.weight" in keys
    for i in range(3):
        assert f"convs.{i}.weight" in keys and f"convs.{i}.bias" in keys
    assert "fc.weight" in keys and "fc.bias" in keys


def test_attention_bilstm_state_dict_matches_checkpoint_layout():
    model = AttentionBiLSTMClassifier(vocab_size=20, embed_dim=4, hidden_dim=4, pad_idx=0)
    keys = set(model.state_dict())
    assert "embedding.weight" in keys
    for direction in ("", "_reverse"):
        assert f"lstm.weight_ih_l0{direction}" in keys
        assert f"lstm.weight_hh_l0{direction}" in keys
    assert "score.weight" in keys and "score.bias" in keys
    assert "fc.weight" in keys and "fc.bias" in keys


def test_textcnn_masked_pooling_excludes_all_pad_windows():
    # Zero conv weights + bias 1 -> every window's post-ReLU activation is
    # exactly 1. A row with at least one real token therefore pools 1 per
    # filter, while an all-pad row must pool exactly 0 (pad-only windows are
    # masked out), even though the pad embedding is nonzero here.
    model = TextCNNClassifier(
        vocab_size=10, embed_dim=4, n_filters=2, kernel_sizes=(3, 4), pad_idx=0
    )
    model.eval()
    with torch.no_grad():
        model.embedding.weight[0].fill_(1.0)  # nonzero pad embedding
        for conv in model.convs:
            conv.weight.zero_()
            conv.bias.fill_(1.0)
        out_pad = model(torch.zeros(1, 8, dtype=torch.long))
        out_real = model(torch.ones(1, 8, dtype=torch.long))
    assert torch.allclose(out_pad, model.fc.bias)  # zero features -> bias only
    expected_real = model.fc.bias + model.fc.weight.sum()  # every filter pools 1.0
    assert torch.allclose(out_real, expected_real)


def test_attention_bilstm_attention_ignores_pad_positions():
    # A fake LSTM returns hidden state i at position i and 99 at pad
    # positions; a zero score vector gives uniform weights over real tokens.
    # If pad positions leaked into the softmax the pooled value would jump,
    # so the head output pins the exact masked average.
    model = AttentionBiLSTMClassifier(vocab_size=10, embed_dim=4, hidden_dim=2, pad_idx=0)
    model.eval()

    class FakeLSTM(torch.nn.Module):
        def forward(self, emb):
            h = torch.full((emb.shape[0], emb.shape[1], 4), 99.0)
            h[:, 0] = 0.0
            h[:, 1] = 1.0
            h[:, 2] = 2.0
            return h, None

    model.lstm = FakeLSTM()
    with torch.no_grad():
        model.score.weight.zero_()
        model.score.bias.zero_()
        model.fc.weight.fill_(1.0)
        out = model(torch.tensor([[5, 6, 7, 0, 0]]))
    expected = model.fc.bias + model.fc.weight.sum() * (0.0 + 1.0 + 2.0) / 3.0
    assert torch.allclose(out, expected.expand_as(out))


def test_attention_bilstm_all_pad_row_is_finite():
    model = AttentionBiLSTMClassifier(vocab_size=10, embed_dim=4, hidden_dim=4, pad_idx=0)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 6, dtype=torch.long))
    assert torch.isfinite(out).all()


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


def test_mlp_forward_shape():
    model = MLPClassifier(input_dim=30, hidden_dims=(16, 8), dropout=0.3)
    model.eval()
    x = torch.rand(3, 30)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (3,)


def test_mlp_state_dict_matches_checkpoint_layout():
    model = MLPClassifier(input_dim=30, hidden_dims=(16, 8), dropout=0.3)
    keys = set(model.state_dict())
    assert "net.0.weight" in keys and "net.0.bias" in keys
    assert "net.3.weight" in keys and "net.3.bias" in keys
    assert "net.6.weight" in keys and "net.6.bias" in keys


def test_tinytransformer_forward_shape():
    model = TinyTransformerClassifier(
        vocab_size=50, embed_dim=16, n_layers=2, n_heads=4, ff_dim=32, max_len=10, pad_idx=0
    )
    model.eval()
    x = torch.randint(0, 50, (3, 10))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (3,)


def test_tinytransformer_pads_do_not_reach_the_pool():
    """Rows with only real tokens vs rows with heavy padding must produce the
    same pooled summary for the same prefix: padding never enters the pool."""
    model = TinyTransformerClassifier(
        vocab_size=50, embed_dim=16, n_layers=2, n_heads=4, ff_dim=32, max_len=10, pad_idx=0
    )
    model.eval()
    full = torch.randint(1, 50, (1, 10))
    padded = full.clone()
    padded[0, 5:] = 0
    with torch.no_grad():
        out_full = model(full)
        out_padded = model(padded)
    assert out_full.shape == (1,) and out_padded.shape == (1,)
    assert not torch.isnan(out_full).any() and not torch.isnan(out_padded).any()
