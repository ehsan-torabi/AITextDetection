"""Per-runtime model loading.

Every loader returns a "bundle" dict holding the loaded model plus whatever
its predictor needs (vocab, config, tokenizer, vectorizer). Shared heavy
objects (tokenizers, the pretrained GPT-2 backbone) are cached at module
level so each runtime pays for them once.
"""

import json
import pickle

import joblib
import torch
from transformers import AutoModel, AutoTokenizer, BertConfig

from aitextd import paths
from aitextd.architectures import (
    AttentionBiLSTMClassifier,
    BiLSTMClassifier,
    GPT2Classifier,
    ParsBERTClassifier,
    SystemOneDecisionModel,
    TextCNNClassifier,
)
from aitextd.features import STYLE_FEATURE_NAMES

_gpt2_tokenizer = None  # shared across every pytorch-gpt2 model
_parsbert_tokenizer = None


def get_gpt2_tokenizer():
    global _gpt2_tokenizer
    if _gpt2_tokenizer is None:
        # Prefer an offline fast (Rust-backed) tokenizer exported by the
        # training notebook via tokenizer.save_pretrained(), which is both
        # faster and less prone to slow-tokenizer pathologies on long
        # documents than the pickled slow GPT2Tokenizer. Falls back to the
        # pickle when that export isn't present.
        if paths.GPT2_TOKENIZER_DIR.is_dir():
            _gpt2_tokenizer = AutoTokenizer.from_pretrained(str(paths.GPT2_TOKENIZER_DIR))
        else:
            with open(paths.GPT2_TOKENIZER_MDL, "rb") as f:
                _gpt2_tokenizer = pickle.load(f)
            _gpt2_tokenizer.pad_token = "<pad>"
    return _gpt2_tokenizer


def get_parsbert_tokenizer():
    global _parsbert_tokenizer
    if _parsbert_tokenizer is None:
        # Prefer the exported tokenizer folder, so the server needs no Hub
        # access at startup -- same offline-first pattern as the GPT-2
        # tokenizer folder above.
        if paths.PARSBERT_TOKENIZER_DIR.is_dir():
            _parsbert_tokenizer = AutoTokenizer.from_pretrained(str(paths.PARSBERT_TOKENIZER_DIR))
        else:
            _parsbert_tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
    return _parsbert_tokenizer


def load_sklearn_model(md: dict, vectorizer) -> dict:
    """sklearn-tfidf bundle; carries the vectorizer its predictor needs.

    styleFeatureCount (models.json) is how many style columns the checkpoint
    was fit on; models trained before the feature list grew only get that many
    columns at inference time (see predictors.predict_sklearn).
    """
    return {
        "model": joblib.load(paths.model_path(md["filename"])),
        "vectorizer": vectorizer,
        "style_feature_count": md.get("styleFeatureCount", len(STYLE_FEATURE_NAMES)),
    }


def load_lstm_model(md: dict) -> dict:
    checkpoint = torch.load(
        paths.model_path(md["filename"]), map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        pad_idx=config["pad_idx"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {"model": model, "vocab": vocab, "config": config}


def load_textcnn_model(md: dict) -> dict:
    checkpoint = torch.load(
        paths.model_path(md["filename"]), map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]
    model = TextCNNClassifier(
        vocab_size=len(vocab),
        embed_dim=config["embed_dim"],
        n_filters=config["n_filters"],
        kernel_sizes=tuple(config["kernel_sizes"]),
        pad_idx=config["pad_idx"],
        dropout=config.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {"model": model, "vocab": vocab, "config": config}


def load_attention_bilstm_model(md: dict) -> dict:
    checkpoint = torch.load(
        paths.model_path(md["filename"]), map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]
    model = AttentionBiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        pad_idx=config["pad_idx"],
        dropout=config.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {"model": model, "vocab": vocab, "config": config}


def load_gpt2_backbone():
    """A PRIVATE copy of the base Persian GPT-2 transformer, for one model
    entry only. Deliberately not cached across entries: load_state_dict
    copies weights into the object in place, so handing every pytorch-gpt2
    model the same backbone object would let a second checkpoint silently
    overwrite the first model's fine-tuned blocks. The tokenizer IS shared
    (read-only)."""
    with open(paths.GPT2_LM_MDL, "rb") as f:
        lm = pickle.load(f)
    get_gpt2_tokenizer()
    return lm.transformer


def load_gpt2_model(md: dict) -> dict:
    backbone = load_gpt2_backbone()
    checkpoint = torch.load(
        paths.model_path(md["filename"]), map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    model = GPT2Classifier(backbone, hidden_size=config["hidden_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {"model": model, "config": config}


def load_parsbert_model(md: dict) -> dict:
    checkpoint = torch.load(
        paths.model_path(md["filename"]), map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    # The saved checkpoint carries the full backbone config (layers, heads,
    # vocab size, ...), so the backbone can be rebuilt with the right shape
    # purely offline -- its fine-tuned weights are about to be loaded from
    # state_dict anyway, so no pretrained checkpoint download is needed here.
    backbone_config = BertConfig.from_dict(config["backbone_config"])
    backbone = AutoModel.from_config(backbone_config)
    model = ParsBERTClassifier(backbone, hidden_size=config["hidden_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    get_parsbert_tokenizer()
    return {"model": model, "config": config}


def load_system_one_model(md: dict) -> dict:
    checkpoint = torch.load(
        paths.model_path(md["filename"]), map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    if "backbone_config" in config:
        # ParsBERT-backed checkpoint (training/train_system_one.py): self-contained --
        # the encoder is rebuilt from the stored config (no pretrained
        # download, same recipe as load_parsbert_model) and the fine-tuned
        # weights (backbone + heads) come from the state_dict. Pooled on [CLS].
        backbone = AutoModel.from_config(BertConfig.from_dict(config["backbone_config"]))
        model = SystemOneDecisionModel(
            backbone,
            hidden_size=config["hidden_size"],
            n_choice=config.get("n_choice", 2),
            n_score_levels=config.get("n_score_levels", 5),
            pool_on_cls=True,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        tokenizer = get_parsbert_tokenizer()
    else:
        # Legacy GPT-2-backed checkpoint (pre-ParsBERT-base System One v2), pooled on
        # the last real token. Removable once a retrained checkpoint replaces
        # systemOneModel.pt. load_gpt2_backbone() hands back a private copy, so the
        # fine-tuned weights loaded below cannot clobber any other model's
        # backbone (~500 MB extra RAM).
        backbone = load_gpt2_backbone()
        model = SystemOneDecisionModel(
            backbone,
            hidden_size=config["hidden_size"],
            n_choice=config.get("n_choice", 2),
            n_score_levels=config.get("n_score_levels", 5),
            pool_on_cls=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        tokenizer = get_gpt2_tokenizer()
    return {"model": model, "config": config, "tokenizer": tokenizer}


def load_laya_model(md: dict) -> dict:
    """Fine-tuned Laya checkpoint (models/laya_persian, the RLCD fine-tune of
    the convaiinnovations/laya multilingual checkpoint). The artifact is a
    standard laya checkpoint directory (weights + rl_agent_config.json +
    encoder/ + tokenizer/), so it is loaded through the laya package itself
    and the agent applies the checkpoint's fitted per-question-type
    temperatures exactly as its own serving stack would."""
    try:
        from laya import load
    except ImportError as exc:
        raise RuntimeError(
            "the 'laya' package is required for the pytorch-laya runtime "
            "(pip install laya, or uv sync after adding it to pyproject)"
        ) from exc
    agent = load(str(paths.LAYA_OUTPUT_DIR), device="cpu")
    with open(paths.LAYA_OUTPUT_DIR / "rl_agent_config.json", encoding="utf-8") as f:
        config = json.load(f)
    return {"agent": agent, "config": config}


LOADERS = {
    "sklearn-tfidf": load_sklearn_model,
    "pytorch-lstm": load_lstm_model,
    "pytorch-textcnn": load_textcnn_model,
    "pytorch-bilstm-attn": load_attention_bilstm_model,
    "pytorch-gpt2": load_gpt2_model,
    "pytorch-parsbert": load_parsbert_model,
    "pytorch-system-one": load_system_one_model,
    "pytorch-laya": load_laya_model,
}
