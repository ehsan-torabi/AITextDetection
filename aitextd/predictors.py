"""Per-runtime prediction.

Every predictor takes the bundle produced by aitextd.loaders plus the raw
text, its preprocessed form, and the style features. All except the System
One decision model return a plain AI probability; the System One model
returns its full typed answers dict (see aitextd.system_one).
"""

import numpy as np
import torch
from scipy.sparse import hstack

from aitextd import loaders
from aitextd.constants import (
    SYSTEM_ONE_CHOICE_CANDIDATES,
    SYSTEM_ONE_DETECTION_QUESTIONS,
    SYSTEM_ONE_SCORE_LEVELS,
)
from aitextd.features import STYLE_FEATURE_NAMES, style_features_vector
from aitextd.text import normalize_for_pretrained


def predict_sklearn(bundle, text: str, cleaned_text: str, style_features: dict) -> float:
    # Every sklearn-tfidf model (SVM, LogReg, NB directly; RF/GB via an
    # internal ColumnTransformer that chi2-selects the TF-IDF slice and
    # passes the style slice through) was fit on TF-IDF columns followed by
    # its style columns, in that order. Checkpoints trained before the style
    # list grew carry a smaller styleFeatureCount and only see the columns
    # they were fit on.
    tfidf_vect = bundle["vectorizer"].transform([cleaned_text])
    n_style = bundle.get("style_feature_count", len(STYLE_FEATURE_NAMES))
    style_vec = style_features_vector(style_features)[:, :n_style]
    vect_text = hstack([tfidf_vect, style_vec]).tocsr()
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vect_text)[0]
        return float(proba[1]) if len(proba) > 1 else float(proba[0])
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(vect_text)[0])
        return 1 / (1 + np.exp(-score))
    return 1.0 if model.predict(vect_text)[0] else 0.0


def _vocab_encode_and_predict(bundle, cleaned_text: str) -> float:
    """Shared forward pass for the vocab-based DL runtimes (pytorch-lstm,
    pytorch-textcnn, pytorch-bilstm-attn): whitespace tokens of the cleaned
    text -> ids via the checkpoint's vocab -> pad/truncate to config max_len
    -> single forward pass -> sigmoid."""
    model = bundle["model"]
    vocab = bundle["vocab"]
    config = bundle["config"]
    tokens = cleaned_text.split()[: config["max_len"]]
    ids = [vocab.get(t, config["unk_idx"]) for t in tokens]
    if len(ids) < config["max_len"]:
        ids = ids + [config["pad_idx"]] * (config["max_len"] - len(ids))
    with torch.no_grad():
        logits = model(torch.tensor([ids], dtype=torch.long))
        prob = torch.sigmoid(logits).item()
    return prob


def predict_lstm(bundle, text: str, cleaned_text: str, style_features: dict) -> float:
    return _vocab_encode_and_predict(bundle, cleaned_text)


def predict_textcnn(bundle, text: str, cleaned_text: str, style_features: dict) -> float:
    return _vocab_encode_and_predict(bundle, cleaned_text)


def predict_attention_bilstm(bundle, text: str, cleaned_text: str, style_features: dict) -> float:
    return _vocab_encode_and_predict(bundle, cleaned_text)


def _encode_and_predict(
    model, tokenizer, normalized: str, seq_len: int, char_budget: int | None
) -> float:
    """Shared forward pass for the pretrained-backbone binary classifiers.

    char_budget guards against the slow-tokenizer fallback path: the pickled
    GPT2Tokenizer BPE-encodes the entire string before truncating, so
    pre-truncating at the character level (a generous upper bound for seq_len
    tokens) keeps it fast. Pass None for fast WordPiece/BPE tokenizers, which
    handle long strings efficiently on their own.
    """
    enc = tokenizer(
        normalized[:char_budget] if char_budget is not None else normalized,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding="max_length",
    )
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        return torch.sigmoid(logits).item()


def predict_gpt2(bundle, text: str, cleaned_text: str, style_features: dict) -> float:
    config = bundle["config"]
    normalized = normalize_for_pretrained(text)
    return _encode_and_predict(
        bundle["model"],
        loaders.get_gpt2_tokenizer(),
        normalized,
        config["seq_len"],
        char_budget=config["seq_len"] * 8,
    )


def predict_parsbert(bundle, text: str, cleaned_text: str, style_features: dict) -> float:
    config = bundle["config"]
    # ParsBERT was fine-tuned on lightly normalized (not lemmatized) text,
    # so reuse the pretrained-model normalization here.
    normalized = normalize_for_pretrained(text)
    return _encode_and_predict(
        bundle["model"],
        loaders.get_parsbert_tokenizer(),
        normalized,
        config["seq_len"],
        char_budget=None,
    )


def predict_system_one(bundle, text: str, cleaned_text: str, style_features: dict) -> dict:
    """One forward pass answers all three typed System One questions at once.
    Returns a dict (not a plain probability like the other predictors) -- the
    store's detection() turns it into the typed-answer block of the response."""
    model = bundle["model"]
    config = bundle["config"]
    tokenizer = bundle["tokenizer"]
    normalized = normalize_for_pretrained(text)
    if "backbone_config" in config:
        # BERT-backed checkpoint: fast WordPiece tokenizer, no character
        # pre-truncation needed.
        char_budget = None
    else:
        # Legacy GPT-2-backed checkpoint: same slow-tokenizer guard as the
        # GPT-2 classifier -- pre-truncate at the character level (generous
        # bound for seq_len tokens in Persian).
        char_budget = config["seq_len"] * 8
    enc = tokenizer(
        normalized[:char_budget] if char_budget is not None else normalized,
        return_tensors="pt",
        truncation=True,
        max_length=config["seq_len"],
        padding="max_length",
    )
    with torch.no_grad():
        choice_logits, score_logits, noul_logit = model(enc["input_ids"], enc["attention_mask"])
        choice_probs = torch.softmax(choice_logits, dim=-1)[0]
        score_probs = torch.softmax(score_logits, dim=-1)[0]
        noul_prob = torch.sigmoid(noul_logit)[0]
    score_value = float((score_probs * torch.arange(len(score_probs), dtype=torch.float32)).sum())
    return {
        # choice probabilities follow SYSTEM_ONE_CHOICE_CANDIDATES order
        # (index 1 == "ai_generated")
        "aiProbability": float(choice_probs[1]),
        "choiceProbs": [float(p) for p in choice_probs],
        "scoreProbs": [float(p) for p in score_probs],
        "scoreValue": score_value,
        "noulProbability": float(noul_prob),
    }


def predict_laya(bundle, text: str, cleaned_text: str, style_features: dict) -> dict:
    """The fine-tuned laya agent answers the same three production questions
    the online path sends (SYSTEM_ONE_DETECTION_QUESTIONS) in one forward
    pass, with the checkpoint's fitted temperatures applied by the laya
    runtime itself. The returned dict mirrors predict_system_one's shape, so
    the typed-answer rendering and the ensemble vote are shared unchanged."""
    result = bundle["agent"].predict({"text": text}, SYSTEM_ONE_DETECTION_QUESTIONS)
    answers = result["answers"]
    choice = answers["source"]["probabilities"]
    score = answers["machine_likeness"]
    score_probs = [
        float(score["probabilities"][str(i)]) for i in range(len(SYSTEM_ONE_SCORE_LEVELS))
    ]
    return {
        # choice probabilities follow SYSTEM_ONE_CHOICE_CANDIDATES order
        # (index 1 == "ai_generated")
        "aiProbability": float(choice["ai_generated"]),
        "choiceProbs": [float(choice[c]) for c in SYSTEM_ONE_CHOICE_CANDIDATES],
        "scoreProbs": score_probs,
        "scoreValue": float(score["score"]),
        "noulProbability": float(answers["needs_review"]["noul"]),
    }


PREDICTORS = {
    "sklearn-tfidf": predict_sklearn,
    "pytorch-lstm": predict_lstm,
    "pytorch-textcnn": predict_textcnn,
    "pytorch-bilstm-attn": predict_attention_bilstm,
    "pytorch-gpt2": predict_gpt2,
    "pytorch-parsbert": predict_parsbert,
    "pytorch-system-one": predict_system_one,
    "pytorch-laya": predict_laya,
}
