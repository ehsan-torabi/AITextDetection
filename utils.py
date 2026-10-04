import json
import math
import os
import pickle
import re
from collections import Counter

import joblib
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel
from scipy.sparse import hstack
from shekar import Lemmatizer, Normalizer, SentenceTokenizer, WordTokenizer
from shekar.preprocessing import EmojiRemover, NonPersianRemover, PunctuationRemover
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer, BertConfig

torch.set_num_threads(4)


class DetectionDetail(BaseModel):
    label: str
    labelFa: str
    value: float  # normalized 0..1
    raw: str
    description: str


class EnsembleVote(BaseModel):
    modelId: str
    label: str
    labelFa: str
    aiProbability: float
    weight: float


class JevChoiceAnswer(BaseModel):
    question: str
    questionFa: str
    candidates: list[str]
    candidateLabelsFa: list[str]
    probabilities: list[float]
    answer: str
    answerFa: str


class JevScoreAnswer(BaseModel):
    question: str
    questionFa: str
    levels: list[str]
    levelLabelsFa: list[str]
    probabilities: list[float]
    value: float  # probability-weighted value on the 0..(n-1) ordinal scale
    answer: str
    answerFa: str


class JevNoulAnswer(BaseModel):
    question: str
    questionFa: str
    probability: float
    answer: bool
    answerFa: str


class JevPolicyDecision(BaseModel):
    tier: str
    tierFa: str
    descriptionFa: str


class JevTypedAnswers(BaseModel):
    choice: JevChoiceAnswer
    score: JevScoreAnswer
    noul: JevNoulAnswer
    policy: JevPolicyDecision


class DetectionResponse(BaseModel):
    isAI: bool
    confidence: float
    verdict: str
    verdictFa: str
    details: list[DetectionDetail]
    modelUsed: str
    textLength: int
    wordCount: int
    ensembleVotes: list[EnsembleVote] | None = None
    # Filled only for the JEV System-One decision model (jev_system_one):
    # typed answers to its three questions instead of a plain probability.
    typedAnswers: JevTypedAnswers | None = None


ENSEMBLE_ID = "ensemble"

# JEV typed-question schema. Keep in sync with training/prepare_jev_data.py
# and training/train_jev.py (the same lists appear in the JSONL targets).
JEV_CHOICE_CANDIDATES = ["human_written", "ai_generated"]
JEV_CHOICE_CANDIDATES_FA = ["نوشته انسان", "تولیدشده توسط هوش مصنوعی"]
JEV_SCORE_LEVELS = [
    "definitely_human",
    "probably_human",
    "mixed_signals",
    "probably_ai",
    "definitely_ai",
]
JEV_SCORE_LEVELS_FA = [
    "قطعاً انسان",
    "احتمالاً انسان",
    "سیگنال‌های ترکیبی",
    "احتمالاً هوش مصنوعی",
    "قطعاً هوش مصنوعی",
]
JEV_QUESTIONS = {
    "choice": {
        "en": "Where does this text come from?",
        "fa": "این متن از کجا آمده است؟",
    },
    "score": {
        "en": "How machine-like is the writing style?",
        "fa": "سبک نوشتار چقدر ماشین‌مانند است؟",
    },
    "noul": {
        "en": "Does this text need human review?",
        "fa": "آیا این متن نیاز به بازبینی انسانی دارد؟",
    },
}

vectorizer = None
models_cache: dict[str, object] = {}  # model_id -> loaded model object/bundle
models_dict: dict[str, dict] = {}  # model_id -> metadata from models.json

_preprocess_pipeline = None
_gpt2_tokenizer = None  # shared across any pytorch-gpt2 models


# ---------------------------------------------------------------------------
# Deep learning model architectures (must match training/train_*.py exactly
# so saved state_dicts load correctly).
# ---------------------------------------------------------------------------
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


class JevDecisionModel(nn.Module):
    """JEV-style 'System One' decision model: a pretrained-transformer backbone
    whose pooled representation feeds three typed decision heads -- choice
    (distribution over candidate answers), score (distribution over an ordinal
    scale, answered as a probability-weighted value) and noul (yes/no
    probability). All heads are scored in one forward pass. Current checkpoints
    pool the bidirectional ParsBERT backbone on [CLS]; legacy checkpoints pooled
    a causal GPT-2 backbone on the last real token, so pooling is selected by
    the pool_on_cls flag the loader sets from the checkpoint format. Must match
    training/train_jev.py exactly so the saved state_dict loads correctly."""

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


def _get_preprocess_pipeline():
    global _preprocess_pipeline
    if _preprocess_pipeline is None:
        _preprocess_pipeline = (
            EmojiRemover()
            | PunctuationRemover()
            | NonPersianRemover()
            | Normalizer()
            | Lemmatizer()
            | WordTokenizer()
        )
    return _preprocess_pipeline


_gpt2_normalizer = None


def _normalize_for_gpt2(text: str) -> str:
    # GPT-2 was pretrained on natural Persian text, so unlike the TF-IDF/BiLSTM
    # pipeline above we deliberately do NOT lemmatize or word-tokenize here —
    # that would strip inflection/morphology and feed the model text unlike
    # anything it saw during pretraining. Just character-level normalization
    # (Arabic->Persian unification, digit/punctuation normalization, ZWNJ and
    # spacing fixes) to clean up inconsistent real-world typing, matching how
    # training/colab_retrain.ipynb preprocesses text for GPT-2 fine-tuning.
    global _gpt2_normalizer
    if _gpt2_normalizer is None:
        _gpt2_normalizer = Normalizer()
    text = re.sub(r"\n", " ", text)
    return _gpt2_normalizer(text)


def _load_sklearn_model(md: dict):
    return joblib.load(md["filename"])


def _load_lstm_model(md: dict):
    checkpoint = torch.load(md["filename"], map_location="cpu", weights_only=False)
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


def _get_gpt2_tokenizer():
    global _gpt2_tokenizer
    if _gpt2_tokenizer is None:
        # Prefer an offline fast (Rust-backed) tokenizer exported by
        # training/colab_retrain.ipynb via tokenizer.save_pretrained(), which is
        # both faster and less prone to slow-tokenizer pathologies on long
        # documents than the pickled slow GPT2Tokenizer. Falls back to the
        # pickle when that export isn't present.
        if os.path.isdir("gpt2_tokenizer"):
            _gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2_tokenizer")
        else:
            with open("gpt2-fa-tokenizer.mdl", "rb") as f:
                _gpt2_tokenizer = pickle.load(f)
            _gpt2_tokenizer.pad_token = "<pad>"
    return _gpt2_tokenizer


def _load_gpt2_backbone():
    """Load the base Persian GPT-2 transformer + tokenizer once, shared by
    every pytorch-gpt2 model entry."""
    with open("gpt2-fa.mdl", "rb") as f:
        lm = pickle.load(f)
    _get_gpt2_tokenizer()
    return lm.transformer


def _load_gpt2_model(md: dict):
    backbone = _load_gpt2_backbone()
    checkpoint = torch.load(md["filename"], map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = GPT2Classifier(backbone, hidden_size=config["hidden_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {"model": model, "config": config}


def _load_jev_model(md: dict):
    checkpoint = torch.load(md["filename"], map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if "backbone_config" in config:
        # ParsBERT-backed checkpoint (training/train_jev.py): self-contained --
        # the encoder is rebuilt from the stored config (no pretrained
        # download, same recipe as _load_parsbert_model) and the fine-tuned
        # weights (backbone + heads) come from the state_dict. Pooled on [CLS].
        backbone = AutoModel.from_config(BertConfig.from_dict(config["backbone_config"]))
        model = JevDecisionModel(
            backbone,
            hidden_size=config["hidden_size"],
            n_choice=config.get("n_choice", 2),
            n_score_levels=config.get("n_score_levels", 5),
            pool_on_cls=True,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        tokenizer = _load_parsbert_tokenizer()
    else:
        # Legacy GPT-2-backed checkpoint (pre-ParsBERT-base JEV), pooled on the
        # last real token. Removable once a retrained checkpoint replaces
        # jevModel.pt. Deliberately NOT using _load_gpt2_backbone(): that
        # helper hands every pytorch-gpt2 model the SAME backbone object, and
        # load_state_dict copies weights into it in place -- a second
        # checkpoint would silently overwrite the first model's fine-tuned
        # blocks. JEV therefore loads its own private backbone copy from the
        # pickle (~500 MB extra RAM).
        with open("gpt2-fa.mdl", "rb") as f:
            lm = pickle.load(f)
        model = JevDecisionModel(
            lm.transformer,
            hidden_size=config["hidden_size"],
            n_choice=config.get("n_choice", 2),
            n_score_levels=config.get("n_score_levels", 5),
            pool_on_cls=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        tokenizer = _get_gpt2_tokenizer()
    return {"model": model, "config": config, "tokenizer": tokenizer}


_gpt2_lm = None  # private copy of the raw pretrained LM for the perplexity signal


def _get_perplexity_lm():
    """The raw pretrained Persian GPT-2 (LM head included), loaded once.

    Deliberately a private copy loaded straight from the pickle, not one of
    the fine-tuned classifiers' backbones: load_state_dict overwrote their
    last blocks, so they no longer model "generic Persian" -- the perplexity
    signal must measure predictability under the untouched pretrained LM.
    Same self-contained-copy tradeoff as _load_jev_model (~500 MB extra RAM).
    """
    global _gpt2_lm
    if _gpt2_lm is None:
        with open("gpt2-fa.mdl", "rb") as f:
            _gpt2_lm = pickle.load(f)
        _gpt2_lm.eval()
    return _gpt2_lm


# Context window for the perplexity signal. The raw LM has no trained seq_len
# of its own; 64 tokens (with the usual char-budget guard) bounds the
# per-request forward cost and matches the local serving path.
PERPLEXITY_SEQ_LEN = 64
PERPLEXITY_BATCH_SIZE = 32


def gpt2_perplexity_batch(texts: list[str]) -> np.ndarray:
    """Perplexity of each text under the raw pretrained GPT-2-fa: exp of the
    mean next-token negative log-likelihood over real (non-pad) tokens.
    Same serving normalization and char-budget guard as _predict_gpt2, so
    the signal sees exactly what the deployed GPT-2 runtime sees."""
    lm = _get_perplexity_lm()
    tokenizer = _get_gpt2_tokenizer()
    char_budget = PERPLEXITY_SEQ_LEN * 8
    results: list[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), PERPLEXITY_BATCH_SIZE):
            enc = tokenizer(
                [
                    _normalize_for_gpt2(str(t))[:char_budget]
                    for t in texts[start : start + PERPLEXITY_BATCH_SIZE]
                ],
                return_tensors="pt",
                truncation=True,
                max_length=PERPLEXITY_SEQ_LEN,
                padding="max_length",
            )
            logits = lm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            # Logits at position j predict token j+1; score only real targets.
            target_ids = enc["input_ids"][:, 1:]
            target_mask = enc["attention_mask"][:, 1:].bool()
            token_nll = -log_probs[:, :-1].gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            for i in range(token_nll.size(0)):
                nll = token_nll[i][target_mask[i]]
                # Texts too short for even one prediction stay NaN.
                results.append(float(nll.mean().exp()) if nll.numel() else float("nan"))
    return np.array(results)


def gpt2_perplexity(text: str) -> float:
    """Single-text convenience wrapper around gpt2_perplexity_batch."""
    return float(gpt2_perplexity_batch([text])[0])


_parsbert_tokenizer = None


def _load_parsbert_tokenizer():
    global _parsbert_tokenizer
    if _parsbert_tokenizer is None:
        # Prefer the tokenizer exported by training/colab_retrain.ipynb via
        # save_pretrained(), so the server needs no Hub access at startup --
        # same offline-first pattern as the gpt2_tokenizer/ folder above.
        if os.path.isdir("parsbert_tokenizer"):
            _parsbert_tokenizer = AutoTokenizer.from_pretrained("parsbert_tokenizer")
        else:
            _parsbert_tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
    return _parsbert_tokenizer


def _load_parsbert_model(md: dict):
    checkpoint = torch.load(md["filename"], map_location="cpu", weights_only=False)
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
    _load_parsbert_tokenizer()
    return {"model": model, "config": config}


_LOADERS = {
    "sklearn-tfidf": _load_sklearn_model,
    "pytorch-lstm": _load_lstm_model,
    "pytorch-gpt2": _load_gpt2_model,
    "pytorch-parsbert": _load_parsbert_model,
    "pytorch-jev": _load_jev_model,
}


def init() -> None:
    """Load the vectorizer and all models once at startup."""
    global vectorizer, models_dict

    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    with open("models.json", encoding="utf-8") as f:
        metadata_list = json.load(f)
    models_dict = {md["id"]: md for md in metadata_list}

    for model_id, md in models_dict.items():
        runtime = md.get("runtime", "sklearn-tfidf")
        loader = _LOADERS.get(runtime, _load_sklearn_model)
        models_cache[model_id] = loader(md)

    # The perplexity signal runs on every detection request; warm it up here
    # so the first request doesn't pay the pickle load.
    _get_perplexity_lm()


def get_model_metadata(model_id: str) -> dict:
    if model_id not in models_dict:
        raise ValueError(f"Model ID '{model_id}' not found.")
    return models_dict[model_id]


def get_ensemble_metadata() -> dict:
    """Virtual "model" metadata for the weighted ensemble, exposed to clients
    alongside the real models so it can be listed/selected like any other."""
    return {
        "id": ENSEMBLE_ID,
        "name": "Ensemble (All Models)",
        "nameFa": "مدل ترکیبی (رأی‌گیری وزن‌دار)",
        "description": (
            "Weighted-vote ensemble that combines the prediction of every "
            "trained model, giving extra weight to the fine-tuned GPT-2 model."
        ),
        "descriptionFa": (
            "ترکیب وزن‌دار پیش‌بینی همه مدل‌های آموزش‌دیده، با وزن بیشتر برای "
            "مدل GPT-2 ریزتنظیم‌شده که معمولاً دقیق‌ترین مدل است."
        ),
        "algorithm": "WeightedEnsemble",
        "type": "ensemble",
        "version": "1.0",
    }


def preprocess_persian_text(text: str) -> str:
    text = re.sub(r"\n", " ", text)
    tokens = _get_preprocess_pipeline()(text)
    return " ".join(tokens)


def _tokenize_persian_text(text: str) -> list[str]:
    text = re.sub(r"\n", " ", text)
    return list(_get_preprocess_pipeline()(text))


def shannon_entropy(text: str) -> float:
    char_counts = Counter(text)
    total_chars = len(text)
    entropy = 0.0
    for count in char_counts.values():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
    return entropy


def sentence_variety(text: str) -> float:
    """Coefficient of variation of sentence lengths (0..1)."""
    sentences = list(SentenceTokenizer().tokenize(text))
    if len(sentences) < 2:
        return 0.0
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    mean_length = float(np.mean(sentence_lengths))
    if mean_length == 0:
        return 0.0
    return min(float(np.std(sentence_lengths)) / mean_length, 1.0)


def pattern_repetition(text: str, n: int = 3) -> float:
    """Fraction of repeated n-grams (0..1)."""
    tokens = list(WordTokenizer().tokenize(text))
    if len(tokens) < n:
        return 0.0
    # The shifted slices intentionally have different lengths (n-gram windows);
    # strict=False is required, strict=True would raise.
    ngrams = zip(*[tokens[i:] for i in range(n)], strict=False)
    ngram_counts = Counter(" ".join(ngram) for ngram in ngrams)
    if not ngram_counts:
        return 0.0
    repeated = sum(1 for count in ngram_counts.values() if count > 1)
    return repeated / len(ngram_counts)


def text_coherence(text: str) -> float:
    """Mean cosine similarity between consecutive sentences of this text (0..1)."""
    sentences = [s for s in SentenceTokenizer().tokenize(text) if s.strip()]
    if len(sentences) < 2:
        return 1.0
    matrix = vectorizer.transform(sentences)
    similarities = [
        float(cosine_similarity(matrix[i], matrix[i + 1])[0][0]) for i in range(matrix.shape[0] - 1)
    ]
    return min(max(float(np.mean(similarities)), 0.0), 1.0)


# Stylometric features appended to the TF-IDF vector for every sklearn-tfidf
# model (SVM, Logistic Regression, Naive Bayes, Random Forest, Gradient
# Boosting). Order here MUST match training/build_style_features.py and the
# column order every sklearn-tfidf .joblib bundle was fit on -- changing it
# (or the formulas below) without retraining those models will silently feed
# them garbage.
STYLE_FEATURE_NAMES = ["entropy", "sentence_variety", "repetition", "coherence"]


def compute_style_features(text: str, cleaned_text: str) -> dict[str, float]:
    """Normalized (0..1) stylometric features for one document."""
    entropy = shannon_entropy(text)
    max_entropy = math.log2(max(len(set(cleaned_text.replace(" ", ""))), 2))
    return {
        "entropy": round(min(entropy / max_entropy, 1.0), 4),
        "sentence_variety": round(sentence_variety(text), 4),
        "repetition": round(pattern_repetition(text, n=3), 4),
        "coherence": round(text_coherence(text), 4),
    }


def style_features_vector(style_features: dict[str, float]) -> np.ndarray:
    """dict -> ordered (1, len(STYLE_FEATURE_NAMES)) array, ready to hstack
    onto a TF-IDF row for sklearn-tfidf models."""
    return np.array([[style_features[name] for name in STYLE_FEATURE_NAMES]])


def predict_ai_probability(model, vect_text) -> tuple[bool, float]:
    """Return (is_ai, confidence) for a fitted sklearn classifier."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vect_text)[0]
        ai_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(vect_text)[0])
        ai_prob = 1 / (1 + np.exp(-score))
    else:
        ai_prob = 1.0 if model.predict(vect_text)[0] else 0.0

    is_ai = ai_prob >= 0.5
    confidence = ai_prob if is_ai else 1 - ai_prob
    return is_ai, round(confidence, 4)


def _predict_sklearn(bundle, text: str, cleaned_text: str, style_features: dict) -> float:
    # Every sklearn-tfidf model (SVM, LogReg, NB directly; RF/GB via an
    # internal ColumnTransformer that chi2-selects the TF-IDF slice and
    # passes the style slice through) was fit on TF-IDF columns followed by
    # the 4 style columns, in that order -- see training/colab_retrain.ipynb.
    tfidf_vect = vectorizer.transform([cleaned_text])
    vect_text = hstack([tfidf_vect, style_features_vector(style_features)]).tocsr()
    if hasattr(bundle, "predict_proba"):
        proba = bundle.predict_proba(vect_text)[0]
        return float(proba[1]) if len(proba) > 1 else float(proba[0])
    if hasattr(bundle, "decision_function"):
        score = float(bundle.decision_function(vect_text)[0])
        return 1 / (1 + np.exp(-score))
    return 1.0 if bundle.predict(vect_text)[0] else 0.0


def _predict_lstm(bundle, text: str, cleaned_text: str, style_features: dict) -> float:
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


def _predict_gpt2(bundle, text: str, cleaned_text: str, style_features: dict) -> float:
    model = bundle["model"]
    config = bundle["config"]
    normalized = _normalize_for_gpt2(text)
    # Guards against the slow-tokenizer fallback path (see _load_gpt2_backbone):
    # the pickled GPT2Tokenizer BPE-encodes the entire string before truncating,
    # so pre-truncate at the character level (generous upper bound for seq_len
    # tokens) to keep it fast. A no-op for the fast tokenizer, which handles
    # long strings efficiently on its own.
    char_budget = config["seq_len"] * 8
    enc = _gpt2_tokenizer(
        normalized[:char_budget],
        return_tensors="pt",
        truncation=True,
        max_length=config["seq_len"],
        padding="max_length",
    )
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        prob = torch.sigmoid(logits).item()
    return prob


def _predict_parsbert(bundle, text: str, cleaned_text: str, style_features: dict) -> float:
    model = bundle["model"]
    config = bundle["config"]
    # Same rationale as _predict_gpt2: ParsBERT was fine-tuned on lightly
    # normalized (not lemmatized) text, so reuse that normalization here.
    normalized = _normalize_for_gpt2(text)
    enc = _parsbert_tokenizer(
        normalized,
        return_tensors="pt",
        truncation=True,
        max_length=config["seq_len"],
        padding="max_length",
    )
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        prob = torch.sigmoid(logits).item()
    return prob


def _predict_jev(bundle, text: str, cleaned_text: str, style_features: dict) -> dict:
    """One forward pass answers all three typed JEV questions at once. Returns
    a dict (not a plain probability like the other predictors) -- detection()
    turns it into the typed-answer block of the response."""
    model = bundle["model"]
    config = bundle["config"]
    tokenizer = bundle["tokenizer"]
    normalized = _normalize_for_gpt2(text)
    if "backbone_config" in config:
        # BERT-backed checkpoint: fast WordPiece tokenizer, same encode path
        # as _predict_parsbert (no character pre-truncation needed).
        enc = tokenizer(
            normalized,
            return_tensors="pt",
            truncation=True,
            max_length=config["seq_len"],
            padding="max_length",
        )
    else:
        # Legacy GPT-2-backed checkpoint: same slow-tokenizer guard as
        # _predict_gpt2 -- pre-truncate at the character level (generous
        # bound for seq_len tokens in Persian).
        char_budget = config["seq_len"] * 8
        enc = tokenizer(
            normalized[:char_budget],
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
        # choice probabilities follow JEV_CHOICE_CANDIDATES order
        # (index 1 == "ai_generated")
        "aiProbability": float(choice_probs[1]),
        "choiceProbs": [float(p) for p in choice_probs],
        "scoreProbs": [float(p) for p in score_probs],
        "scoreValue": score_value,
        "noulProbability": float(noul_prob),
    }


_PREDICTORS = {
    "sklearn-tfidf": _predict_sklearn,
    "pytorch-lstm": _predict_lstm,
    "pytorch-gpt2": _predict_gpt2,
    "pytorch-parsbert": _predict_parsbert,
    "pytorch-jev": _predict_jev,
}


def _predict_single(model_id: str, text: str, cleaned_text: str, style_features: dict):
    """Run one model: returns its AI probability as a float, except the JEV
    decision model which returns the full typed-answers dict."""
    bundle = models_cache[model_id]
    md = models_dict[model_id]
    runtime = md.get("runtime", "sklearn-tfidf")
    predictor = _PREDICTORS.get(runtime, _predict_sklearn)
    return predictor(bundle, text, cleaned_text, style_features)


def _jev_typed_answers(bundle, result: dict) -> JevTypedAnswers:
    """Wrap the raw JEV forward-pass output into the typed-answer response
    block, including the three-tier decision policy (proceed / needs review /
    low confidence) taken from the checkpoint's thresholds."""
    choice_probs = result["choiceProbs"]
    choice_idx = int(np.argmax(choice_probs))
    score_probs = result["scoreProbs"]
    score_idx = int(np.argmax(score_probs))

    thresholds = bundle["config"].get("thresholds", {})
    proceed_thr = float(thresholds.get("proceed", 0.8))
    review_thr = float(thresholds.get("needsReview", 0.6))
    confidence = max(choice_probs)
    if confidence >= proceed_thr:
        policy = JevPolicyDecision(
            tier="proceed",
            tierFa="ادامه",
            descriptionFa="اطمینان مدل بالاست؛ تصمیم خودکار قابل‌اعتماد است.",
        )
    elif confidence >= review_thr:
        policy = JevPolicyDecision(
            tier="needs_review",
            tierFa="نیازمند بازبینی انسانی",
            descriptionFa="پاسخ مدل در ناحیه مرزی است؛ پیش از اقدام، بازبینی انسانی توصیه می‌شود.",
        )
    else:
        policy = JevPolicyDecision(
            tier="low_confidence",
            tierFa="اطمینان پایین",
            descriptionFa="اطمینان مدل کم است؛ تصمیم خودکار توصیه نمی‌شود.",
        )

    return JevTypedAnswers(
        choice=JevChoiceAnswer(
            question=JEV_QUESTIONS["choice"]["en"],
            questionFa=JEV_QUESTIONS["choice"]["fa"],
            candidates=JEV_CHOICE_CANDIDATES,
            candidateLabelsFa=JEV_CHOICE_CANDIDATES_FA,
            probabilities=[round(p, 4) for p in choice_probs],
            answer=JEV_CHOICE_CANDIDATES[choice_idx],
            answerFa=JEV_CHOICE_CANDIDATES_FA[choice_idx],
        ),
        score=JevScoreAnswer(
            question=JEV_QUESTIONS["score"]["en"],
            questionFa=JEV_QUESTIONS["score"]["fa"],
            levels=JEV_SCORE_LEVELS,
            levelLabelsFa=JEV_SCORE_LEVELS_FA,
            probabilities=[round(p, 4) for p in score_probs],
            value=round(result["scoreValue"], 3),
            answer=JEV_SCORE_LEVELS[score_idx],
            answerFa=JEV_SCORE_LEVELS_FA[score_idx],
        ),
        noul=JevNoulAnswer(
            question=JEV_QUESTIONS["noul"]["en"],
            questionFa=JEV_QUESTIONS["noul"]["fa"],
            probability=round(result["noulProbability"], 4),
            answer=result["noulProbability"] >= 0.5,
            answerFa="بله" if result["noulProbability"] >= 0.5 else "خیر",
        ),
        policy=policy,
    )


def _ensemble_predict(
    text: str, cleaned_text: str, style_features: dict
) -> tuple[bool, float, list[EnsembleVote]]:
    """Weighted-average vote across every loaded model. Each model's weight
    comes from its "ensembleWeight" metadata field (models.json) -- the
    fine-tuned GPT-2 model is weighted higher there since it consistently
    outperforms the classical ML models on held-out text."""
    votes: list[EnsembleVote] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for model_id, md in models_dict.items():
        weight = float(md.get("ensembleWeight", 1.0))
        if weight <= 0.0:
            # Models can opt out of the vote via "ensembleWeight": 0.
            continue
        result = _predict_single(model_id, text, cleaned_text, style_features)
        if isinstance(result, dict):
            # pytorch-jev returns the full typed-answers dict; vote with its
            # choice-question probability, same as detection() does.
            ai_prob = result["aiProbability"]
        else:
            ai_prob = result
        weighted_sum += ai_prob * weight
        weight_total += weight
        votes.append(
            EnsembleVote(
                modelId=model_id,
                label=md.get("name", model_id),
                labelFa=md.get("nameFa", md.get("name", model_id)),
                aiProbability=round(ai_prob, 4),
                weight=weight,
            )
        )

    ai_prob_avg = weighted_sum / weight_total if weight_total else 0.0
    is_ai = ai_prob_avg >= 0.5
    confidence = round(ai_prob_avg if is_ai else 1 - ai_prob_avg, 4)
    return is_ai, confidence, votes


def detection(text: str, model_id: str) -> DetectionResponse:
    """Detect whether the given Persian text is AI-generated using the specified
    model, or every model at once via the weighted ensemble (model_id == "ensemble")."""
    is_ensemble = model_id == ENSEMBLE_ID
    if not is_ensemble and (model_id not in models_cache or model_id not in models_dict):
        raise ValueError(f"Model ID '{model_id}' not found.")

    cleaned_text = preprocess_persian_text(text)

    if not cleaned_text.strip():
        return DetectionResponse(
            isAI=False,
            confidence=0.0,
            verdict="Unable to analyze empty text",
            verdictFa="متن خالی قابل تحلیل نیست",
            details=[],
            modelUsed=model_id,
            textLength=len(text),
            wordCount=len(text.split()),
        )

    # Computed once and reused both as extra model input (sklearn-tfidf
    # models) and for the human-readable detail breakdown below.
    style_features = compute_style_features(text, cleaned_text)
    entropy = shannon_entropy(text)  # raw bits, for the "raw" display field only

    ensemble_votes = None
    typed_answers = None
    if is_ensemble:
        is_ai, confidence, ensemble_votes = _ensemble_predict(text, cleaned_text, style_features)
    else:
        result = _predict_single(model_id, text, cleaned_text, style_features)
        if isinstance(result, dict):
            # pytorch-jev: one forward pass, three typed answers
            typed_answers = _jev_typed_answers(models_cache[model_id], result)
            ai_prob = result["aiProbability"]
        else:
            ai_prob = result
        is_ai = ai_prob >= 0.5
        confidence = round(ai_prob if is_ai else 1 - ai_prob, 4)

    word_count = len(text.split())
    char_count = len(text)

    details = [
        DetectionDetail(
            label="entropy",
            labelFa="آنتروپی",
            value=style_features["entropy"],
            raw=f"{entropy:.2f} bits",
            description=f"آنتروپی کاراکترها: {entropy:.2f} بیت (نرمال‌شده بر حسب تنوع کاراکتر)",
        ),
        DetectionDetail(
            label="sentence_variety",
            labelFa="تنوع جملات",
            value=style_features["sentence_variety"],
            raw=f"{style_features['sentence_variety']:.2f}",
            description=f"پراکندگی طول جمله‌ها (ضریب تغییرات): {style_features['sentence_variety']:.2f}",
        ),
        DetectionDetail(
            label="repetition",
            labelFa="تکرار",
            value=style_features["repetition"],
            raw=f"{style_features['repetition'] * 100:.1f}%",
            description=f"درصد تکرار سه‌گرام‌ها: {style_features['repetition'] * 100:.1f}%",
        ),
        DetectionDetail(
            label="coherence",
            labelFa="انسجام",
            value=style_features["coherence"],
            raw=f"{style_features['coherence']:.2f}",
            description=f"میانگین شباهت جمله‌های مجاور: {style_features['coherence']:.2f}",
        ),
    ]

    # Raw-LM perplexity: a training-free view of the text (how "surprised" the
    # untouched pretrained GPT-2 is), independent of every fine-tuned vote.
    ppl = gpt2_perplexity(text)
    if not math.isnan(ppl):
        details.append(
            DetectionDetail(
                label="perplexity",
                labelFa="سرگشتگی",
                value=round(ppl, 2),
                raw=f"{ppl:.2f}",
                description=(
                    "سرگشتگی متن تحت مدل زبانی خام GPT-2 فارسی (بدون آموزش): "
                    "متنِ قابل‌پیش‌بینی‌تر امتیاز کمتری می‌گیرد؛ سیگنالی مستقل از "
                    "مدل‌های آموزش‌دیده."
                ),
            )
        )

    if is_ai:
        verdict = "This text appears to be AI-generated."
        verdictFa = "این متن به نظر تولید شده توسط هوش مصنوعی است."
    else:
        verdict = "This text appears to be human-written."
        verdictFa = "این متن به نظر توسط انسان نوشته شده است."

    return DetectionResponse(
        isAI=is_ai,
        confidence=confidence,
        verdict=verdict,
        verdictFa=verdictFa,
        details=details,
        modelUsed=model_id,
        textLength=char_count,
        wordCount=word_count,
        ensembleVotes=ensemble_votes,
        typedAnswers=typed_answers,
    )
