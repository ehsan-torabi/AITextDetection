import json
import math
import re
from collections import Counter

import joblib
import numpy as np
from pydantic import BaseModel
from shekar import Normalizer, WordTokenizer, Lemmatizer, SentenceTokenizer
from shekar.preprocessing import EmojiRemover, PunctuationRemover, NonPersianRemover
from sklearn.metrics.pairwise import cosine_similarity


class DetectionDetail(BaseModel):
    label: str
    labelFa: str
    value: float  # normalized 0..1
    raw: str
    description: str


class DetectionResponse(BaseModel):
    isAI: bool
    confidence: float
    verdict: str
    verdictFa: str
    details: list[DetectionDetail]
    modelUsed: str
    textLength: int
    wordCount: int


vectorizer = None
models_cache: dict[str, object] = {}  # model_id -> loaded model object
models_dict: dict[str, dict] = {}  # model_id -> metadata from models.json

_preprocess_pipeline = None


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


def init() -> None:
    """Load the vectorizer and all models once at startup."""
    global vectorizer, models_cache, models_dict

    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    with open("models.json", encoding="utf-8") as f:
        metadata_list = json.load(f)
    models_dict = {md["id"]: md for md in metadata_list}

    for model_id, md in models_dict.items():
        models_cache[model_id] = joblib.load(md["filename"])


def get_model_metadata(model_id: str) -> dict:
    if model_id not in models_dict:
        raise ValueError(f"Model ID '{model_id}' not found.")
    return models_dict[model_id]


def preprocess_persian_text(text: str) -> str:
    text = re.sub(r"\n", " ", text)
    tokens = _get_preprocess_pipeline()(text)
    return " ".join(tokens)


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
    ngrams = zip(*[tokens[i:] for i in range(n)])
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
        float(cosine_similarity(matrix[i], matrix[i + 1])[0][0])
        for i in range(matrix.shape[0] - 1)
    ]
    return min(max(float(np.mean(similarities)), 0.0), 1.0)


def predict_ai_probability(model, vect_text) -> tuple[bool, float]:
    """Return (is_ai, confidence) for a fitted classifier."""
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


def detection(text: str, model_id: str) -> DetectionResponse:
    """Detect whether the given Persian text is AI-generated using the specified model."""
    model = models_cache.get(model_id)
    if model is None:
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

    vect_text = vectorizer.transform([cleaned_text])
    is_ai, confidence = predict_ai_probability(model, vect_text)

    word_count = len(text.split())
    char_count = len(text)
    entropy = shannon_entropy(text)
    variety = sentence_variety(text)
    repetition = pattern_repetition(text, n=3)
    coherence = text_coherence(text)

    max_entropy = math.log2(max(len(set(cleaned_text.replace(" ", ""))), 2))
    details = [
        DetectionDetail(
            label="entropy",
            labelFa="آنتروپی",
            value=round(min(entropy / max_entropy, 1.0), 4),
            raw=f"{entropy:.2f} bits",
            description=f"آنتروپی کاراکترها: {entropy:.2f} بیت (نرمال‌شده بر حسب تنوع کاراکتر)",
        ),
        DetectionDetail(
            label="sentence_variety",
            labelFa="تنوع جملات",
            value=round(variety, 4),
            raw=f"{variety:.2f}",
            description=f"پراکندگی طول جمله‌ها (ضریب تغییرات): {variety:.2f}",
        ),
        DetectionDetail(
            label="repetition",
            labelFa="تکرار",
            value=round(repetition, 4),
            raw=f"{repetition * 100:.1f}%",
            description=f"درصد تکرار سه‌گرام‌ها: {repetition * 100:.1f}%",
        ),
        DetectionDetail(
            label="coherence",
            labelFa="انسجام",
            value=round(coherence, 4),
            raw=f"{coherence:.2f}",
            description=f"میانگین شباهت جمله‌های مجاور: {coherence:.2f}",
        ),
    ]

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
    )
