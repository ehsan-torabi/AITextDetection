"""Stylometric features appended to the TF-IDF vector for every sklearn-tfidf
model (SVM, Logistic Regression, Naive Bayes, Random Forest, Gradient
Boosting), and shown as the human-readable detail breakdown of every response.

Column order is STYLE_FEATURE_NAMES. training/build_style_features.py imports
this same list, and every sklearn-tfidf .joblib bundle was fit on TF-IDF
columns followed by these style columns, in this order -- changing the
formulas (or the 4-decimal rounding) without retraining those models will
silently feed them garbage.
"""

import math
from collections import Counter
from itertools import pairwise

import numpy as np
from shekar import SentenceTokenizer, WordTokenizer
from sklearn.metrics.pairwise import cosine_similarity

STYLE_FEATURE_NAMES = [
    "entropy",
    "sentence_variety",
    "repetition",
    "coherence",
    "lexical_diversity",
    "sentence_length",
    "function_word_ratio",
    "punctuation_rate",
    "lexical_burstiness",
    "clause_variety",
]

# Bump when any formula (or its 4-decimal rounding) changes, so on-disk
# style-feature caches keyed on it are rebuilt instead of silently reused.
STYLE_FEATURES_VERSION = 2

# Small closed-class core for function_word_ratio. Deliberately not a
# full stopword list: this measures style register, and open-class synonyms
# would blur it. ponytail: learn the set from corpus DF if more is needed.
FUNCTION_WORDS = frozenset(
    {
        "و",
        "در",
        "به",
        "از",
        "که",
        "این",
        "آن",
        "با",
        "برای",
        "را",
        "تا",
        "اما",
        "یا",
        "اگر",
        "چون",
        "نه",
        "هم",
        "خود",
        "ما",
        "شما",
        "او",
        "من",
        "تو",
        "نیز",
        "فقط",
        "همه",
        "باید",
        "است",
        "بود",
        "نیست",
        "شد",
        "می",
        "نمی",
        "ها",
    }
)

PUNCTUATION = frozenset(".,!?؟:؛،…'\"()[]{}«»ـ–—-")


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


def text_coherence(text: str, vectorizer) -> float:
    """Mean cosine similarity between consecutive sentences of this text (0..1)."""
    sentences = [s for s in SentenceTokenizer().tokenize(text) if s.strip()]
    if len(sentences) < 2:
        return 1.0
    matrix = vectorizer.transform(sentences)
    similarities = [
        float(cosine_similarity(matrix[i], matrix[i + 1])[0][0]) for i in range(matrix.shape[0] - 1)
    ]
    return min(max(float(np.mean(similarities)), 0.0), 1.0)


def lexical_diversity(text: str) -> float:
    """Type-token ratio of the word tokens (0..1)."""
    tokens = list(WordTokenizer().tokenize(text))
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def sentence_length(text: str) -> float:
    """Mean sentence length in words, scaled to 0..1 (50 words = 1.0)."""
    sentences = [s for s in SentenceTokenizer().tokenize(text) if s.strip()]
    if not sentences:
        return 0.0
    mean_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
    return min(mean_length / 50.0, 1.0)


def function_word_ratio(text: str) -> float:
    """Share of word tokens that are closed-class function words (0..1)."""
    tokens = list(WordTokenizer().tokenize(text))
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token in FUNCTION_WORDS) / len(tokens)


def punctuation_rate(text: str) -> float:
    """Share of characters that are punctuation (0..1)."""
    if not text:
        return 0.0
    return sum(1 for char in text if char in PUNCTUATION) / len(text)


def lexical_burstiness(text: str) -> float:
    """Coefficient of variation of inter-arrival gaps of repeated words (0..1).

    High when word recurrences cluster in bursts, low when they arrive evenly.
    """
    tokens = list(WordTokenizer().tokenize(text))
    if len(tokens) < 2:
        return 0.0
    positions: dict[str, list[int]] = {}
    for i, token in enumerate(tokens):
        positions.setdefault(token, []).append(i)
    gaps = [b - a for idxs in positions.values() for a, b in pairwise(idxs)]
    if not gaps:
        return 0.0
    mean_gap = float(np.mean(gaps))
    if mean_gap == 0:
        return 0.0
    return min(float(np.std(gaps)) / mean_gap, 1.0)


_CLAUSE_BREAKS = set(".,!?؟:؛،…\n")


def clause_variety(text: str) -> float:
    """Coefficient of variation of clause lengths in words (0..1); clauses are
    the runs between punctuation pauses."""
    clause_lengths = []
    for raw_clause in text.splitlines():
        start = 0
        for i, char in enumerate(raw_clause):
            if char in _CLAUSE_BREAKS:
                clause_lengths.append(len(raw_clause[start:i].split()))
                start = i + 1
        clause_lengths.append(len(raw_clause[start:].split()))
    clause_lengths = [length for length in clause_lengths if length]
    if len(clause_lengths) < 2:
        return 0.0
    mean_length = float(np.mean(clause_lengths))
    if mean_length == 0:
        return 0.0
    return min(float(np.std(clause_lengths)) / mean_length, 1.0)


def compute_style_features(text: str, cleaned_text: str, vectorizer) -> dict[str, float]:
    """Normalized (0..1) stylometric features for one document."""
    entropy = shannon_entropy(text)
    max_entropy = math.log2(max(len(set(cleaned_text.replace(" ", ""))), 2))
    return {
        "entropy": round(min(entropy / max_entropy, 1.0), 4),
        "sentence_variety": round(sentence_variety(text), 4),
        "repetition": round(pattern_repetition(text, n=3), 4),
        "coherence": round(text_coherence(text, vectorizer), 4),
        "lexical_diversity": round(lexical_diversity(text), 4),
        "sentence_length": round(sentence_length(text), 4),
        "function_word_ratio": round(function_word_ratio(text), 4),
        "punctuation_rate": round(punctuation_rate(text), 4),
        "lexical_burstiness": round(lexical_burstiness(text), 4),
        "clause_variety": round(clause_variety(text), 4),
    }


def style_features_vector(style_features: dict[str, float]) -> np.ndarray:
    """dict -> ordered (1, len(STYLE_FEATURE_NAMES)) array, ready to hstack
    onto a TF-IDF row for sklearn-tfidf models."""
    return np.array([[style_features[name] for name in STYLE_FEATURE_NAMES]])
