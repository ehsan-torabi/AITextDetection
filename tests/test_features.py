"""Pin the stylometric features: ranges, edge cases, and exact formulas."""

import math

import pytest
from conftest import SAMPLE_AI_TEXT, SAMPLE_HUMAN_TEXT

from aitextd.features import (
    STYLE_FEATURE_NAMES,
    clause_variety,
    compute_style_features,
    function_word_ratio,
    lexical_burstiness,
    lexical_diversity,
    pattern_repetition,
    punctuation_rate,
    sentence_length,
    sentence_variety,
    shannon_entropy,
    style_features_vector,
    text_coherence,
)


def test_shannon_entropy_values():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    # Two distinct chars -> 1 bit.
    assert shannon_entropy("aabb") == pytest.approx(1.0)


def test_sentence_variety_bounds_and_short_text():
    assert sentence_variety("یک جمله کوتاه") == 0.0
    v = sentence_variety(SAMPLE_HUMAN_TEXT)
    assert 0.0 <= v <= 1.0


def test_pattern_repetition_bounds():
    assert pattern_repetition("کوتاه") == 0.0
    assert 0.0 <= pattern_repetition(SAMPLE_HUMAN_TEXT, n=3) <= 1.0
    # Fully repeated trigram -> every distinct n-gram repeats.
    assert pattern_repetition("آب آب آب آب آب", n=3) == 1.0


def test_lexical_diversity():
    assert lexical_diversity("") == 0.0
    assert lexical_diversity("آب") == 1.0
    assert lexical_diversity("آب آب آب") == pytest.approx(1 / 3)


def test_sentence_length_scaled_by_50():
    assert sentence_length("") == 0.0
    assert sentence_length("یک جمله کوتاه") == pytest.approx(3 / 50)


def test_function_word_ratio():
    assert function_word_ratio("") == 0.0
    assert function_word_ratio("و در به") == 1.0
    assert function_word_ratio("سلام") == 0.0


def test_punctuation_rate():
    assert punctuation_rate("") == 0.0
    assert punctuation_rate("abc!!!") == pytest.approx(0.5)
    assert punctuation_rate("بدون علامت") == 0.0


def test_lexical_burstiness_gap_clustering():
    # equal gaps between repeats -> CV 0
    assert lexical_burstiness("کتاب دفتر کتاب دفتر کتاب دفتر") == 0.0
    # one tight cluster plus one distant return -> CV > 0
    assert lexical_burstiness("آب آب نان شیر آب") > 0.0
    # no repeated word -> nothing to burst
    assert lexical_burstiness("یک دو سه چهار") == 0.0


def test_clause_variety_pause_segments():
    assert clause_variety("") == 0.0
    assert clause_variety("یک جمله ساده") == 0.0  # single clause
    # clause lengths 1, 3, 1 -> nonzero CV
    assert clause_variety("یک، یک دو سه، یک") > 0.0


def test_compute_style_features_shape_and_range(vectorizer):
    cleaned = " ".join(SAMPLE_AI_TEXT.split())
    features = compute_style_features(SAMPLE_AI_TEXT, cleaned, vectorizer)
    assert list(features) == STYLE_FEATURE_NAMES
    for value in features.values():
        assert 0.0 <= value <= 1.0
        assert value == round(value, 4)


def test_style_features_vector_order():
    features = {name: i / 10 for i, name in enumerate(STYLE_FEATURE_NAMES)}
    vector = style_features_vector(features)
    expected = [features[name] for name in STYLE_FEATURE_NAMES]
    assert vector.shape == (1, len(STYLE_FEATURE_NAMES))
    assert vector.tolist() == [expected]


def test_entropy_normalization_divides_by_cleaned_alphabet(vectorizer):
    text = SAMPLE_AI_TEXT
    cleaned = text.replace(" ", "")
    max_entropy = math.log2(max(len(set(cleaned)), 2))
    raw = shannon_entropy(text)
    assert compute_style_features(text, text, vectorizer)["entropy"] == pytest.approx(
        round(min(raw / max_entropy, 1.0), 4)
    )


def test_text_coherence_short_text_is_one(vectorizer):
    assert text_coherence("", vectorizer) == 1.0
    assert text_coherence("یک جمله تنها.", vectorizer) == 1.0


def test_compute_style_features_empty_text(vectorizer):
    features = compute_style_features("", "", vectorizer)
    assert list(features) == STYLE_FEATURE_NAMES
    assert features["coherence"] == 1.0
    for value in features.values():
        assert 0.0 <= value <= 1.0


def test_predict_sklearn_slices_style_columns_to_checkpoint_width(vectorizer):
    from aitextd.predictors import predict_sklearn

    class Recorder:
        def predict_proba(self, X):
            self.n_features = X.shape[1]
            return [[0.2, 0.8]]

    cleaned = " ".join(SAMPLE_HUMAN_TEXT.split())
    feats = compute_style_features(SAMPLE_HUMAN_TEXT, cleaned, vectorizer)
    recorder = Recorder()
    bundle = {"model": recorder, "vectorizer": vectorizer, "style_feature_count": 4}
    assert predict_sklearn(bundle, SAMPLE_HUMAN_TEXT, cleaned, feats) == pytest.approx(0.8)
    assert recorder.n_features == len(vectorizer.get_feature_names_out()) + 4
