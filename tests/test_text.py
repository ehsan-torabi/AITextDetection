"""Pin the Persian preprocessing behavior that every model depends on."""

import re

from conftest import SAMPLE_AI_TEXT

from aitextd.text import (
    normalize_for_pretrained,
    preprocess_persian_text,
    tokenize_persian_text,
)


def test_preprocess_persian_text_strips_newlines_and_punctuation():
    out = preprocess_persian_text(SAMPLE_AI_TEXT)
    assert isinstance(out, str)
    assert "\n" not in out
    assert not re.search(r"[.,؛!؟]", out)
    assert out.strip()  # never empty for non-empty Persian input


def test_preprocess_is_deterministic():
    assert preprocess_persian_text(SAMPLE_AI_TEXT) == preprocess_persian_text(SAMPLE_AI_TEXT)


def test_tokenize_matches_joined_preprocess():
    text = "این یک آزمایش ساده است."
    tokens = tokenize_persian_text(text)
    assert isinstance(tokens, list)
    assert " ".join(tokens) == preprocess_persian_text(text)


def test_normalize_for_pretrained_keeps_word_forms_but_normalizes_characters():
    # Raw newlines vanish, Arabic yeh/kaf unify to Persian forms.
    raw = "سلام\nعلیکم — كيف حال شما؟"
    out = normalize_for_pretrained(raw)
    assert "\n" not in out
    assert "کیف" in out  # Arabic kaf -> Persian kaf
    assert "علیکم" in out  # word forms survive (no lemmatization)
