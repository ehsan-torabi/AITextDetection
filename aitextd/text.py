"""Persian text preprocessing.

Two deliberately different pipelines:

- the full shekar pipeline (lemmatize + word-tokenize) for the classic
  TF-IDF models and the BiLSTM, which see whitespace-joined tokens;
- character-level normalization only for the pretrained transformer models
  (GPT-2 classifier, ParsBERT, the System One decision model, and the
  raw-LM perplexity signal), whose
  tokenizers expect natural word forms.
"""

import re
from functools import lru_cache

from shekar import Lemmatizer, Normalizer, SentenceTokenizer, WordTokenizer
from shekar.preprocessing import EmojiRemover, NonPersianRemover, PunctuationRemover


@lru_cache(maxsize=1)
def _preprocess_pipeline():
    return (
        EmojiRemover()
        | PunctuationRemover()
        | NonPersianRemover()
        | Normalizer()
        | Lemmatizer()
        | WordTokenizer()
    )


@lru_cache(maxsize=1)
def _normalizer():
    return Normalizer()


def preprocess_persian_text(text: str) -> str:
    """Full pipeline: clean, normalize, lemmatize, word-tokenize, re-join."""
    text = re.sub(r"\n", " ", text)
    return " ".join(_preprocess_pipeline()(text))


def tokenize_persian_text(text: str) -> list[str]:
    """Same pipeline as preprocess_persian_text, but returning the tokens."""
    text = re.sub(r"\n", " ", text)
    return list(_preprocess_pipeline()(text))


def normalize_for_pretrained(text: str) -> str:
    """Character-level normalization for the pretrained-model runtimes.

    GPT-2 and ParsBERT were pretrained on natural Persian text, so we
    deliberately do NOT lemmatize or word-tokenize here — that would strip
    inflection/morphology and feed the models text unlike anything they saw
    during pretraining. Just character-level normalization (Arabic->Persian
    unification, digit/punctuation normalization, ZWNJ and spacing fixes) to
    clean up inconsistent real-world typing.
    """
    text = re.sub(r"\n", " ", text)
    return _normalizer()(text)


def split_sentences(text: str) -> list[str]:
    """shekar sentence segmentation, used by the style features."""
    return SentenceTokenizer().tokenize(text)
