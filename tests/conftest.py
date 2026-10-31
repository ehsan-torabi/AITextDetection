"""Shared fixtures for the AITextDetection test suite.

Fast tests pin the behavior of pure logic (preprocessing, style features,
architectures, System One answer wrapping) with no model artifacts required.
The slow golden test loads the full serving stack and pins exact detection()
outputs so a refactor can prove it did not change any prediction.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aitextd import paths  # noqa: E402  (needs PROJECT_ROOT on sys.path first)

GOLDEN_PATH = Path(__file__).parent / "golden" / "detection_golden.json"

# Persian samples: one chatty/informal, one formal/expository. Any fixed pair
# works — the tests pin identical outputs for identical inputs.
SAMPLE_HUMAN_TEXT = (
    "دیروز رفتم بازار تا برای عید نوروز خرید کنم. شلوغ بود و صف طولی بسته بودیم "
    "تا نان بگیریم. خواهرم می‌گوید امسال قیمت‌ها خیلی بالا رفته است."
)
SAMPLE_AI_TEXT = (
    "هوش مصنوعی یکی از مهم‌ترین فناوری‌های قرن بیست‌ویکم است. این فناوری تأثیر "
    "عمیقی بر اقتصاد، آموزش و سلامت گذاشته است. کارشناسان پیش‌بینی می‌کنند که "
    "کاربردهای آن در سال‌های آینده گسترش بیشتری خواهد یافت."
)

REQUIRED_ARTIFACTS = [
    paths.MODELS_JSON,
    paths.TFIDF_VECTORIZER_JOBLIB,
    paths.GPT2_LM_MDL,
    paths.model_path("lstmModel.pt"),
    paths.model_path("textcnnModel.pt"),
    paths.model_path("attnBilstmModel.pt"),
    paths.model_path("gpt2ClassifierModel.pt"),
    paths.model_path("parsbertClassifierModel.pt"),
    paths.model_path("systemOneModel.pt"),
    paths.LAYA_OUTPUT_DIR / "model.safetensors",
]

ALL_MODEL_IDS = [
    "svm",
    "logreg",
    "naive_bayes",
    "random_forest",
    "gradient_boosting",
    "bilstm",
    "gpt2_transformer",
    "parsbert",
    "system_one",
    "laya",
    "textcnn",
    "attention_bilstm",
    "ensemble",
]


def artifacts_available() -> bool:
    return all(p.exists() for p in REQUIRED_ARTIFACTS)


@pytest.fixture(scope="session")
def vectorizer():
    """The deployed TF-IDF vectorizer (needed by the style features)."""
    if not paths.TFIDF_VECTORIZER_JOBLIB.exists():
        pytest.skip("tfidf_vectorizer.joblib not present on this machine")
    import joblib

    return joblib.load(paths.TFIDF_VECTORIZER_JOBLIB)


@pytest.fixture(scope="session")
def store():
    """The ModelStore with every model loaded once per session."""
    if not artifacts_available():
        pytest.skip("model artifacts not present on this machine")
    from aitextd.store import ModelStore

    model_store = ModelStore()
    model_store.init()
    return model_store
