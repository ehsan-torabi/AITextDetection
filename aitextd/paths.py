"""Single source of truth for every filesystem path the project touches.

Datasets live under data/, trained model artifacts and exported tokenizers
under models/, and the model registry (models.json) at the project root.
Resolving them here keeps both the serving stack and the training scripts
independent of the current working directory.

Set AITEXTD_ROOT to point the whole stack at a different checkout.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("AITEXTD_ROOT", Path(__file__).resolve().parents[1]))

# --- data (datasets and derived feature matrices; gitignored) ---------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_AI_CSV = DATA_DIR / "persian_ai_generated_final.csv"
RAW_HUMAN_CSV = DATA_DIR / "blogs.csv"
PROCESSED_CSV = DATA_DIR / "processedDataset.csv"
TFIDF_X_JOBLIB = DATA_DIR / "tfidf_vectorizer_X.joblib"

# --- models (trained artifacts; gitignored) ---------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_JSON = PROJECT_ROOT / "models.json"
TFIDF_VECTORIZER_JOBLIB = MODELS_DIR / "tfidf_vectorizer.joblib"
GPT2_LM_MDL = MODELS_DIR / "gpt2-fa.mdl"
GPT2_TOKENIZER_MDL = MODELS_DIR / "gpt2-fa-tokenizer.mdl"
GPT2_TOKENIZER_DIR = MODELS_DIR / "gpt2_tokenizer"
PARSBERT_TOKENIZER_DIR = MODELS_DIR / "parsbert_tokenizer"


def model_path(filename: str) -> Path:
    """Resolve a bare artifact filename from models.json against models/."""
    return MODELS_DIR / filename


# --- training (caches and outputs; artifacts gitignored) --------------------
TRAINING_DIR = PROJECT_ROOT / "training"
TOKENIZED_TEXTS_JOBLIB = TRAINING_DIR / "tokenized_texts.joblib"
SPLIT_INDICES_JOBLIB = TRAINING_DIR / "split_indices.joblib"
STYLE_FEATURES_JOBLIB = DATA_DIR / "style_features.joblib"
SYSTEM_ONE_DATA_DIR = TRAINING_DIR / "system_one_data"

# --- laya (open-weights System One fine-tuning; artifacts gitignored) --------
# Base checkpoint = convaiinnovations/laya "multilingual" subfolder (mmBERT-base,
# 100+ languages -- the English root checkpoint cannot read Persian script).
# Flattened layout so laya.load() can take the directory directly.
LAYA_BASE_DIR = MODELS_DIR / "laya_multilingual"
LAYA_OUTPUT_DIR = MODELS_DIR / "laya_persian"
LAYA_ITEMS_DIR = TRAINING_DIR / "laya_data"
