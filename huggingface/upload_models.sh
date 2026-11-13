#!/usr/bin/env bash
# Upload every trained serving artifact to the Hugging Face model repo.
#
# The repo layout mirrors models/ one-to-one (plus the model card), so a
# single download puts everything in place:
#
#     hf download <user>/aitextd-models --local-dir models/
#
# Usage (from the project root, logged in via `hf auth login`):
#
#     HF_USER=<your-huggingface-username> ./huggingface/upload_models.sh
#
# Deliberately skipped:
#   - models/laya_multilingual/          (untouched convaiinnovations/laya base)
#   - models/laya_persian/checkpoint_latest/  (mid-training duplicate of final weights)
set -euo pipefail

HF_USER="${HF_USER:?set HF_USER=<your-huggingface-username>}"
REPO_ID="$HF_USER/aitextd-models"

# Prefer the project venv's hf (uv sync installs huggingface-hub there); fall
# back to whatever `hf` is on PATH.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -x "$PROJECT_ROOT/.venv/bin/hf" ]; then
    HF="$PROJECT_ROOT/.venv/bin/hf"
else
    HF="hf"
fi

run() {
    echo "+ $HF upload $REPO_ID $*"
    "$HF" upload "$REPO_ID" "$@"
}

# --- model card ---------------------------------------------------------------
run huggingface/README.md README.md

# --- classical + deep learning checkpoints (registry filenames, flat) ----------
run models/lstmModel.pt lstmModel.pt
run models/textcnnModel.pt textcnnModel.pt
run models/textcnnRlcdModel.pt textcnnRlcdModel.pt
run models/attnBilstmModel.pt attnBilstmModel.pt
run models/mlpModel.pt mlpModel.pt
run models/charcnnModel.pt charcnnModel.pt
run models/tinyTransformerModel.pt tinyTransformerModel.pt
run models/gpt2ClassifierModel.pt gpt2ClassifierModel.pt
run models/parsbertClassifierModel.pt parsbertClassifierModel.pt
run models/systemOneModel.pt systemOneModel.pt
run models/svModel.joblib svModel.joblib
run models/logregModel.joblib logregModel.joblib
run models/nbModel.joblib nbModel.joblib
run models/rfModel.joblib rfModel.joblib
run models/gbModel.joblib gbModel.joblib

# --- shared serving artifacts ---------------------------------------------------
run models/tfidf_vectorizer.joblib tfidf_vectorizer.joblib
run models/gpt2-fa.mdl gpt2-fa.mdl
run models/gpt2-fa-tokenizer.mdl gpt2-fa-tokenizer.mdl
run models/gpt2_tokenizer gpt2_tokenizer
run models/parsbert_tokenizer parsbert_tokenizer

# --- Laya RLCD fine-tune (final checkpoint only) --------------------------------
run models/laya_persian laya_persian --exclude "checkpoint_latest/*"

echo "Done: https://huggingface.co/$REPO_ID"
