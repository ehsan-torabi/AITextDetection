# AITextDetection — Persian AI-Generated Text Detection

Detect whether Persian text is AI-generated or human-written. A FastAPI serving
stack scores text with sixteen trained models plus a weighted ensemble, and a
System One decision model answers typed questions (source, machine-likeness,
need-for-review) with calibrated probabilities instead of a bare score.

## Layout

```
aitextd/                  the shared Python package (serving)
├── paths.py              every filesystem path (data/, models/, models.json)
├── constants.py          System One typed-question schema, ensemble id
├── schemas.py            Pydantic API response models
├── text.py               Persian preprocessing (full pipeline + normalization)
├── features.py           10 stylometric features (fixed order, pinned by checkpoints)
├── architectures.py      torch model architectures (checkpoint-compatible)
├── loaders.py            per-runtime model loading
├── predictors.py         per-runtime prediction
├── perplexity.py         training-free raw-LM perplexity signal
├── system_one.py         System One typed answers + decision policy
├── store.py              ModelStore: model registry + detection orchestration
├── api.py                FastAPI application
└── dataset.py            canonical v2 dataset schema (used by notebook 01)

server.py                 launcher: python server.py  (API on :8000)
models.json               the model registry (ids, filenames, weights, metrics)
notebooks/                the whole retraining pipeline, four notebooks
data/                     datasets and derived artifacts (gitignored)
models/                   trained artifacts (gitignored binaries + exported tokenizers)
huggingface/              HF model card + upload script (publishes models/ 1:1)
report/                   the Persian LaTeX project report
reports/                  frozen evaluation evidence (leak-free v2 protocol)
tests/                    pytest suite; golden test pins exact serving outputs
web-app/                  the Next.js front end (talks to the API on :8000)
```

## Models on Hugging Face

The trained checkpoints are not stored in git — they are published, one-to-one
mirroring `models/`, in the [aitextd-models](https://huggingface.co/ehsan-torabi/aitextd-models)
repo. After cloning this repo, download them into place and start serving:

```bash
hf download ehsan-torabi/aitextd-models --local-dir models/
```

To re-publish after retraining (any machine logged in with `hf auth login`):

```bash
HF_USER=ehsan-torabi ./huggingface/upload_models.sh
```

## Notebooks

The full pipeline lives in `notebooks/`, in run order. Every stage loads the
existing artifacts by default and only recomputes when its `FORCE_*` flag is
set, so the notebooks are safe to run top-to-bottom on a machine that already
has the artifacts:

1. **`01_data_preparation.ipynb`** — raw CSVs → `processedDataset.csv` →
   tokenization cache → TF-IDF + style features → canonical split; plus the v2
   dataset, group-aware splits and paraphrase-attack artifacts (parity-checked
   against the frozen versions).
2. **`02_classical_models.ipynb`** — Linear SVM and Logistic Regression on
   TF-IDF; Naive Bayes, Random Forest and Gradient Boosting on TF-IDF + style.
3. **`03_deep_learning_models.ipynb`** — BiLSTM, TextCNN and
   Attention-BiLSTM trained from scratch on the tokenized cache.
4. **`04_llm_models.ipynb`** — GPT-2 and ParsBERT fine-tunes, the System One
   decision model (soft targets distilled from the GPT-2 classifier) and the
   Laya RLCD fine-tune. By default each section loads the deployed checkpoint
   and scores it through the exact serving path.
5. **`05_reinforcement_learning.ipynb`** — a REINFORCE contextual-bandit
   decision policy (auto-human / review / auto-AI) and an RLCD fine-tune of
   the supervised TextCNN (registered as `textcnn_rlcd`, ensemble weight 0).
6. **`06_neural_networks.ipynb`** — three more from-scratch networks: an MLP
   over the chi2-reduced TF-IDF + style features, a character-level CharCNN
   and a small from-scratch Transformer encoder.

## Running the API

```bash
.venv/bin/python server.py        # serves on 0.0.0.0:8000
```

Endpoints: `POST /detect` (`{"text": ..., "modelId": ...}`), `POST /details`
(`{"text": ...}` → the model-independent style + perplexity breakdown) and
`GET /models`. Model ids come from `models.json` plus the virtual `ensemble`
id.

## Web app

```bash
cd web-app && pnpm install && pnpm dev
```

## Tests

```bash
.venv/bin/python -m pytest tests -m "not slow"   # fast behavior tests
.venv/bin/python -m pytest tests -m slow         # full serving stack + golden parity
```

The golden test pins exact `detection()` outputs for all sixteen models and the
ensemble: any change that alters a served prediction fails the suite. To
re-baseline (e.g. after registering a new model), delete
`tests/golden/detection_golden.json` and run the slow suite twice.

## License

MIT — see [LICENSE](LICENSE). The fine-tunes build on ParsBERT
([HooshvareLab/bert-fa-base-uncased](https://huggingface.co/HooshvareLab/bert-fa-base-uncased)),
Persian GPT-2 ([HooshvareLab/gpt2-fa](https://huggingface.co/HooshvareLab/gpt2-fa)) and
Laya ([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)).
