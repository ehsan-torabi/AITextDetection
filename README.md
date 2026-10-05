# AITextDetection — Persian AI-Generated Text Detection

Detect whether Persian text is AI-generated or human-written. A FastAPI
serving stack scores text with eight trained models plus a weighted ensemble
and a training-free raw-LM perplexity signal; a System One decision model
answers typed questions (source, machine-likeness, need-for-review) —
returning typed answers and calibrated probabilities instead of generated
text — instead of emitting a bare probability.

## Layout

```
aitextd/                  the shared Python package (serving + training)
├── paths.py              every filesystem path (data/, models/, models.json)
├── constants.py          System One typed-question schema, ensemble id
├── schemas.py            Pydantic API response models
├── text.py               Persian preprocessing (full pipeline + normalization)
├── features.py           stylometric features (entropy, variety, repetition, coherence, lexical diversity, sentence length, function words, punctuation, lexical burstiness, clause variety)
├── architectures.py      torch model architectures (checkpoint-compatible)
├── loaders.py            per-runtime model loading (sklearn/lstm/gpt2/parsbert/system-one)
├── predictors.py         per-runtime prediction
├── perplexity.py         raw-LM perplexity signal (training-free detail)
├── system_one.py         System One typed answers + decision policy
├── store.py              ModelStore: model registry + detection orchestration
└── api.py                FastAPI application

server.py                 launcher: python server.py  (API on :8000)
models.json               the model registry (ids, filenames, weights, metrics)
data/                     datasets (gitignored): raw CSVs, processedDataset.csv
models/                   trained artifacts (gitignored binaries: *.pt, *.joblib, *.mdl)
                          plus the git-tracked exported tokenizer/ folders
                          (gpt2_tokenizer/, parsbert_tokenizer/)
training/                 training pipeline (run from the project root)
├── build_dataset.py      rebuild processedDataset.csv from the raw sources
├── preprocess_cache.py   cache lemmatized/tokenized text for the whole dataset
├── split.py              the canonical stratified train/test split
├── build_style_features.py   style-feature matrix + the TF-IDF+style training matrix
├── prepare_system_one_data.py   distill System One soft targets from the GPT-2 teacher
├── train_ml_models.py    classical ML (style features for trees/NB only — see below)
├── eval_style_ablation.py   proof: when the style block helps/hurts each model
├── train_lstm.py         BiLSTM from scratch
├── train_gpt2_classifier.py   GPT-2 fine-tune (CPU budget)
├── train_parsbert.py     ParsBERT fine-tune (CPU budget)
├── train_system_one.py   System One decision-head training
├── eval_perplexity.py    measure perplexity's standalone detection quality
└── system_one_data/      generated System One JSONL targets

notebooks/
└── colab_training.ipynb  the unified Colab GPU retraining notebook

The deleted legacy `main.ipynb` (the original one-off SVM experiment) survives
as `training/build_dataset.py` — its dataset-building step. Its
model-interpretability plots (top TF-IDF coefficients, ROC/PR curves, Persian
matplotlib font fix) were one-off scratch work and were not carried over; git
history still has them if ever needed.

tests/                    pytest suite; golden test pins exact serving outputs
report/                   the LaTeX project report
web-app/                  the Next.js front end (talks to the API on :8000)
backups/                  local safety backups of model artifacts (gitignored)
```

Style features feed the tree models and Naive Bayes only (plus the shared
detail breakdown of every response). SVM and Logistic Regression train on
TF-IDF alone: `training/eval_style_ablation.py` shows the style block in the
joint vector slightly hurts those linear solvers, while the trees genuinely
use it (7% and 59% importance share).

## Running the API

```bash
.venv/bin/python server.py        # serves on 0.0.0.0:8000
```

Endpoints: `POST /detect` (`{"text": ..., "modelId": ...}`), `POST /details`
(`{"text": ...}` → the model-independent style + perplexity breakdown) and
`GET /models`.
Model ids come from `models.json` plus the virtual `ensemble` id.

## Retraining

Locally (CPU): run the scripts in `training/` from the project root, in the
order implied by the README layout above. On Colab (GPU): open
`notebooks/colab_training.ipynb` and follow its Drive-setup section — it
retrains everything and emits a `models.json` fragment ready to merge.

Shared logic (preprocessing, features, architectures, System One schema) lives only
in `aitextd/`; both the server and every training path import the same code,
so train/serve drift is impossible by construction.

## Tests

```bash
.venv/bin/python -m pytest tests/ -m "not slow"   # fast behavior tests
.venv/bin/python -m pytest tests/ -m slow         # full serving stack + golden parity
```

The golden test captures exact `detection()` outputs on first run and pins
them afterwards: any change that alters a served prediction fails the suite.

## Code style

```bash
uvx ruff format . && uvx ruff check --fix .
```
