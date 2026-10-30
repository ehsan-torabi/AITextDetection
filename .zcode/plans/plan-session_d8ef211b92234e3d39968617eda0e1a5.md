# AITextDetection — Rewrite into a Simple Notebook-Based Project

## Context
The project works but is messy: ~30 training scripts in `training/` (two generations, ~6k lines), stale colab notebooks, stray root files, and a 3,156-line report. You are presenting this project and want: trained models + serving code retained, all training scripts replaced by **4 clean notebooks** (data prep / classical / DL / LLM), **2 new deep-learning models trained & deployed**, and `report.tex` rewritten with the **humanizer** + **persian-writing** skills. Serving stack stays as-is; explanatory text in English everywhere except `report.tex`.

## Decisions locked with you
1. Serving stack (`aitextd/`, `server.py`, `web-app/`, `tests/`) kept as-is — only extended for the 2 new models.
2. v2 rigorous-eval layer → condensed into notebooks; `reports/` artifacts stay frozen as evidence; the 1,200-line eval battery + LaTeX table generator are dropped.
3. TextCNN + Attention-BiLSTM trained locally now, registered in serving, golden tests re-baselined.
4. Cleanup = project junk only; your `aether*` VPN files and `backups/` untouched (just gitignored).

## Target layout
```
AITextDetection/
├── README.md            rewritten (concise English)
├── pyproject.toml       dev deps += ipykernel, nbconvert; drop training/ ruff ignores
├── server.py            unchanged
├── models.json          10 existing + 2 new entries
├── aitextd/             unchanged except architectures.py (+2 classes), loaders.py/predictors.py (+2 runtimes)
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_classical_models.ipynb
│   ├── 03_deep_learning_models.ipynb
│   └── 04_llm_models.ipynb
├── data/  models/       artifacts (unchanged; a few training/ caches moved here)
├── report/report.tex    rewritten; keeps \input{generated/*.tex} frozen tables
├── reports/             frozen v2 evidence (kept; predictions/*.npz gitignored)
├── tests/               conftest += 2 model ids; golden re-baselined
└── web-app/             untouched
```

## Stage 0 — Snapshot & guardrails (orchestrator)
- `.gitignore`: add `*.parquet`, `reports/predictions/`, `aether*.toml`, `aether.toml-psiphon/`, `colab_laya_files.zip`.
- `uv sync` (adds jupyter dev deps; you fix env breakage per your preference).
- **Snapshot commit** of current state incl. untracked v2 sources (safety net before deletions).
- Run `pytest -m slow` once to prove the current golden baseline is green **before** any change.

## Stage 1 — Four notebooks (4 parallel writer agents)
Conventions for all: markdown cells in concise English; minimal code comments; config cell uses `aitextd.paths`; `FORCE_*` flags default `False` → **load existing artifacts by default, retrain/rebuild only when flagged**; each ends with a results-summary cell; self-contained (one compact ~25-line eval helper per notebook: acc/P/R/F1/AUC/ECE); valid nbformat 4.5 JSON.
- **01_data_preparation** — ports `build_dataset.py`, `preprocess_cache.py`, `build_style_features.py`, `split.py`, plus condensed v2 layer from `build_dataset_v2.py` / `splits_v2.py` / `build_features_v2.py` / `collect_human_domains.py` / `paraphrase.py` (split logic ported faithfully; assertion batteries trimmed to key checks; `aitextd/dataset.py` reused for the v2 schema). Dataset overview stats at the end.
- **02_classical_models** — ports `train_ml_models.py` (SVM/LogReg on TF-IDF; NB/RF/GB on TF-IDF+style): trains or loads, evaluates, saves joblibs + `models.json` entries.
- **03_deep_learning_models** — existing BiLSTM (`train_lstm.py`) **plus new TextCNN and Attention-BiLSTM** (see Stage 2); saves `models/*.pt`, registers in `models.json`.
- **04_llm_models** — ports `train_gpt2_classifier.py`, `train_parsbert.py`, `prepare_system_one_data.py`, `train_system_one.py`, plus a condensed Laya RLCD port (from `train_laya.py` + colab `aitextd_colab/laya_finetune.py`). Loads existing checkpoints/JSONLs by default. Agent loads the `typesafe-ai` skill for System One terminology.

## Stage 2 — New DL models (part of agent 03's work)
- `aitextd/architectures.py`: `TextCNNClassifier` (embedding + parallel conv1d kernels + max-pool + linear) and `AttentionBiLSTMClassifier` (BiLSTM + attention pooling + linear), vocab-based like `BiLSTMClassifier`.
- `aitextd/loaders.py` / `predictors.py`: runtimes `pytorch-textcnn`, `pytorch-bilstm-attn` mirroring the `pytorch-lstm` pattern.
- Train on the canonical split (`data/tokenized_texts.joblib` moved out of `training/`), save `models/textcnnModel.pt`, `models/attnBilstmModel.pt`, add honest metrics to `models.json` with conservative ensemble weights (~0.5).
- Tests: `conftest.py` `REQUIRED_ARTIFACTS` + `ALL_MODEL_IDS` extended; add shape tests to `test_architectures.py`; **golden re-baseline**: delete `tests/golden/detection_golden.json`, run slow suite twice.

## Stage 3 — Integration verification (orchestrator)
- Execute all 4 notebooks head-to-toe (`jupyter nbconvert --execute`, load-default path); strip outputs before commit.
- `pytest -m "not slow"` + `-m slow`; `uvx ruff format . && uvx ruff check --fix .`.
- Server smoke test: boot `server.py`, `GET /models` shows 12, `POST /detect` with a new model id.

## Stage 4 — report.tex rewrite (dedicated agent)
Loads `humanizer` + `persian-writing` skills. Keeps the XeLaTeX/xepersian preamble and `\input{generated/...}` tables; tightens the narrative to match the simplified repo; deep-learning chapter gains TextCNN/Attention-BiLSTM subsections with real trained metrics; infrastructure + appendix sections rewritten around the notebook workflow (no `training/` references). Attempt compile (xelatex −no-pdf + xdvipdfmx per your recipe); if TeX Live is still uninstalled, report the blocker instead of pretending. Sync `web-app/public/report.pdf` after a successful build.

## Stage 5 — README + final review loop
- README rewritten for the new layout (run notebooks, run API, run tests).
- **Final reviewer agent** runs a full checklist (no `.py` scripts outside `aitextd/`/`tests/`/`server.py`; 4 valid executable notebooks; pytest green; ruff clean; report consistency; root clean; README accurate). Findings → fix → re-review until it passes.

## Deletions (after snapshot commit + porting)
`training/` entirely (its `config.json` with your relay key → moved to `backups/`; `tokenized_texts.joblib`, `split_indices.joblib`, `system_one_data/`, `laya_data/` → moved under `data/`), 3 old `notebooks/colab_*.ipynb`, `colab_laya_files.zip`, empty `data/{data,models,outputs}`, stale `edt_params.txt` (reviewed first), report build residue.

## Commits
Snapshot → notebooks+serving additions → report rewrite → final polish. (Plan approval = authorization for these commits on `master`.)

## Verification
Fast + slow pytest suites, notebook executions, ruff, live server smoke test against the 2 new models, report compile (or documented blocker), final reviewer checklist pass.
