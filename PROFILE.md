# Deployment profiling — AI Text Detection (Persian)

Measured with `aitextd/profiling.py` on the deployed checkpoint set (16 models,
all loaded at startup via `ModelStore.init()`, exactly like production).

## How to re-run

```bash
PYTHONPATH="$PWD/.venv/lib/python3.11/site-packages:$PWD" \
/home/ehsan/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11 \
-m aitextd.profiling full-stack
```

Subcommands: `disk` | `per-model` | `full-stack` | `all`.

## Measured numbers

### Disk footprint

| scope | size |
|-------|------|
| `models/` (artifacts + tokenizers) | ~4.0 GiB |
| `data/` (datasets + caches) | ~3.3 GiB |
| `models.json` | 30 KiB |
| **total** | **7.33 GiB** |

Only `models/` + `models.json` are required at serving time. `data/` is needed for
training/retraining, not for the API.

### RAM (steady-state, one process, all 16 models)

| metric | value |
|--------|-------|
| RSS after `store.init()` (all models loaded) | **5.112 GiB** |
| RSS growth after sample requests | +0.0 GiB (no leak) |
| Process startup time (`store.init()`) | ~9.9 s |
| Per-request CPU work (single forward pass) | 0.2–0.7 s per model; ensemble ~2.6–3.9 s |

**Bottom line:** size one worker at **≥6 GiB RAM** (5.1 GiB steady-state + OS +
Python/runtime overhead + per-request buffers). Deploy with **8 GiB per worker**
for headroom.

Important: `ModelStore.init()` loads **all 16 models into every worker process**.
RAM scales linearly with worker count: 2 workers ≈ 10.2 GiB, 4 workers ≈ 20.4 GiB.
Do not over-provision workers for this stack.

### Per-request latency (CPU, torch limited to 4 threads)

| model | short text | medium text | long text |
|-------|-----------|-------------|-----------|
| SVM (sklearn-tfidf) | 0.40 s | 0.20 s | 0.20 s |
| GPT-2 transformer | 0.47 s | 0.46 s | 0.47 s |
| ParsBERT | 0.65 s | 0.67 s | 0.68 s |
| **ensemble (all models)** | **2.63 s** | **2.90 s** | **3.93 s** |

`/details` (style + perplexity breakdown, no classifier): ~0.18 s.

## Sizing recommendations

| deployment | RAM | CPU cores | workers | notes |
|------------|-----|-----------|---------|-------|
| minimal (single worker) | 8 GiB | 2–4 | 1 | ~20 ensemble req/min |
| standard (2 workers) | 16 GiB | 4–8 | 2 | ~40 ensemble req/min |
| heavy (4 workers) | 32 GiB | 8–16 | 4 | ~80 ensemble req/min |

- CPU bound: each forward pass is single-process CPU work (torch.set_num_threads(4)).
  Scale workers horizontally, not threads.
- RAM bound by worker count (each worker = 5.1 GiB base). Prefer fewer workers
  with a queue over many small workers.
- Ensemble (`modelId=ensemble`) runs every model per request (~3 s). Route
  latency-sensitive traffic to a single model (`svm`, `gpt2_transformer`, …)
  and reserve the ensemble for batch/offline scoring.

## What the profiler measures

- `disk` — on-disk bytes of models/, data/, models.json.
- `per-model` — load time + peak RSS for each model in an isolated subprocess.
  (isolated runs overcount shared tokenizers; use full-stack for the real number)
- `full-stack` — one process calling `store.init()` (all 16 models), then warmup
  requests to `/detect` (svm/gpt2/parsbert/ensemble) and `/details`; reports
  steady-state RSS and per-endpoint latency.
