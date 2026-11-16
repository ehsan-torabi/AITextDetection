"""aitextd -- Persian AI-generated text detection.

Layout:
- aitextd.paths          every filesystem path (data/, models/, models.json)
- aitextd.constants      shared constants (System One typed-question schema, ensemble id)
- aitextd.schemas        Pydantic API response models
- aitextd.text           Persian preprocessing pipelines
- aitextd.features       stylometric features
- aitextd.architectures  torch model architectures (checkpoint-compatible)
- aitextd.loaders        per-runtime model loading
- aitextd.predictors     per-runtime prediction
- aitextd.perplexity     raw-LM perplexity signal (training-free detail)
- aitextd.system_one     System One typed answers + decision policy
- aitextd.store          ModelStore: model registry + detection orchestration
- aitextd.api            FastAPI application

The FastAPI app is imported lazily (aitextd.api) so training scripts can use
the pure modules without pulling in the web stack.
"""

__version__ = "2.0.0"

__all__ = [
    "architectures",
    "constants",
    "features",
    "loaders",
    "paths",
    "perplexity",
    "predictors",
    "schemas",
    "store",
    "system_one",
    "text",
]
