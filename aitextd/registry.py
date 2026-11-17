"""models.json maintenance shared by the training notebooks.

refresh_entry() updates only the numeric blocks (metrics / confusionMatrix /
training) of an existing entry. Identity fields -- names, bilingual
descriptions, algorithm, runtime, ensemble weight -- always stay untouched,
so re-running a notebook can never silently rewrite the serving metadata.
"""

from __future__ import annotations

import json

from aitextd import metrics, paths


def refresh_entry(model_id: str, results: dict, training: dict) -> None:
    """Refresh one entry's numeric blocks from metrics.evaluate() output."""
    registry = json.loads(paths.MODELS_JSON.read_text(encoding="utf-8"))
    entry = next((e for e in registry if e["id"] == model_id), None)
    if entry is None:
        raise KeyError(f"models.json has no entry '{model_id}'")
    entry["metrics"] = metrics.registry_metrics(results)
    entry["confusionMatrix"] = metrics.registry_confusion(results)
    entry["training"] = training
    paths.MODELS_JSON.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"refreshed models.json entry '{model_id}'")
