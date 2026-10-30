"""The mandatory metric suite (review section 7) in one place.

Every evaluation in this project reports, for each model and test set:

- accuracy, macro-F1
- per-class precision / recall / F1 (binary and three-class views)
- ROC-AUC, PR-AUC
- FPR@TPR95 (false-positive rate when the threshold is set for 95% recall)
- expected calibration error (equal-width and equal-mass bins)
- Brier score
- confusion matrix
- stratified bootstrap confidence intervals

The point estimates come from scikit-learn (trusted implementations); the
bootstrap uses the local vectorised equivalents, which a unit test pins
against scikit-learn on the full sample. Accuracy alone is never enough here:
`fpr_at_tpr95` is the number that matters for an academic-integrity use case,
because it states how many human documents get flagged when the detector is
tuned to catch 95% of AI text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

DEFAULT_N_BOOT = 1000
ECE_BINS = 15


# --- fast vectorised primitives (bootstrap-inner-loop safe) -----------------


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based ROC-AUC; ties handled by average ranks."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sorted_p = p[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Tie-aware average precision, matching sklearn.average_precision_score:
    thresholds are the distinct score values, and precision/recall are taken
    at the end of each tie group."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    p_sorted = p[order]
    # end index (inclusive) of each tie group
    group_starts = np.flatnonzero(np.diff(p_sorted)) + 1
    group_ends = np.concatenate([group_starts, [len(p_sorted)]]) - 1
    tp_cum = np.cumsum(y_sorted)
    tp = tp_cum[group_ends].astype(float)
    precision = tp / (group_ends + 1)
    recall = tp / float(y.sum())
    delta_recall = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * delta_recall))


def _prf(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def fpr_at_tpr(y: np.ndarray, p: np.ndarray, target_tpr: float = 0.95) -> float:
    """FPR at the threshold that achieves at least `target_tpr` on positives.
    Threshold is picked on the *given* sample only; callers must pass the
    validation set when tuning a real operating point."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    pos_scores = np.sort(p[y == 1])
    neg_scores = p[y == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    rank = max(int(math.ceil(target_tpr * len(pos_scores))) - 1, 0)
    threshold = pos_scores[rank]
    return float((neg_scores >= threshold).mean())


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS, strategy: str = "width"
) -> float:
    """ECE = sum_b (n_b / N) * |accuracy_b - confidence_b|.

    strategy="width" uses equal-width probability bins; strategy="mass" uses
    equal-mass bins. Both are reported because equal-width ECE hides errors in
    the tails of a heavily skewed detector.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if n == 0:
        return float("nan")
    if strategy == "mass":
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        edges[0], edges[-1] = -np.inf, np.inf
    elif strategy == "width":
        edges = np.linspace(0.0, 1.0, bins + 1)
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        raise ValueError(f"unknown ECE strategy {strategy!r}")
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p > lo) & (p <= hi)
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


# --- full metric bundle -----------------------------------------------------


@dataclass
class EvalResult:
    n: int
    metrics: dict[str, float]
    confusion: dict[str, int]
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "metrics": {k: _round(v) for k, v in self.metrics.items()},
            "confusion": self.confusion,
            **({"extra": self.extra} if self.extra else {}),
        }


def _round(value: float) -> float | None:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return round(float(value), 6)


def binary_metrics(
    y_true, y_score, threshold: float = 0.5, prefix: str = ""
) -> EvalResult:
    """Complete binary metric bundle for human(=0) vs machine(=1)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_score, dtype=float)
    pred = (p >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y, pred, labels=[0, 1])
    metrics = {
        f"{prefix}accuracy": accuracy_score(y, pred),
        f"{prefix}macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        f"{prefix}precision_human": precision[0],
        f"{prefix}recall_human": recall[0],
        f"{prefix}f1_human": f1[0],
        f"{prefix}precision_ai": precision[1],
        f"{prefix}recall_ai": recall[1],
        f"{prefix}f1_ai": f1[1],
        f"{prefix}roc_auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan"),
        f"{prefix}pr_auc": (
            average_precision_score(y, p) if 0 < y.sum() < len(y) else float("nan")
        ),
        f"{prefix}brier": brier_score_loss(y, np.clip(p, 0, 1)),
        f"{prefix}ece_width": expected_calibration_error(y, p, strategy="width"),
        f"{prefix}ece_mass": expected_calibration_error(y, p, strategy="mass"),
        f"{prefix}fpr_at_tpr95": fpr_at_tpr(y, p, 0.95),
        f"{prefix}fpr": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        f"{prefix}fnr": float(((pred == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)),
        f"{prefix}npv_human": (
            float(((pred == 0) & (y == 0)).sum() / max((pred == 0).sum(), 1))
        ),
    }
    return EvalResult(
        n=int(len(y)),
        metrics=metrics,
        confusion={
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        },
    )


def multiclass_metrics(y_true, proba, classes: list[str]) -> EvalResult:
    """Three-class view: per-class one-vs-rest P/R/F1 plus macro aggregates
    and macro one-vs-rest ROC-AUC / PR-AUC."""
    y = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = proba.argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(range(len(classes))), zero_division=0
    )
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
    }
    for idx, name in enumerate(classes):
        metrics[f"precision_{name}"] = float(precision[idx])
        metrics[f"recall_{name}"] = float(recall[idx])
        metrics[f"f1_{name}"] = float(f1[idx])
        metrics[f"support_{name}"] = float(support[idx])
        metrics[f"pr_auc_{name}"] = (
            average_precision_score((y == idx).astype(int), proba[:, idx])
            if 0 < (y == idx).sum() < len(y)
            else float("nan")
        )
    try:
        metrics["macro_roc_auc_ovr"] = float(
            roc_auc_score(y, proba, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["macro_roc_auc_ovr"] = float("nan")
    cm = confusion_matrix(y, pred, labels=list(range(len(classes))))
    confusion = {
        f"true_{classes[i]}_as_{classes[j]}": int(cm[i, j])
        for i in range(len(classes))
        for j in range(len(classes))
    }
    return EvalResult(n=int(len(y)), metrics=metrics, confusion=confusion)


# --- bootstrap --------------------------------------------------------------

BOOTSTRAP_METRICS = (
    "accuracy",
    "macro_f1",
    "precision_human",
    "recall_human",
    "precision_ai",
    "recall_ai",
    "roc_auc",
    "pr_auc",
    "fpr_at_tpr95",
    "ece_width",
    "brier",
)


def _fast_binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    prec_h, rec_h, f1_h = _prf(1 - y, 1 - pred)  # human as positive
    prec_a, rec_a, f1_a = _prf(y, pred)
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": (f1_h + f1_a) / 2.0,
        "precision_human": prec_h,
        "recall_human": rec_h,
        "precision_ai": prec_a,
        "recall_ai": rec_a,
        "roc_auc": _auc(y, p),
        "pr_auc": _average_precision(y, p),
        "fpr_at_tpr95": fpr_at_tpr(y, p, 0.95),
        "ece_width": expected_calibration_error(y, p, strategy="width"),
        "brier": float(np.mean((np.clip(p, 0, 1) - y) ** 2)),
    }


def bootstrap_ci(
    y_true,
    y_score,
    threshold: float = 0.5,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 20260101,
    metrics: tuple[str, ...] = BOOTSTRAP_METRICS,
    confidence: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Stratified bootstrap CIs: resample positives and negatives separately so
    each replicate keeps the class mix (important for small test sets)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_score, dtype=float)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        sample = np.concatenate(
            [
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ]
        )
        vals = _fast_binary_metrics(y[sample], p[sample], threshold)
        for m in metrics:
            draws[m].append(vals[m])
    alpha = (1 - confidence) / 2
    out = {}
    for m in metrics:
        arr = np.asarray(draws[m], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            out[m] = {"lo": None, "hi": None}
            continue
        out[m] = {
            "lo": round(float(np.quantile(arr, alpha)), 6),
            "hi": round(float(np.quantile(arr, 1 - alpha)), 6),
        }
    return out


# --- paired / single-sample bootstrap ---------------------------------------


def bootstrap_mean_ci(
    values,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 20260101,
    confidence: float = 0.95,
) -> dict[str, float | None]:
    """Percentile CI for the mean of one sample (e.g. the share of attacked
    documents still detected). Non-finite values are dropped first."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    n = arr.size
    for i in range(n_boot):
        draws[i] = arr[rng.integers(0, n, size=n)].mean()
    alpha = (1 - confidence) / 2
    return {
        "mean": round(float(arr.mean()), 6),
        "lo": round(float(np.quantile(draws, alpha)), 6),
        "hi": round(float(np.quantile(draws, 1 - alpha)), 6),
    }


def paired_bootstrap_diff(
    reference,
    candidate,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 20260101,
    confidence: float = 0.95,
) -> dict[str, float | None]:
    """CI for ``mean(reference) - mean(candidate)`` with the same documents in
    both arms (the paraphrase-attack case: each attacked document scores a
    clean source document). Resampling is done over *documents*, so the
    correlation between the two arms is preserved and the interval is much
    tighter than two independent samples would give."""
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.size != cand.size:
        raise ValueError("paired_bootstrap_diff: arms must be aligned")
    keep = np.isfinite(ref) & np.isfinite(cand)
    ref, cand = ref[keep], cand[keep]
    if ref.size == 0:
        return {"mean_difference": None, "lo": None, "hi": None, "p_difference_positive": None}
    diff = ref - cand
    rng = np.random.default_rng(seed)
    n = diff.size
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = diff[rng.integers(0, n, size=n)].mean()
    alpha = (1 - confidence) / 2
    return {
        "mean_difference": round(float(diff.mean()), 6),
        "lo": round(float(np.quantile(draws, alpha)), 6),
        "hi": round(float(np.quantile(draws, 1 - alpha)), 6),
        "p_difference_positive": round(float((draws > 0).mean()), 6),
        "n_pairs": int(n),
    }


# --- multi-seed aggregation -------------------------------------------------


def aggregate_seeds(results: list[dict], metrics: list[str] | None = None) -> dict:
    """mean ± std over seeds, for the 'report mean and standard deviation'
    requirement. Missing metrics in some seeds are skipped, not imputed."""
    if not results:
        return {}
    keys = metrics or sorted(
        set().union(*(set(r.get("metrics", {})) for r in results))
    )
    out: dict[str, dict[str, float | int | None]] = {"n_seeds": len(results)}
    for key in keys:
        vals = [
            r["metrics"][key]
            for r in results
            if key in r.get("metrics", {}) and r["metrics"][key] is not None
        ]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[key] = {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
        }
    return out
