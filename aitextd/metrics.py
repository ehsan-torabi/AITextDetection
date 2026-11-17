"""Shared evaluation for every training notebook and the models.json registry.

One metric shape everywhere: evaluate(y_true, probs) -> accuracy / precision /
recall / f1 / roc_auc / ece / confusion, with the AI class as positive, and
evaluation_report(y_true, probs) as the full battery (ranking quality, low-FPR
operating points, agreement, calibration). The registry helpers map the basic
dict onto the camelCase models.json blocks, so a notebook never hand-writes
its own metric code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_COLUMNS = ("accuracy", "precision", "recall", "f1", "roc_auc", "ece")

# Column groups of evaluation_report(), for readable per-topic tables.
RANKING_COLUMNS = ("roc_auc", "pr_auc", "pauc_5", "tpr_at_fpr_1", "tpr_at_fpr_5", "fpr_at_tpr_90")
THRESHOLD_COLUMNS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "kappa",
    "precision",
    "recall",
    "f1",
    "fpr",
)
CALIBRATION_COLUMNS = ("ece", "ace", "mce", "brier", "nll", "calibration_slope", "calibration_intercept")


# --- basic bundle -----------------------------------------------------------


def expected_calibration_error(y_true, probs, n_bins: int = 10) -> float:
    """ECE over equal-width probability bins: sum_b (n_b / N) * |acc_b - conf_b|."""
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.sum() / len(probs) * abs(y_true[mask].mean() - probs[mask].mean())
    return float(ece)


def confusion(y_true, pred) -> dict:
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    return {
        "tn": int(((pred == 0) & (y_true == 0)).sum()),
        "fp": int(((pred == 1) & (y_true == 0)).sum()),
        "fn": int(((pred == 0) & (y_true == 1)).sum()),
        "tp": int(((pred == 1) & (y_true == 1)).sum()),
    }


def evaluate(y_true, probs, threshold: float = 0.5) -> dict:
    """Binary metric bundle for human (=0) vs AI (=1) at the given threshold."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
        "ece": round(expected_calibration_error(y_true, probs), 4),
        "confusion": confusion(y_true, pred),
    }


# --- operating points (RAID-style low-FPR regime) ---------------------------


def tpr_at_fpr(y_true, probs, fpr_budget: float) -> float:
    """Best TPR whose FPR stays within the budget (read off the ROC curve)."""
    fpr, tpr, _ = roc_curve(y_true, probs)
    reachable = fpr <= fpr_budget
    return float(tpr[reachable].max()) if reachable.any() else 0.0


def fpr_at_tpr(y_true, probs, tpr_target: float) -> float:
    """Smallest FPR that still reaches the TPR target (1.0 when unreachable)."""
    fpr, tpr, _ = roc_curve(y_true, probs)
    reachable = tpr >= tpr_target
    return float(fpr[reachable].min()) if reachable.any() else 1.0


def partial_auc(y_true, probs, fpr_max: float) -> float:
    """AUROC restricted to FPR in [0, fpr_max], linearly normalised (chance = 0.5)."""
    fpr, tpr, _ = roc_curve(y_true, probs)
    keep = fpr <= fpr_max
    if not keep.any() or fpr_max <= 0:
        return 0.0
    # roc_curve(drop_intermediate=True) removes the collinear tail points, so
    # the area must be extended horizontally from the last kept point to the
    # budget, otherwise early-saturating curves get an unfairly small pAUC.
    area = np.trapezoid(tpr[keep], fpr[keep])
    area += (fpr_max - fpr[keep][-1]) * tpr[keep][-1]
    return float(area / fpr_max)


# --- calibration beyond ECE --------------------------------------------------


def _binned_gaps(y_true, probs, n_bins: int, equal_mass: bool) -> np.ndarray:
    """|accuracy - confidence| of each probability bin, weighted by bin share."""
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if equal_mass:
        edges = np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        bins = np.clip(np.searchsorted(edges[1:-1], probs, side="right"), 0, n_bins - 1)
    else:
        bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    gaps = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            gaps[b] = abs(y_true[mask].mean() - probs[mask].mean()) * mask.mean()
    return gaps


def adaptive_ece(y_true, probs, n_bins: int = 15) -> float:
    """ACE: ECE over equal-mass bins, so crowded probability regions cannot hide."""
    return float(_binned_gaps(y_true, probs, n_bins, equal_mass=True).sum())


def max_calibration_error(y_true, probs, n_bins: int = 15) -> float:
    """MCE: the worst single equal-width calibration gap."""
    gaps = np.zeros(n_bins)
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            gaps[b] = abs(y_true[mask].mean() - probs[mask].mean())
    return float(gaps.max())


def calibration_slope_intercept(y_true, probs) -> tuple[float, float]:
    """Logistic-recalibration diagnostic: slope 1 / intercept 0 = calibrated."""
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000).fit(logit, np.asarray(y_true))
    return float(model.coef_[0][0]), float(model.intercept_[0])


# --- full battery ------------------------------------------------------------


def evaluation_report(y_true, probs, threshold: float = 0.5) -> dict:
    """The complete evaluation battery for one model, as one flat dict:
    ranking quality (AUROC / PR-AUC / partial AUC), low-FPR operating points
    (TPR@1%, TPR@5%, FPR@TPR90), threshold metrics (accuracy / macro-F1 /
    MCC / kappa / balanced accuracy / error rates) and calibration
    (ECE / ACE / MCE / Brier / NLL / slope-intercept)."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)
    n_human = max(int((y_true == 0).sum()), 1)
    n_ai = max(int((y_true == 1).sum()), 1)
    slope, intercept = calibration_slope_intercept(y_true, probs)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probs)), 4),
        "pauc_5": round(partial_auc(y_true, probs, 0.05), 4),
        "tpr_at_fpr_1": round(tpr_at_fpr(y_true, probs, 0.01), 4),
        "tpr_at_fpr_5": round(tpr_at_fpr(y_true, probs, 0.05), 4),
        "fpr_at_tpr_90": round(fpr_at_tpr(y_true, probs, 0.90), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "macro_f1": round(float(f1_score(y_true, pred, average="macro", zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, pred)), 4),
        "kappa": round(float(cohen_kappa_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "fpr": round(float(((pred == 1) & (y_true == 0)).sum()) / n_human, 4),
        "fnr": round(float(((pred == 0) & (y_true == 1)).sum()) / n_ai, 4),
        "ece": round(expected_calibration_error(y_true, probs), 4),
        "ace": round(adaptive_ece(y_true, probs), 4),
        "mce": round(max_calibration_error(y_true, probs), 4),
        "brier": round(float(np.mean((probs - y_true) ** 2)), 4),
        "nll": round(float(log_loss(y_true, np.clip(probs, 1e-6, 1 - 1e-6))), 4),
        "calibration_slope": round(slope, 4),
        "calibration_intercept": round(intercept, 4),
        "confusion": confusion(y_true, pred),
    }


def evaluation_table(reports: dict[str, dict], columns: tuple = RANKING_COLUMNS) -> pd.DataFrame:
    """One row per model over a chosen column group of evaluation_report()."""
    frame = pd.DataFrame({mid: {k: m[k] for k in columns} for mid, m in reports.items()}).T
    frame.index.name = "model"
    return frame


def results_frame(results: dict[str, dict]) -> pd.DataFrame:
    """One row per model, basic metric columns in the shared fixed order."""
    frame = pd.DataFrame({mid: {k: m[k] for k in METRIC_COLUMNS} for mid, m in results.items()}).T
    frame.index.name = "model"
    return frame


# --- uncertainty on the metrics ---------------------------------------------


def bootstrap_ci(
    y_true, probs, metrics: tuple = ("roc_auc", "f1", "tpr_at_fpr_1"), n_boot: int = 1000,
    seed: int = 42, ci: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Stratified bootstrap 95% CI per metric, resampling within each class."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    positives = np.flatnonzero(y_true == 1)
    negatives = np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in metrics}
    for _ in range(n_boot):
        take = np.concatenate([
            rng.choice(positives, len(positives), replace=True),
            rng.choice(negatives, len(negatives), replace=True),
        ])
        report = evaluation_report(y_true[take], probs[take])
        for metric in metrics:
            samples[metric].append(report[metric])
    alpha = (1 - ci) / 2
    return {
        metric: (round(float(np.quantile(values, alpha)), 4), round(float(np.quantile(values, 1 - alpha)), 4))
        for metric, values in samples.items()
    }


# --- paired model comparison -------------------------------------------------


def mcnemar_test(y_true, probs_a, probs_b, threshold: float = 0.5) -> dict:
    """Exact paired McNemar test on the binary errors of two models."""
    y_true = np.asarray(y_true)
    wrong_a = (np.asarray(probs_a) >= threshold).astype(int) != y_true
    wrong_b = (np.asarray(probs_b) >= threshold).astype(int) != y_true
    n01 = int((wrong_a & ~wrong_b).sum())  # A wrong, B right
    n10 = int((~wrong_a & wrong_b).sum())  # A right, B wrong
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0}
    p_value = binomtest(n01, n01 + n10, 0.5).pvalue
    return {"n01": n01, "n10": n10, "p_value": round(float(p_value), 4)}


def delong_test(y_true, probs_a, probs_b) -> dict:
    """Paired DeLong test for two AUROCs on the same documents (Sun & Xu 2014
    fast covariance over the pos/neg placement kernels)."""
    y_true = np.asarray(y_true)
    probs_a = np.asarray(probs_a, dtype=float)
    probs_b = np.asarray(probs_b, dtype=float)
    positives = y_true == 1
    negatives = ~positives

    def kernel(probs):
        pos, neg = probs[positives][:, None], probs[negatives][None, :]
        return (pos > neg) + 0.5 * (pos == neg)  # (n_pos, n_neg) placements

    kernel_a, kernel_b = kernel(probs_a), kernel(probs_b)
    auc_a, auc_b = float(kernel_a.mean()), float(kernel_b.mean())
    rows = np.cov(np.stack([kernel_a.mean(axis=1), kernel_b.mean(axis=1)]), ddof=1) / int(positives.sum())
    cols = np.cov(np.stack([kernel_a.mean(axis=0), kernel_b.mean(axis=0)]), ddof=1) / int(negatives.sum())
    variance = float(
        rows[0, 0] + rows[1, 1] - 2 * rows[0, 1] + cols[0, 0] + cols[1, 1] - 2 * cols[0, 1]
    )
    z = (auc_a - auc_b) / np.sqrt(variance) if variance > 0 else 0.0
    return {
        "auc_a": round(auc_a, 4),
        "auc_b": round(auc_b, 4),
        "z": round(float(z), 4),
        "p_value": round(float(2 * norm.sf(abs(z))), 4),
    }


# --- abstention (PAN Voight-Kampff conventions) ------------------------------


def c_at_1(y_true, pred, undecided) -> float:
    """PAN c@1: accuracy that credits undecided cases proportionally to the
    accuracy of the answered ones: (n_c + n_u * n_c / n_a) / n."""
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    undecided = np.asarray(undecided, dtype=bool)
    answered = ~undecided
    n_correct = int((pred[answered] == y_true[answered]).sum())
    n_undecided = int(undecided.sum())
    n = len(y_true)
    return float((n_correct + n_undecided * n_correct / max(n - n_undecided, 1)) / n)


def f0_5u(y_true, pred, undecided) -> float:
    """PAN F0.5u (Potha & Stamatatos 2019): undecided answers count as
    non-answers, i.e. negative predictions -- they never create false
    positives but always count as missed AI cases."""
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    resolved = np.where(np.asarray(undecided, dtype=bool), 0, pred)
    return round(float(fbeta_score(y_true, resolved, beta=0.5, zero_division=0)), 4)


# --- models.json registry mappings -------------------------------------------


def registry_metrics(results: dict) -> dict:
    """models.json metrics block (AI class = positive)."""
    return {
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1Score": results["f1"],
        "rocAuc": results["roc_auc"],
    }


def registry_confusion(results: dict) -> dict:
    c = results["confusion"]
    return {
        "trueNegative": c["tn"],
        "falsePositive": c["fp"],
        "falseNegative": c["fn"],
        "truePositive": c["tp"],
    }
