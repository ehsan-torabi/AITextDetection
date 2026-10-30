"""Post-hoc calibration and decision thresholds (review sections 1, 5, 8).

Two separate jobs, done in this order and both fitted on the *validation*
split only (never on test):

1. Calibration -- temperature scaling (and optionally Platt / isotonic) so a
   reported probability means what it says. Reported with ECE before/after.
2. Operating points -- a two-stage policy for a human-review workflow:

       p <  review_threshold          -> "human"      (no flag, automated)
       review_threshold <= p < auto_threshold
                                      -> "review"     (referred to a human)
       p >= auto_threshold            -> "ai"         (automated AI label)

   `review_threshold` is set from a recall target on validation (deferral must
   catch most machine text); `auto_threshold` is set from an FPR budget on
   validation (a wrong automated accusation is the costly error, so it needs a
   strict budget). Both are *validation* quantities; the test set only ever
   measures the outcome.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from eval_metrics import expected_calibration_error  # type: ignore[import-not-found]


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class TemperatureScaler:
    """Single-parameter calibration on the logit scale. Monotone, so ranking
    metrics (ROC-AUC, PR-AUC) are unchanged -- only the probabilities move."""

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, scores: np.ndarray, y: np.ndarray) -> TemperatureScaler:
        z = _logit(scores)
        y = np.asarray(y, dtype=float)

        def nll(log_temp: np.ndarray) -> float:
            temp = float(np.exp(log_temp[0]))
            p = _sigmoid(z / temp)
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        result = minimize(nll, x0=np.zeros(1), method="Nelder-Mead")
        self.temperature = float(np.exp(result.x[0]))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return _sigmoid(_logit(scores) / self.temperature)


class PlattScaler:
    """Two-parameter logistic recalibration (Platt scaling)."""

    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e6, solver="lbfgs")

    def fit(self, scores: np.ndarray, y: np.ndarray) -> PlattScaler:
        self.model.fit(_logit(scores).reshape(-1, 1), np.asarray(y, dtype=int))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]


class IsotonicScaler:
    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, scores: np.ndarray, y: np.ndarray) -> IsotonicScaler:
        self.model.fit(np.asarray(scores, dtype=float), np.asarray(y, dtype=float))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(scores, dtype=float))


CALIBRATORS = {
    "temperature": TemperatureScaler,
    "platt": PlattScaler,
    "isotonic": IsotonicScaler,
    "none": None,
}


def fit_best_calibrator(
    scores_val: np.ndarray,
    y_val: np.ndarray,
    candidates: tuple[str, ...] = ("none", "temperature", "platt", "isotonic"),
) -> tuple[object | None, str, dict[str, float]]:
    """Pick the calibrator with the lowest validation log-loss (ties -> the
    simpler/more monotone one, i.e. the earlier candidate). Returns
    (calibrator, name, validation report)."""
    y_val = np.asarray(y_val, dtype=int)
    scored: list[tuple[float, str, object | None]] = []
    report: dict[str, float] = {}
    for name in candidates:
        cls = CALIBRATORS[name]
        if cls is None:
            report["none"] = _nll(scores_val, y_val)
            report["none_ece"] = expected_calibration_error(y_val, scores_val)
            scored.append((report["none"], name, None))
            continue
        calibrator = cls().fit(scores_val, y_val)
        transformed = calibrator.transform(scores_val)
        report[name] = _nll(transformed, y_val)
        report[f"{name}_ece"] = expected_calibration_error(y_val, transformed)
        scored.append((report[name], name, calibrator))
    scored.sort(key=lambda item: item[0])
    best_nll, best_name, best_cal = scored[0]
    report["best"] = best_name
    report["best_nll"] = best_nll
    return best_cal, best_name, report


def _nll(scores: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(scores, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# --- thresholds -------------------------------------------------------------


def threshold_at_recall(y: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    """Largest threshold whose recall on the given sample is >= target."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = np.sort(scores[y == 1])
    if len(pos) == 0:
        return 0.0
    # scores are probabilities: keep the quantile that retains the target share
    q = np.quantile(pos, 1 - target_recall)
    return float(q)


def threshold_at_fpr(y: np.ndarray, scores: np.ndarray, fpr_budget: float) -> float:
    """Smallest threshold whose FPR on the given sample is <= budget."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    neg = np.sort(scores[y == 0])
    if len(neg) == 0:
        return 1.0
    q = np.quantile(neg, 1 - fpr_budget)
    return float(q)


class TwoStagePolicy:
    """review_threshold / auto_threshold fitted on validation, applied to test."""

    def __init__(
        self,
        review_recall: float = 0.95,
        auto_fpr: float = 0.005,
    ) -> None:
        self.review_recall = review_recall
        self.auto_fpr = auto_fpr
        self.review_threshold_ = 0.5
        self.auto_threshold_ = 0.5
        self.val_report_: dict[str, float] = {}

    def fit(self, y_val, scores_val) -> TwoStagePolicy:
        y_val = np.asarray(y_val, dtype=int)
        scores_val = np.asarray(scores_val, dtype=float)
        self.review_threshold_ = threshold_at_recall(y_val, scores_val, self.review_recall)
        self.auto_threshold_ = threshold_at_fpr(y_val, scores_val, self.auto_fpr)
        if self.auto_threshold_ < self.review_threshold_:
            # Inseparable on validation: keep the review band non-empty by
            # collapsing the automated threshold onto the deferral one.
            self.auto_threshold_ = self.review_threshold_
        self.val_report_ = self.report(y_val, scores_val, prefix="val_")
        return self

    def act(self, scores):
        scores_arr = np.atleast_1d(np.asarray(scores, dtype=float))
        actions = np.where(
            scores_arr >= self.auto_threshold_,
            "ai",
            np.where(scores_arr >= self.review_threshold_, "review", "human"),
        )
        return actions[0] if np.isscalar(scores) or scores_arr.size == 1 else actions

    def report(self, y, scores, prefix: str = "") -> dict[str, float]:
        y = np.asarray(y, dtype=int)
        actions = np.atleast_1d(self.act(scores))
        review_mask = actions == "review"
        auto_ai = actions == "ai"
        auto_human = actions == "human"
        n = len(y)
        return {
            f"{prefix}review_threshold": self.review_threshold_,
            f"{prefix}auto_threshold": self.auto_threshold_,
            f"{prefix}share_auto_human": float((auto_human & (y == 0)).sum() / max((y == 0).sum(), 1)),
            f"{prefix}share_deferred_or_flagged_ai": float(
                ((review_mask | auto_ai) & (y == 1)).sum() / max((y == 1).sum(), 1)
            ),
            f"{prefix}ai_called_ai_automatically": float(
                (auto_ai & (y == 1)).sum() / max((y == 1).sum(), 1)
            ),
            f"{prefix}human_called_ai_automatically": float(
                (auto_ai & (y == 0)).sum() / max((y == 0).sum(), 1)
            ),
            f"{prefix}deferred_share": float(review_mask.mean()),
            f"{prefix}n": float(n),
        }
