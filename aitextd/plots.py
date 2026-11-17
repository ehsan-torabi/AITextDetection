"""Charts for the training notebooks, following the project's matplotlib skill
(opinionated despined style: whitegrid base, DejaVu Sans, dimgrey text,
ColorBrewer / cubehelix palettes, value labels instead of value axes).

Every helper renders one figure and returns an axes (so the parent figure is
displayed exactly once); the notebook cell output IS the deliverable, so the
saved .ipynb carries the charts for reports and slides. Labels stay English
(matplotlib has no Persian shaping out of the box).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve

from aitextd import metrics

sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")

CATEGORICAL = ("#d73027", "#4575b4", "#555555", "#fc8d59", "#91bfdb", "#1f253f")
ACCENT_RED = "#bd0c0c"
GREY = "#bdbdbd"


def _frame(ax: plt.Axes) -> None:
    """Lightgrey panel frame for curves/point clouds that float in the plot area."""
    ax.patch.set_edgecolor("lightgrey")
    ax.patch.set_linewidth(0.8)


def _legend(ax: plt.Axes, loc: str = "upper left") -> None:
    ax.legend(frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey",
              labelcolor="dimgrey", loc=loc, fontsize=9)


def _ticks(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")


def _title(ax: plt.Axes, title: str, takeaway: str | None) -> None:
    ax.set_title(title, loc="left", fontsize=13, color="dimgrey", weight="bold", pad=24)
    if takeaway:
        ax.annotate(takeaway, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=9.5, color="dimgrey")


def _roc_label(y_true, probs, model_id: str) -> str:
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = float(np.trapezoid(tpr, fpr))
    return (f"{model_id} (AUROC {auc:.4f}, "
            f"TPR@1%FPR {metrics.tpr_at_fpr(y_true, probs, 0.01):.2f})")


def metric_bars(results: dict[str, dict], title: str) -> plt.Axes:
    """Grouped accuracy / F1 / ROC-AUC bars per model (P2 style: value labels
    carry the numbers, the value axis is hidden)."""
    columns = ("accuracy", "f1", "roc_auc")
    frame = pd.DataFrame({mid: {k: m[k] for k in columns} for mid, m in results.items()}).T
    ax = frame.plot.bar(rot=0, figsize=(max(6, 4 + len(frame)), 5.5), color=CATEGORICAL[:3])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, weight="bold", size=9, color="dimgrey")
    best = frame["f1"].idxmax()
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.grid(False)
    _ticks(ax)
    sns.despine(left=True, bottom=True)
    _title(ax, title, f"{best} leads on F1 ({frame.loc[best, 'f1']:.2f}); differences of <0.01 separate the rest")
    _legend(ax, "upper right")
    return ax


def confusion_grid(results: dict[str, dict], title: str = "Confusion matrices") -> plt.Axes:
    """One figure, one 2x2 heatmap per model; rows/cols = truth/prediction.
    Returns the first axes so the notebook renders the parent figure once."""
    ids = list(results)
    fig, axes = plt.subplots(1, len(ids), figsize=(2.5 * len(ids), 3.4))
    axes = np.atleast_1d(axes)
    palette = sns.color_palette("rocket", 5 + 2)[:4]
    for ax, mid in zip(axes, ids):
        cm = results[mid]["confusion"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        sns.heatmap(matrix, cmap=palette, annot=[[f"{v:,}" for v in row] for row in matrix],
                    fmt="", cbar=False, linewidths=0.8, linecolor="white", ax=ax,
                    annot_kws={"size": 10, "weight": "bold"}, square=True)
        ax.set_xticklabels(("human", "AI"), fontsize=9)
        ax.set_yticklabels(("human", "AI"), fontsize=9, rotation=0)
        ax.set_xlabel("prediction", fontsize=10, color="dimgrey")
        ax.set_title(mid, fontsize=10, color="dimgrey", weight="bold")
        ax.tick_params(length=0, labelcolor="dimgrey")
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, color="dimgrey", weight="bold")
    fig.tight_layout()
    return axes[0]


def roc_curves(y_true, probs_by_model: dict[str, np.ndarray], title: str) -> plt.Axes:
    """ROC curves per model with AUROC and TPR@1%FPR in the legend (P9 style)."""
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    palette = sns.cubehelix_palette(len(probs_by_model), rot=-0.25, light=0.7)
    for color, (model_id, probs) in zip(palette, probs_by_model.items()):
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=_roc_label(y_true, probs, model_id))
    ax.plot((0, 1), (0, 1), linestyle="--", color="orange", linewidth=1.5, zorder=0, label="no skill")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=12, labelpad=8, color="dimgrey")
    ax.set_ylabel("True Positive Rate", fontsize=12, labelpad=8, color="dimgrey")
    ax.grid(False)
    _frame(ax)
    _ticks(ax)
    sns.despine(left=True, bottom=True)
    _title(ax, title, None)
    _legend(ax, "lower right")
    return ax


def roc_curves_log(y_true, probs_by_model: dict[str, np.ndarray], title: str) -> plt.Axes:
    """Log-scale ROC focused on the low-FPR region that matters in practice,
    with reference lines at the 1% and 5% false-positive budgets (RAID style)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    palette = sns.cubehelix_palette(len(probs_by_model), rot=-0.25, light=0.7)
    for color, (model_id, probs) in zip(palette, probs_by_model.items()):
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax.step(fpr, tpr, where="post", color=color, linewidth=2,
                label=_roc_label(y_true, probs, model_id))
    ax.plot((1e-4, 1), (1e-4, 1), linestyle="--", color="orange", linewidth=1.5, zorder=0, label="no skill")
    for budget in (0.01, 0.05):
        ax.axvline(budget, color="grey", linestyle=":", linewidth=1)
        ax.text(budget * 1.15, 0.04, f"FPR {budget:.0%}", fontsize=9, color="dimgrey")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate (log scale)", fontsize=12, labelpad=8, color="dimgrey")
    ax.set_ylabel("True Positive Rate", fontsize=12, labelpad=8, color="dimgrey")
    ax.grid(False)
    _frame(ax)
    _ticks(ax)
    sns.despine(left=True, bottom=True)
    _title(ax, title, None)
    _legend(ax, "lower right")
    return ax


def reliability_grid(y_true, probs_by_model: dict[str, np.ndarray], n_bins: int = 15,
                     title: str = "Reliability diagrams") -> plt.Axes:
    """Calibration curve (top) + confidence histogram (bottom) per model, with
    the equal-width-bin ECE in each panel title."""
    ids = list(probs_by_model)
    y = np.asarray(y_true, dtype=float)
    fig, axes = plt.subplots(
        2, len(ids), figsize=(2.6 * len(ids), 4.6), sharex="col", sharey="row",
        gridspec_kw={"height_ratios": (3, 1)}, dpi=150,
    )
    axes = np.atleast_2d(axes)
    centers = (np.arange(n_bins) + 0.5) / n_bins
    for col, mid in enumerate(ids):
        probs = np.asarray(probs_by_model[mid], dtype=float)
        empirical, confidence, share = [], [], []
        for b in range(n_bins):
            mask = (probs >= b / n_bins) & (probs < (b + 1) / n_bins)
            share.append(mask.mean())
            if mask.any():
                empirical.append(y[mask].mean())
                confidence.append(probs[mask].mean())
            else:
                empirical.append(np.nan)
                confidence.append(np.nan)
        ax_top, ax_bottom = axes[0, col], axes[1, col]
        ax_top.plot((0, 1), (0, 1), linestyle="--", color="orange", linewidth=1.2, zorder=0, label="ideal")
        ax_top.plot(centers, empirical, "o-", color="#446485", markersize=3, label="empirical")
        gap = np.abs(np.nan_to_num(np.array(empirical)) - np.nan_to_num(np.array(confidence)))
        ax_top.bar(centers, gap, width=0.06, alpha=0.35, color=ACCENT_RED, label="|gap|")
        ax_top.set_title(
            f"{mid}\nECE {metrics.expected_calibration_error(y, probs, n_bins):.3f}",
            loc="left", fontsize=9, color="dimgrey", weight="bold",
        )
        ax_bottom.bar(centers, share, width=0.06, color=GREY)
        ax_bottom.set_xlabel("confidence", fontsize=9, color="dimgrey")
        if col == 0:
            ax_top.set_ylabel("empirical accuracy", fontsize=9, color="dimgrey")
            ax_bottom.set_ylabel("share", fontsize=9, color="dimgrey")
        for ax in (ax_top, ax_bottom):
            ax.tick_params(length=0, labelcolor="dimgrey", labelsize=7)
            ax.set_xlim(0, 1)
        ax_top.set_ylim(0, 1.02)
        ax_top.grid(False)
        ax_bottom.grid(False)
        _frame(ax_top)
        ax_top.legend(fontsize=6, frameon=True, facecolor="white", framealpha=0.8,
                      edgecolor="lightgrey", labelcolor="dimgrey", loc="upper left")
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, color="dimgrey", weight="bold")
    fig.tight_layout()
    return axes[0, 0]


def score_distributions(y_true, probs_by_model: dict[str, np.ndarray],
                        title: str = "AI-probability distributions") -> plt.Axes:
    """Human vs AI probability density per model, with the review zone shaded."""
    ids = list(probs_by_model)
    y = np.asarray(y_true)
    fig, axes = plt.subplots(1, len(ids), figsize=(2.6 * len(ids), 3.4), sharey=True, dpi=150)
    axes = np.atleast_1d(axes)
    grid = np.linspace(0, 1, 200)
    for ax, mid in zip(axes, ids):
        for label, name, color, linestyle in ((0, "human", "#4575b4", "--"), (1, "AI", "#d73027", "-")):
            values = np.asarray(probs_by_model[mid], dtype=float)[y == label]
            if len(values) > 1:
                density = gaussian_kde(np.clip(values, 0.0, 1.0))(grid)
                ax.plot(grid, density, linestyle, color=color, linewidth=1.5, label=name)
        ax.axvspan(0.45, 0.55, color="grey", alpha=0.25, linewidth=0)
        ax.set_title(mid, loc="left", fontsize=10, color="dimgrey", weight="bold")
        ax.set_xlabel("AI probability", fontsize=9, color="dimgrey")
        _ticks(ax)
        ax.grid(False)
        _frame(ax)
        ax.legend(fontsize=7, frameon=True, facecolor="white", framealpha=0.8,
                  edgecolor="lightgrey", labelcolor="dimgrey", loc="upper center")
    axes[0].set_ylabel("density", fontsize=9, color="dimgrey")
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, color="dimgrey", weight="bold")
    fig.tight_layout()
    return axes[0]


def bar_series(series: pd.Series, title: str, ylabel: str) -> plt.Axes:
    """Horizontal bars, one value per model (P1 style: value labels, no grid,
    grey baseline); supports negative values; the best value is annotated."""
    frame = series.sort_values()
    palette = sns.color_palette("rocket", len(frame) + 2)[:len(frame)]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(frame) + 1.5), dpi=150)
    ax.barh(frame.index, frame.values, color=palette)
    for i, value in enumerate(frame.values):
        ax.text(value + (0.015 if value >= 0 else -0.015), i, f"{value:.3f}",
                ha="left" if value >= 0 else "right", va="center",
                weight="bold", size=10, color="dimgrey")
    ax.axvline(0, color="lightgrey", linewidth=0.8, zorder=0)
    ax.set_xlim(min(0.0, float(frame.min()) * 1.1), max(1.1, float(frame.max()) * 1.12))
    ax.set_xlabel(ylabel, fontsize=11, color="dimgrey")
    ax.set_yticks([])
    ax.grid(False)
    _ticks(ax)
    sns.despine(left=True, bottom=True)
    _title(ax, title, f"best: {frame.idxmax()} at {frame.max():.3f}")
    return ax


def dumbbell(clean: pd.Series, attacked: pd.Series, title: str, xlabel: str) -> plt.Axes:
    """Clean-vs-attacked paired dot plot (drop magnitude between the dots),
    sorted by the attacked value; filled = clean, open = attacked."""
    frame = pd.DataFrame({"clean": clean, "attacked": attacked}).sort_values("attacked")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(frame) + 1.5), dpi=150)
    rows = np.arange(len(frame))
    for i, (_, row) in enumerate(frame.iterrows()):
        ax.plot((row["attacked"], row["clean"]), (i, i), color=GREY, linewidth=1.5, zorder=1)
        ax.annotate(f"delta {row['clean'] - row['attacked']:+.2f}",
                    (max(row["clean"], row["attacked"]), i), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, color="dimgrey", va="center")
    ax.scatter(frame["clean"], rows, s=50, color="#4575b4", zorder=2, label="clean")
    ax.scatter(frame["attacked"], rows, s=50, facecolors="none", edgecolors="#d73027",
               linewidths=1.6, zorder=2, label="attacked")
    ax.set_yticks(rows, frame.index, fontsize=10)
    ax.tick_params(axis="y", length=0, labelcolor="dimgrey")
    ax.set_xlim(-0.03, max(1.08, float(frame["clean"].max()) * 1.15))
    ax.set_xlabel(xlabel, fontsize=11, color="dimgrey")
    ax.grid(False)
    _ticks(ax)
    sns.despine(left=True, bottom=True)
    _title(ax, title, None)
    _legend(ax, "lower right")
    return ax


def training_curve(history: list[dict], title: str, metric_label: str = "eval F1") -> plt.Axes:
    """Loss (left axis) and the per-epoch quality metric (right axis), from the
    fit_classifier history records {epoch, loss, metric}."""
    epochs = [record["epoch"] for record in history]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(epochs, [record["loss"] for record in history], color="#d73027", marker="o", label="train loss")
    ax.set_xlabel("epoch", fontsize=11, color="dimgrey")
    ax.set_ylabel("train loss", fontsize=11, color="#d73027")
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(epochs, [record["metric"] for record in history], color="#4575b4", marker="s", label=metric_label)
    ax2.set_ylabel(metric_label, fontsize=11, color="#4575b4")
    ax2.tick_params(axis="y", which="both", length=0, labelcolor="dimgrey")
    ax2.grid(False)
    sns.despine(left=True, bottom=True, right=True)
    _title(ax, title, f"final {metric_label} {history[-1]['metric']:.3f} after {len(history)} epochs")
    _legend(ax, "center right")
    return ax


def risk_coverage(coverages, risks, operating_point=None, title: str = "Risk-coverage curve") -> plt.Axes:
    """Selective-classification curve: error among auto-decided documents as
    the accepted share grows; the policy's operating point gets a marker."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(coverages, risks, color="#446485", linewidth=2)
    if operating_point is not None:
        coverage, risk = operating_point
        ax.scatter((coverage,), (risk,), s=60, color=ACCENT_RED, zorder=3,
                   label=f"policy ({coverage:.0%} coverage, {risk:.1%} error)")
    ax.set_xlabel("coverage (share of documents auto-decided)", fontsize=11, color="dimgrey")
    ax.set_ylabel("selective error", fontsize=11, color="dimgrey")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_ylim(0, max(0.05, float(np.nanmax(risks)) * 1.15))
    _frame(ax)
    _ticks(ax)
    sns.despine(left=True, bottom=True)
    _title(ax, title, None)
    _legend(ax, "lower left")
    return ax


def heatmap(frame: pd.DataFrame, title: str, fmt: str = ".3f") -> plt.Axes:
    """Annotated value matrix (rocket cmap, P7 style); the diagonal gets a bold
    border when the matrix is square (train -> test transfer maps)."""
    fig, ax = plt.subplots(figsize=(1.1 * len(frame.columns) + 2, 0.55 * len(frame) + 1.8), dpi=150)
    palette = sns.color_palette("rocket", as_cmap=True)
    sns.heatmap(frame, cmap=palette, annot=True, fmt=fmt, linewidths=0.8, linecolor="white",
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 9}, ax=ax)
    if frame.shape[0] == frame.shape[1]:
        for i in range(len(frame.index)):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", linewidth=2))
    ax.tick_params(length=0, labelcolor="dimgrey", labelsize=9)
    ax.set_title(title, loc="left", fontsize=12, color="dimgrey", weight="bold")
    return ax


def grouped_counts(counts: pd.DataFrame, title: str, ylabel: str = "documents") -> plt.Axes:
    """Grouped vertical bars: rows = groups (e.g. splits), columns = series."""
    ax = counts.plot.bar(rot=0, figsize=(8, 5), color=CATEGORICAL[:len(counts.columns)])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=2, size=9, weight="bold", color="dimgrey")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.grid(False)
    _ticks(ax)
    sns.despine(left=True, bottom=True)
    total = int(counts.sum().sum())
    _title(ax, title, f"{total:,} documents, class ratios preserved in every split")
    _legend(ax, "upper right")
    return ax
