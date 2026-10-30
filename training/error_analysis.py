"""Error analysis: inspect at least 100 false positives and 100 false negatives.

The review requires that the headline numbers are not the end of the story:
the two error directions must be looked at document by document, with a
taxonomy and a review workbook a human can work through.

What this script produces (all under ``reports/error_analysis/``):

- ``fp_review.csv``   the highest-scoring *human* documents on the main test
                      split (at least 100, more if the model produces more
                      above the 0.5 decision point), with the metadata a
                      reviewer needs: domain, length, formality, calibrated
                      probability, decision band under the validation-tuned
                      two-stage policy, and a text excerpt.
- ``fn_review.csv``   the lowest-scoring *machine* documents, same columns.
- ``taxonomy.md``     aggregate counts per rule-based taxonomy bucket, so the
                      prose in the report can say *what kind* of error each
                      direction is dominated by.
- ``feature_contrasts.md``  median style features for FP vs true-negative
                      humans and FN vs true-positive machines, with a
                      Mann-Whitney U p-value: this is the "why" behind the
                      errors, in the feature space the models actually use.
- ``summary.json``    the same numbers in machine-readable form, plus the
                      seeds/models covered (seeds are pooled; every reviewed
                      row records its seed).

Run from the project root (after ``training/eval_suite.py``):
    python training/error_analysis.py --model tfidf_svm
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from aitextd import dataset as ds
from aitextd import paths

DATASET = paths.DATA_DIR / "dataset_v2.parquet"
STYLE = paths.DATA_DIR / "style_features_v2.parquet"
SPLITS = paths.DATA_DIR / "splits_v2.joblib"
REPORTS = PROJECT_ROOT / "reports"
PREDICTIONS = REPORTS / "predictions"
OUT_DIR = REPORTS / "error_analysis"

MIN_PER_DIRECTION = 100
EXCERPT_CHARS = 400
TAXONOMY_MIN_N = 5


def load_predictions(model: str, condition: str, split: str) -> list[dict]:
    """Every seed's saved prediction bundle for one model/condition/split."""
    files = sorted(PREDICTIONS.glob(f"{model}__seed*__{condition}__{split}.npz"))
    out = []
    for path in files:
        seed = int(path.name.split("__seed")[1].split("__")[0])
        with np.load(path) as payload:
            out.append(
                {
                    "seed": seed,
                    "idx": payload["idx"],
                    "y": payload["y"],
                    "raw": payload["raw"],
                    "calibrated": payload["calibrated"],
                }
            )
    return out


def rule_taxonomy_fp(row: pd.Series, zwnj: float, informal: float, formal: float) -> str:
    """Rule-based bucket for a flagged human document.

    Deliberately crude and readable: the point is to give the report honest
    sentences like "most false positives are short informal social-media
    posts", not a second model.
    """
    if row["origin"] == "translated":
        return "translated_text"
    if row["length_bucket"] == "short":
        return "short_document"
    if row["domain"] == "social_media":
        return "social_media_register"
    if row["domain"] == "legal_admin":
        return "legal_admin_register"
    if row["formality"] == "informal":
        return "informal_register"
    if row["domain"] == "encyclopedia":
        return "encyclopedic_style"
    if row["domain"] == "news":
        return "news_style"
    if formal > 0.0 and row["length_words"] >= 500:
        return "long_formal_text"
    return "unclassified"


def rule_taxonomy_fn(row: pd.Series, zwnj: float, informal: float) -> str:
    """Rule-based bucket for an undetected machine document."""
    if row["length_bucket"] == "short":
        return "short_generation"
    if row["prompt_style"] == "qa_explainer":
        return "qa_style_generation"
    if row["prompt_style"] == "summary_overview":
        return "summary_style_generation"
    if informal >= 0.004:
        return "informal_generation"
    if row["length_bucket"] == "long":
        return "long_generation"
    if row["formality"] == "formal":
        return "formal_generation"
    return "unclassified"


def build_review_frame(
    df: pd.DataFrame,
    style: pd.DataFrame,
    payload: dict,
    direction: str,
    policy_thresholds: dict[str, float],
) -> pd.DataFrame:
    """Rows for one direction, sorted so the worst calls come first."""
    idx = np.asarray(payload["idx"])
    scores = np.asarray(payload["calibrated"], dtype=float)
    y = np.asarray(payload["y"])
    subset = df.iloc[idx].copy()
    subset["calibrated_probability"] = scores
    subset["raw_probability"] = np.asarray(payload["raw"], dtype=float)
    subset["seed"] = payload["seed"]
    subset["model"] = payload.get("model", "")

    review_threshold = policy_thresholds.get("review_threshold", 0.5)
    auto_threshold = policy_thresholds.get("auto_threshold", 0.5)
    subset["decision"] = np.where(
        scores >= auto_threshold,
        "automated_ai",
        np.where(scores >= review_threshold, "review", "human"),
    )

    if direction == "fp":
        candidates = subset[subset["label"] == ds.HUMAN].sort_values(
            "calibrated_probability", ascending=False
        )
        n_above = int((candidates["calibrated_probability"] >= 0.5).sum())
    else:
        candidates = subset[subset["label"] != ds.HUMAN].sort_values(
            "calibrated_probability", ascending=True
        )
        n_above = int((candidates["calibrated_probability"] < 0.5).sum())

    keep = max(MIN_PER_DIRECTION, n_above)
    candidates = candidates.head(keep).copy()

    # taxonomy + style context
    style_lookup = style.set_index("doc_id")
    zwnj, informal, formal = [], [], []
    for doc_id in candidates["doc_id"]:
        row_styles = style_lookup.loc[doc_id] if doc_id in style_lookup.index else None
        zwnj.append(float(row_styles["zwnj_rate"]) if row_styles is not None else np.nan)
        informal.append(float(row_styles["informal_marker_rate"]) if row_styles is not None else np.nan)
        formal.append(float(row_styles["formal_marker_rate"]) if row_styles is not None else np.nan)
    candidates["zwnj_rate"] = zwnj
    candidates["informal_marker_rate"] = informal
    candidates["formal_marker_rate"] = formal

    taxonomy = []
    for _, row in candidates.iterrows():
        if direction == "fp":
            taxonomy.append(rule_taxonomy_fp(row, row["zwnj_rate"], row["informal_marker_rate"], row["formal_marker_rate"]))
        else:
            taxonomy.append(rule_taxonomy_fn(row, row["zwnj_rate"], row["informal_marker_rate"]))
    candidates["taxonomy_bucket"] = taxonomy
    candidates["text_excerpt"] = candidates["text"].str.slice(0, EXCERPT_CHARS).str.replace(r"\s+", " ", regex=True)
    # columns a human reviewer fills in while working through the file
    candidates["review_verdict"] = ""
    candidates["review_notes"] = ""

    columns = [
        "model",
        "seed",
        "doc_id",
        "label",
        "domain",
        "origin",
        "generator",
        "prompt_style",
        "length_words",
        "length_bucket",
        "formality",
        "calibrated_probability",
        "raw_probability",
        "decision",
        "taxonomy_bucket",
        "zwnj_rate",
        "informal_marker_rate",
        "formal_marker_rate",
        "text_excerpt",
        "review_verdict",
        "review_notes",
    ]
    return candidates[columns].reset_index(drop=True)


def feature_contrasts(
    df: pd.DataFrame, style: pd.DataFrame, pooled: pd.DataFrame, direction: str
) -> pd.DataFrame:
    """Median style features for the error group vs the correctly handled group."""
    from scipy.stats import mannwhitneyu

    style_features = [c for c in style.columns if c != "doc_id"]
    merged = style.merge(df[["doc_id", "label"]], on="doc_id", how="left")
    if direction == "fp":
        error_ids = set(pooled[pooled["calibrated_probability"] >= 0.5]["doc_id"])
        group = merged[(merged["label"] == ds.HUMAN) & merged["doc_id"].isin(error_ids)]
        other = merged[(merged["label"] == ds.HUMAN) & ~merged["doc_id"].isin(error_ids)]
    else:
        error_ids = set(pooled[pooled["calibrated_probability"] < 0.5]["doc_id"])
        group = merged[(merged["label"] != ds.HUMAN) & merged["doc_id"].isin(error_ids)]
        other = merged[(merged["label"] != ds.HUMAN) & ~merged["doc_id"].isin(error_ids)]

    rows = []
    for feature in style_features:
        a = group[feature].to_numpy(dtype=float)
        b = other[feature].to_numpy(dtype=float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size == 0 or b.size == 0:
            continue
        try:
            p_value = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except ValueError:
            p_value = float("nan")
        rows.append(
            {
                "feature": feature,
                "error_median": round(float(np.median(a)), 4),
                "correct_median": round(float(np.median(b)), 4),
                "error_n": int(a.size),
                "correct_n": int(b.size),
                "mann_whitney_p": p_value,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["significant_at_0.01"] = frame["mann_whitney_p"] < 0.01
        frame = frame.sort_values("mann_whitney_p")
    return frame


def taxonomy_table(frame: pd.DataFrame) -> str:
    lines = ["| taxonomy bucket | n | share | median probability | median words |", "|---|---|---|---|---|"]
    counts = frame["taxonomy_bucket"].value_counts()
    for bucket, count in counts.items():
        if count < TAXONOMY_MIN_N and len(counts) > 5:
            continue
        block = frame[frame["taxonomy_bucket"] == bucket]
        lines.append(
            f"| {bucket} | {int(count)} | {count / len(frame):.3f} | "
            f"{block['calibrated_probability'].median():.3f} | {block['length_words'].median():.0f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="tfidf_svm")
    ap.add_argument("--condition", default="in_domain_group")
    ap.add_argument("--split", default="in_domain_group__test")
    ap.add_argument("--thresholds-source", default="val")
    args = ap.parse_args()

    df = pd.read_parquet(DATASET).reset_index(drop=True)
    style = pd.read_parquet(STYLE)
    results_path = REPORTS / "eval_results.json"
    if not results_path.exists():
        raise SystemExit(f"{results_path} missing; run training/eval_suite.py first")
    results = json.loads(results_path.read_text(encoding="utf-8"))

    condition = next(
        (c for c in results["conditions"] if c.get("condition") == args.condition), None
    )
    if condition is None:
        raise SystemExit(f"condition {args.condition!r} not found in eval_results.json")
    model_records = condition["models"].get(args.model)
    if not model_records:
        raise SystemExit(f"model {args.model!r} not found in condition {args.condition!r}")

    thresholds = {
        "review_threshold": float(
            np.mean([r["policy_validation"]["val_review_threshold"] for r in model_records])
        ),
        "auto_threshold": float(
            np.mean([r["policy_validation"]["val_auto_threshold"] for r in model_records])
        ),
    }

    frames = load_predictions(args.model, args.condition, args.split)
    if not frames:
        raise SystemExit(f"no saved predictions for {args.model} / {args.split}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "condition": args.condition,
        "split": args.split,
        "seeds": [f["seed"] for f in frames],
        "policy_thresholds": thresholds,
        "min_per_direction": MIN_PER_DIRECTION,
    }

    for direction in ("fp", "fn"):
        pieces = [
            build_review_frame(df, style, {**payload, "model": args.model}, direction, thresholds)
            for payload in frames
        ]
        pooled = pd.concat(pieces, ignore_index=True)
        # one row per document: keep the seed with the most extreme score
        if direction == "fp":
            pooled = pooled.sort_values("calibrated_probability", ascending=False)
        else:
            pooled = pooled.sort_values("calibrated_probability", ascending=True)
        pooled = pooled.drop_duplicates(subset="doc_id", keep="first").reset_index(drop=True)

        above = int((pooled["calibrated_probability"] >= 0.5).sum()) if direction == "fp" else int((pooled["calibrated_probability"] < 0.5).sum())
        csv_path = OUT_DIR / f"{direction}_review.csv"
        pooled.to_csv(csv_path, index=False, encoding="utf-8-sig")
        contrasts = feature_contrasts(df, style, pooled, direction)
        if not contrasts.empty:
            contrasts.to_csv(OUT_DIR / f"{direction}_feature_contrasts.csv", index=False, encoding="utf-8-sig")
        (OUT_DIR / f"{direction}_taxonomy.md").write_text(
            f"### {'False positives (flagged human documents)' if direction == 'fp' else 'False negatives (missed machine documents)'}\n\n"
            + taxonomy_table(pooled)
            + f"\n_reviewed rows: {len(pooled)}; above the 0.5 decision point: {above}; "
            f"pooled over {len(frames)} seed(s), one row per document._\n",
            encoding="utf-8",
        )
        if not contrasts.empty:
            (OUT_DIR / f"{direction}_feature_contrasts.md").write_text(
                contrasts.to_string(index=False) + "\n", encoding="utf-8"
            )
        summary[direction] = {
            "reviewed_rows": int(len(pooled)),
            "above_decision_point": above,
            "decision_mix": pooled["decision"].value_counts().to_dict(),
            "domain_mix": pooled["domain"].value_counts().to_dict(),
            "length_bucket_mix": pooled["length_bucket"].value_counts().to_dict(),
            "formality_mix": pooled["formality"].value_counts().to_dict(),
            "taxonomy_mix": pooled["taxonomy_bucket"].value_counts().to_dict(),
            "csv": csv_path.name,
        }
        print(f"[{direction}] {len(pooled)} documents reviewed -> {csv_path}", flush=True)

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
