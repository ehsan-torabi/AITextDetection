"""Rigorous evaluation suite: every model against every required condition.

What it trains (all CPU-feasible, seeds recorded):

- ``tfidf_svm``        TF-IDF (1-2 gram, chi-free) + LinearSVC
- ``tfidf_logreg``     TF-IDF + Logistic Regression
- ``tfidf_nb``         TF-IDF + Multinomial Naive Bayes
- ``style_rf``         the 20 Persian style features + Random Forest
- ``tfidf_style_rf``   TF-IDF (top 30k by chi2) + style features + Random Forest
- ``length_only``      document length alone + Logistic Regression (diagnostic
                       baseline: quantifies how much of the headline number is
                       just "AI texts are longer / shorter")
- ``perplexity_only``  raw-LM perplexity + Logistic Regression (sign learned on
                       train; the local signal was found to be inverted)
- ``ensemble``         mean of the calibrated members' probabilities

What it evaluates (conditions from review sections 3 and 7):

- in_domain_group (group-aware test set), per model and per seed
- cross_generator (unseen LLMs), cross_domain (held-out human domain),
  cross_style (held-out AI prompt style), few_shot (8/32/128 rows per class),
  length buckets, v1 comparability, paraphrase attack (clean vs attacked),
  the three-class human / ai / ai_revised problem, and the balanced binary
  revision-detection condition (human vs machine-revised; "three_class" runs
  both revision views).

Every model/split pair is reported with the full metric suite, stratified
bootstrap CIs, and a two-stage decision policy fitted on *validation* only.
Predictions are stored under reports/predictions/ so tables can be rebuilt
without retraining; summary tables land in reports/tables/.

Run from the project root:
    python training/eval_suite.py --conditions all --models core
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import calibration  # type: ignore[import-not-found]
import eval_metrics as em  # type: ignore[import-not-found]

from aitextd import dataset as ds
from aitextd import paths

DATA = paths.DATA_DIR
DATASET = DATA / "dataset_v2.parquet"
STYLE_FEATURES = DATA / "style_features_v2.parquet"
SPLITS = DATA / "splits_v2.joblib"
ATTACK = DATA / "paraphrase_attack.parquet"
PERPLEXITY = DATA / "perplexity_v2.joblib"

REPORTS = PROJECT_ROOT / "reports"
TABLES = REPORTS / "tables"
PREDICTIONS = REPORTS / "predictions"
RESULTS_JSON = REPORTS / "eval_results.json"

CORE_MODELS = (
    "tfidf_svm",
    "tfidf_logreg",
    "tfidf_nb",
    "style_rf",
    "tfidf_style_rf",
    "length_only",
    "perplexity_only",
)
STYLE_FEATURE_COLUMNS: list[str] = []


# ---------------------------------------------------------------------------
# feature assembly
# ---------------------------------------------------------------------------


class Bundle:
    """TF-IDF + style + auxiliary features fitted on a specific train split."""

    def __init__(self, df: pd.DataFrame, style: pd.DataFrame, train_idx: np.ndarray):
        train_texts = df.iloc[train_idx]["text"].tolist()
        self.vectorizer = TfidfVectorizer(
            max_features=150_000, min_df=3, ngram_range=(1, 2), sublinear_tf=True
        )
        self.vectorizer.fit(train_texts)
        self.df = df
        self.style = style
        style_cols = [c for c in style.columns if c != "doc_id"]
        self.style_matrix = style[style_cols].to_numpy(dtype=np.float32)
        self.style_columns = style_cols
        self._tfidf_cache: csr_matrix | None = None
        self._chi_selector: SelectKBest | None = None
        self.train_idx = train_idx

    def tfidf(self) -> csr_matrix:
        if self._tfidf_cache is None:
            self._tfidf_cache = self.vectorizer.transform(self.df["text"].tolist())
        return self._tfidf_cache

    def tfidf_style(self, top_k: int = 30_000) -> csr_matrix:
        tfidf = self.tfidf()
        if self._chi_selector is None:
            y = (self.df["label"].to_numpy() != ds.HUMAN).astype(int)
            self._chi_selector = SelectKBest(chi2, k=min(top_k, tfidf.shape[1]))
            self._chi_selector.fit(tfidf[self.train_idx], y[self.train_idx])
        selected = self._chi_selector.transform(tfidf)
        return hstack([selected, csr_matrix(self.style_matrix)]).tocsr()

    def with_texts(self, texts: list[str]) -> "Bundle":
        """Return a sibling bundle that additionally contains ``texts`` as new
        trailing rows.

        The vectorizer and the chi2 selector stay exactly as fitted on the
        training split, so attack documents can never influence the model; only
        the scored matrix grows. Used by the paraphrase-attack condition.
        """
        import copy

        from aitextd.features import compute_style_features

        clone = copy.copy(self)
        rows = []
        for text in texts:
            base = compute_style_features(text, text, self.vectorizer)
            merged = {**base, **ds.extended_style_features(text)}
            rows.append({c: float(merged.get(c, 0.0)) for c in self.style_columns})

        new_frame = pd.DataFrame(
            {"text": texts, "label": ds.AI, "length_words": [len(t.split()) for t in texts]}
        )
        for column in self.df.columns:
            if column not in new_frame:
                new_frame[column] = ""
        new_frame = new_frame[self.df.columns]

        style_frame = pd.DataFrame(rows, columns=self.style_columns)
        clone.df = pd.concat([self.df, new_frame], ignore_index=True)
        clone.style = style_frame
        clone.style_matrix = np.vstack(
            [self.style_matrix, style_frame.to_numpy(dtype=np.float32)]
        )
        clone._tfidf_cache = None
        clone.new_positions = np.arange(len(self.df), len(clone.df))
        return clone


def length_features(df: pd.DataFrame) -> np.ndarray:
    words = df["length_words"].to_numpy(dtype=np.float32)
    chars = df["text"].str.len().to_numpy(dtype=np.float32)
    return np.vstack([np.log1p(words), np.log1p(chars)]).T


# ---------------------------------------------------------------------------
# model zoo
# ---------------------------------------------------------------------------


class Scorer:
    """Uniform interface: fit on train indices, score any index array."""

    def __init__(self, key: str, seed: int):
        self.key = key
        self.seed = seed

    def fit(self, bundle: Bundle, train_idx: np.ndarray):  # pragma: no cover - interface
        raise NotImplementedError

    def score(self, bundle: Bundle, idx: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class TfidfScorer(Scorer):
    def __init__(self, key: str, seed: int, estimator):
        super().__init__(key, seed)
        self.estimator = estimator
        self.needs_style = False

    def _matrix(self, bundle: Bundle):
        return bundle.tfidf()

    def fit(self, bundle: Bundle, train_idx: np.ndarray):
        y = (bundle.df.iloc[train_idx]["label"].to_numpy() != ds.HUMAN).astype(int)
        self.estimator.fit(self._matrix(bundle)[train_idx], y)
        return self

    def score(self, bundle: Bundle, idx: np.ndarray) -> np.ndarray:
        matrix = self._matrix(bundle)[idx]
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(matrix)[:, 1]
        # LinearSVC: the serving stack maps the decision value through a
        # sigmoid; the calibrator fitted on validation refines it afterwards.
        decision = self.estimator.decision_function(matrix)
        return 1.0 / (1.0 + np.exp(-decision))


class TfidfStyleScorer(TfidfScorer):
    def __init__(self, key: str, seed: int, estimator, top_k: int = 30_000):
        super().__init__(key, seed, estimator)
        self.top_k = top_k
        self.needs_style = True

    def _matrix(self, bundle: Bundle):
        return bundle.tfidf_style(self.top_k)


class StyleScorer(Scorer):
    needs_style = True

    def __init__(self, key: str, seed: int, estimator):
        super().__init__(key, seed)
        self.estimator = estimator
        self.scaler = StandardScaler()

    def fit(self, bundle: Bundle, train_idx: np.ndarray):
        y = (bundle.df.iloc[train_idx]["label"].to_numpy() != ds.HUMAN).astype(int)
        matrix = bundle.style_matrix
        matrix = self.scaler.fit_transform(matrix[train_idx])
        self.estimator.fit(matrix, y)
        return self

    def score(self, bundle: Bundle, idx: np.ndarray) -> np.ndarray:
        matrix = self.scaler.transform(bundle.style_matrix[idx])
        return self.estimator.predict_proba(matrix)[:, 1]


class LengthScorer(Scorer):
    def __init__(self, key: str, seed: int):
        super().__init__(key, seed)
        self.scaler = StandardScaler()
        self.estimator = LogisticRegression(max_iter=2000, class_weight="balanced")

    def fit(self, bundle: Bundle, train_idx: np.ndarray):
        y = (bundle.df.iloc[train_idx]["label"].to_numpy() != ds.HUMAN).astype(int)
        matrix = length_features(bundle.df)
        matrix = self.scaler.fit_transform(matrix[train_idx])
        self.estimator.fit(matrix, y)
        return self

    def score(self, bundle: Bundle, idx: np.ndarray) -> np.ndarray:
        matrix = self.scaler.transform(length_features(bundle.df)[idx])
        return self.estimator.predict_proba(matrix)[:, 1]


class PerplexityScorer(Scorer):
    """Uses the cached raw-LM perplexity of every document (computed once by
    `--compute-perplexity`). The direction is learned on the training split so
    an inverted signal cannot masquerade as a detector.

    The cache is keyed by ``doc_id``, not by row position, so it survives a
    dataset rebuild (rows are added, order changes) and can be extended
    incrementally: documents with no cached value are scored as "no signal"
    (0.5) rather than silently taking a neighbouring row's perplexity.
    """

    def __init__(self, key: str, seed: int):
        super().__init__(key, seed)
        self.estimator = LogisticRegression(max_iter=2000, class_weight="balanced")
        self._values_cache: np.ndarray | None = None
        self._coverage: dict[str, float] = {}

    def _values(self, bundle: Bundle) -> np.ndarray:
        if self._values_cache is None or len(self._values_cache) != len(bundle.df):
            vector, coverage = perplexity_vector(bundle.df)
            self._values_cache = vector
            self._coverage = coverage
        return self._values_cache

    def fit(self, bundle: Bundle, train_idx: np.ndarray):
        y = (bundle.df.iloc[train_idx]["label"].to_numpy() != ds.HUMAN).astype(int)
        x = np.log1p(self._values(bundle)[train_idx]).reshape(-1, 1)
        keep = np.isfinite(x[:, 0])
        self.estimator.fit(x[keep], y[keep])
        return self

    def score(self, bundle: Bundle, idx: np.ndarray) -> np.ndarray:
        x = np.log1p(self._values(bundle)[idx]).reshape(-1, 1)
        out = np.full(len(idx), 0.5)
        keep = np.isfinite(x[:, 0])
        if keep.any():
            out[keep] = self.estimator.predict_proba(x[keep])[:, 1]
        return out


def make_model(key: str, seed: int) -> Scorer:
    if key == "tfidf_svm":
        return TfidfScorer(key, seed, LinearSVC(C=1.0, class_weight="balanced", random_state=seed))
    if key == "tfidf_logreg":
        return TfidfScorer(
            key, seed, LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        )
    if key == "tfidf_style_logreg":
        return TfidfStyleScorer(
            key,
            seed,
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
    if key == "tfidf_nb":
        return TfidfScorer(key, seed, MultinomialNB(alpha=0.1))
    if key == "style_rf":
        return StyleScorer(
            key,
            seed,
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
            ),
        )
    if key == "tfidf_style_rf":
        return TfidfStyleScorer(
            key,
            seed,
            RandomForestClassifier(
                n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
            ),
        )
    if key == "length_only":
        return LengthScorer(key, seed)
    if key == "perplexity_only":
        return PerplexityScorer(key, seed)
    raise ValueError(f"unknown model {key!r}")


# ---------------------------------------------------------------------------
# training + evaluation of one model on one condition
# ---------------------------------------------------------------------------


def train_and_score(
    key: str,
    df: pd.DataFrame,
    bundle: Bundle,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None,
    score_sets: dict[str, np.ndarray],
    seed: int,
) -> dict:
    """Fit one model, calibrate on validation, score every requested set."""
    started = time.time()
    scorer = make_model(key, seed).fit(bundle, train_idx)

    val_scores = scorer.score(bundle, val_idx) if val_idx is not None and len(val_idx) else None
    y_val = (
        (df.iloc[val_idx]["label"].to_numpy() != ds.HUMAN).astype(int)
        if val_idx is not None and len(val_idx)
        else None
    )

    calibrator, calibrator_name, calibration_report = (None, "none", {})
    if val_scores is not None:
        calibrator, calibrator_name, calibration_report = calibration.fit_best_calibrator(
            val_scores, y_val
        )

    predictions: dict[str, dict] = {}
    for name, idx in score_sets.items():
        if len(idx) == 0:
            continue
        raw = scorer.score(bundle, idx)
        calibrated = calibrator.transform(raw) if calibrator is not None else raw
        y = (df.iloc[idx]["label"].to_numpy() != ds.HUMAN).astype(int)
        predictions[name] = {"idx": idx, "y": y, "raw": raw, "calibrated": calibrated}

    result = {
        "model": key,
        "seed": seed,
        "n_train": int(len(train_idx)),
        "train_seconds": round(time.time() - started, 1),
        "calibrator": calibrator_name,
        "calibration_validation": {k: round(v, 6) for k, v in calibration_report.items() if isinstance(v, float)},
        "predictions": predictions,
    }
    return result


def summarize_predictions(
    df: pd.DataFrame,
    result: dict,
    thresholds_from: str | None = "val",
    primary_split: str | None = None,
    n_boot: int = 1000,
) -> dict:
    """Full metric suite + CIs + decision policy for each scored set."""
    out: dict[str, dict] = {"model": result["model"], "seed": result["seed"], "splits": {}}
    # thresholds come from the validation split when it was scored
    policy: calibration.TwoStagePolicy | None = None
    if thresholds_from and thresholds_from in result["predictions"]:
        val = result["predictions"][thresholds_from]
        policy = calibration.TwoStagePolicy().fit(val["y"], val["calibrated"])
        out["policy_validation"] = policy.val_report_

    for name, payload in result["predictions"].items():
        if name == thresholds_from:
            continue
        metrics = em.binary_metrics(payload["y"], payload["calibrated"])
        raw_metrics = em.binary_metrics(payload["y"], payload["raw"])
        record = metrics.to_dict()
        record["raw_uncalibrated"] = {
            k: v for k, v in raw_metrics.to_dict()["metrics"].items() if k in
            ("ece_width", "ece_mass", "brier", "roc_auc")
        }
        record["ci95"] = em.bootstrap_ci(payload["y"], payload["calibrated"], n_boot=n_boot)
        if policy is not None:
            record["policy_test"] = policy.report(payload["y"], payload["calibrated"])
        if primary_split is not None and name == primary_split:
            record["domain_breakdown"] = _breakdown(
                df.iloc[payload["idx"]], payload["y"], payload["calibrated"]
            )
        out["splits"][name] = record
    return out


def _breakdown(subset: pd.DataFrame, y: np.ndarray, scores: np.ndarray, min_n: int = 15) -> dict:
    """Per-group FPR / recall, for error analysis by domain, generator, style,
    formality and length."""
    out: dict[str, dict] = {}
    for column in ("domain", "prompt_style", "formality", "length_bucket"):
        if column not in subset:
            continue
        for value, group in subset.groupby(column):
            if len(group) < min_n:
                continue
            positions = subset.index.get_indexer(group.index)
            y_g, s_g = y[positions], scores[positions]
            fp = ((s_g >= 0.5) & (y_g == 0)).sum() / max((y_g == 0).sum(), 1)
            rec = ((s_g >= 0.5) & (y_g == 1)).sum() / max((y_g == 1).sum(), 1)
            out[f"{column}={value}"] = {
                "n": int(len(group)),
                "false_positive_rate": round(float(fp), 4),
                "recall": round(float(rec), 4),
            }
    if "generator" in subset:
        ai_groups = subset[subset["label"] != ds.HUMAN]
        if len(ai_groups) >= min_n:
            by_generator = []
            for value, group in ai_groups.groupby("generator"):
                if len(group) < 5:
                    continue
                positions = subset.index.get_indexer(group.index)
                s_g = scores[positions]
                by_generator.append(
                    {"generator": value, "n": int(len(group)), "recall": round(float((s_g >= 0.5).mean()), 4)}
                )
            if by_generator:
                out["generator_recall"] = sorted(
                    by_generator, key=lambda item: item["recall"]
                )
    return out


# ---------------------------------------------------------------------------
# perplexity cache
# ---------------------------------------------------------------------------


def _load_perplexity_cache() -> dict:
    if PERPLEXITY.exists():
        cache = joblib.load(PERPLEXITY)
        if "perplexity_by_doc_id" in cache:
            return cache
        # legacy positional cache: kept only so an old file cannot be silently
        # misinterpreted as doc_id-keyed data
        print("ignoring positional perplexity cache; rebuilding keyed by doc_id", flush=True)
    return {"perplexity_by_doc_id": {}, "seq_len": 64}


def perplexity_vector(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Align the doc_id-keyed cache to this frame; NaN where a document has no
    cached value yet."""
    cache = _load_perplexity_cache()
    lookup = cache["perplexity_by_doc_id"]
    values = np.array([lookup.get(doc_id, np.nan) for doc_id in df["doc_id"]], dtype=float)
    coverage = {
        "rows": int(len(df)),
        "cached": int(np.isfinite(values).sum()),
        "missing": int((~np.isfinite(values)).sum()),
    }
    return values, coverage


def compute_perplexity(limit: int = 0) -> None:
    """Incrementally fill the doc_id-keyed perplexity cache.

    Only documents that are not in the cache are scored, so this is safe to
    re-run after any dataset rebuild (and cheap when only a few thousand
    machine-revised rows were added).
    """
    from aitextd.perplexity import gpt2_perplexity_batch

    df = pd.read_parquet(DATASET)
    cache = _load_perplexity_cache()
    lookup: dict[str, float] = cache["perplexity_by_doc_id"]
    pending = [
        (doc_id, text)
        for doc_id, text in zip(df["doc_id"], df["text"], strict=True)
        if doc_id not in lookup
    ]
    if limit:
        pending = pending[:limit]
    print(f"perplexity: {len(lookup)} cached, {len(pending)} to score", flush=True)
    started = time.time()
    step = 500
    for start in range(0, len(pending), step):
        chunk = pending[start : start + step]
        values = gpt2_perplexity_batch([text for _, text in chunk])
        for (doc_id, _), value in zip(chunk, values, strict=True):
            lookup[doc_id] = float(value)
        if (start // step) % 5 == 0:
            print(
                f"  {min(start + step, len(pending))}/{len(pending)} documents "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )
    cache["perplexity_by_doc_id"] = lookup
    cache["seq_len"] = 64
    cache["computed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    joblib.dump(cache, PERPLEXITY)
    print(f"cached perplexity for {len(lookup)} documents -> {PERPLEXITY}")


# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------


def run_condition(
    name: str,
    df: pd.DataFrame,
    style: pd.DataFrame,
    splits: dict[str, np.ndarray],
    models: list[str],
    seeds: list[int],
    extra_score_sets: dict[str, np.ndarray] | None = None,
    seeds_for_model: dict[str, list[int]] | None = None,
) -> dict:
    """Train `models` on {name}__train, evaluate on {name}__test (+ extras)."""
    train_idx = splits[f"{name}__train"]
    val_idx = splits.get(f"{name}__val", splits.get("in_domain_group__val"))
    primary = f"{name}__test"
    score_sets = {primary: splits[primary]}
    score_sets["val"] = val_idx
    if extra_score_sets:
        score_sets.update(extra_score_sets)

    condition: dict = {"condition": name, "models": {}, "train_rows": int(len(train_idx))}
    # one vectorizer per condition: the vocabulary of a split is part of its
    # definition, so it is fitted once and shared by every model and seed
    bundle = Bundle(df, style, train_idx)
    for model_key in models:
        model_seeds = (seeds_for_model or {}).get(model_key, seeds)
        per_seed = []
        for seed in model_seeds:
            result = train_and_score(model_key, df, bundle, train_idx, val_idx, score_sets, seed)
            per_seed.append(
                summarize_predictions(df, result, thresholds_from="val", primary_split=primary)
            )
            _save_predictions(result, condition=name)
        condition["models"][model_key] = per_seed
        first = per_seed[0]["splits"].get(primary, {})
        headline = first.get("metrics", {}).get("macro_f1")
        print(
            f"  [{name}] {model_key}: macro-F1={headline} "
            f"({len(model_seeds)} seed(s))",
            flush=True,
        )
    return condition


def _save_predictions(result: dict, condition: str) -> None:
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    for split_name, payload in result["predictions"].items():
        path = PREDICTIONS / f"{result['model']}__seed{result['seed']}__{condition}__{split_name}.npz"
        np.savez_compressed(
            path,
            idx=payload["idx"],
            y=payload["y"],
            raw=payload["raw"],
            calibrated=payload["calibrated"],
        )


def run_attack_condition(
    df: pd.DataFrame, style: pd.DataFrame, splits: dict[str, np.ndarray], models: list[str], seeds: list[int]
) -> dict:
    """Clean vs paraphrased comparison on the reserved attack rows. Each model
    is trained on the main in-domain training split (attack data is never used
    for training); the attack texts are appended as new scoring positions and
    the recall drop relative to the clean source documents is reported with a
    *paired* bootstrap CI over documents."""
    if not ATTACK.exists():
        return {"condition": "paraphrase_attack", "skipped": "attack parquet missing"}
    attack = pd.read_parquet(ATTACK)
    train_idx = splits["in_domain_group__train"]
    val_idx = splits["in_domain_group__val"]

    source_ids = attack["extra"].map(lambda e: json.loads(e)["source_doc_id"]).tolist()
    id_to_pos = {doc_id: pos for pos, doc_id in enumerate(df["doc_id"])}
    missing = [i for i in source_ids if i not in id_to_pos]
    clean_idx = np.array([id_to_pos[i] for i in source_ids if i in id_to_pos], dtype=int)

    record: dict = {
        "condition": "paraphrase_attack",
        "n_attacked": int(len(attack)),
        "n_clean_sources": int(len(clean_idx)),
        "n_unmatched_sources": int(len(missing)),
        "methods": attack["revision_type"].value_counts().to_dict(),
        "models": {},
    }
    attack_bundle = Bundle(df, style, train_idx)
    for model_key in models:
        per_seed = []
        for seed in seeds:
            scorer = make_model(model_key, seed).fit(attack_bundle, train_idx)
            y_val = (df.iloc[val_idx]["label"].to_numpy() != ds.HUMAN).astype(int)
            calibrator, calibrator_name, _ = calibration.fit_best_calibrator(
                scorer.score(attack_bundle, val_idx), y_val
            )

            clean_scores = calibrator.transform(scorer.score(attack_bundle, clean_idx))
            clean_detected = (clean_scores >= 0.5).astype(float)
            clean_recall = float(clean_detected.mean())

            bundle = attack_bundle.with_texts(attack["text"].tolist())
            attack_scores = calibrator.transform(scorer.score(bundle, bundle.new_positions))
            attack_detected = (attack_scores >= 0.5).astype(float)

            by_method = {}
            method_series = attack["revision_type"].to_numpy()
            for method in sorted(set(method_series)):
                mask = method_series == method
                scores = attack_scores[mask]
                detected = attack_detected[mask]
                by_method[method] = {
                    "n": int(mask.sum()),
                    "mean_ai_probability": round(float(scores.mean()), 4),
                    "recall_at_0.5": round(float(detected.mean()), 4),
                    "recall_at_0.5_ci95": em.bootstrap_mean_ci(detected, seed=seed),
                    "recall_drop_vs_clean": round(clean_recall - float(detected.mean()), 4),
                }
            paired = em.paired_bootstrap_diff(clean_detected, attack_detected, seed=seed)
            per_seed.append(
                {
                    "model": model_key,
                    "seed": seed,
                    "calibrator": calibrator_name,
                    "clean_recall_at_0.5": round(clean_recall, 4),
                    "clean_recall_ci95": em.bootstrap_mean_ci(clean_detected, seed=seed),
                    "attack_mean_probability": round(float(attack_scores.mean()), 4),
                    "attack_recall_at_0.5": round(float(attack_detected.mean()), 4),
                    "attack_recall_ci95": em.bootstrap_mean_ci(attack_detected, seed=seed),
                    "recall_drop_paired_ci95": paired,
                    "by_method": by_method,
                }
            )
        record["models"][model_key] = per_seed
        first = per_seed[0]
        print(
            f"  [attack] {model_key}: clean recall {first['clean_recall_at_0.5']} -> "
            f"attacked {first['attack_recall_at_0.5']} "
            f"(paired drop {first['recall_drop_paired_ci95']['mean_difference']} "
            f"[{first['recall_drop_paired_ci95']['lo']}, {first['recall_drop_paired_ci95']['hi']}])",
            flush=True,
        )
    return record


def run_three_class(
    df: pd.DataFrame, style: pd.DataFrame, splits: dict[str, np.ndarray], seeds: list[int]
) -> dict:
    """Three-class formulation (human / ai / ai_revised) plus its binary
    collapse, trained on the group-aware in-domain split, which carries all
    three classes (revised rows inherit their source document's partition).

    Two feature sets are reported: TF-IDF alone and TF-IDF + style features.
    Thresholds do not apply to the argmax decision, but the binary collapse
    (human vs everything-machine) is reported with its full metric suite so it
    can be compared with the binary models.
    """
    revised = df[df["label"] == ds.AI_REVISED]
    if revised.empty:
        return {"condition": "three_class", "skipped": "no ai_revised rows yet"}
    test_idx = splits["in_domain_group__test"]
    train_idx = splits["in_domain_group__train"]
    classes = [ds.HUMAN, ds.AI, ds.AI_REVISED]
    label_to_int = {name: i for i, name in enumerate(classes)}

    bundle = Bundle(df, style, train_idx)
    matrices = {
        "tfidf_logreg_3class": bundle.tfidf(),
        "tfidf_style_logreg_3class": bundle.tfidf_style(),
    }
    y_train = np.array([label_to_int[l] for l in df.iloc[train_idx]["label"]])
    y_test = np.array([label_to_int[l] for l in df.iloc[test_idx]["label"]])
    results = []
    for matrix_name, matrix in matrices.items():
        for seed in seeds:
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
            clf.fit(matrix[train_idx], y_train)
            proba = clf.predict_proba(matrix[test_idx])
            multi = em.multiclass_metrics(y_test, proba, classes)
            binary_y = (y_test != label_to_int[ds.HUMAN]).astype(int)
            binary_p = 1 - proba[:, label_to_int[ds.HUMAN]]
            binary = em.binary_metrics(binary_y, binary_p)
            results.append(
                {
                    "model": matrix_name,
                    "seed": seed,
                    "multiclass": multi.to_dict(),
                    "binary_collapse": binary.to_dict(),
                    "ci95_binary_collapse": em.bootstrap_ci(binary_y, binary_p),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "train_class_counts": df.iloc[train_idx]["label"].value_counts().to_dict(),
                    "test_class_counts": df.iloc[test_idx]["label"].value_counts().to_dict(),
                }
            )
            print(f"  [three_class] {matrix_name} seed {seed}: "
                  f"macro-F1={multi.metrics['macro_f1']:.4f} "
                  f"revised-recall={multi.metrics['recall_' + ds.AI_REVISED]:.4f} "
                  f"binary macro-F1={binary.metrics['macro_f1']:.4f}", flush=True)
    return {"condition": "three_class", "results": results}


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

FLAT_METRICS = (
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

# Feature-heavy models: reported with fewer seeds on purpose (cost), and the
# seed count is written into every table so the difference is visible.
HEAVY_MODELS = ("tfidf_style_rf",)


def _mean_std(records: list[dict], key: str, source: str = "metrics") -> tuple[float, float]:
    values = []
    for record in records:
        block = record.get(source) or {}
        value = block.get(key)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def _fmt(mean: float, std: float | None = None) -> str:
    if mean is None or not np.isfinite(mean):
        return "—"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}" if np.isfinite(std) else f"{mean:.4f}"


def _condition(results: dict, name: str) -> dict | None:
    for condition in results["conditions"]:
        if condition.get("condition") == name:
            return condition
    return None


def _split_records(condition: dict | None, split_name: str) -> dict[str, list[dict]]:
    """model key -> per-seed records for one split of one condition."""
    out: dict[str, list[dict]] = {}
    if not condition:
        return out
    for model_key, per_seed in condition.get("models", {}).items():
        records = [
            seed_result["splits"][split_name]
            for seed_result in per_seed
            if split_name in seed_result.get("splits", {})
        ]
        if records:
            out[model_key] = records
    return out


def table_in_domain(results: dict) -> str:
    records_by_model = _split_records(_condition(results, "in_domain_group"), "in_domain_group__test")
    lines = [
        "| model | seeds | n | macro-F1 (mean±sd) | recall AI | FPR@TPR95 | ROC-AUC | PR-AUC | ECE | auto-flagged AI | humans auto-flagged | deferred |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model_key, records in records_by_model.items():
        row = [model_key, str(len(records)), str(records[0]["n"])]
        for key in ("macro_f1", "recall_ai", "fpr_at_tpr95", "roc_auc", "pr_auc", "ece_width"):
            mean, std = _mean_std(records, key)
            row.append(_fmt(mean, std) if key == "macro_f1" else _fmt(mean))
        for key in (
            "ai_called_ai_automatically",
            "human_called_ai_automatically",
            "deferred_share",
        ):
            mean, _ = _mean_std(records, key, source="policy_test")
            row.append(_fmt(mean))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def table_conditions(results: dict) -> str:
    lines = [
        "| condition | model | seeds | n train | n test | macro-F1 (mean±sd) | recall AI | FPR@TPR95 | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for condition in results["conditions"]:
        name = condition.get("condition", "")
        if name in ("in_domain_group", "paraphrase_attack", "three_class") or condition.get("skipped"):
            continue
        primary = f"{name}__test"
        for model_key, records in _split_records(condition, primary).items():
            row = [
                name,
                model_key,
                str(len(records)),
                str(condition.get("train_rows", "—")),
                str(records[0]["n"]),
            ]
            for key in ("macro_f1", "recall_ai", "fpr_at_tpr95", "roc_auc", "pr_auc"):
                mean, std = _mean_std(records, key)
                row.append(_fmt(mean, std) if key == "macro_f1" else _fmt(mean))
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def table_length_and_v1(results: dict) -> str:
    lines = [
        "| subset | model | n | macro-F1 | recall AI | FPR@TPR95 | ROC-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    condition = _condition(results, "in_domain_group")
    for split_name in (
        "length__short__test",
        "length__long__test",
        "v1_test",
        "revision_sensitivity__all",
    ):
        for model_key, records in _split_records(condition, split_name).items():
            row = [split_name, model_key, str(records[0]["n"])]
            for key in ("macro_f1", "recall_ai", "fpr_at_tpr95", "roc_auc"):
                mean, _ = _mean_std(records, key)
                row.append(_fmt(mean))
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def table_domain_breakdown(results: dict, model_key: str = "tfidf_svm") -> str:
    condition = _condition(results, "in_domain_group")
    if not condition or model_key not in condition.get("models", {}):
        return ""
    records = _split_records(condition, "in_domain_group__test").get(model_key, [])
    lines = ["| group | n | human FPR | AI recall |", "|---|---|---|---|"]
    aggregated: dict[str, list[dict]] = {}
    for record in records:
        for group, values in (record.get("domain_breakdown") or {}).items():
            if group == "generator_recall":
                continue
            aggregated.setdefault(group, []).append(values)
    for group, values in aggregated.items():
        n = int(values[0].get("n", 0))
        fpr = np.mean([v["false_positive_rate"] for v in values])
        recall = np.mean([v["recall"] for v in values])
        lines.append(f"| {group} | {n} | {_fmt(float(fpr))} | {_fmt(float(recall))} |")
    return "\n".join(lines) + "\n"


def table_attack(results: dict) -> str:
    condition = _condition(results, "paraphrase_attack")
    if not condition or condition.get("skipped"):
        return ""
    lines = [
        "| model | clean recall | attacked recall | paired drop (95% CI) | P(drop>0) |",
        "|---|---|---|---|---|",
    ]
    for model_key, per_seed in condition["models"].items():
        clean = np.mean([s["clean_recall_at_0.5"] for s in per_seed])
        attacked = np.mean([s["attack_recall_at_0.5"] for s in per_seed])
        drop = np.mean([s["recall_drop_paired_ci95"]["mean_difference"] for s in per_seed])
        lo = np.mean([s["recall_drop_paired_ci95"]["lo"] for s in per_seed])
        hi = np.mean([s["recall_drop_paired_ci95"]["hi"] for s in per_seed])
        p_pos = np.mean([s["recall_drop_paired_ci95"]["p_difference_positive"] for s in per_seed])
        lines.append(
            f"| {model_key} | {clean:.4f} | {attacked:.4f} | {drop:.4f} [{lo:.4f}, {hi:.4f}] | {p_pos:.3f} |"
        )
    return "\n".join(lines) + "\n"


def table_attack_methods(results: dict) -> str:
    condition = _condition(results, "paraphrase_attack")
    if not condition or condition.get("skipped"):
        return ""
    methods = sorted(
        {m for per_seed in condition["models"].values() for s in per_seed for m in s["by_method"]}
    )
    lines = [
        "| model | method | n | mean AI prob. | recall | recall 95% CI | drop vs clean |",
        "|---|---|---|---|---|---|---|",
    ]
    for model_key, per_seed in condition["models"].items():
        for method in methods:
            rows = [s["by_method"][method] for s in per_seed if method in s["by_method"]]
            if not rows:
                continue
            prob = np.mean([r["mean_ai_probability"] for r in rows])
            recall = np.mean([r["recall_at_0.5"] for r in rows])
            lo = np.mean([r["recall_at_0.5_ci95"]["lo"] for r in rows])
            hi = np.mean([r["recall_at_0.5_ci95"]["hi"] for r in rows])
            drop = np.mean([r["recall_drop_vs_clean"] for r in rows])
            lines.append(
                f"| {model_key} | {method} | {rows[0]['n']} | {prob:.4f} | {recall:.4f} "
                f"[{lo:.4f}, {hi:.4f}] | {drop:.4f} |"
            )
    return "\n".join(lines) + "\n"


def table_three_class(results: dict) -> str:
    condition = _condition(results, "three_class")
    if not condition or condition.get("skipped"):
        return "no ai_revised rows in the dataset yet\n"
    lines = [
        "| model | seeds | n test | accuracy | macro-F1 | P/R/F1 human | P/R/F1 ai | P/R/F1 ai_revised | macro ROC-AUC OvR | binary collapse macro-F1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    by_model: dict[str, list[dict]] = {}
    for record in condition["results"]:
        by_model.setdefault(record["model"], []).append(record)
    for model_key, records in by_model.items():
        multi = [r["multiclass"]["metrics"] for r in records]
        binary = [r["binary_collapse"]["metrics"] for r in records]

        def agg(values: list[float]) -> float:
            values = [v for v in values if v is not None]
            return float(np.mean(values)) if values else float("nan")

        def prf(class_name: str) -> str:
            return (
                f"{agg([m.get(f'precision_{class_name}') for m in multi]):.3f} / "
                f"{agg([m.get(f'recall_{class_name}') for m in multi]):.3f} / "
                f"{agg([m.get(f'f1_{class_name}') for m in multi]):.3f}"
            )

        row = [
            model_key,
            str(len(records)),
            str(records[0]["n_test"]),
            f"{agg([m['accuracy'] for m in multi]):.4f}",
            f"{agg([m['macro_f1'] for m in multi]):.4f}",
            prf(ds.HUMAN),
            prf(ds.AI),
            prf(ds.AI_REVISED),
            f"{agg([m.get('macro_roc_auc_ovr') for m in multi]):.4f}",
            f"{agg([m['macro_f1'] for m in binary]):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def write_tables(results: dict) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    tables = {
        "in_domain.md": table_in_domain(results),
        "conditions.md": table_conditions(results),
        "length_and_v1.md": table_length_and_v1(results),
        "domain_breakdown.md": table_domain_breakdown(results),
        "attack.md": table_attack(results),
        "attack_by_method.md": table_attack_methods(results),
        "three_class.md": table_three_class(results),
    }
    for name, body in tables.items():
        if body:
            (TABLES / name).write_text(body, encoding="utf-8")
    with open(TABLES / "run_metadata.md", "w", encoding="utf-8") as handle:
        handle.write(
            "| field | value |\n|---|---|\n"
            f"| generated_at | {results['generated_at']} |\n"
            f"| dataset_fingerprint | {results['dataset_fingerprint']} |\n"
            f"| rows | {results['rows']} |\n"
            f"| models | {', '.join(results['models'])} |\n"
            f"| seeds | {', '.join(str(s) for s in results['seeds'])} |\n"
            f"| heavy models (fewer seeds) | {', '.join(HEAVY_MODELS)} |\n"
        )
    RESULTS_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8"
    )
    print(f"tables -> {TABLES}")
    print(f"raw results -> {RESULTS_JSON}")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conditions", default="main")
    ap.add_argument("--models", default="core")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--heavy-seeds", type=int, default=1, help="seeds for the costly feature-heavy models")
    ap.add_argument("--compute-perplexity", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="ignore an existing eval_results.json from the same dataset build",
    )
    args = ap.parse_args()

    if args.compute_perplexity:
        compute_perplexity()
        return

    df = pd.read_parquet(DATASET).reset_index(drop=True)
    style = pd.read_parquet(STYLE_FEATURES)
    if "doc_id" in style.columns and not np.array_equal(
        style["doc_id"].to_numpy(), df["doc_id"].to_numpy()
    ):
        raise SystemExit(
            "style features are not row-aligned with dataset_v2.parquet; "
            "rerun training/build_features_v2.py after the dataset was rebuilt"
        )
    splits = joblib.load(SPLITS)
    models = list(CORE_MODELS) if args.models == "core" else [m.strip() for m in args.models.split(",")]
    if "perplexity_only" in models and not PERPLEXITY.exists():
        print("perplexity cache missing -> dropping perplexity_only from this run", flush=True)
        models = [m for m in models if m != "perplexity_only"]
    seeds = [int(s) for s in args.seeds.split(",")]
    conditions = args.conditions.split(",")
    seeds_for_model = {
        key: (seeds[: max(1, args.heavy_seeds)] if key in HEAVY_MODELS else seeds) for key in models
    }

    # Long runs are split across invocations (a full pass is hours). Results are
    # therefore merged into the existing file: a condition that is recomputed
    # replaces its old entry, everything else is kept -- but only when the
    # dataset fingerprint still matches, so numbers from different dataset
    # builds can never be mixed in one table.
    previous: dict | None = None
    if RESULTS_JSON.exists() and not args.fresh:
        try:
            loaded = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            if loaded.get("dataset_fingerprint") == ds.dataset_fingerprint(df):
                previous = loaded
                print(
                    "resuming: previous conditions kept -> "
                    f"{[c.get('condition') for c in loaded.get('conditions', [])]}",
                    flush=True,
                )
            else:
                print(
                    "existing eval_results.json is from a different dataset build; "
                    "starting a fresh file (old file kept as eval_results.previous.json)",
                    flush=True,
                )
                shutil.copy2(RESULTS_JSON, RESULTS_JSON.with_name("eval_results.previous.json"))

    results: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_fingerprint": ds.dataset_fingerprint(df),
        "rows": int(len(df)),
        "models": models,
        "seeds": seeds,
        "conditions": list(previous.get("conditions", [])) if previous else [],
    }

    def _replace_condition(record: dict) -> None:
        name = record.get("condition")
        results["conditions"] = [
            c for c in results["conditions"] if c.get("condition") != name
        ] + [record]

    if "main" in conditions or "all" in conditions:
        print("condition: in_domain_group", flush=True)
        # the in-domain model is also scored on the two length buckets and on
        # the v1-comparable rows, so those views cost no extra training; when
        # machine-revised documents exist they are scored separately as well
        # (recall on revised text at the clean-data operating point)
        extras = {
            name: idx
            for name, idx in splits.items()
            if name.startswith("length__") or name == "v1_test"
        }
        revised_all = np.flatnonzero(df["label"].to_numpy() == ds.AI_REVISED)
        if revised_all.size:
            # every machine-revised row, scored at the clean-data operating
            # point: a *sensitivity* probe (how often revision hides text from
            # the detector), not a held-out generalisation number -- the
            # group-aware three-class split below is the generalisation view
            extras["revision_sensitivity__all"] = revised_all
        _replace_condition(
            run_condition(
                "in_domain_group",
                df,
                style,
                splits,
                models,
                seeds,
                extra_score_sets=extras,
                seeds_for_model=seeds_for_model,
            )
        )
        write_tables(results)
    if "all" in conditions or "attack" in conditions:
        print("condition: paraphrase_attack", flush=True)
        _replace_condition(run_attack_condition(df, style, splits, models, seeds))
        write_tables(results)
    if "all" in conditions or "three_class" in conditions or "human_vs_ai_revised" in conditions:
        # the revision family has two views: a balanced binary condition
        # (human vs machine-revised, tuned on its own validation split) and the
        # three-class model above. "three_class" runs both.
        print("condition: human_vs_ai_revised", flush=True)
        _replace_condition(
            run_condition(
                "human_vs_ai_revised",
                df,
                style,
                splits,
                models,
                seeds,
                seeds_for_model=seeds_for_model,
            )
        )
        write_tables(results)
        print("condition: three_class", flush=True)
        _replace_condition(run_three_class(df, style, splits, seeds))
        write_tables(results)
    for prefix in ("cross_generator", "cross_domain", "cross_style", "few_shot"):
        if "all" not in conditions and prefix not in conditions:
            continue
        names = sorted(
            name[: -len("__train")] for name in splits if name.startswith(prefix) and name.endswith("__train")
        )
        for name in names:
            print(f"condition: {name}", flush=True)
            _replace_condition(
                run_condition(
                    name,
                    df,
                    style,
                    splits,
                    models,
                    seeds[:1] if prefix == "few_shot" else seeds,
                    seeds_for_model=seeds_for_model,
                )
            )
            write_tables(results)  # incremental, so a long run is never lost

    write_tables(results)


if __name__ == "__main__":
    main()
