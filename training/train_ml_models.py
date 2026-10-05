"""Train classic ML models on the TF-IDF + style-feature matrix and merge
their metadata into models.json.

Expects data/tfidf_vectorizer_X.joblib in the current layout (TF-IDF columns
followed by the style columns). Run from the project root:

    python training/build_style_features.py
    python training/train_ml_models.py
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from aitextd import paths
from aitextd.features import STYLE_FEATURE_NAMES
from split import RANDOM_SEED, TEST_SIZE, load_tfidf_split

MODELS_JSON_PATH = paths.MODELS_JSON
# Trees don't scale to the full TF-IDF space, so they train on a chi2-selected
# subset of the TF-IDF columns; the style columns always pass through (they
# must not compete for a chi2 slot). The selector is bundled into the saved
# pipeline so the server keeps feeding it the full vector at inference time.
TREE_FEATURE_COUNT = 30_000
N_STYLE = len(STYLE_FEATURE_NAMES)


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "f1Score": round(float(f1_score(y_test, y_pred)), 4),
        },
        "confusionMatrix": {
            "trueNegative": int(tn),
            "falsePositive": int(fp),
            "falseNegative": int(fn),
            "truePositive": int(tp),
        },
    }


def training_info(feature_count, train_idx, test_idx):
    return {
        "totalSamples": len(train_idx) + len(test_idx),
        "trainSamples": len(train_idx),
        "testSamples": len(test_idx),
        "featureCount": feature_count,
        "testSplitRatio": TEST_SIZE,
        "randomSeed": RANDOM_SEED,
        "dataset": "persian_ai_generated_final.csv + blogs.csv",
    }


def commit(model, entry):
    """Dump one model and pin its metadata immediately, so an interrupted run
    never leaves a model file paired with a stale models.json entry (the
    serving shim slices style columns per that entry)."""
    joblib.dump(model, paths.model_path(entry["filename"]))
    with open(MODELS_JSON_PATH, encoding="utf-8") as f:
        existing = json.load(f)
    prior = next((m for m in existing if m["id"] == entry["id"]), None)
    # Hand-tuned ensemble weights survive a retrain; fresh models default to 1.0.
    entry.setdefault("ensembleWeight", (prior or {}).get("ensembleWeight", 1.0))
    existing = [m for m in existing if m["id"] != entry["id"]]
    existing.append(entry)
    with open(MODELS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"  saved {entry['filename']} + models.json entry '{entry['id']}'")


def tree_pipeline(clf, tfidf_dim):
    return Pipeline(
        [
            (
                "select",
                ColumnTransformer(
                    [
                        (
                            "tfidf_select",
                            SelectKBest(chi2, k=TREE_FEATURE_COUNT),
                            slice(0, tfidf_dim),
                        ),
                        ("style", "passthrough", slice(tfidf_dim, None)),
                    ],
                    sparse_threshold=1.0,
                ),
            ),
            ("clf", clf),
        ]
    )


def main():
    X_train, X_test, y_train, y_test, _, train_idx, test_idx = load_tfidf_split()
    full_feature_count = X_train.shape[1]
    tfidf_dim = full_feature_count - N_STYLE
    vectorizer = joblib.load(paths.TFIDF_VECTORIZER_JOBLIB)
    expected_tfidf_dim = len(vectorizer.get_feature_names_out())
    assert tfidf_dim == expected_tfidf_dim, (
        f"data/tfidf_vectorizer_X.joblib has {tfidf_dim} TF-IDF columns, expected "
        f"{expected_tfidf_dim}; run training/build_style_features.py first"
    )

    def entry_fields(result, feature_count, n_style=N_STYLE):
        return {
            "type": "ml",
            "version": "4.0",
            "runtime": "sklearn-tfidf",
            "vectorizer": "vectorizer.joblib",
            "styleFeatureCount": n_style,
            **result,
            "training": training_info(feature_count, train_idx, test_idx),
        }

    # --- Linear SVM on TF-IDF ------------------------------------------------
    # Linear models train on the TF-IDF slice only: appending the style
    # columns to the joint vector measurably hurts them (see
    # training/eval_style_ablation.py). Trees and NB keep the style columns.
    print("Training Linear SVM...")
    t0 = time.time()
    svm = LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_SEED)
    svm.fit(X_train[:, :tfidf_dim], y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    commit(
        svm,
        {
            "id": "svm",
            "name": "SVM",
            "nameFa": "ماشین بردار پشتیبان",
            "description": "Linear Support Vector Machine (LinearSVC) trained on TF-IDF features of Persian text to distinguish AI-generated content from human writing.",
            "descriptionFa": "ماشین بردار پشتیبان خطی (LinearSVC) که روی ویژگی‌های TF-IDF متن فارسی آموزش دیده تا متن تولیدشده توسط هوش مصنوعی را از نوشته انسان جدا کند.",
            "algorithm": "LinearSVC",
            "filename": "svModel.joblib",
            **entry_fields(evaluate(svm, X_test[:, :tfidf_dim], y_test), tfidf_dim, n_style=0),
        },
    )

    # --- Logistic Regression on TF-IDF --------------------------------------
    print("Training Logistic Regression...")
    t0 = time.time()
    logreg = LogisticRegression(
        solver="liblinear", dual=True, C=32.0, max_iter=5000, random_state=RANDOM_SEED
    )
    logreg.fit(X_train[:, :tfidf_dim], y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    commit(
        logreg,
        {
            "id": "logreg",
            "name": "Logistic Regression",
            "nameFa": "رگرسیون لجستیک",
            "description": "Linear Logistic Regression trained on TF-IDF features of Persian text to distinguish AI-generated content from human writing.",
            "descriptionFa": "رگرسیون لجستیک خطی که روی ویژگی‌های TF-IDF متن فارسی آموزش دیده تا متن تولیدشده توسط هوش مصنوعی را از نوشته انسان جدا کند.",
            "algorithm": "LogisticRegression",
            "filename": "logregModel.joblib",
            **entry_fields(evaluate(logreg, X_test[:, :tfidf_dim], y_test), tfidf_dim, n_style=0),
        },
    )

    # --- Multinomial Naive Bayes on full TF-IDF + style space ---------------
    print("Training Multinomial Naive Bayes...")
    t0 = time.time()
    nb = MultinomialNB(alpha=0.01)
    nb.fit(X_train, y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    commit(
        nb,
        {
            "id": "naive_bayes",
            "name": "Naive Bayes",
            "nameFa": "بیز ساده",
            "description": "Multinomial Naive Bayes trained on TF-IDF and stylometric text-style features of Persian text to distinguish AI-generated content from human writing.",
            "descriptionFa": "بیز ساده چندجمله‌ای که روی ویژگی‌های TF-IDF و سبک‌شناسی متن فارسی آموزش دیده تا متن تولیدشده توسط هوش مصنوعی را از نوشته انسان جدا کند.",
            "algorithm": "MultinomialNB",
            "filename": "nbModel.joblib",
            **entry_fields(evaluate(nb, X_test, y_test), full_feature_count),
        },
    )

    # --- Random Forest (chi2 TF-IDF subset + style passthrough) -------------
    print("Training Random Forest...")
    t0 = time.time()
    rf_pipeline = tree_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        ),
        tfidf_dim,
    )
    rf_pipeline.fit(X_train, y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    commit(
        rf_pipeline,
        {
            "id": "random_forest",
            "name": "Random Forest",
            "nameFa": "جنگل تصادفی",
            "description": "Random Forest ensemble trained on the top 30,000 chi2-selected TF-IDF features plus the stylometric text-style features of Persian text to distinguish AI-generated content from human writing.",
            "descriptionFa": "مدل جنگل تصادفی که روی ۳۰٬۰۰۰ ویژگی برتر TF-IDF (انتخاب‌شده با آزمون chi2) و ویژگی‌های سبک‌شناسی متن فارسی آموزش دیده تا متن تولیدشده توسط هوش مصنوعی را از نوشته انسان جدا کند.",
            "algorithm": "RandomForestClassifier",
            "filename": "rfModel.joblib",
            **entry_fields(evaluate(rf_pipeline, X_test, y_test), TREE_FEATURE_COUNT + N_STYLE),
        },
    )

    # --- Gradient Boosting (chi2 TF-IDF subset + style passthrough) ----------
    print("Training Gradient Boosting...")
    t0 = time.time()
    gb_pipeline = tree_pipeline(
        GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
        ),
        tfidf_dim,
    )
    gb_pipeline.fit(X_train, y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    commit(
        gb_pipeline,
        {
            "id": "gradient_boosting",
            "name": "Gradient Boosting",
            "nameFa": "گرادیان بوستینگ",
            "description": "Gradient Boosting ensemble trained on the top 30,000 chi2-selected TF-IDF features plus the stylometric text-style features of Persian text to distinguish AI-generated content from human writing.",
            "descriptionFa": "مدل گرادیان بوستینگ که روی ۳۰٬۰۰۰ ویژگی برتر TF-IDF (انتخاب‌شده با آزمون chi2) و ویژگی‌های سبک‌شناسی متن فارسی آموزش دیده تا متن تولیدشده توسط هوش مصنوعی را از نوشته انسان جدا کند.",
            "algorithm": "GradientBoostingClassifier",
            "filename": "gbModel.joblib",
            **entry_fields(evaluate(gb_pipeline, X_test, y_test), TREE_FEATURE_COUNT + N_STYLE),
        },
    )

    print("\nSummary:")
    with open(MODELS_JSON_PATH, encoding="utf-8") as f:
        for entry in json.load(f):
            if entry.get("type") == "ml":
                print(f"  {entry['id']}: {entry['metrics']}")


if __name__ == "__main__":
    main()
