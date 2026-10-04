"""Train additional classic ML models on the existing TF-IDF features and
append their metadata to models.json.

Run from the project root: python training/train_ml_models.py
"""

import json
import time

import joblib
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

from split import RANDOM_SEED, TEST_SIZE, load_tfidf_split

MODELS_JSON_PATH = "models.json"
# Trees don't scale well to the full 9.3M-dim TF-IDF space, so they train on
# a chi2-selected subset. The selector is bundled into the saved pipeline so
# the server keeps feeding it the full TF-IDF vector at inference time.
TREE_FEATURE_COUNT = 20_000


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


def main():
    X_train, X_test, y_train, y_test, df, train_idx, test_idx = load_tfidf_split()
    full_feature_count = X_train.shape[1]

    new_entries = []

    # --- Logistic Regression on full TF-IDF space -------------------------
    print("Training Logistic Regression...")
    t0 = time.time()
    logreg = LogisticRegression(
        solver="liblinear", dual=True, C=1.0, max_iter=2000, random_state=RANDOM_SEED
    )
    logreg.fit(X_train, y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    joblib.dump(logreg, "logregModel.joblib")
    result = evaluate(logreg, X_test, y_test)
    new_entries.append(
        {
            "id": "logreg",
            "name": "Logistic Regression",
            "nameFa": "رگرسیون لجستیک",
            "description": "Linear Logistic Regression trained on TF-IDF features of Persian text to distinguish AI-generated content from human writing.",
            "descriptionFa": "رگرسیون لجستیک خطی که روی ویژگی‌های TF-IDF متن فارسی آموزش دیده تا متن تولیدشده توسط هوش مصنوعی را از نوشته انسان جدا کند.",
            "algorithm": "LogisticRegression",
            "type": "ml",
            "version": "1.0",
            "filename": "logregModel.joblib",
            "runtime": "sklearn-tfidf",
            "vectorizer": "tfidf_vectorizer.joblib",
            **result,
            "training": training_info(full_feature_count, train_idx, test_idx),
        }
    )

    # --- Multinomial Naive Bayes on full TF-IDF space ----------------------
    print("Training Multinomial Naive Bayes...")
    t0 = time.time()
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    joblib.dump(nb, "nbModel.joblib")
    result = evaluate(nb, X_test, y_test)
    new_entries.append(
        {
            "id": "naive_bayes",
            "name": "Naive Bayes",
            "nameFa": "بیز ساده",
            "description": "Multinomial Naive Bayes trained on TF-IDF features of Persian text to distinguish AI-generated content from human writing.",
            "descriptionFa": "بیز ساده چندجمله‌ای که روی ویژگی‌های TF-IDF متن فارسی آموزش دیده تا متن تولیدشده توسط هوش مصنوعی را از نوشته انسان جدا کند.",
            "algorithm": "MultinomialNB",
            "type": "ml",
            "version": "1.0",
            "filename": "nbModel.joblib",
            "runtime": "sklearn-tfidf",
            "vectorizer": "tfidf_vectorizer.joblib",
            **result,
            "training": training_info(full_feature_count, train_idx, test_idx),
        }
    )

    # --- Feature selection for tree ensembles ------------------------------
    print(f"Selecting top {TREE_FEATURE_COUNT} features via chi2 for tree models...")
    t0 = time.time()
    selector = SelectKBest(chi2, k=TREE_FEATURE_COUNT)
    selector.fit(X_train, y_train)
    print(f"  done in {time.time() - t0:.1f}s")

    # --- Random Forest (selector + classifier bundled as one pipeline) -----
    print("Training Random Forest...")
    t0 = time.time()
    rf_pipeline = Pipeline(
        [
            ("select", selector),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    joblib.dump(rf_pipeline, "rfModel.joblib")
    result = evaluate(rf_pipeline, X_test, y_test)
    new_entries.append(
        {
            "id": "random_forest",
            "name": "Random Forest",
            "nameFa": "جنگل تصادفی",
            "description": "Random Forest ensemble trained on the top 20,000 chi2-selected TF-IDF features of Persian text to distinguish AI-generated content from human writing.",
            "descriptionFa": "مدل جنگل تصادفی که روی ۲۰٬۰۰۰ ویژگی برتر TF-IDF (انتخاب‌شده با آزمون chi2) متن فارسی آموزش دیده تا متن تولیدشده توسط هوش مصنوعی را از نوشته انسان جدا کند.",
            "algorithm": "RandomForestClassifier",
            "type": "ml",
            "version": "1.0",
            "filename": "rfModel.joblib",
            "runtime": "sklearn-tfidf",
            "vectorizer": "tfidf_vectorizer.joblib",
            **result,
            "training": training_info(TREE_FEATURE_COUNT, train_idx, test_idx),
        }
    )

    # --- Gradient Boosting (selector + classifier bundled as one pipeline) -
    print("Training Gradient Boosting...")
    t0 = time.time()
    gb_pipeline = Pipeline(
        [
            ("select", selector),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    gb_pipeline.fit(X_train, y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    joblib.dump(gb_pipeline, "gbModel.joblib")
    result = evaluate(gb_pipeline, X_test, y_test)
    new_entries.append(
        {
            "id": "gradient_boosting",
            "name": "Gradient Boosting",
            "nameFa": "گرادیان بوستینگ",
            "description": "Gradient Boosting ensemble trained on the top 20,000 chi2-selected TF-IDF features of Persian text to distinguish AI-generated content from human writing.",
            "descriptionFa": "مدل گرادیان بوستینگ که روی ۲۰٬۰۰۰ ویژگی برتر TF-IDF (انتخاب‌شده با آزمون chi2) متن فارسی آموزش دیده تا متن تولیدشده توسط هوش مصنوعی را از نوشته انسان جدا کند.",
            "algorithm": "GradientBoostingClassifier",
            "type": "ml",
            "version": "1.0",
            "filename": "gbModel.joblib",
            "runtime": "sklearn-tfidf",
            "vectorizer": "tfidf_vectorizer.joblib",
            **result,
            "training": training_info(TREE_FEATURE_COUNT, train_idx, test_idx),
        }
    )

    # --- Merge into models.json ---------------------------------------------
    with open(MODELS_JSON_PATH, encoding="utf-8") as f:
        existing = json.load(f)
    existing_ids = {m["id"] for m in existing}
    for entry in new_entries:
        if entry["id"] in existing_ids:
            existing = [m for m in existing if m["id"] != entry["id"]]
        existing.append(entry)

    with open(MODELS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print("\nSummary:")
    for entry in new_entries:
        print(f"  {entry['id']}: {entry['metrics']}")


if __name__ == "__main__":
    main()
