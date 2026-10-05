"""Ablation proof: do the 10 style columns actually move the sklearn-tfidf models?

Hypothesis under test: the ~150k TF-IDF columns swamp the 10 stylometric
columns so they barely affect learning or prediction. Measures on the
canonical split (training/split.py), with the exact hyperparameters of
train_ml_models.py:

  1. style-only vs tfidf-only vs combined logistic regression
  2. error complementarity of the style-only model on tfidf-only mistakes
  3. late fusion of the two single-view models (unique style signal)
  4. learned weight share of the style block (LR |coef|, RF/GBG importance)
  5. style-block scale sweep (style * alpha) as the minimal rebalance test

Run from the project root:

    python training/eval_style_ablation.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from aitextd import paths
from aitextd.features import STYLE_FEATURE_NAMES
from split import RANDOM_SEED, load_tfidf_split

N_STYLE = len(STYLE_FEATURE_NAMES)
ALPHAS = (3.0, 10.0)


def make_lr():
    # Same recipe as training/train_ml_models.py (logreg entry).
    return LogisticRegression(
        solver="liblinear", dual=True, C=32.0, max_iter=5000, random_state=RANDOM_SEED
    )


def metrics(model, X, y):
    pred = model.predict(X)
    return round(float(accuracy_score(y, pred)), 4), round(float(f1_score(y, pred)), 4)


def row(name, acc, f1):
    print(f"  {name:<28} acc={acc:.4f}  f1={f1:.4f}")


def main():
    X_train, X_test, y_train, y_test, *_ = load_tfidf_split()
    tfidf_dim = X_train.shape[1] - N_STYLE
    Xtr_t = X_train[:, :tfidf_dim]
    Xte_t = X_test[:, :tfidf_dim]
    Xtr_s = X_train[:, tfidf_dim:].toarray()
    Xte_s = X_test[:, tfidf_dim:].toarray()

    print("== linear models: tfidf-only vs combined ==")
    m_style = make_lr().fit(Xtr_s, y_train)
    acc_s, f1_s = metrics(m_style, Xte_s, y_test)
    row("style only (10)", acc_s, f1_s)

    m_tfidf = make_lr().fit(Xtr_t, y_train)
    acc_t, f1_t = metrics(m_tfidf, Xte_t, y_test)
    row(f"tfidf only ({tfidf_dim})", acc_t, f1_t)

    m_full = make_lr().fit(X_train, y_train)
    acc_f, f1_f = metrics(m_full, X_test, y_test)
    row("combined logreg (current)", acc_f, f1_f)

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    for name, clf in (
        ("svm", LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_SEED)),
        ("nb", MultinomialNB(alpha=0.01)),
    ):
        acc_c, f1_c = metrics(clf.fit(X_train, y_train), X_test, y_test)
        row(f"combined {name} (current)", acc_c, f1_c)
        acc_o, f1_o = metrics(clf.fit(Xtr_t, y_train), Xte_t, y_test)
        row(f"tfidf-only {name}", acc_o, f1_o)

    with open(paths.MODELS_JSON, encoding="utf-8") as f:
        registry = json.load(f)
    pinned = next(m for m in registry if m["id"] == "logreg")["metrics"]["f1Score"]
    assert abs(f1_t - pinned) < 0.001, f"tfidf-only logreg f1 {f1_t} != models.json {pinned}"

    print("== error complementarity (tfidf-only mistakes) ==")
    miss = m_tfidf.predict(Xte_t) != y_test
    fixed = int((m_style.predict(Xte_s)[miss] == y_test[miss]).sum())
    print(f"  tfidf-only errors={int(miss.sum())}, style-only fixes {fixed}")

    print("== late fusion (base 75%, fusion 25% of train) ==")
    Xb_t, Xf_t, Xb_s, Xf_s, yb, yf = train_test_split(
        Xtr_t, Xtr_s, y_train, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train
    )
    base_t = make_lr().fit(Xb_t, yb)
    base_s = make_lr().fit(Xb_s, yb)
    stack = np.column_stack(
        [
            base_t.predict_proba(Xf_t)[:, 1],
            base_s.predict_proba(Xf_s)[:, 1],
        ]
    )
    fusion = LogisticRegression().fit(stack, yf)
    stack_test = np.column_stack(
        [
            base_t.predict_proba(Xte_t)[:, 1],
            base_s.predict_proba(Xte_s)[:, 1],
        ]
    )
    acc_u, f1_u = metrics(fusion, stack_test, y_test)
    row("fusion(tfidf, style)", acc_u, f1_u)
    acc_bt, f1_bt = metrics(base_t, Xte_t, y_test)
    row("its base: tfidf-only 75%", acc_bt, f1_bt)
    acc_bs, f1_bs = metrics(base_s, Xte_s, y_test)
    row("its base: style-only 75%", acc_bs, f1_bs)
    print(f"  fusion weights: tfidf={fusion.coef_[0][0]:.3f}, style={fusion.coef_[0][1]:.3f}")

    print("== learned weight share of the style block ==")
    coef = np.abs(m_full.coef_[0])
    share = float(coef[-N_STYLE:].sum() / coef.sum())
    per_feat = float(coef[-N_STYLE:].mean() / coef[:tfidf_dim].mean())
    print(f"  LR |coef| share={share:.4%}  mean|coef| style/tfidf={per_feat:.2f}x")
    for name, filename in (
        ("random_forest", "rfModel.joblib"),
        ("gradient_boosting", "gbModel.joblib"),
    ):
        clf = joblib.load(paths.model_path(filename)).named_steps["clf"]
        imp = clf.feature_importances_
        imp_share = float(imp[-N_STYLE:].sum() / imp.sum())
        print(f"  {name} importance share={imp_share:.4%}")

    print("== style-block scale sweep (combined LR) ==")
    row("alpha=1 (current)", acc_f, f1_f)
    for alpha in ALPHAS:
        Xa = hstack([Xtr_t, csr_matrix(Xtr_s * alpha)], format="csr")
        Xae = hstack([Xte_t, csr_matrix(Xte_s * alpha)], format="csr")
        acc_a, f1_a = metrics(make_lr().fit(Xa, y_train), Xae, y_test)
        row(f"style * {alpha:g}", acc_a, f1_a)


if __name__ == "__main__":
    main()
