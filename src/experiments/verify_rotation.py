"""
Verify Finding 2: does PCA rotation cost downstream accuracy, systematically?
Run:  python -m src.experiments.verify_rotation

Fair comparison: raw and full-PCA inputs get the SAME preprocessing.
- raw input      = StandardScaler(raw features)
- full-pca input = PCA(StandardScaler(raw features))   [a pure rotation of the above]
Every model is trained the same way on both. Any gap is due to the ROTATION,
not to preprocessing asymmetry. (SVM no longer has its own extra scaler.)

Families:
  rotation-invariant : logreg, svm   -> expect gap ~ 0
  axis-aligned (tree): rf, gbm       -> expect gap > 0
"""
import numpy as np
import pandas as pd
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score, r2_score

from src.datasets import get_dataset, split_train_calib_test

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
TABDIR = os.path.join(RESULTS, "tables")
os.makedirs(TABDIR, exist_ok=True)

DATASETS = ["madelon", "mnist", "fashion"]
MODELS = ["logreg", "svm", "rf", "gbm"]
SEEDS = [0, 1, 2, 3, 4]
FAMILY = {"logreg": "rotation-invariant", "svm": "rotation-invariant",
          "rf": "axis-aligned(tree)", "gbm": "axis-aligned(tree)"}


def model_fn(task, name):
    if name == "rf":
        return (RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
                if task == "classification"
                else RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))
    if name == "gbm":
        return (GradientBoostingClassifier(random_state=0)
                if task == "classification"
                else GradientBoostingRegressor(random_state=0))
    if name == "svm":
        # NO internal scaler now: inputs are already scaled/rotated consistently.
        # High max_iter so it converges on both raw and PCA inputs.
        return (LinearSVC(max_iter=50000, dual="auto") if task == "classification"
                else LinearSVR(max_iter=50000))
    if task == "classification":
        return LogisticRegression(max_iter=5000)
    return Ridge()


def score(task, y_true, y_pred):
    return accuracy_score(y_true, y_pred) if task == "classification" else r2_score(y_true, y_pred)


def main():
    rows = []
    for ds in DATASETS:
        for seed in SEEDS:
            X, y, task = get_dataset(ds, seed=seed)
            Xtr, ytr, Xca, yca, Xte, yte = split_train_calib_test(X, y, task, seed=seed)

            # ---- one consistent scaling, fit on TRAIN only ----
            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xca_s = scaler.transform(Xtr), scaler.transform(Xca)

            # ---- full PCA = pure rotation of the SAME scaled input ----
            pca = PCA(svd_solver="full").fit(Xtr_s)
            Ztr, Zca = pca.transform(Xtr_s), pca.transform(Xca_s)

            for mname in MODELS:
                raw = score(task, yca, model_fn(task, mname).fit(Xtr_s, ytr).predict(Xca_s))
                fpc = score(task, yca, model_fn(task, mname).fit(Ztr, ytr).predict(Zca))
                rows.append({"dataset": ds, "model": mname, "family": FAMILY[mname],
                             "seed": seed, "raw": round(raw, 4),
                             "full_pca": round(fpc, 4), "gap": round(raw - fpc, 4)})
                print(f"{ds:10s} {mname:7s} seed{seed}  raw={raw:.3f}  "
                      f"full_pca={fpc:.3f}  gap={raw-fpc:+.3f}")

    df = pd.DataFrame(rows)
    out = os.path.join(TABDIR, "verify_rotation.csv")
    df.to_csv(out, index=False)

    print("\n=== MEAN gap by dataset x model ===")
    print(df.groupby(["dataset", "model"])["gap"].agg(["mean", "std"]).round(4))
    print("\n=== MEAN gap by FAMILY (the key test) ===")
    print(df.groupby("family")["gap"].agg(["mean", "std"]).round(4))
    print(f"\nsaved: {out}")
    print("Expect rotation-invariant ~ 0, axis-aligned(tree) > 0. "
          "If SVM now ~0 on MNIST too, the 2x2 story is clean.")


if __name__ == "__main__":
    main()