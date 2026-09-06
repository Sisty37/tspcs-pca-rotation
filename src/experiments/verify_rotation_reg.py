"""
Regression version of the rotation-loss verification.
Run:  python -m src.experiments.verify_rotation_reg

Same logic as the classification test, on Superconductivity:
  raw R2 vs full-PCA R2, per (model x seed).
  rotation-invariant (ridge, linsvr) -> expect gap ~ 0
  axis-aligned tree (rf, histgbm)     -> expect gap > 0
R2 is the metric; gap = R2_raw - R2_fullpca (positive => rotation hurt).
"""
import numpy as np
import pandas as pd
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score

from src.datasets import get_dataset, split_train_calib_test

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
TABDIR = os.path.join(RESULTS, "tables")
os.makedirs(TABDIR, exist_ok=True)

MODELS = ["ridge", "linsvr", "rf", "histgbm"]
SEEDS = [0, 1, 2, 3, 4]
FAMILY = {"ridge": "rotation-invariant", "linsvr": "rotation-invariant",
          "rf": "axis-aligned(tree)", "histgbm": "axis-aligned(tree)"}


def model_fn(name):
    if name == "rf":
        return RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    if name == "histgbm":
        return HistGradientBoostingRegressor(random_state=0)
    if name == "linsvr":
        return LinearSVR(max_iter=50000)
    return Ridge()


def main():
    rows = []
    for seed in SEEDS:
        X, y, task = get_dataset("superconduct", seed=seed)
        Xtr, ytr, Xca, yca, Xte, yte = split_train_calib_test(X, y, task, seed=seed)

        # target scaling helps linear models; fit on train only
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xca_s = scaler.transform(Xtr), scaler.transform(Xca)
        pca = PCA(svd_solver="full").fit(Xtr_s)
        Ztr, Zca = pca.transform(Xtr_s), pca.transform(Xca_s)

        for mname in MODELS:
            raw = r2_score(yca, model_fn(mname).fit(Xtr_s, ytr).predict(Xca_s))
            fpc = r2_score(yca, model_fn(mname).fit(Ztr, ytr).predict(Zca))
            rows.append({"dataset": "superconduct", "model": mname,
                         "family": FAMILY[mname], "seed": seed,
                         "raw_r2": round(raw, 4), "full_pca_r2": round(fpc, 4),
                         "gap": round(raw - fpc, 4)})
            print(f"superconduct {mname:8s} seed{seed}  raw_r2={raw:.3f}  "
                  f"full_pca_r2={fpc:.3f}  gap={raw-fpc:+.3f}")

    df = pd.DataFrame(rows)
    out = os.path.join(TABDIR, "verify_rotation_reg.csv")
    df.to_csv(out, index=False)

    print("\n=== MEAN gap by model (R2_raw - R2_fullpca) ===")
    print(df.groupby("model")["gap"].agg(["mean", "std"]).round(4))
    print("\n=== MEAN gap by FAMILY ===")
    print(df.groupby("family")["gap"].agg(["mean", "std"]).round(4))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
