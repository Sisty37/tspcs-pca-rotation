"""
Diagnostic — why does Madelon fail after PCA?
Run:  python -m src.experiments.diag_madelon

Answers three questions:
  1. Does RF work on RAW Madelon (no PCA)?  -> isolates PCA vs model problem
  2. Does RF work on scaled-then-PCA (all 500 PCs)?  -> does PCA itself destroy signal
  3. Which individual PCs carry predictive signal?  -> is signal in LOW-variance PCs?
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.datasets import get_dataset, split_train_calib_test


def rf():
    return RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)


def main():
    X, y, task = get_dataset("madelon", seed=0)
    Xtr, ytr, Xca, yca, Xte, yte = split_train_calib_test(X, y, task, seed=0)
    print(f"Madelon: X={X.shape}  train={Xtr.shape[0]} calib={Xca.shape[0]}")

    # --- Q1: RF on RAW features (no PCA, no scaling) ---
    acc_raw = accuracy_score(yca, rf().fit(Xtr, ytr).predict(Xca))
    print(f"\n[Q1] RF on RAW features (no PCA):        acc = {acc_raw:.3f}")
    print("     -> if this is high (~0.8+), the signal IS there; PCA is the problem.")
    print("     -> if this is also ~0.6, Madelon is just hard for this split.")

    # --- Q2: RF on scaled + full PCA (all components) ---
    scaler = StandardScaler().fit(Xtr)
    pca = PCA(svd_solver="full").fit(scaler.transform(Xtr))
    Ztr = pca.transform(scaler.transform(Xtr))
    Zca = pca.transform(scaler.transform(Xca))
    acc_pca_full = accuracy_score(yca, rf().fit(Ztr, ytr).predict(Zca))
    print(f"\n[Q2] RF on ALL PCA components:           acc = {acc_pca_full:.3f}")
    print("     -> PCA is a rotation; with ALL components RF should ~= raw.")
    print("     -> if much lower than Q1, scaling/PCA is hurting (investigate).")

    # --- Q3: which single PCs are predictive? (univariate signal per PC) ---
    # train a 1-feature LogReg on each PC alone, rank by calib accuracy
    per_pc = []
    for j in range(Ztr.shape[1]):
        m = LogisticRegression(max_iter=500).fit(Ztr[:, [j]], ytr)
        a = accuracy_score(yca, m.predict(Zca[:, [j]]))
        per_pc.append((j + 1, a, pca.explained_variance_ratio_[j]))  # 1-indexed PC
    per_pc.sort(key=lambda t: t[1], reverse=True)

    print("\n[Q3] Top-10 most predictive individual PCs (PC#, acc, var_ratio):")
    for pc, a, v in per_pc[:10]:
        print(f"     PC{pc:<4d}  acc={a:.3f}   var_ratio={v:.4f}")
    print("     -> if top PCs have HIGH pc-number / LOW var_ratio,")
    print("        the signal lives in LOW-variance PCs = your paper's key point.")


if __name__ == "__main__":
    main()
