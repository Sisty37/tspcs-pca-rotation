"""
E-sanity : does the method behave sensibly in controlled/pathological cases?
Run:  python -m src.experiments.exp_sanity

Two controls, reusing the SAME LTT selection as exp_crc:

(1) NO-SIGNAL : shuffle y on a real dataset so features carry no information.
    Expectation: no genuine compression is justified. The full-model error is
    ~chance, and LTT should NOT certify a small k that "retains" performance
    (there is nothing to retain). We report k_hat and full accuracy (~chance).

(2) PLANTED-SIGNAL : synthetic data where the label depends ONLY on a few
    leading directions. Expectation: LTT selects a SMALL k close to the planted
    dimension, and downstream accuracy is high.

These answer the reviewer question: "is the selector fooled by noise, and does
it recover a known sufficient dimension?"
"""
import numpy as np
import pandas as pd
import os
import warnings
from scipy.stats import binom
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from src.datasets import get_dataset, split_train_calib_test

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
TABDIR = os.path.join(RESULTS, "tables")
os.makedirs(TABDIR, exist_ok=True)

K = 100
SLACK = 0.05
DELTA = 0.05
SEEDS = [0, 1, 2]


def bentkus_pvalue(Rhat_k, alpha, n):
    if Rhat_k >= alpha:
        return 1.0
    k_succ = int(np.ceil(n * Rhat_k))
    return float(min(1.0, np.e * binom.cdf(k_succ, n, alpha)))


def ltt_select_k(R_cal, alpha, delta, n):
    K_eff = len(R_cal)
    k_hat = K_eff
    for k in range(K_eff, 0, -1):
        if bentkus_pvalue(R_cal[k - 1], alpha, n) <= delta:
            k_hat = k
        else:
            break
    return k_hat


def risks(Ztr, ytr, Zev, yev, K_eff):
    full = LogisticRegression(max_iter=5000).fit(Ztr, ytr)
    R = np.empty(K_eff)
    for k in range(1, K_eff + 1):
        p = LogisticRegression(max_iter=5000).fit(Ztr[:, :k], ytr).predict(Zev[:, :k])
        R[k - 1] = np.mean(p != yev)
    R_full = np.mean(full.predict(Zev) != yev)
    return R, R_full


def run_case(Xtr, ytr, Xca, yca, K_eff, scale=True):
    if scale:
        scaler = StandardScaler().fit(Xtr)
        Xtr_p, Xca_p = scaler.transform(Xtr), scaler.transform(Xca)
    else:
        # keep raw variance structure (needed for the planted-variance control)
        Xtr_p, Xca_p = Xtr, Xca
    pca = PCA(n_components=K_eff, svd_solver="full").fit(Xtr_p)
    Ztr = pca.transform(Xtr_p)
    Zca = pca.transform(Xca_p)
    R_ca, Rfull = risks(Ztr, ytr, Zca, yca, K_eff)
    alpha = Rfull + SLACK
    k_hat = ltt_select_k(R_ca, alpha, DELTA, len(yca))
    return k_hat, 1 - Rfull   # k_hat, full accuracy


def make_planted(n=3000, d=50, n_informative=3, seed=0):
    """
    Label depends only on the first n_informative directions, and those directions
    are made HIGH-VARIANCE so PCA ranks them first. No rotation: signal stays in the
    leading principal components, so a small prefix should be sufficient.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype("float32")
    # inflate variance of the informative features so PCA puts them first
    scale = np.ones(d, dtype="float32")
    scale[:n_informative] = 8.0                      # informative dirs dominate variance
    X = X * scale
    w = np.zeros(d); w[:n_informative] = rng.randn(n_informative)
    logits = X[:, :n_informative] @ w[:n_informative]
    y = (logits > np.median(logits)).astype(int)
    return X, y


def main():
    rows = []

    # (1) NO-SIGNAL on real datasets (shuffle y)
    for ds in ["mnist", "fashion"]:
        for seed in SEEDS:
            X, y, task = get_dataset(ds, seed=seed)
            K_eff = min(K, X.shape[1])
            Xtr, ytr, Xca, yca, Xte, yte = split_train_calib_test(X, y, task, seed=seed)
            rng = np.random.RandomState(seed)
            ytr_s = rng.permutation(ytr)          # destroy signal
            yca_s = rng.permutation(yca)
            k_hat, full_acc = run_case(Xtr, ytr_s, Xca, yca_s, K_eff)
            rows.append({"case": "no_signal", "data": ds, "seed": seed,
                         "k_hat": k_hat, "K": K_eff, "full_acc": round(full_acc, 3)})
            print(f"no_signal   {ds:8s} seed{seed}  k_hat={k_hat:3d}/{K_eff}  "
                  f"full_acc={full_acc:.3f} (chance ~0.1)")

    # (2) PLANTED-SIGNAL synthetic (only 3 informative directions)
    for seed in SEEDS:
        X, y = make_planted(seed=seed)
        K_eff = min(K, X.shape[1])
        # simple split
        n = len(X); idx = np.random.RandomState(seed).permutation(n)
        a, b = int(.6 * n), int(.8 * n)
        tr, ca = idx[:a], idx[a:b]
        k_hat, full_acc = run_case(X[tr], y[tr], X[ca], y[ca], K_eff, scale=False)
        rows.append({"case": "planted_k3", "data": "synthetic", "seed": seed,
                     "k_hat": k_hat, "K": K_eff, "full_acc": round(full_acc, 3)})
        print(f"planted_k3  synth    seed{seed}  k_hat={k_hat:3d}/{K_eff}  "
              f"full_acc={full_acc:.3f} (planted dim=3)")

    df = pd.DataFrame(rows)
    out = os.path.join(TABDIR, "exp_sanity.csv")
    df.to_csv(out, index=False)
    print("\n=== sanity summary ===")
    print(df.groupby(["case", "data"]).agg(
        k_hat_mean=("k_hat", "mean"),
        full_acc_mean=("full_acc", "mean")).round(2))
    print(f"\nsaved: {out}")
    print("Sensible if: no_signal -> full_acc ~ chance (no real compression story); "
          "planted -> small k_hat with high full_acc.")


if __name__ == "__main__":
    main()
