"""
E-ltt : guaranteed task-sufficient PCA dimension via Learn-Then-Test
        (Angelopoulos et al., 2021). Run: python -m src.experiments.exp_crc

Why LTT (not CRC): choosing the number of PCs by downstream accuracy is a
selection over hypotheses whose per-example loss is NOT monotone in k, so the
Conformal Risk Control monotonicity assumption fails. LTT gives a finite-sample,
distribution-free guarantee WITHOUT requiring monotonicity, by testing each
candidate k as a hypothesis with a valid p-value and a fixed-sequence procedure.

Risk (simple, bounded, Bernoulli-mean -> Hoeffding p-value applies):
    R(k) = P(k-prefix model misclassifies)          [classification]
    R(k) = P(|k-pred - y| > full-model tolerance)   [regression, binarised]

Target level alpha is set relative to the FULL model:
    alpha = R_full / tau     (retain a tau-fraction of full accuracy)
Confidence: delta (e.g. 0.05).

Procedure (fixed-sequence, safe->risky):
    test k = K, K-1, ..., 1.
    valid p-value for "R(k) <= alpha":  Hoeffding
        p_k = exp(-2 n (alpha - Rhat(k))^2)  if Rhat(k) < alpha else 1.0
    walk down while p_k <= delta (reject H_k: R(k)>alpha, i.e. k is certified good);
    stop at the first k with p_k > delta. k_hat = last certified k.

Guarantee: P(R_test(k_hat) <= alpha) >= 1 - delta.
We verify on TEST: report R_test(k_hat) and whether it is <= alpha.

Strict split: TRAIN fits scaler+PCA+models; CALIBRATION runs LTT; TEST once.
"""
import numpy as np
import pandas as pd
import os
import warnings
from scipy.stats import binom
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.datasets import get_dataset, split_train_calib_test

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
TABDIR = os.path.join(RESULTS, "tables")
os.makedirs(TABDIR, exist_ok=True)

CONFIGS = [("mnist", "logreg"), ("mnist", "rf"),
           ("fashion", "logreg"), ("superconduct", "ridge")]
SEEDS = [0, 1, 2, 3, 4]
K = 100
SLACK = 0.05      # tolerate full-model error + 5 percentage points
DELTA = 0.05      # confidence level for the guarantee


def model_fn(task, name):
    if name == "rf":
        return (RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
                if task == "classification"
                else RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))
    if task == "classification":
        return LogisticRegression(max_iter=5000)
    return Ridge()


def per_k_risk(task, model, Ztr, ytr, Zeval, yeval, K_eff, reg_tol=None):
    """
    R(k) for k=1..K_eff on the eval set. Bounded in [0,1].
    classification: misclassification rate.
    regression: fraction of examples with abs error > reg_tol.
    Also returns the full-model risk R_full and (for regression) the tolerance used.
    """
    full = model_fn(task, model).fit(Ztr, ytr)
    if task == "classification":
        R = np.empty(K_eff)
        for k in range(1, K_eff + 1):
            p = model_fn(task, model).fit(Ztr[:, :k], ytr).predict(Zeval[:, :k])
            R[k - 1] = np.mean(p != yeval)
        R_full = np.mean(full.predict(Zeval) != yeval)
        return R, R_full, None
    else:
        # tolerance = 90th percentile of full-model abs error on TRAIN-fit, eval-measured
        if reg_tol is None:
            full_abs = np.abs(full.predict(Zeval) - yeval)
            reg_tol = np.quantile(full_abs, 0.90)
        R = np.empty(K_eff)
        for k in range(1, K_eff + 1):
            p = model_fn(task, model).fit(Ztr[:, :k], ytr).predict(Zeval[:, :k])
            R[k - 1] = np.mean(np.abs(p - yeval) > reg_tol)
        R_full = np.mean(np.abs(full.predict(Zeval) - yeval) > reg_tol)
        return R, R_full, reg_tol


def bentkus_pvalue(Rhat_k, alpha, n):
    """
    Valid p-value for H_k: R(k) > alpha, using the Bentkus inequality
    (tighter than Hoeffding; used in the LTT paper). Small p => certify good.
    p = e * P(Binomial(n, alpha) <= ceil(n * Rhat_k)).
    """
    if Rhat_k >= alpha:
        return 1.0
    k_succ = int(np.ceil(n * Rhat_k))
    p = np.e * binom.cdf(k_succ, n, alpha)
    return float(min(1.0, p))


def ltt_select_k(R_cal, alpha, delta, n):
    """
    Fixed-sequence from k=K down to k=1. Certify while p<=delta; stop at first p>delta.
    Returns smallest certified k (k_hat). If none certified, return K (full model).
    """
    K_eff = len(R_cal)
    k_hat = K_eff
    for k in range(K_eff, 0, -1):
        p = bentkus_pvalue(R_cal[k - 1], alpha, n)
        if p <= delta:
            k_hat = k            # k is certified; keep walking down
        else:
            break                # first failure -> stop
    return k_hat


def main():
    rows = []
    for ds, model in CONFIGS:
        for seed in SEEDS:
            X, y, task = get_dataset(ds, seed=seed)
            K_eff = min(K, X.shape[1])
            Xtr, ytr, Xca, yca, Xte, yte = split_train_calib_test(X, y, task, seed=seed)
            scaler = StandardScaler().fit(Xtr)
            pca = PCA(n_components=K_eff, svd_solver="full").fit(scaler.transform(Xtr))
            Ztr = pca.transform(scaler.transform(Xtr))
            Zca = pca.transform(scaler.transform(Xca))
            Zte = pca.transform(scaler.transform(Xte))

            # calibration risks
            R_ca, Rfull_ca, tol = per_k_risk(task, model, Ztr, ytr, Zca, yca, K_eff)
            n_ca = len(yca)
            alpha = Rfull_ca + SLACK             # target risk level (interpretable)
            k_hat = ltt_select_k(R_ca, alpha, DELTA, n_ca)

            # verify on TEST (reuse same tolerance for regression)
            R_te, Rfull_te, _ = per_k_risk(task, model, Ztr, ytr, Zte, yte, K_eff, reg_tol=tol)
            test_risk = float(R_te[k_hat - 1])
            alpha_te = Rfull_te + SLACK
            ok = test_risk <= alpha_te
            rows.append({"dataset": ds, "model": model, "seed": seed,
                         "k_hat": k_hat, "K": K_eff,
                         "alpha": round(alpha_te, 4),
                         "test_risk": round(test_risk, 4),
                         "within_alpha": bool(ok)})
            print(f"{ds:12s} {model:7s} seed{seed}  k_hat={k_hat:3d}/{K_eff}  "
                  f"alpha={alpha_te:.3f}  test_risk={test_risk:.3f}  "
                  f"{'OK' if ok else 'VIOLATED'}")

    df = pd.DataFrame(rows)
    out = os.path.join(TABDIR, "exp_ltt.csv")
    df.to_csv(out, index=False)

    print("\n=== LTT summary (per dataset x model) ===")
    g = df.groupby(["dataset", "model"]).agg(
        k_hat_mean=("k_hat", "mean"),
        k_hat_std=("k_hat", "std"),
        mean_test_risk=("test_risk", "mean"),
        frac_within_alpha=("within_alpha", "mean")).round(3)
    print(g)
    print(f"\nGuarantee (delta={DELTA}): per-seed test_risk should be <= alpha in "
          f">= {1-DELTA:.0%} of runs. frac_within_alpha near 1.0 confirms it.")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
