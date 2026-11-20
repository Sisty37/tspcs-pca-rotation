"""
Shared Learn-Then-Test utilities, exactly as described in the paper.

Reference ("full") model: the same learner trained on all K PCA components.
Per-example loss l in {0,1}:
    classification: misclassification
    regression    : 1[|y_hat - y| > tau], tau = 90th percentile of the full
                    model's absolute error on the TRAINING partition.
Paired excess-loss indicator: D_i(k) = 1[l_i(k) > l_i(full)].
For 0/1 losses, R(k) - R(full) <= E[D(k)].
Test H_k: E[D(k)] > s with a Bentkus p-value; fixed sequence k = K..1,
certify while p_k <= delta, stop at the first failure.
"""
import numpy as np
from scipy.stats import binom


def regression_tau(full_model, Ztr, ytr, q=0.90):
    return float(np.quantile(np.abs(full_model.predict(Ztr) - ytr), q))


def losses(task, y_pred, y_true, tau=None):
    if task == "classification":
        return (y_pred != y_true).astype(int)
    return (np.abs(y_pred - y_true) > tau).astype(int)


def bentkus_pvalue(dbar, s, n):
    """Valid p-value for H: E[D] > s, D in {0,1}, n i.i.d. samples."""
    if dbar >= s:
        return 1.0
    return float(min(1.0, np.e * binom.cdf(int(np.ceil(n * dbar)), n, s)))


def fixed_sequence(dbar_of_k, K, s, delta, n):
    """dbar_of_k(k) -> mean D(k) on calibration. Evaluated lazily from k=K down.
    Returns (k_hat, number of k evaluated)."""
    k_hat, evaluated = K, 0
    for k in range(K, 0, -1):
        evaluated += 1
        if bentkus_pvalue(dbar_of_k(k), s, n) <= delta:
            k_hat = k
        else:
            break
    return k_hat, evaluated
