"""
PCA + per-k downstream performance for TS-PCS.
CRITICAL: scaler and PCA are fit on TRAIN ONLY. No exceptions (leakage = desk reject).
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score


# ---------- default downstream models ----------
def default_clf():
    return LogisticRegression(max_iter=2000)


def default_reg():
    return Ridge()


# ---------- PCA fit on TRAIN only ----------
def fit_pca(X_tr, n_components):
    scaler = StandardScaler().fit(X_tr)
    pca = PCA(n_components=n_components, svd_solver="full").fit(scaler.transform(X_tr))
    return scaler, pca


def transform(scaler, pca, X):
    return pca.transform(scaler.transform(X))


# ---------- per-k performance (Exp 1 core) ----------
def perf_vs_k(X_tr, y_tr, X_eval, y_eval, K, task="classification",
              model_fn=None):
    """
    Fit PCA on TRAIN, then for k=1..K train a fresh downstream model on the
    first-k PCs and score it on the EVAL set (calibration or test).
    Returns:
        perf : array [K]  (accuracy for clf, R2 for reg) at each k
        cum_var : array [K]  cumulative explained-variance ratio
    """
    if model_fn is None:
        model_fn = default_clf if task == "classification" else default_reg
    score_fn = accuracy_score if task == "classification" else r2_score

    scaler, pca = fit_pca(X_tr, n_components=K)
    Ztr = transform(scaler, pca, X_tr)
    Zev = transform(scaler, pca, X_eval)
    cum_var = np.cumsum(pca.explained_variance_ratio_)[:K]

    perf = np.empty(K)
    for k in range(1, K + 1):
        m = model_fn().fit(Ztr[:, :k], y_tr)
        perf[k - 1] = score_fn(y_eval, m.predict(Zev[:, :k]))
    return perf, cum_var


# ---------- divergence quantifier ----------
def variance_vs_utility_gap(perf, cum_var, tau=0.95):
    """
    How many PCs does VARIANCE say you need vs PERFORMANCE?
    - k_var : smallest k reaching tau of total variance
    - k_perf: smallest k reaching tau * full performance
    Positive gap (k_var > k_perf) => variance is WASTEFUL.
    Negative gap (k_var < k_perf) => variance UNDER-selects (misses predictive PCs).
    """
    full = perf[-1]
    k_perf = next((k for k, p in enumerate(perf, 1) if p >= tau * full), len(perf))
    k_var = next((k for k, v in enumerate(cum_var, 1) if v >= tau), len(cum_var))
    return {"k_var": k_var, "k_perf": k_perf, "gap": k_var - k_perf,
            "full_perf": float(full)}


# ---------- univariate predictive ranking (leakage-free) ----------
def predictive_order(X_tr, y_tr, K, task="classification", cv=5, model_fn=None):
    """
    Rank the first-K PCs by their INDIVIDUAL predictive signal, measured by
    cross-validation ON THE TRAINING SET ONLY (no calib/test peeking -> no
    selection-stage overfitting; each PC scored independently, once).

    Returns:
        order : array [K] of PC indices (0-based), best predictive first.
        scores: array [K] the CV score of each PC in ORIGINAL (variance) order.
    """
    if model_fn is None:
        model_fn = default_clf if task == "classification" else default_reg
    scoring = "accuracy" if task == "classification" else "r2"

    scaler, pca = fit_pca(X_tr, n_components=K)
    Ztr = transform(scaler, pca, X_tr)

    scores = np.empty(K)
    for j in range(K):
        s = cross_val_score(model_fn(), Ztr[:, [j]], y_tr, cv=cv, scoring=scoring)
        scores[j] = s.mean()
    order = np.argsort(scores)[::-1]        # best-first
    return order, scores


def perf_along_order(X_tr, y_tr, X_eval, y_eval, order, task="classification",
                     model_fn=None):
    """
    Given a PC ORDER (list of PC indices), evaluate prefix performance:
    use the first-m PCs of `order` for m=1..len(order), score on EVAL set.
    Works for both variance order (0,1,2,..) and predictive order.
    Returns perf array [len(order)].
    """
    if model_fn is None:
        model_fn = default_clf if task == "classification" else default_reg
    score_fn = accuracy_score if task == "classification" else r2_score

    K = len(order)
    scaler, pca = fit_pca(X_tr, n_components=K)
    Ztr = transform(scaler, pca, X_tr)
    Zev = transform(scaler, pca, X_eval)

    perf = np.empty(K)
    for m in range(1, K + 1):
        cols = list(order[:m])
        mdl = model_fn().fit(Ztr[:, cols], y_tr)
        perf[m - 1] = score_fn(y_eval, mdl.predict(Zev[:, cols]))
    return perf


def k_for_tau(perf, tau=0.95):
    """Smallest prefix length reaching tau * full performance (full = perf[-1])."""
    full = perf[-1]
    return next((m for m, p in enumerate(perf, 1) if p >= tau * full), len(perf))
