"""
Experiment 1 — Variance ordering vs Predictive ordering of PCA components.
Run from project root:  python -m src.experiments.exp1

Compares two ways to pick a k-dim PCA prefix:
  (A) VARIANCE order   : PCs 1,2,3,...  (standard PCA)
  (B) PREDICTIVE order : PCs ranked by individual train-CV signal (leakage-free)
For each, we find the smallest prefix reaching tau*full accuracy.
Headline question: does predictive ordering reach the target with FEWER PCs?
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.datasets import get_dataset, split_train_calib_test
from src.pca_utils import (perf_along_order, predictive_order, k_for_tau,
                           fit_pca)

   

DATASET= "madelon"  
MODEL= "logreg" 
           # "logreg" (linear) | "rf" (nonlinear)
SEED = 0
K = 100
TAU = 0.95

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
FIGDIR = os.path.join(RESULTS, "figures")
TABDIR = os.path.join(RESULTS, "tables")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)


def make_model_fn(task, model):
    if model == "rf":
        if task == "classification":
            return lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        return lambda: RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    return None  # linear default


def main():
    X, y, task = get_dataset(DATASET, seed=SEED)
    K_eff = min(K, X.shape[1])
    Xtr, ytr, Xca, yca, Xte, yte = split_train_calib_test(X, y, task, seed=SEED)
    model_fn = make_model_fn(task, MODEL)
    tag = f"{DATASET}_{MODEL}"

    # variance order = 0,1,2,...  (PCA already sorts by variance)
    var_order = np.arange(K_eff)
    # predictive order (ranked on TRAIN CV only -> leakage-free)
    pred_order, _ = predictive_order(Xtr, ytr, K_eff, task=task, model_fn=model_fn)

    # prefix performance along each order, scored on CALIBRATION set
    perf_var = perf_along_order(Xtr, ytr, Xca, yca, var_order, task=task, model_fn=model_fn)
    perf_pred = perf_along_order(Xtr, ytr, Xca, yca, pred_order, task=task, model_fn=model_fn)

    k_var = k_for_tau(perf_var, tau=TAU)
    k_pred = k_for_tau(perf_pred, tau=TAU)

    metric = "Accuracy" if task == "classification" else "R2"
    ks = np.arange(1, K_eff + 1)
    df = pd.DataFrame({"k": ks, "perf_variance_order": perf_var,
                       "perf_predictive_order": perf_pred})
    csv_path = os.path.join(TABDIR, f"exp1_{tag}.csv")
    df.to_csv(csv_path, index=False)

    # figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, perf_var, "-", color="tab:gray", label="Variance order (standard PCA)")
    ax.plot(ks, perf_pred, "-", color="tab:blue", label="Predictive order (ours)")
    ax.axvline(k_var, color="tab:gray", ls=":", alpha=.7, label=f"k_var={k_var}")
    ax.axvline(k_pred, color="tab:blue", ls=":", alpha=.7, label=f"k_pred={k_pred}")
    ax.set_xlabel("Number of PCs in prefix (k)")
    ax.set_ylabel(metric)
    ax.set_title(f"Exp 1 - {tag}: variance vs predictive ordering")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig_path = os.path.join(FIGDIR, f"exp1_{tag}.png")
    fig.savefig(fig_path, dpi=150)

    # report
    full = perf_var[-1]
    print(f"[{tag}] task={task}  K={K_eff}  full {metric}={full:.4f}")
    print(f"  k to reach {TAU}*full  |  variance order = {k_var}   predictive order = {k_pred}")
    saving = k_var - k_pred
    print(f"  PC saving from predictive ordering = {saving} "
          f"({'predictive wins' if saving > 0 else 'no advantage' if saving == 0 else 'variance better'})")
    print(f"  saved: {csv_path}")
    print(f"  saved: {fig_path}")


if __name__ == "__main__":
    main()
