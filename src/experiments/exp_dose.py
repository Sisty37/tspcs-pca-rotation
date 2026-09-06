"""
E-dose : does the rotation gap scale with how CONCENTRATED the predictive signal is?
Run:  python -m src.experiments.exp_dose

For each dataset we compute:
  concentration : how unevenly predictive signal is spread across raw features,
                  measured as the Gini coefficient of per-feature mutual information
                  with the target. High Gini = signal in few features (concentrated).
  rotation_gap  : RF gap (raw acc - full-PCA acc), the tree-model rotation cost.
Then we plot gap vs concentration across datasets. Hypothesis: positive relationship.
Reuses the same protocol; RF only (the clearest tree effect). Fast.
"""
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, r2_score

from src.datasets import get_dataset, split_train_calib_test

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
FIGDIR = os.path.join(RESULTS, "figures")
TABDIR = os.path.join(RESULTS, "tables")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

DATASETS = ["madelon", "mnist", "fashion", "superconduct"]
SEEDS = [0, 1, 2]


def gini(x):
    """Gini coefficient of a non-negative array. 0 = perfectly even, ->1 = concentrated."""
    x = np.sort(np.asarray(x, dtype=float))
    x = x - x.min() if x.min() < 0 else x            # ensure non-negative
    if x.sum() == 0:
        return 0.0
    n = len(x)
    cum = np.cumsum(x)
    # standard Gini formula
    return (2.0 * np.sum((np.arange(1, n + 1)) * x) / (n * cum[-1])) - (n + 1.0) / n


def concentration_of_signal(Xtr, ytr, task, seed):
    """Gini of per-feature mutual information with the target (on TRAIN)."""
    if task == "classification":
        mi = mutual_info_classif(Xtr, ytr, random_state=seed)
    else:
        mi = mutual_info_regression(Xtr, ytr, random_state=seed)
    return gini(mi)


def rf_for(task):
    return (RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
            if task == "classification"
            else RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))


def score(task, yt, yp):
    return accuracy_score(yt, yp) if task == "classification" else r2_score(yt, yp)


def main():
    rows = []
    for ds in DATASETS:
        for seed in SEEDS:
            X, y, task = get_dataset(ds, seed=seed)
            Xtr, ytr, Xca, yca, Xte, yte = split_train_calib_test(X, y, task, seed=seed)
            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xca_s = scaler.transform(Xtr), scaler.transform(Xca)
            pca = PCA(svd_solver="full").fit(Xtr_s)
            Ztr, Zca = pca.transform(Xtr_s), pca.transform(Xca_s)

            conc = concentration_of_signal(Xtr_s, ytr, task, seed)
            raw = score(task, yca, rf_for(task).fit(Xtr_s, ytr).predict(Xca_s))
            fpc = score(task, yca, rf_for(task).fit(Ztr, ytr).predict(Zca))
            gap = raw - fpc
            rows.append({"dataset": ds, "task": task, "seed": seed,
                         "concentration": round(conc, 4), "rf_gap": round(gap, 4)})
            print(f"{ds:12s} seed{seed}  concentration(Gini)={conc:.3f}  rf_gap={gap:+.3f}")

    df = pd.DataFrame(rows)
    out = os.path.join(TABDIR, "exp_dose.csv")
    df.to_csv(out, index=False)

    # aggregate per dataset (mean over seeds)
    agg = df.groupby(["dataset", "task"]).agg(
        concentration=("concentration", "mean"),
        rf_gap=("rf_gap", "mean")).reset_index()
    print("\n=== per-dataset means ===")
    print(agg.round(4).to_string(index=False))

    # correlation
    r = np.corrcoef(agg["concentration"], agg["rf_gap"])[0, 1]
    print(f"\nPearson corr(concentration, rf_gap) across datasets = {r:.3f}")

    # figure
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for _, row in agg.iterrows():
        ax.scatter(row["concentration"], row["rf_gap"], s=80)
        ax.annotate(row["dataset"], (row["concentration"], row["rf_gap"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Signal concentration (Gini of per-feature MI)")
    ax.set_ylabel("RF rotation gap (raw - full-PCA)")
    ax.set_title(f"Dose-response: gap vs signal concentration (r={r:.2f})")
    ax.grid(alpha=.3)
    fig.tight_layout()
    figpath = os.path.join(FIGDIR, "exp_dose.png")
    fig.savefig(figpath, dpi=150)

    print(f"\nsaved: {out}")
    print(f"saved: {figpath}")
    print("Hypothesis supported if higher concentration -> larger gap (positive r).")


if __name__ == "__main__":
    main()
