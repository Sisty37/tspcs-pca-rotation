"""
E-stab : how STABLE is the selected task-sufficient dimension across seeds?
Run:  python -m src.experiments.exp_stab

For each dataset x model, across seeds we compute:
  k_tau   = smallest prefix k (variance order) with acc_k >= tau * acc_full
            (measured on the CALIBRATION set; TRAIN-only PCA/scaler)
  report  : mean k, std k, and mean pairwise Jaccard of the selected PC prefix {1..k}
Low std + high Jaccard => the task-sufficient dimension is reproducible.
"""
import numpy as np
import pandas as pd
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

from src.datasets import get_dataset, split_train_calib_test

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
TABDIR = os.path.join(RESULTS, "tables")
os.makedirs(TABDIR, exist_ok=True)

# (dataset, model) pairs to test. logreg=rotation-invariant anchor, rf=tree.
CONFIGS = [("mnist", "logreg"), ("mnist", "rf"),
           ("fashion", "logreg"), ("fashion", "rf"),
           ("superconduct", "ridge"), ("superconduct", "rf")]
SEEDS = [0, 1, 2, 3, 4]
K = 100
TAU = 0.95


def model_fn(task, name):
    if name == "rf":
        return (RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
                if task == "classification"
                else RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))
    if task == "classification":
        return LogisticRegression(max_iter=5000)
    return Ridge()


def score(task, yt, yp):
    return accuracy_score(yt, yp) if task == "classification" else r2_score(yt, yp)


def k_tau_for(ds, model, seed):
    X, y, task = get_dataset(ds, seed=seed)
    K_eff = min(K, X.shape[1])
    Xtr, ytr, Xca, yca, Xte, yte = split_train_calib_test(X, y, task, seed=seed)
    scaler = StandardScaler().fit(Xtr)
    pca = PCA(n_components=K_eff, svd_solver="full").fit(scaler.transform(Xtr))
    Ztr = pca.transform(scaler.transform(Xtr))
    Zca = pca.transform(scaler.transform(Xca))

    perf = np.empty(K_eff)
    for k in range(1, K_eff + 1):
        m = model_fn(task, model).fit(Ztr[:, :k], ytr)
        perf[k - 1] = score(task, yca, m.predict(Zca[:, :k]))
    full = perf[-1]
    k_tau = next((k for k, p in enumerate(perf, 1) if p >= TAU * full), K_eff)
    return k_tau


def jaccard_prefix(k1, k2):
    """Jaccard of prefixes {1..k1} and {1..k2} = min/max."""
    return min(k1, k2) / max(k1, k2) if max(k1, k2) > 0 else 1.0


def main():
    rows = []
    for ds, model in CONFIGS:
        ks = [k_tau_for(ds, model, s) for s in SEEDS]
        pair_j = [jaccard_prefix(ks[i], ks[j])
                  for i in range(len(ks)) for j in range(i + 1, len(ks))]
        rows.append({"dataset": ds, "model": model,
                     "k_mean": round(float(np.mean(ks)), 2),
                     "k_std": round(float(np.std(ks)), 2),
                     "jaccard_mean": round(float(np.mean(pair_j)), 3),
                     "ks": ks})
        print(f"{ds:12s} {model:7s}  k={ks}  mean={np.mean(ks):.1f} "
              f"std={np.std(ks):.2f}  jaccard={np.mean(pair_j):.3f}")

    df = pd.DataFrame(rows)
    out = os.path.join(TABDIR, "exp_stab.csv")
    df.to_csv(out, index=False)
    print("\n=== stability summary ===")
    print(df[["dataset", "model", "k_mean", "k_std", "jaccard_mean"]].to_string(index=False))
    print(f"\nsaved: {out}")
    print("Low k_std and high jaccard_mean => task-sufficient dimension is reproducible.")


if __name__ == "__main__":
    main()
