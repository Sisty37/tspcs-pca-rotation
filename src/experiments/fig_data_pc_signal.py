"""
Data for BONUS FIG 1: where does predictive signal sit across PCs?
Run: python -m src.experiments.fig_data_pc_signal
Saves results/tables/pc_signal.csv  (per-PC: variance ratio + univariate predictive power)
"""
import numpy as np, pandas as pd, os, warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.datasets import get_dataset, split_train_calib_test
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
TAB = os.path.join(RES, "tables"); os.makedirs(TAB, exist_ok=True)

DATASETS = ["madelon", "mnist", "fashion"]
K = 100
SEED = 0

rows = []
for ds in DATASETS:
    X, y, task = get_dataset(ds, seed=SEED)
    K_eff = min(K, X.shape[1])
    Xtr, ytr, Xca, yca, Xte, yte = split_train_calib_test(X, y, task, seed=SEED)
    scaler = StandardScaler().fit(Xtr)
    pca = PCA(n_components=K_eff, svd_solver="full").fit(scaler.transform(Xtr))
    Ztr = pca.transform(scaler.transform(Xtr)); Zca = pca.transform(scaler.transform(Xca))
    for j in range(K_eff):
        acc = accuracy_score(yca, LogisticRegression(max_iter=500)
                             .fit(Ztr[:, [j]], ytr).predict(Zca[:, [j]]))
        rows.append({"dataset": ds, "pc": j+1,
                     "var_ratio": float(pca.explained_variance_ratio_[j]),
                     "uni_acc": float(acc)})
    print(f"{ds}: done ({K_eff} PCs)")

pd.DataFrame(rows).to_csv(os.path.join(TAB, "pc_signal.csv"), index=False)
print("saved results/tables/pc_signal.csv")
