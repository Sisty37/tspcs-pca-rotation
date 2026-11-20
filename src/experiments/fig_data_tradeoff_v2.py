"""Compression vs tolerance s (camera-ready), same LTT as Table III.
Run: python -m src.experiments.fig_data_tradeoff_v2  ->  results/tables/tradeoff_v2.csv"""
import numpy as np, pandas as pd, os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from src.datasets import get_dataset, split_train_calib_test
from src.ltt import losses, fixed_sequence
from src.experiments._common import TABDIR, K, DELTA

SLACKS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
rows = []
for ds in ["mnist", "fashion"]:
    for seed in [0, 1, 2]:
        X, y, task = get_dataset(ds, seed=seed)
        Xtr, ytr, Xca, yca, _, _ = split_train_calib_test(X, y, task, seed=seed)
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=K, svd_solver="full").fit(sc.transform(Xtr))
        Ztr, Zca = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xca))
        preds = {k: LogisticRegression(max_iter=5000).fit(Ztr[:, :k], ytr).predict(Zca[:, :k])
                 for k in range(1, K + 1)}
        lf = losses(task, preds[K], yca)
        D = {k: float(np.mean(losses(task, preds[k], yca) > lf)) for k in preds}
        for s in SLACKS:
            k_hat, _ = fixed_sequence(lambda k: D[k], K, s, DELTA, len(yca))
            rows.append(dict(dataset=ds, seed=seed, s=s, k_hat=k_hat))
        print(f"{ds} seed{seed} done")
df = pd.DataFrame(rows); df.to_csv(os.path.join(TABDIR, "tradeoff_v2.csv"), index=False)
print(df.groupby(["dataset", "s"])["k_hat"].mean().round(1))
