"""Stability of the task-sufficient dimension (camera-ready), using the paper's
additive definition: smallest k with R_cal(k) <= R_cal(full) + s, with the same
0/1 losses as LTT. Run: python -m src.experiments.exp_stab_v2"""
import numpy as np, pandas as pd, os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.datasets import get_dataset, split_train_calib_test
from src.ltt import regression_tau, losses
from src.experiments._common import TABDIR, K, S, model_fn

CONFIGS = [("mnist", "logreg"), ("mnist", "rf"), ("fashion", "logreg"),
           ("fashion", "rf"), ("superconduct", "ridge"), ("superconduct", "rf")]
SEEDS = [0, 1, 2, 3, 4]


def k_s(ds, mdl, seed):
    X, y, task = get_dataset(ds, seed=seed); K_eff = min(K, X.shape[1])
    Xtr, ytr, Xca, yca, _, _ = split_train_calib_test(X, y, task, seed=seed)
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=K_eff, svd_solver="full").fit(sc.transform(Xtr))
    Ztr, Zca = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xca))
    full = model_fn(task, mdl).fit(Ztr, ytr)
    tau = regression_tau(full, Ztr, ytr) if task == "regression" else None
    r_full = losses(task, full.predict(Zca), yca, tau).mean()
    for k in range(1, K_eff + 1):
        m = full if k == K_eff else model_fn(task, mdl).fit(Ztr[:, :k], ytr)
        if losses(task, m.predict(Zca[:, :k]), yca, tau).mean() <= r_full + S:
            return k
    return K_eff


rows = []
for ds, mdl in CONFIGS:
    ks = [k_s(ds, mdl, s) for s in SEEDS]
    jac = [min(a, b) / max(a, b) for i, a in enumerate(ks) for b in ks[i + 1:]]
    rows.append(dict(dataset=ds, model=mdl, k_mean=np.mean(ks), k_std=np.std(ks),
                     jaccard_mean=np.mean(jac), ks=ks))
    print(f"{ds:12s} {mdl:6s} k={ks} mean={np.mean(ks):.1f} std={np.std(ks):.2f} jaccard={np.mean(jac):.3f}")
pd.DataFrame(rows).round(3).to_csv(os.path.join(TABDIR, "exp_stab_v2.csv"), index=False)
