"""Controls (camera-ready), same LTT as Table III.
Run: python -m src.experiments.exp_sanity_v2  ->  results/tables/exp_sanity_v2.csv"""
import numpy as np, pandas as pd, os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from src.datasets import get_dataset, split_train_calib_test
from src.ltt import losses, fixed_sequence
from src.experiments._common import TABDIR, K, S, DELTA

SEEDS = [0, 1, 2]
LR = lambda: LogisticRegression(max_iter=5000)


def run(Xtr, ytr, Xca, yca, K_eff, scale=True):
    if scale:
        sc = StandardScaler().fit(Xtr); Xtr, Xca = sc.transform(Xtr), sc.transform(Xca)
    pca = PCA(n_components=K_eff, svd_solver="full").fit(Xtr)
    Ztr, Zca = pca.transform(Xtr), pca.transform(Xca)
    full = LR().fit(Ztr, ytr); lf = losses("classification", full.predict(Zca), yca)
    def dbar(k):
        m = full if k == K_eff else LR().fit(Ztr[:, :k], ytr)
        return float(np.mean(losses("classification", m.predict(Zca[:, :k]), yca) > lf))
    k_hat, _ = fixed_sequence(dbar, K_eff, S, DELTA, len(yca))
    return k_hat, 1 - lf.mean()


def planted(n=3000, d=50, m=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype("float32"); X[:, :m] *= 8.0
    w = rng.randn(m); z = X[:, :m] @ w
    return X, (z > np.median(z)).astype(int)


rows = []
for ds in ["mnist", "fashion"]:
    for seed in SEEDS:
        X, y, task = get_dataset(ds, seed=seed); K_eff = min(K, X.shape[1])
        Xtr, ytr, Xca, yca, _, _ = split_train_calib_test(X, y, task, seed=seed)
        rng = np.random.RandomState(seed)
        k_hat, acc = run(Xtr, rng.permutation(ytr), Xca, rng.permutation(yca), K_eff)
        rows.append(dict(case="no_signal", data=ds, seed=seed, K=K_eff, k_hat=k_hat, full_acc=round(acc, 3)))
        print(f"no_signal  {ds:8s} seed{seed} k_hat={k_hat:3d}/{K_eff} full_acc={acc:.3f}")
for seed in SEEDS:
    X, y = planted(seed=seed); idx = np.random.RandomState(seed).permutation(len(X))
    a, b = int(.6 * len(X)), int(.8 * len(X))
    k_hat, acc = run(X[idx[:a]], y[idx[:a]], X[idx[a:b]], y[idx[a:b]], 50, scale=False)
    rows.append(dict(case="planted_k3", data="synthetic", seed=seed, K=50, k_hat=k_hat, full_acc=round(acc, 3)))
    print(f"planted_k3 synth    seed{seed} k_hat={k_hat:3d}/50 full_acc={acc:.3f}")
df = pd.DataFrame(rows); df.to_csv(os.path.join(TABDIR, "exp_sanity_v2.csv"), index=False)
print(df.groupby(["case", "data"])[["k_hat", "full_acc"]].mean().round(3))
