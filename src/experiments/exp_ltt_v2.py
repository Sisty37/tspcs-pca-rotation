"""Table III (camera-ready): LTT selection exactly as in the paper.
Run: python -m src.experiments.exp_ltt_v2   ->  results/tables/exp_ltt_v2.csv"""
import time
import numpy as np, pandas as pd, os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.datasets import get_dataset, split_train_calib_test
from src.ltt import regression_tau, losses, fixed_sequence
from src.experiments._common import TABDIR, K, S, DELTA, model_fn

CONFIGS = [("mnist", "logreg"), ("mnist", "rf"), ("fashion", "logreg"), ("superconduct", "ridge")]
SEEDS = [0, 1, 2, 3, 4]

rows = []
for ds, mdl in CONFIGS:
    for seed in SEEDS:
        X, y, task = get_dataset(ds, seed=seed); K_eff = min(K, X.shape[1])
        Xtr, ytr, Xca, yca, Xte, yte = split_train_calib_test(X, y, task, seed=seed)
        sc = StandardScaler().fit(Xtr)
        pca = PCA(n_components=K_eff, svd_solver="full").fit(sc.transform(Xtr))
        Ztr, Zca, Zte = (pca.transform(sc.transform(A)) for A in (Xtr, Xca, Xte))

        t0 = time.perf_counter()
        full = model_fn(task, mdl).fit(Ztr, ytr)
        tau = regression_tau(full, Ztr, ytr) if task == "regression" else None
        lf_ca = losses(task, full.predict(Zca), yca, tau)
        fitted = {}
        def dbar(k):
            fitted[k] = full if k == K_eff else model_fn(task, mdl).fit(Ztr[:, :k], ytr)
            return float(np.mean(losses(task, fitted[k].predict(Zca[:, :k]), yca, tau) > lf_ca))
        k_hat, n_eval = fixed_sequence(dbar, K_eff, S, DELTA, len(yca))
        secs = time.perf_counter() - t0

        lf_te = losses(task, full.predict(Zte), yte, tau)
        lk_te = losses(task, fitted[k_hat].predict(Zte[:, :k_hat]), yte, tau)
        d_te = float(np.mean(lk_te > lf_te))
        rows.append(dict(dataset=ds, model=mdl, seed=seed, K=K_eff, k_hat=k_hat,
                         n_cal=len(yca), k_evaluated=n_eval, seconds=round(secs, 2),
                         test_Dbar=round(d_te, 4), covered=d_te <= S,
                         test_risk_full=round(float(lf_te.mean()), 4),
                         test_risk_khat=round(float(lk_te.mean()), 4),
                         excess_risk=round(float(lk_te.mean() - lf_te.mean()), 4)))
        print(f"{ds:12s} {mdl:6s} seed{seed} k_hat={k_hat:3d}/{K_eff} test_Dbar={d_te:.3f} "
              f"excess={lk_te.mean()-lf_te.mean():+.3f} evaluated={n_eval} {secs:.1f}s")

df = pd.DataFrame(rows); df.to_csv(os.path.join(TABDIR, "exp_ltt_v2.csv"), index=False)
pd.set_option("display.width", 200)
print("\n=== Table III (camera-ready) ===")
print(df.groupby(["dataset", "model"]).agg(
    k_hat_mean=("k_hat", "mean"), k_hat_std=("k_hat", "std"),
    test_Dbar=("test_Dbar", "mean"), test_risk_khat=("test_risk_khat", "mean"),
    test_risk_full=("test_risk_full", "mean"), excess=("excess_risk", "mean"),
    coverage=("covered", "mean"), k_evaluated=("k_evaluated", "mean"),
    seconds=("seconds", "mean")).round(3))
