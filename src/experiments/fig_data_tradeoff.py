"""
Data for BONUS FIG 2: compression vs guarantee trade-off.
Run: python -m src.experiments.fig_data_tradeoff
Saves results/tables/tradeoff.csv  (selected k_hat as slack varies)
"""
import numpy as np, pandas as pd, os, warnings
from scipy.stats import binom
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from src.datasets import get_dataset, split_train_calib_test
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
TAB = os.path.join(RES, "tables"); os.makedirs(TAB, exist_ok=True)

K = 100; DELTA = 0.05
SLACKS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
CONFIGS = [("mnist","logreg"), ("fashion","logreg")]
SEEDS = [0,1,2]

def bentkus(Rhat, alpha, n):
    if Rhat >= alpha: return 1.0
    return float(min(1.0, np.e*binom.cdf(int(np.ceil(n*Rhat)), n, alpha)))

def select_k(R, alpha, n):
    kh = len(R)
    for k in range(len(R),0,-1):
        if bentkus(R[k-1], alpha, n) <= DELTA: kh = k
        else: break
    return kh

rows=[]
for ds,mdl in CONFIGS:
    for seed in SEEDS:
        X,y,task = get_dataset(ds, seed=seed); K_eff=min(K,X.shape[1])
        Xtr,ytr,Xca,yca,Xte,yte = split_train_calib_test(X,y,task,seed=seed)
        sc=StandardScaler().fit(Xtr); pca=PCA(n_components=K_eff,svd_solver="full").fit(sc.transform(Xtr))
        Ztr=pca.transform(sc.transform(Xtr)); Zca=pca.transform(sc.transform(Xca))
        R=np.array([np.mean(LogisticRegression(max_iter=2000).fit(Ztr[:,:k],ytr).predict(Zca[:,:k])!=yca)
                    for k in range(1,K_eff+1)])
        Rfull=R[-1]; n=len(yca)
        for s in SLACKS:
            kh=select_k(R, Rfull+s, n)
            rows.append({"dataset":ds,"model":mdl,"seed":seed,"slack":s,"k_hat":kh})
        print(f"{ds} {mdl} seed{seed} done")

df=pd.DataFrame(rows)
df.to_csv(os.path.join(TAB,"tradeoff.csv"), index=False)
print("saved results/tables/tradeoff.csv")
print(df.groupby(["dataset","slack"])["k_hat"].mean().round(1))
