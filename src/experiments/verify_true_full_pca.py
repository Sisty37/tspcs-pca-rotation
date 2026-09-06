"""
CRITICAL FIX (review issue #1): TRUE full-dimensional PCA rotation experiment.
Run: python -m src.experiments.verify_true_full_pca

The earlier experiment used K=100 components for Madelon(500)/MNIST(784)/Fashion(784),
which is TRUNCATED PCA (dimensionality reduction), not a pure rotation. Here we
retain ALL components so that raw and PCA representations have identical dimension
and (up to numerical precision) identical information. Any gap is then attributable
to the coordinate rotation alone.

For each (dataset x model x seed):
  raw_perf      = model on standardized raw features (d dims)
  fullpca_perf  = model on ALL d principal components (pure rotation)
  gap           = raw_perf - fullpca_perf
Compare tree (axis-aligned) vs linear (rotation-invariant) families.
"""
import numpy as np, pandas as pd, os, warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score, r2_score
from src.datasets import get_dataset, split_train_calib_test

warnings.filterwarnings("ignore", category=ConvergenceWarning)
RES=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"results")
TAB=os.path.join(RES,"tables"); os.makedirs(TAB,exist_ok=True)

# classification datasets + their models; superconduct handled separately (regression)
CLF_DATA=["madelon","mnist","fashion"]
CLF_MODELS=["logreg","svm","rf","gbm"]
SEEDS=[0,1,2,3,4]
FAMILY={"logreg":"rotation-invariant","svm":"rotation-invariant",
        "rf":"axis-aligned(tree)","gbm":"axis-aligned(tree)",
        "ridge":"rotation-invariant","linsvr":"rotation-invariant",
        "histgbm":"axis-aligned(tree)"}

def clf_model(name):
    if name=="rf": return RandomForestClassifier(n_estimators=200,random_state=0,n_jobs=-1)
    if name=="gbm": return HistGradientBoostingClassifier(random_state=0)
    if name=="svm": return LinearSVC(max_iter=50000,dual="auto")
    return LogisticRegression(max_iter=5000)

def reg_model(name):
    if name=="rf": return RandomForestRegressor(n_estimators=200,random_state=0,n_jobs=-1)
    if name=="histgbm": return HistGradientBoostingRegressor(random_state=0)
    if name=="linsvr": return LinearSVR(max_iter=50000)
    return Ridge()

rows=[]

# ---- classification: FULL PCA (all d components) ----
for ds in CLF_DATA:
    for seed in SEEDS:
        X,y,task=get_dataset(ds,seed=seed)
        Xtr,ytr,Xca,yca,Xte,yte=split_train_calib_test(X,y,task,seed=seed)
        sc=StandardScaler().fit(Xtr); Xtr_s,Xca_s=sc.transform(Xtr),sc.transform(Xca)
        # FULL PCA: n_components = min(n_samples-1, n_features) -> keep all informative dims
        ncomp=min(Xtr_s.shape[0]-1, Xtr_s.shape[1])
        pca=PCA(n_components=ncomp,svd_solver="full").fit(Xtr_s)
        Ztr,Zca=pca.transform(Xtr_s),pca.transform(Xca_s)
        for m in CLF_MODELS:
            raw=accuracy_score(yca,clf_model(m).fit(Xtr_s,ytr).predict(Xca_s))
            fpc=accuracy_score(yca,clf_model(m).fit(Ztr,ytr).predict(Zca))
            rows.append({"dataset":ds,"model":m,"family":FAMILY[m],"seed":seed,
                         "d":Xtr_s.shape[1],"pca_dim":ncomp,
                         "raw":round(raw,4),"full_pca":round(fpc,4),"gap":round(raw-fpc,4)})
            print(f"{ds:8s} {m:7s} seed{seed}  d={Xtr_s.shape[1]} pca={ncomp}  gap={raw-fpc:+.4f}")

# ---- regression: Superconductivity (already full at 81) ----
for seed in SEEDS:
    X,y,task=get_dataset("superconduct",seed=seed)
    Xtr,ytr,Xca,yca,Xte,yte=split_train_calib_test(X,y,task,seed=seed)
    sc=StandardScaler().fit(Xtr); Xtr_s,Xca_s=sc.transform(Xtr),sc.transform(Xca)
    ncomp=min(Xtr_s.shape[0]-1, Xtr_s.shape[1])
    pca=PCA(n_components=ncomp,svd_solver="full").fit(Xtr_s)
    Ztr,Zca=pca.transform(Xtr_s),pca.transform(Xca_s)
    for m in ["ridge","linsvr","rf","histgbm"]:
        raw=r2_score(yca,reg_model(m).fit(Xtr_s,ytr).predict(Xca_s))
        fpc=r2_score(yca,reg_model(m).fit(Ztr,ytr).predict(Zca))
        rows.append({"dataset":"superconduct","model":m,"family":FAMILY[m],"seed":seed,
                     "d":Xtr_s.shape[1],"pca_dim":ncomp,
                     "raw":round(raw,4),"full_pca":round(fpc,4),"gap":round(raw-fpc,4)})
        print(f"supercon {m:7s} seed{seed}  d={Xtr_s.shape[1]} pca={ncomp}  gap={raw-fpc:+.4f}")

df=pd.DataFrame(rows); df.to_csv(os.path.join(TAB,"verify_true_full_pca.csv"),index=False)
print("\n=== TRUE FULL PCA: mean gap by family (classification) ===")
clf=df[df.dataset.isin(CLF_DATA)]
print(clf.groupby("family")["gap"].agg(["mean","std"]).round(4))
print("\n=== TRUE FULL PCA: mean gap by family (regression) ===")
reg=df[df.dataset=="superconduct"]
print(reg.groupby("family")["gap"].agg(["mean","std"]).round(4))
print("\n=== by dataset x model ===")
print(df.groupby(["dataset","model"])["gap"].mean().round(4))
print("\nsaved: results/tables/verify_true_full_pca.csv")
print("KEY: if tree gap stays >0 and linear ~0 with FULL PCA, the rotation claim holds cleanly.")
