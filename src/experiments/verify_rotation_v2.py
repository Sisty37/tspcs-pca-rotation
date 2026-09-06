"""
Review issues #8 (more seeds) + #23-Exp4 (extra tree baseline).
Run: python -m src.experiments.verify_rotation_v2

Upgraded rotation experiment: 10 seeds and an added Extra Trees model
(another axis-aligned learner) plus the existing families. Uses TRUE FULL PCA
(all components) so the gap is a pure rotation effect, addressing review issue #1
in the same run. Reports mean gap with std by family and by model.
"""
import numpy as np, pandas as pd, os, warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score, r2_score
from src.datasets import get_dataset, split_train_calib_test

warnings.filterwarnings("ignore", category=ConvergenceWarning)
RES=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"results")
TAB=os.path.join(RES,"tables"); os.makedirs(TAB,exist_ok=True)

CLF_DATA=["madelon","mnist","fashion"]
CLF_MODELS=["logreg","svm","rf","extratrees","gbm"]
REG_MODELS=["ridge","linsvr","rf","extratrees","histgbm"]
SEEDS=list(range(10))   # 10 seeds
FAMILY={"logreg":"rotation-invariant","svm":"rotation-invariant","ridge":"rotation-invariant",
        "linsvr":"rotation-invariant","rf":"axis-aligned(tree)","extratrees":"axis-aligned(tree)",
        "gbm":"axis-aligned(tree)","histgbm":"axis-aligned(tree)"}

def clf_model(name):
    if name=="rf": return RandomForestClassifier(n_estimators=150,random_state=0,n_jobs=-1)
    if name=="extratrees": return ExtraTreesClassifier(n_estimators=150,random_state=0,n_jobs=-1)
    if name=="gbm": return HistGradientBoostingClassifier(random_state=0)
    if name=="svm": return LinearSVC(max_iter=50000,dual="auto")
    return LogisticRegression(max_iter=5000)

def reg_model(name):
    if name=="rf": return RandomForestRegressor(n_estimators=150,random_state=0,n_jobs=-1)
    if name=="extratrees": return ExtraTreesRegressor(n_estimators=150,random_state=0,n_jobs=-1)
    if name=="histgbm": return HistGradientBoostingRegressor(random_state=0)
    if name=="linsvr": return LinearSVR(max_iter=50000)
    return Ridge()

rows=[]
def run(ds, models, mk, task, score):
    for seed in SEEDS:
        X,y,_=get_dataset(ds,seed=seed)
        Xtr,ytr,Xca,yca,Xte,yte=split_train_calib_test(X,y,task,seed=seed)
        sc=StandardScaler().fit(Xtr); Xtr_s,Xca_s=sc.transform(Xtr),sc.transform(Xca)
        d=min(Xtr_s.shape[0]-1,Xtr_s.shape[1])
        pca=PCA(n_components=d,svd_solver="full").fit(Xtr_s)
        Ztr,Zca=pca.transform(Xtr_s),pca.transform(Xca_s)
        for m in models:
            raw=score(yca,mk(m).fit(Xtr_s,ytr).predict(Xca_s))
            fpc=score(yca,mk(m).fit(Ztr,ytr).predict(Zca))
            rows.append({"dataset":ds,"model":m,"family":FAMILY[m],"seed":seed,
                         "raw":round(raw,4),"full_pca":round(fpc,4),"gap":round(raw-fpc,4)})
        print(f"{ds:8s} seed{seed} done")

for ds in CLF_DATA:
    run(ds, CLF_MODELS, clf_model, "classification", accuracy_score)
run("superconduct", REG_MODELS, reg_model, "regression", r2_score)

df=pd.DataFrame(rows); df.to_csv(os.path.join(TAB,"verify_rotation_v2.csv"),index=False)
clf=df[df.dataset.isin(CLF_DATA)]; reg=df[df.dataset=="superconduct"]
print("\n=== FULL PCA, 10 seeds : classification gap by family ===")
print(clf.groupby("family")["gap"].agg(["mean","std"]).round(4))
print("\n=== classification gap by model (incl. Extra Trees) ===")
print(clf.groupby("model")["gap"].agg(["mean","std"]).round(4))
print("\n=== regression gap by family ===")
print(reg.groupby("family")["gap"].agg(["mean","std"]).round(4))
print("\nsaved: results/tables/verify_rotation_v2.csv")
