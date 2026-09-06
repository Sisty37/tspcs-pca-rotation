"""
Review issue #3: separate ROTATION from DIMENSIONALITY REDUCTION.
Run: python -m src.experiments.verify_truncation

We already know (verify_true_full_pca) that FULL PCA rotation alone produces the
gap. Here we vary the number of retained components to quantify how much the gap
grows as we ALSO truncate. This disentangles the two effects:
  - full PCA (all d)      -> pure rotation gap (baseline)
  - k < d components       -> rotation + truncation
For each dataset x model, we report the gap (raw - PCA_k) at several k.
Focus on RF (clearest tree effect) and logreg (rotation-invariant control).
"""
import numpy as np, pandas as pd, os, warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from src.datasets import get_dataset, split_train_calib_test

warnings.filterwarnings("ignore", category=ConvergenceWarning)
RES=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"results")
TAB=os.path.join(RES,"tables"); os.makedirs(TAB,exist_ok=True)

# (dataset, model) : RF (tree) and logreg/ridge (linear control)
CONFIGS=[("madelon","rf"),("madelon","logreg"),
         ("mnist","rf"),("mnist","logreg"),
         ("fashion","rf"),("fashion","logreg"),
         ("superconduct","rf"),("superconduct","ridge")]
# fractions of components to keep (relative to full)
KFRACS=[1.0, 0.5, 0.25, 0.10, 0.05]
SEEDS=[0,1,2,3,4]

def model_fn(task,name):
    if name=="rf":
        return (RandomForestClassifier(n_estimators=150,random_state=0,n_jobs=-1)
                if task=="classification" else
                RandomForestRegressor(n_estimators=150,random_state=0,n_jobs=-1))
    if task=="classification": return LogisticRegression(max_iter=5000)
    return Ridge()

def score(task,yt,yp):
    return accuracy_score(yt,yp) if task=="classification" else r2_score(yt,yp)

rows=[]
for ds,mdl in CONFIGS:
    for seed in SEEDS:
        X,y,task=get_dataset(ds,seed=seed)
        Xtr,ytr,Xca,yca,Xte,yte=split_train_calib_test(X,y,task,seed=seed)
        sc=StandardScaler().fit(Xtr); Xtr_s,Xca_s=sc.transform(Xtr),sc.transform(Xca)
        dmax=min(Xtr_s.shape[0]-1, Xtr_s.shape[1])
        raw=score(task,yca,model_fn(task,mdl).fit(Xtr_s,ytr).predict(Xca_s))
        pca_full=PCA(n_components=dmax,svd_solver="full").fit(Xtr_s)
        for frac in KFRACS:
            k=max(2,int(round(frac*dmax)))
            Ztr=pca_full.transform(Xtr_s)[:,:k]; Zca=pca_full.transform(Xca_s)[:,:k]
            pcperf=score(task,yca,model_fn(task,mdl).fit(Ztr,ytr).predict(Zca))
            rows.append({"dataset":ds,"model":mdl,"seed":seed,"dmax":dmax,
                         "k":k,"frac":frac,"raw":round(raw,4),
                         "pca_k":round(pcperf,4),"gap":round(raw-pcperf,4)})
        print(f"{ds:8s} {mdl:7s} seed{seed} done (dmax={dmax})")

df=pd.DataFrame(rows); df.to_csv(os.path.join(TAB,"verify_truncation.csv"),index=False)
print("\n=== gap by (dataset, model, frac) : mean over seeds ===")
piv=df.groupby(["dataset","model","frac"])["gap"].mean().round(4)
print(piv)
print("\nReading: at frac=1.0 gap is PURE ROTATION; as frac drops, extra gap is TRUNCATION.")
print("saved: results/tables/verify_truncation.csv")
