"""
Regenerate the two figures affected by the new experiments:
  1. fig_rotation_bar.pdf  -> now from verify_rotation_v2.csv (10 seeds, full PCA),
     shown BY MODEL so Extra Trees is visible (tree family) alongside linear.
  2. fig_synth_concentration.pdf -> polished version from exp_synth_concentration.csv.
Run from project root: python -m src.experiments.update_figs_v2
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE="#0072B2"; ORANGE="#E69F00"; GREEN="#009E73"; VERM="#D55E00"; PURPLE="#CC79A7"; GREY="#888888"
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.linewidth":0.9,
    "axes.grid":True,"grid.alpha":0.25,"grid.linewidth":0.5,
    "xtick.direction":"in","ytick.direction":"in",
    "xtick.labelsize":9,"ytick.labelsize":9,"axes.titlesize":10.5,"axes.labelsize":10,"legend.fontsize":8.5})
RES=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"results")
TAB=os.path.join(RES,"tables"); FIG=os.path.join(RES,"figures"); os.makedirs(FIG,exist_ok=True)
def T(f): return pd.read_csv(os.path.join(TAB,f))

# ========== FIG rotation bar (by model, v2 data) ==========
v2=T("verify_rotation_v2.csv")
clf=v2[v2.dataset.isin(["madelon","mnist","fashion"])]
# per-model mean gap (classification)
order=["logreg","svm","gbm","rf","extratrees"]
labels={"logreg":"LogReg","svm":"SVM","gbm":"GBM","rf":"RF","extratrees":"ExtraTrees"}
famcol={"logreg":BLUE,"svm":BLUE,"gbm":ORANGE,"rf":VERM,"extratrees":PURPLE}
means=clf.groupby("model")["gap"].mean(); stds=clf.groupby("model")["gap"].std()
fig,ax=plt.subplots(figsize=(3.5,2.7))
xs=np.arange(len(order))
bars=ax.bar(xs,[means[m] for m in order],yerr=[stds[m] for m in order],
            color=[famcol[m] for m in order],edgecolor="black",linewidth=.6,capsize=3)
ax.axhline(0,color="black",linewidth=.8)
ax.set_xticks(xs); ax.set_xticklabels([labels[m] for m in order],rotation=20,ha="right")
ax.set_ylabel("Rotation gap $\\Delta$")
# legend for families
from matplotlib.patches import Patch
leg=[Patch(facecolor=BLUE,edgecolor="black",label="Rotation-invariant"),
     Patch(facecolor=VERM,edgecolor="black",label="Axis-aligned (tree)")]
ax.legend(handles=leg,frameon=False,loc="upper left",fontsize=8)
ax.set_title("Rotation gap by model (10 seeds, full PCA)")
fig.tight_layout(pad=0.4); fig.savefig(os.path.join(FIG,"fig_rotation_bar.pdf"),bbox_inches="tight"); plt.close(fig)
print("fig_rotation_bar.pdf  (by model, Extra Trees included)")

# ========== FIG synthetic concentration (polished) ==========
sc=T("exp_synth_concentration.csv")
agg=sc.groupby("n_informative")["gap"].agg(["mean","std"]).reset_index()
fig,ax=plt.subplots(figsize=(3.5,2.7))
ax.errorbar(agg["n_informative"],agg["mean"],yerr=agg["std"],marker="o",
            color=BLUE,markeredgecolor="black",markeredgewidth=.6,linewidth=1.8,
            markersize=6,capsize=3)
ax.set_xscale("log")
ax.set_xlabel("Number of informative dimensions (log)")
ax.set_ylabel("RF rotation gap $\\Delta$")
ax.set_title("Controlled signal concentration")
fig.tight_layout(pad=0.4); fig.savefig(os.path.join(FIG,"fig_synth_concentration.pdf"),bbox_inches="tight"); plt.close(fig)
print("fig_synth_concentration.pdf  (polished)")
print("\nBoth figures regenerated. Copy them into your paper/ folder.")
