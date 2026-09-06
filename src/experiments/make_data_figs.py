"""
Data figures for IEEE two-column format (COLORFUL, colorblind-safe Okabe-Ito).
Most figures are single-column width (~3.5in); the 3-panel pc-signal is full-width.
Fonts sized so text is readable at IEEE column scale.
Run: python -m src.experiments.make_data_figs
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.datasets import get_dataset, split_train_calib_test

BLUE="#0072B2"; ORANGE="#E69F00"; GREEN="#009E73"; VERM="#D55E00"; PURPLE="#CC79A7"; SKY="#56B4E9"; GREY="#888888"
# Fonts tuned for IEEE single-column (~3.5in wide) legibility
plt.rcParams.update({
    "font.family":"serif","font.size":10,"axes.linewidth":0.9,
    "axes.grid":True,"grid.alpha":0.25,"grid.linewidth":0.5,
    "xtick.direction":"in","ytick.direction":"in",
    "xtick.labelsize":9,"ytick.labelsize":9,"axes.titlesize":10.5,"axes.labelsize":10,
    "legend.fontsize":8.5,
    "axes.prop_cycle":plt.cycler(color=[BLUE,ORANGE,GREEN,VERM,PURPLE,SKY]),
})
RES=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"results")
TAB=os.path.join(RES,"tables"); FIG=os.path.join(RES,"figures"); os.makedirs(FIG,exist_ok=True)
def T(f): return pd.read_csv(os.path.join(TAB,f))
def save(fig,name): fig.tight_layout(pad=0.4); fig.savefig(os.path.join(FIG,name),bbox_inches="tight"); plt.close(fig); print(name)

# single-column canvas
SC=(3.5,2.7)

# ===== FIG: rotation gap bar (single-col) =====
clf=T("verify_rotation.csv"); reg=T("verify_rotation_reg.csv")
cf=clf.groupby("family")["gap"].mean(); rf_=reg.groupby("family")["gap"].mean()
fams=["axis-aligned(tree)","rotation-invariant"]; clf_v=[cf.get(f,0) for f in fams]; reg_v=[rf_.get(f,0) for f in fams]
fig,ax=plt.subplots(figsize=SC); x=np.arange(2); w=0.36
ax.bar(x-w/2,clf_v,w,label="Classification",color=BLUE,edgecolor="black",linewidth=.6)
ax.bar(x+w/2,reg_v,w,label="Regression",color=ORANGE,edgecolor="black",linewidth=.6)
ax.axhline(0,color="black",linewidth=.8)
ax.set_xticks(x); ax.set_xticklabels(["Axis-aligned\n(tree)","Rotation-\ninvariant"])
ax.set_ylabel("Rotation gap $\\Delta$")
ax.legend(frameon=False,ncol=2,loc="lower center",bbox_to_anchor=(0.5,1.01),handlelength=1.2,columnspacing=1.0)
save(fig,"fig_rotation_bar.pdf")

# ===== FIG: dose scatter (single-col) =====
d=T("exp_dose.csv").groupby("dataset").agg(conc=("concentration","mean"),gap=("rf_gap","mean")).reset_index()
r=np.corrcoef(d["conc"],d["gap"])[0,1]
colors={"madelon":VERM,"mnist":BLUE,"fashion":GREEN,"superconduct":PURPLE}
fig,ax=plt.subplots(figsize=SC)
m,b=np.polyfit(d["conc"],d["gap"],1); xs=np.linspace(d["conc"].min(),d["conc"].max(),50)
ax.plot(xs,m*xs+b,"--",color=GREY,linewidth=1.3,zorder=1,label=f"fit ($r={r:.2f}$)")
for _,row in d.iterrows():
    ax.scatter(row["conc"],row["gap"],s=90,color=colors.get(row["dataset"],BLUE),zorder=3,edgecolor="black",linewidth=.7)
    ax.annotate(row["dataset"],(row["conc"],row["gap"]),textcoords="offset points",xytext=(5,5),fontsize=8.5)
ax.set_xlabel("Signal concentration (Gini of MI)"); ax.set_ylabel("RF rotation gap $\\Delta$")
ax.legend(frameon=False,loc="upper left")
save(fig,"fig_dose.pdf")

# ===== FIG: sufficiency curve (single-col) =====
X,y,task=get_dataset("mnist",seed=0); K=100
Xtr,ytr,Xca,yca,Xte,yte=split_train_calib_test(X,y,task,seed=0)
sc=StandardScaler().fit(Xtr); pca=PCA(n_components=K,svd_solver="full").fit(sc.transform(Xtr))
Ztr=pca.transform(sc.transform(Xtr)); Zca=pca.transform(sc.transform(Xca))
accs=[accuracy_score(yca,LogisticRegression(max_iter=2000).fit(Ztr[:,:k],ytr).predict(Zca[:,:k])) for k in range(1,K+1)]
cumvar=np.cumsum(pca.explained_variance_ratio_)[:K]
ltt=T("exp_ltt.csv"); khat=int(ltt[(ltt.dataset=="mnist")&(ltt.model=="logreg")]["k_hat"].mean())
fig,ax=plt.subplots(figsize=SC); ks=np.arange(1,K+1)
ax.plot(ks,accs,"-",color=BLUE,linewidth=1.8,label="Accuracy")
ax.plot(ks,cumvar,"--",color=ORANGE,linewidth=1.6,label="Cum. variance")
ax.axvline(khat,color=VERM,linestyle=":",linewidth=1.8,label=f"$\\hat k={khat}$")
ax.set_xlabel("Number of components $k$"); ax.set_ylabel("Fraction")
ax.legend(frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(0.5,1.01),handlelength=1.1,columnspacing=0.8)
save(fig,"fig_sufficiency.pdf")

# ===== FIG: pc-signal (FULL-WIDTH, 3 panels) =====
ps=T("pc_signal.csv"); dcolors={"madelon":VERM,"mnist":BLUE,"fashion":GREEN}
fig,axes=plt.subplots(1,3,figsize=(7.0,2.4),sharey=True)
for ax,ds in zip(axes,["madelon","mnist","fashion"]):
    sub=ps[ps.dataset==ds]; sizes=260*(sub["var_ratio"]/sub["var_ratio"].max())+6
    ax.scatter(sub["pc"],sub["uni_acc"],s=sizes,alpha=.55,color=dcolors[ds],edgecolor="black",linewidth=.35)
    ax.set_title(ds,color=dcolors[ds],fontsize=11); ax.set_xlabel("PC index",fontsize=10)
    ax.tick_params(labelsize=9)
axes[0].set_ylabel("Univariate acc.",fontsize=10)
fig.suptitle("Predictive power vs PC index (dot size $\\propto$ variance)",fontsize=11,y=1.02)
fig.tight_layout(pad=0.4); fig.savefig(os.path.join(FIG,"fig_pc_signal.pdf"),bbox_inches="tight"); plt.close(fig); print("fig_pc_signal.pdf")

# ===== FIG: tradeoff (single-col) =====
td=T("tradeoff.csv").groupby(["dataset","slack"])["k_hat"].mean().reset_index()
fig,ax=plt.subplots(figsize=SC); style={"mnist":(BLUE,"o"),"fashion":(GREEN,"s")}
for ds in ["mnist","fashion"]:
    sub=td[td.dataset==ds]; col,mk=style[ds]
    ax.plot(sub["slack"],sub["k_hat"],marker=mk,color=col,linewidth=1.8,markersize=6,markeredgecolor="black",markeredgewidth=.5,label=ds)
ax.set_xlabel("Risk slack ($\\alpha - R_{\\mathrm{full}}$)"); ax.set_ylabel("Selected $\\hat k$")
ax.legend(frameon=False,ncol=2,loc="lower center",bbox_to_anchor=(0.5,1.01))
save(fig,"fig_tradeoff.pdf")
print("\nDone: 4 single-column + 1 full-width (pc_signal), IEEE-ready, colorful.")
