"""
Regenerate ONLY fig_pc_signal.pdf at single-column size (colorful).
Run from project root: python -m src.experiments.make_pc_signal_fig
Does not touch any other figure.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE="#0072B2"; GREEN="#009E73"; VERM="#D55E00"
plt.rcParams.update({"font.family":"serif","axes.linewidth":0.8,
    "axes.grid":True,"grid.alpha":0.25,"grid.linewidth":0.4,
    "xtick.direction":"in","ytick.direction":"in"})

RES=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"results")
TAB=os.path.join(RES,"tables"); FIG=os.path.join(RES,"figures"); os.makedirs(FIG,exist_ok=True)

ps=pd.read_csv(os.path.join(TAB,"pc_signal.csv"))
dcolors={"madelon":VERM,"mnist":BLUE,"fashion":GREEN}

# single-column: 3 small panels side by side
fig,axes=plt.subplots(1,3,figsize=(3.5,1.7),sharey=True)
for ax,ds in zip(axes,["madelon","mnist","fashion"]):
    sub=ps[ps.dataset==ds]
    sizes=90*(sub["var_ratio"]/sub["var_ratio"].max())+3
    ax.scatter(sub["pc"],sub["uni_acc"],s=sizes,alpha=.55,color=dcolors[ds],
               edgecolor="black",linewidth=.25)
    ax.set_title(ds,color=dcolors[ds],fontsize=8)
    ax.set_xlabel("PC idx",fontsize=7)
    ax.tick_params(labelsize=6)
axes[0].set_ylabel("Uni. acc.",fontsize=7)
fig.suptitle("Predictive power vs PC index (size $\\propto$ var.)",fontsize=7.5,y=1.06)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG,"fig_pc_signal.pdf"),bbox_inches="tight")
plt.close(fig)
print("fig_pc_signal.pdf regenerated (single-column). Other figures untouched.")
