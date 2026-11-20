"""Camera-ready data figures with larger, clearer labels (Reviewer 1).
Needs: verify_rotation_v2.csv, exp_dose.csv, exp_synth_concentration.csv, exp_ltt_v2.csv
Run: python -m src.experiments.make_figs_camera_ready  -> results/figures/*.pdf"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.datasets import get_dataset, split_train_calib_test
from src.experiments._common import TABDIR, FIGDIR

BLUE, ORANGE, GREEN, VERM, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.labelsize": 12,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
                     "axes.titlesize": 11.5, "axes.linewidth": 0.9, "axes.grid": True,
                     "grid.alpha": 0.25, "xtick.direction": "in", "ytick.direction": "in"})
SC = (3.5, 2.6)
T = lambda f: pd.read_csv(os.path.join(TABDIR, f))
def save(fig, name):
    fig.tight_layout(pad=0.3); fig.savefig(os.path.join(FIGDIR, name), bbox_inches="tight")
    plt.close(fig); print(name)

# Fig 3: rotation gap by model (classification, 10 seeds, full PCA)
v2 = T("verify_rotation_v2.csv"); clf = v2[v2.dataset.isin(["madelon", "mnist", "fashion"])]
order = ["logreg", "svm", "gbm", "rf", "extratrees"]
lab = {"logreg": "LogReg", "svm": "SVM", "gbm": "GBM", "rf": "RF", "extratrees": "ExtraT."}
col = {"logreg": BLUE, "svm": BLUE, "gbm": ORANGE, "rf": VERM, "extratrees": PURPLE}
mu, sd = clf.groupby("model")["gap"].mean(), clf.groupby("model")["gap"].std()
fig, ax = plt.subplots(figsize=SC)
ax.bar(range(5), [mu[m] for m in order], yerr=[sd[m] for m in order], capsize=3,
       color=[col[m] for m in order], edgecolor="black", linewidth=.6)
ax.axhline(0, color="black", lw=.8)
ax.set_xticks(range(5)); ax.set_xticklabels([lab[m] for m in order], fontsize=10)
ax.set_ylabel(r"Rotation gap $\Delta$ (accuracy)")
ax.legend(handles=[Patch(fc=BLUE, ec="black", label="Rotation-invariant"),
                   Patch(fc=VERM, ec="black", label="Axis-aligned (tree)")],
          frameon=False, loc="upper left")
save(fig, "fig_rotation_bar.pdf")

# Fig 4: gap vs signal concentration (real datasets)
d = T("exp_dose.csv").groupby("dataset")[["concentration", "rf_gap"]].mean().reset_index()
names = {"madelon": "Madelon", "mnist": "MNIST", "fashion": "Fashion", "superconduct": "Supercond."}
fig, ax = plt.subplots(figsize=SC)
for _, r in d.iterrows():
    ax.scatter(r.concentration, r.rf_gap, s=80, color=VERM, edgecolor="black", zorder=3)
    ax.annotate(names[r.dataset], (r.concentration, r.rf_gap), xytext=(6, -3),
                textcoords="offset points", fontsize=10)
ax.set_xlabel("Signal concentration (Gini of MI)"); ax.set_ylabel(r"RF rotation gap $\Delta$")
ax.set_xlim(0, 0.95)
save(fig, "fig_dose.pdf")

# Fig 5: controlled synthetic concentration
a = T("exp_synth_concentration.csv").groupby("n_informative")["gap"].agg(["mean", "std"]).reset_index()
fig, ax = plt.subplots(figsize=SC)
ax.errorbar(a.n_informative, a["mean"], yerr=a["std"], marker="o", color=BLUE,
            mec="black", lw=1.8, ms=6, capsize=3)
ax.set_xscale("log"); ax.set_xlabel("Informative directions $m$ (log scale)")
ax.set_ylabel(r"RF rotation gap $\Delta$")
save(fig, "fig_synth_concentration.pdf")

# Fig 6: task sufficiency on MNIST with LTT-selected dimension (seed 0)
X, y, task = get_dataset("mnist", seed=0); K = 100
Xtr, ytr, Xca, yca, _, _ = split_train_calib_test(X, y, task, seed=0)
sc = StandardScaler().fit(Xtr); pca = PCA(n_components=K, svd_solver="full").fit(sc.transform(Xtr))
Ztr, Zca = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xca))
acc = [accuracy_score(yca, LogisticRegression(max_iter=5000).fit(Ztr[:, :k], ytr).predict(Zca[:, :k]))
       for k in range(1, K + 1)]
ltt = T("exp_ltt_v2.csv")
khat = int(ltt[(ltt.dataset == "mnist") & (ltt.model == "logreg") & (ltt.seed == 0)]["k_hat"].iloc[0])
ks = np.arange(1, K + 1)
fig, ax = plt.subplots(figsize=SC)
ax.plot(ks, acc, color=BLUE, lw=1.8, label="Accuracy")
ax.plot(ks, np.cumsum(pca.explained_variance_ratio_)[:K], "--", color=ORANGE, lw=1.6, label="Cum. variance")
ax.axvline(khat, color=VERM, ls=":", lw=2, label=rf"LTT $\hat k={khat}$")
ax.set_xlabel("Number of components $k$"); ax.set_ylabel("Fraction"); ax.set_ylim(0, 1.02)
ax.legend(frameon=False, loc="lower right")
save(fig, "fig_sufficiency.pdf")
