"""Two methodology diagrams as COLORFUL vector PDFs (colorblind-safe, print-friendly)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family":"serif","font.size":8.5})

# Okabe-Ito palette (soft tints for boxes)
BLUE_F="#cfe6f7"; ORANGE_F="#fbe2c0"; GREEN_F="#c9ecdf"; PURPLE_F="#efd6e6"
BLUE_E="#0072B2"; ORANGE_E="#E69F00"; GREEN_E="#009E73"; PURPLE_E="#CC79A7"; VERM_E="#D55E00"
WHITE="#ffffff"

def box(ax,x,y,w,h,text,fc=WHITE,ec="black",fontsize=8,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008,rounding_size=0.02",
                 linewidth=1.2,edgecolor=ec,facecolor=fc,zorder=2))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fontsize,zorder=3,
            fontweight="bold" if bold else "normal")

def arrow(ax,x1,y1,x2,y2,color="black"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,
                 linewidth=1.1,color=color,zorder=1))

# ===== Figure A: rotation-comparison protocol =====
figA,ax=plt.subplots(figsize=(3.3,3.1)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
box(ax,0.30,0.86,0.40,0.10,"Raw features $X$",fc="#e9e9e9",ec="black",bold=True)
box(ax,0.04,0.60,0.42,0.11,"Standardize\n(train-fit)",fc=BLUE_F,ec=BLUE_E)
box(ax,0.54,0.60,0.42,0.11,"Standardize\n+ full PCA\n(rotation, all $k$)",fc=ORANGE_F,ec=ORANGE_E)
arrow(ax,0.45,0.86,0.25,0.71,BLUE_E); arrow(ax,0.55,0.86,0.75,0.71,ORANGE_E)
box(ax,0.04,0.36,0.42,0.11,"Downstream\nmodel $\\to$ perf$_{\\mathrm{raw}}$",fc=BLUE_F,ec=BLUE_E)
box(ax,0.54,0.36,0.42,0.11,"Downstream\nmodel $\\to$ perf$_{\\mathrm{PCA}}$",fc=ORANGE_F,ec=ORANGE_E)
arrow(ax,0.25,0.60,0.25,0.47,BLUE_E); arrow(ax,0.75,0.60,0.75,0.47,ORANGE_E)
box(ax,0.20,0.10,0.60,0.12,"Gap $\\Delta=$ perf$_{\\mathrm{raw}}-$perf$_{\\mathrm{PCA}}$",fc=GREEN_F,ec=GREEN_E,bold=True)
arrow(ax,0.25,0.36,0.42,0.22,GREEN_E); arrow(ax,0.75,0.36,0.58,0.22,GREEN_E)
ax.text(0.5,0.02,"Same model both sides; full PCA is a pure rotation,\nso $\\Delta$ isolates the rotation effect.",
        ha="center",va="bottom",fontsize=6.8,style="italic")
figA.tight_layout(pad=0.2); figA.savefig("paper/fig_protocol.pdf",bbox_inches="tight"); print("fig_protocol.pdf")

# ===== Figure B: LTT selection flow =====
figB,ax=plt.subplots(figsize=(3.3,3.5)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
box(ax,0.16,0.88,0.68,0.09,"Train / Calibration / Test split",fc="#e9e9e9",ec="black",bold=True)
box(ax,0.06,0.71,0.88,0.10,"TRAIN: fit scaler + PCA;\ntrain model on each prefix $k$",fc=BLUE_F,ec=BLUE_E)
arrow(ax,0.5,0.88,0.5,0.81,BLUE_E)
box(ax,0.06,0.54,0.88,0.10,"CALIB: per-$k$ risk $\\hat R(k)$\n(misclass. / error rate)",fc=ORANGE_F,ec=ORANGE_E)
arrow(ax,0.5,0.71,0.5,0.64,ORANGE_E)
box(ax,0.06,0.37,0.88,0.10,"Bentkus $p$-value for $H_k:\\ R(k)>\\alpha$\n$\\alpha=R_{\\mathrm{full}}+$slack",fc=PURPLE_F,ec=PURPLE_E)
arrow(ax,0.5,0.54,0.5,0.47,PURPLE_E)
box(ax,0.06,0.20,0.88,0.10,"Fixed-sequence $k=K\\!\\to\\!1$;\ncertify while $p\\leq\\delta$",fc=PURPLE_F,ec=PURPLE_E)
arrow(ax,0.5,0.37,0.5,0.30,PURPLE_E)
box(ax,0.18,0.03,0.64,0.10,"Select $\\hat k$\n(guarantee: $R_{\\mathrm{test}}(\\hat k)\\leq\\alpha$)",fc=GREEN_F,ec=GREEN_E,bold=True)
arrow(ax,0.5,0.20,0.5,0.13,GREEN_E)
figB.tight_layout(pad=0.2); figB.savefig("paper/fig_ltt_flow.pdf",bbox_inches="tight"); print("fig_ltt_flow.pdf")
print("Both method diagrams now colorful (colorblind-safe).")
