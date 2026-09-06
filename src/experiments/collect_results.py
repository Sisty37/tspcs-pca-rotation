"""
Collect all experiment results into ONE file for paper writing.
Run from project root:  python -m src.experiments.collect_results
Writes results/PAPER_RESULTS.txt and prints it.
"""
import os
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
TAB = os.path.join(RESULTS, "tables")
OUT = os.path.join(RESULTS, "PAPER_RESULTS.txt")

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

lines = []
def show(title, fn):
    lines.append("=" * 70)
    lines.append(title)
    lines.append("=" * 70)
    path = os.path.join(TAB, fn)
    if not os.path.exists(path):
        lines.append(f"(missing: {fn})\n")
        return
    df = pd.read_csv(path)
    lines.append(fn)
    lines.append(df.to_string(index=False))
    lines.append("")

# E-rot classification: gap by family
show("E-rot-clf : rotation gap, classification (raw rows)", "verify_rotation.csv")
# E-rot regression
show("E-rot-reg : rotation gap, regression (raw rows)", "verify_rotation_reg.csv")
# E-dose
show("E-dose : gap vs signal concentration", "exp_dose.csv")
# E-stab
show("E-stab : stability of task-sufficient k", "exp_stab.csv")
# E-ltt
show("E-ltt : guaranteed selection (Learn-Then-Test)", "exp_ltt.csv")
# E-sanity
show("E-sanity : no-signal + planted controls", "exp_sanity.csv")

# ---- computed summaries the paper actually quotes ----
lines.append("#" * 70)
lines.append("# KEY SUMMARY NUMBERS FOR THE PAPER")
lines.append("#" * 70)

try:
    r = pd.read_csv(os.path.join(TAB, "verify_rotation.csv"))
    fam = r.groupby("family")["gap"].agg(["mean", "std"]).round(4)
    lines.append("\n[clf] rotation gap by family:")
    lines.append(fam.to_string())
except Exception as e:
    lines.append(f"clf family summary failed: {e}")

try:
    rr = pd.read_csv(os.path.join(TAB, "verify_rotation_reg.csv"))
    famr = rr.groupby("family")["gap"].agg(["mean", "std"]).round(4)
    lines.append("\n[reg] rotation gap by family:")
    lines.append(famr.to_string())
except Exception as e:
    lines.append(f"reg family summary failed: {e}")

try:
    d = pd.read_csv(os.path.join(TAB, "exp_dose.csv"))
    agg = d.groupby("dataset").agg(concentration=("concentration","mean"),
                                   rf_gap=("rf_gap","mean")).round(4)
    import numpy as np
    rho = np.corrcoef(agg["concentration"], agg["rf_gap"])[0,1]
    lines.append("\n[dose] per-dataset means + Pearson r:")
    lines.append(agg.to_string())
    lines.append(f"Pearson r = {rho:.3f}")
except Exception as e:
    lines.append(f"dose summary failed: {e}")

try:
    l = pd.read_csv(os.path.join(TAB, "exp_ltt.csv"))
    g = l.groupby(["dataset","model"]).agg(
        k_hat_mean=("k_hat","mean"), mean_test_risk=("test_risk","mean"),
        coverage=("within_alpha","mean")).round(3)
    lines.append("\n[ltt] guaranteed selection summary:")
    lines.append(g.to_string())
except Exception as e:
    lines.append(f"ltt summary failed: {e}")

text = "\n".join(lines)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(text)
print(text)
print(f"\n\n>>> saved everything to: {OUT}")
