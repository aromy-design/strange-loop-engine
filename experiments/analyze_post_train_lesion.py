"""
E7 post-training lesion analysis.

Reads results/post_train_lesion/results.csv and reports:
  - Post-training eats rate per condition
  - FULL vs LESION_MB t-test + Cohen's d
  - Bar chart: pre vs post by condition
"""
import csv
from pathlib import Path
import numpy as np
try:
    from scipy import stats as sst
    import math
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INDIR = Path(__file__).parent / "results" / "post_train_lesion"
PLOT = Path(__file__).parent / "plots"
PLOT.mkdir(parents=True, exist_ok=True)


def load():
    fpath = INDIR / "results.csv"
    rows = []
    with open(fpath, newline="") as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                try: r[k] = float(v)
                except: pass
            rows.append(r)
    return rows


def cohen_d(a, b):
    n1, n2 = len(a), len(b)
    var1 = np.var(a, ddof=1) if n1 > 1 else 0
    var2 = np.var(b, ddof=1) if n2 > 1 else 0
    sp = math.sqrt((var1 + var2) / 2.0) if (var1 + var2) > 0 else 1e-9
    return (np.mean(a) - np.mean(b)) / sp


def main():
    rows = load()
    conditions = sorted(set(r["lesion"] for r in rows))
    out_lines = []
    out_lines.append("=" * 72)
    out_lines.append("E7 POST-TRAINING LESION ANALYSIS")
    out_lines.append("=" * 72)
    n_seeds = len([r for r in rows if r["lesion"] == conditions[0]])
    out_lines.append(f"Seeds per condition: {n_seeds}  |  Conditions: {conditions}")
    out_lines.append(f"Train steps: {int(rows[0]['train_steps'])}  |  Test steps: {int(rows[0]['test_steps'])}")
    out_lines.append("")

    post_rates = {}
    pre_rates = {}
    for cond in conditions:
        subset = [r for r in rows if r["lesion"] == cond]
        post = [r["post_eats_rate"] for r in subset]
        pre = [r["pre_eats_rate"] for r in subset]
        post_rates[cond] = post
        pre_rates[cond] = pre
        mb_before = np.mean([r["mb_w_before"] for r in subset])
        mb_after = np.mean([r["mb_w_after"] for r in subset])
        out_lines.append(f"  {cond:15s}: pre={np.mean(pre)*100:.2f}%±{np.std(pre,ddof=1)*100:.2f}%  "
                         f"post={np.mean(post)*100:.2f}%±{np.std(post,ddof=1)*100:.2f}%  "
                         f"mb_w {mb_before:.4f}->{mb_after:.4f}")

    out_lines.append("")
    out_lines.append("=== KEY COMPARISON: FULL vs LESION_MB (post-training phase) ===")
    if "FULL" in post_rates and "LESION_MB" in post_rates:
        full = np.array(post_rates["FULL"])
        lesion = np.array(post_rates["LESION_MB"])
        diff_pct = (full.mean() - lesion.mean()) / max(lesion.mean(), 1e-9) * 100
        if HAS_SCIPY:
            t, p = sst.ttest_ind(full, lesion)
            d = cohen_d(full.tolist(), lesion.tolist())
            out_lines.append(f"  FULL:      {full.mean()*100:.2f}% ± {full.std(ddof=1)*100:.2f}%")
            out_lines.append(f"  LESION_MB: {lesion.mean()*100:.2f}% ± {lesion.std(ddof=1)*100:.2f}%")
            out_lines.append(f"  Difference: FULL is {diff_pct:+.1f}% higher")
            out_lines.append(f"  t = {t:.3f}  p = {p:.4f}  Cohen's d = {d:.3f}")
            if p < 0.05 and diff_pct > 0:
                out_lines.append("  >>> SIGNIFICANT: trained MB causally drives behavior! ⭐⭐⭐")
            elif p < 0.05 and diff_pct < 0:
                out_lines.append("  >>> SURPRISING: LESION_MB outperforms FULL — MB interferes?")
            else:
                out_lines.append("  >>> NULL: MB weight growth doesn't cause behavioral advantage")

    out_lines.append("")
    out_lines.append("=== ALL PAIRWISE vs FULL (t-test) ===")
    if HAS_SCIPY and "FULL" in post_rates:
        full = np.array(post_rates["FULL"])
        for cond in conditions:
            if cond == "FULL": continue
            other = np.array(post_rates[cond])
            t, p = sst.ttest_ind(full, other)
            d = cohen_d(full.tolist(), other.tolist())
            delta = (full.mean() - other.mean()) / max(other.mean(), 1e-9) * 100
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            out_lines.append(f"  FULL vs {cond:15s}: delta={delta:+.1f}%  t={t:.3f}  p={p:.4f}  d={d:.3f}  {sig}")

    text = "\n".join(out_lines)
    print(text)
    out_file = INDIR / "analysis.txt"
    with open(out_file, "w", encoding="utf-8") as f: f.write(text)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = {"FULL": "#3affc8", "LESION_MB": "#ffd47a", "LESION_CX": "#a87aff", "LESION_ALL": "#ff7a7a"}
    conds = [c for c in ["FULL", "LESION_MB", "LESION_CX", "LESION_ALL"] if c in post_rates]

    ax = axes[0]
    x = np.arange(len(conds))
    pre_m = [np.mean(pre_rates[c]) * 100 for c in conds]
    post_m = [np.mean(post_rates[c]) * 100 for c in conds]
    pre_e = [np.std(pre_rates[c], ddof=1) * 100 for c in conds]
    post_e = [np.std(post_rates[c], ddof=1) * 100 for c in conds]
    ax.bar(x - 0.2, pre_m, 0.35, yerr=pre_e, color=[colors.get(c, "#aaa") for c in conds],
           alpha=0.4, label="pre-lesion (train phase)")
    ax.bar(x + 0.2, post_m, 0.35, yerr=post_e, color=[colors.get(c, "#aaa") for c in conds],
           alpha=1.0, label="post-lesion (test phase)")
    ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=8)
    ax.set_ylabel("Eats rate (%)")
    ax.set_title("Pre vs Post eats rate by condition")
    ax.legend(fontsize=8)
    ax.grid(linestyle=":", alpha=0.3)

    ax = axes[1]
    if HAS_SCIPY and "FULL" in post_rates:
        full = np.array(post_rates["FULL"])
        for i, cond in enumerate(conds):
            other = np.array(post_rates[cond])
            for v in other:
                ax.scatter(i, v * 100, color=colors.get(cond, "#aaa"), alpha=0.5, s=40)
            ax.bar(i, other.mean() * 100, 0.5, color=colors.get(cond, "#aaa"), alpha=0.3)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, fontsize=8)
    ax.set_ylabel("Post-test eats rate (%)")
    ax.set_title("Distribution of post-test performance")
    ax.grid(linestyle=":", alpha=0.3)

    fig.suptitle(f"E7 Post-training lesion (n={n_seeds} seeds/condition)", fontweight="bold")
    fig.tight_layout()
    out_png = PLOT / "post_train_lesion.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out_png}")
    return text


if __name__ == "__main__":
    main()
