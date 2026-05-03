"""
E9 MB necessity analysis.
Key question: does MB-forced training outperform random walk?
"""
import csv, math
from pathlib import Path
import numpy as np
try:
    from scipy import stats as sst
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INDIR = Path(__file__).parent / "results" / "e9_mb_necessity"
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
    v1 = np.var(a, ddof=1) if n1 > 1 else 0.0
    v2 = np.var(b, ddof=1) if n2 > 1 else 0.0
    sp = math.sqrt((v1 + v2) / 2) if (v1 + v2) > 0 else 1e-9
    return (np.mean(a) - np.mean(b)) / sp


def main():
    rows = load()
    conditions = sorted(set(r["condition"] for r in rows))
    out = ["=" * 72, "E9 MB NECESSITY ANALYSIS", "=" * 72]
    n_seeds = len([r for r in rows if r["condition"] == conditions[0]])
    out.append(f"Seeds per condition: {n_seeds}")
    out.append(f"Train steps: {int(rows[0]['train_steps'])}  Test steps: {int(rows[0]['test_steps'])}")
    out.append("")

    post_rates = {}
    pre_rates = {}
    for cond in conditions:
        sub = [r for r in rows if r["condition"] == cond]
        post = [r["post_eats_rate"] * 100 for r in sub]
        pre  = [r["pre_eats_rate"] * 100  for r in sub]
        post_rates[cond] = post
        pre_rates[cond] = pre
        mb_w = np.mean([r["mb_w_before"] for r in sub])
        out.append(f"  {cond:30s}: pre={np.mean(pre):.2f}%  "
                   f"post={np.mean(post):.2f}%±{np.std(post,ddof=1):.2f}%  mb_w={mb_w:.4f}")

    out += ["", "=== KEY COMPARISONS ==="]
    comparisons = [
        ("TRAIN_LESION_TEST_LESION", "NAIVE_TEST_LESION",
         "MB-forced training vs pure random walk — THE KEY TEST"),
        ("TRAIN_FULL_TEST_LESION",   "NAIVE_TEST_LESION",
         "Shortcuts-training → test without shortcuts (passive MB transfer)"),
        ("TRAIN_LESION_TEST_LESION", "TRAIN_FULL_TEST_LESION",
         "Forced-MB training vs passive MB (both tested without shortcuts)"),
        ("TRAIN_FULL_TEST_FULL",     "NAIVE_TEST_FULL",
         "Training benefit with shortcuts intact (sanity)"),
    ]
    for ca, cb, label in comparisons:
        if ca not in post_rates or cb not in post_rates: continue
        a = post_rates[ca]; b = post_rates[cb]
        out.append(f"\n  {label}")
        out.append(f"    {ca}: {np.mean(a):.2f}%±{np.std(a,ddof=1):.2f}%")
        out.append(f"    {cb}: {np.mean(b):.2f}%±{np.std(b,ddof=1):.2f}%")
        if HAS_SCIPY:
            t, p = sst.ttest_ind(a, b)
            d = cohen_d(a, b)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            out.append(f"    delta={np.mean(a)-np.mean(b):+.3f}%  t={t:.3f}  p={p:.4f}  d={d:.3f}  {sig}")
            if ca == "TRAIN_LESION_TEST_LESION" and cb == "NAIVE_TEST_LESION":
                if p < 0.05 and np.mean(a) > np.mean(b):
                    out.append("    >>> SIGNIFICANT: MB CAN learn goal-directed behavior! ⭐⭐⭐")
                elif p < 0.05 and np.mean(a) < np.mean(b):
                    out.append("    >>> SURPRISING: MB-trained performs WORSE than random??")
                else:
                    out.append("    >>> NULL: MB training fails to beat random walk without shortcuts")

    text = "\n".join(out)
    print(text)
    out_file = INDIR / "analysis.txt"
    with open(out_file, "w", encoding="utf-8") as f: f.write(text)

    # Plot
    cond_order = [c for c in [
        "NAIVE_TEST_LESION", "TRAIN_FULL_TEST_LESION",
        "TRAIN_LESION_TEST_LESION", "NAIVE_TEST_FULL", "TRAIN_FULL_TEST_FULL",
    ] if c in post_rates]
    colors = {
        "NAIVE_TEST_LESION":       "#ff7a7a",
        "TRAIN_FULL_TEST_LESION":  "#ffd47a",
        "TRAIN_LESION_TEST_LESION":"#3affc8",
        "NAIVE_TEST_FULL":         "#a87aff",
        "TRAIN_FULL_TEST_FULL":    "#7ad7ff",
    }
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(len(cond_order))
    means = [np.mean(post_rates[c]) for c in cond_order]
    stds  = [np.std(post_rates[c], ddof=1) for c in cond_order]
    bars = ax.bar(x, means, 0.6, yerr=stds,
                  color=[colors.get(c, "#aaa") for c in cond_order],
                  capsize=5, alpha=0.85)
    for i, (c, m) in enumerate(zip(cond_order, means)):
        for v in post_rates[c]:
            ax.scatter(i, v, color=colors.get(c, "#aaa"), s=25, alpha=0.5, zorder=5)
    ax.axhline(2.3, color="#888", linestyle="--", linewidth=1, label="random walk baseline ~2.3%")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Post-test eat rate (%)")
    ax.set_title("E9: MB Necessity — does MB beat random when shortcuts removed?", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", alpha=0.3)
    fig.tight_layout()
    out_png = PLOT / "e9_mb_necessity.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out_png}")
    return text


if __name__ == "__main__":
    main()
