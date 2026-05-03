"""
Publication-quality plots for E1 lesion study.

Output: experiments/plots/lesion_*.png
"""
import csv
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

RESULTS = Path(__file__).parent / "results" / "lesion" / "summary.csv"
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                try:
                    if "." in str(v): r[k] = float(v)
                    elif str(v).lstrip("-").isdigit(): r[k] = int(v)
                except (ValueError, AttributeError):
                    pass
            rows.append(r)
    return rows


def group_by(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return g


def order_conditions(by_cond):
    """FULL first, then alphabetical."""
    keys = list(by_cond.keys())
    keys.sort(key=lambda k: (0 if k == "FULL" else 1, k))
    return keys


def bar_plot(by_cond, metric, ylabel, fname, color="#7ad7ff"):
    if not HAS_MPL:
        print("matplotlib not installed; skipping plots.")
        return
    keys = order_conditions(by_cond)
    means = [np.mean([r[metric] for r in by_cond[k]]) for k in keys]
    sds = [np.std([r[metric] for r in by_cond[k]], ddof=1) if len(by_cond[k]) > 1 else 0 for k in keys]
    sems = [sd / np.sqrt(max(1, len(by_cond[k]))) for sd, k in zip(sds, keys)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(keys))
    colors = ["#3affc8" if k == "FULL" else color for k in keys]
    bars = ax.bar(x, means, yerr=sems, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.6, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"E1 lesion study — {metric}", fontsize=13)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = PLOT_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    if not RESULTS.exists():
        print(f"ERROR: {RESULTS} not found.")
        return
    rows = load(RESULTS)
    by_cond = group_by(rows, "condition")
    print(f"Loaded {len(rows)} rows across {len(by_cond)} conditions.")

    bar_plot(by_cond, "total_eats", "eats per run", "lesion_eats.png", "#7ad7ff")
    bar_plot(by_cond, "total_danger", "danger steps", "lesion_danger.png", "#ff5d8f")
    bar_plot(by_cond, "deaths", "deaths", "lesion_deaths.png", "#ff3a3a")
    bar_plot(by_cond, "final_mirror", "mirror score", "lesion_mirror.png", "#a87aff")
    bar_plot(by_cond, "final_awareness_idx", "awareness index", "lesion_awareness.png", "#ffd47a")
    bar_plot(by_cond, "final_collapse_index", "obs collapse", "lesion_collapse.png", "#3affc8")

    print(f"\nAll plots in {PLOT_DIR}")


if __name__ == "__main__":
    main()
