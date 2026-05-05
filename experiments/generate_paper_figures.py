"""
Generate the main paper figures from the experiment CSVs.

Outputs to ./figures/ as 300-dpi PNGs:
  fig2_dissociation_e10.png       — bar plot trained vs naive danger (E10)
  fig3_flyem_replication.png      — bar plot trained vs naive danger (E12 FlyEM)
  fig4_e7v2_null.png              — boxplot eat rate FULL vs LESION (E7v2)
  fig5_e13_dispensable.png        — bar plot all E13 lesion conditions
  fig6_summary_three_paradigms.png — combined summary panel

Reads:
  experiments/results/e10_scaffolded/results.csv
  experiments/results/e12_flyem/results.csv
  experiments/results/e7v2/results.csv
  experiments/results/e13_selective/results.csv

Style: legibility first. Annotations never overlap with titles or bars.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

DPI = 300
plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.titlepad": 12,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.15,
    "figure.constrained_layout.w_pad": 0.15,
})


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def col(rows, key, cast=float):
    return np.array([cast(r[key]) for r in rows])


def by_condition(rows):
    out = {}
    for r in rows:
        out.setdefault(r["condition"], []).append(r)
    return out


def add_padding(ax, top_frac=0.20, bottom=0.0):
    """Expand y-axis upper limit by top_frac to leave room for annotations
    above bars without colliding with the title."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(bottom, hi + (hi - bottom) * top_frac)


# ------------------------------------------------------------------
# Figure 2 — E10 dissociation
# ------------------------------------------------------------------
def fig2_e10():
    rows = load_csv(RESULTS / "e10_scaffolded" / "results.csv")
    cond = by_condition(rows)
    order = ["NAIVE_LESION", "FULL_LESION_TEST", "SCAFFOLDED_4PH"]
    order = [c for c in order if c in cond]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0))

    # Left panel: danger steps
    means = [col(cond[c], "post_danger").mean() for c in order]
    sds = [col(cond[c], "post_danger").std(ddof=1) for c in order]
    xs = np.arange(len(order))
    colors = ["#c33", "#39c", "#39c"]
    ax1.bar(xs, means, yerr=sds, capsize=5, color=colors,
            edgecolor="black", linewidth=0.7)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["NAIVE\n(no train)", "FULL\nLESION_TEST",
                         "SCAFFOLDED\n4PH"], fontsize=9)
    ax1.set_ylabel("Danger steps (8000-tick test)")
    ax1.set_title("Aversive: trained agents avoid danger", fontsize=10)
    add_padding(ax1, top_frac=0.30)
    # Annotation above the gap, in clear space between bars
    ax1.annotate("d ≈ −6.5\np < 10⁻⁶",
                 xy=(1.5, 0), xytext=(1.5, max(means) * 0.45),
                 ha="center", va="center",
                 fontsize=10, fontweight="bold", color="#222",
                 bbox=dict(boxstyle="round,pad=0.4",
                           facecolor="white", edgecolor="#888", linewidth=0.6))

    # Right panel: eat rate
    eats = [col(cond[c], "post_eats_rate").mean() * 100 for c in order]
    eats_sd = [col(cond[c], "post_eats_rate").std(ddof=1) * 100 for c in order]
    ax2.bar(xs, eats, yerr=eats_sd, capsize=5, color=colors,
            edgecolor="black", linewidth=0.7)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["NAIVE\n(no train)", "FULL\nLESION_TEST",
                         "SCAFFOLDED\n4PH"], fontsize=9)
    ax2.set_ylabel("Eat rate (%)")
    ax2.axhline(0.495, color="grey", linestyle="--", linewidth=0.9,
                label="random walk (0.50%)")
    ax2.set_title("Appetitive: no transfer, all near random walk", fontsize=10)
    ax2.legend(fontsize=9, loc="upper right", frameon=False)
    add_padding(ax2, top_frac=0.25)

    fig.suptitle("Figure 2 — E10: aversive transfer, appetitive null "
                 "(random wiring, n=8 per condition)",
                 fontsize=11, y=1.04)
    fig.savefig(FIGURES / "fig2_dissociation_e10.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("saved fig2_dissociation_e10.png")


# ------------------------------------------------------------------
# Figure 3 — E12 FlyEM
# ------------------------------------------------------------------
def fig3_e12_flyem():
    rows = load_csv(RESULTS / "e12_flyem" / "results.csv")
    cond = by_condition(rows)
    order = ["NAIVE_LESION", "FULL_LESION_TEST"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0))

    means = [col(cond[c], "post_danger").mean() for c in order]
    sds = [col(cond[c], "post_danger").std(ddof=1) for c in order]
    xs = np.arange(len(order))
    colors = ["#c33", "#39c"]
    ax1.bar(xs, means, yerr=sds, capsize=6, color=colors,
            edgecolor="black", linewidth=0.7)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["Naive\n(no train)",
                         "Trained\n(MB intact at test)"], fontsize=9)
    ax1.set_ylabel("Danger steps (8000-tick test)")
    ax1.set_title("Aversive (real Drosophila wiring)", fontsize=10)
    add_padding(ax1, top_frac=0.30)
    ax1.annotate("d = −4.76\np < 10⁻⁶\nn = 8 / cond",
                 xy=(0.5, 0), xytext=(0.5, max(means) * 0.5),
                 ha="center", va="center",
                 fontsize=10, fontweight="bold", color="#222",
                 bbox=dict(boxstyle="round,pad=0.4",
                           facecolor="white", edgecolor="#888", linewidth=0.6))

    eats = [col(cond[c], "post_eats_rate").mean() * 100 for c in order]
    eats_sd = [col(cond[c], "post_eats_rate").std(ddof=1) * 100 for c in order]
    ax2.bar(xs, eats, yerr=eats_sd, capsize=6, color=colors,
            edgecolor="black", linewidth=0.7)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["Naive\n(no train)",
                         "Trained\n(MB intact at test)"], fontsize=9)
    ax2.set_ylabel("Eat rate (%)")
    ax2.axhline(0.495, color="grey", linestyle="--", linewidth=0.9,
                label="random walk (0.50%)")
    ax2.set_title("Appetitive (real Drosophila wiring)", fontsize=10)
    ax2.legend(fontsize=9, loc="upper right", frameon=False)
    add_padding(ax2, top_frac=0.25)

    fig.suptitle("Figure 3 — E12: dissociation replicates with FlyEM hemibrain "
                 "(21,728 neurons mapped to 7 circuits)",
                 fontsize=11, y=1.04)
    fig.savefig(FIGURES / "fig3_flyem_replication.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("saved fig3_flyem_replication.png")


# ------------------------------------------------------------------
# Figure 4 — E7v2 null (pre-registered)
# ------------------------------------------------------------------
def fig4_e7v2():
    rows = load_csv(RESULTS / "e7v2" / "results.csv")
    cond = by_condition(rows)
    full = col(cond["FULL"], "post_eat_rate") * 100
    les = col(cond["LESION_FULL_MB"], "post_eat_rate") * 100
    full_d = col(cond["FULL"], "post_danger")
    les_d = col(cond["LESION_FULL_MB"], "post_danger")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.4))

    # Left: eat rate boxplot
    bp1 = ax1.boxplot([full, les], tick_labels=["FULL\n(MB intact)",
                                                "LESION_FULL_MB\n(MB silenced)"],
                      widths=0.55, patch_artist=True, showmeans=True,
                      meanprops={"marker": "D", "markerfacecolor": "white",
                                 "markeredgecolor": "black", "markersize": 7})
    for box, color in zip(bp1["boxes"], ["#39c", "#c93"]):
        box.set_facecolor(color)
        box.set_alpha(0.7)
    ax1.set_ylabel("Eat rate (%)")
    ax1.set_title("Appetitive — H1 null (pre-registered)",
                  fontsize=10, pad=14)
    add_padding(ax1, top_frac=0.30,
                bottom=min(full.min(), les.min()) - 0.2)
    ax1.text(1.5, ax1.get_ylim()[1] * 0.99,
             "d = +0.07 · BF₀₁ = 3.71 · p_Bonf = 1.00",
             ha="center", va="top", fontsize=9, style="italic",
             color="#222",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="white", edgecolor="#aaa", linewidth=0.5))

    # Right: danger boxplot
    bp2 = ax2.boxplot([full_d, les_d], tick_labels=["FULL\n(MB intact)",
                                                    "LESION_FULL_MB\n(MB silenced)"],
                      widths=0.55, patch_artist=True, showmeans=True,
                      meanprops={"marker": "D", "markerfacecolor": "white",
                                 "markeredgecolor": "black", "markersize": 7})
    for box, color in zip(bp2["boxes"], ["#39c", "#c93"]):
        box.set_facecolor(color)
        box.set_alpha(0.7)
    ax2.set_ylabel("Danger steps")
    ax2.set_title("Aversive — H2 positive control (NOT confirmed)",
                  fontsize=10, pad=14)
    add_padding(ax2, top_frac=0.30, bottom=0)
    ax2.text(1.5, ax2.get_ylim()[1] * 0.99,
             "d = +0.22 · p_Bonf = 0.81",
             ha="center", va="top", fontsize=9, style="italic",
             color="#222",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="white", edgecolor="#aaa", linewidth=0.5))

    fig.suptitle("Figure 4 — E7v2 (n=30 per condition, pre-registered "
                 "OSF DOI 10.17605/OSF.IO/397GJ)",
                 fontsize=11, y=1.02)
    fig.savefig(FIGURES / "fig4_e7v2_null.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved fig4_e7v2_null.png")


# ------------------------------------------------------------------
# Figure 5 — E13 selective lesion dispensability
# ------------------------------------------------------------------
def fig5_e13():
    rows = load_csv(RESULTS / "e13_selective" / "results.csv")
    cond = by_condition(rows)
    order = ["NAIVE_LESION", "CONTROL", "LESION_VIS_TO_MUSH",
             "LESION_MUSH_TO_MOTOR", "LESION_MUSH_TO_LH", "LESION_ALL_MB"]

    means = [col(cond[c], "post_danger").mean() for c in order]
    sds = [col(cond[c], "post_danger").std(ddof=1) for c in order]
    xs = np.arange(len(order))
    colors = ["#c33", "#39c", "#963", "#963", "#963", "#a30"]
    labels = ["NAIVE\n(no train)", "CONTROL\n(MB intact)",
              "vis→MB\nsilenced", "MB→motor\nsilenced",
              "MB→LH\nsilenced", "ALL THREE\nsilenced"]

    fig, ax = plt.subplots(1, 1, figsize=(10.0, 4.6))
    ax.bar(xs, means, yerr=sds, capsize=5, color=colors,
           edgecolor="black", linewidth=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Danger steps (8000-tick test)")
    ax.set_title("Figure 5 — E13 (n=8 per condition): MB silencing produces "
                 "no large effect on trained avoidance",
                 fontsize=11, pad=14)

    add_padding(ax, top_frac=0.18)

    # NAIVE annotation: place at right edge of the NAIVE bar, in white space
    ax.annotate("p < 10⁻⁵\nd = −6.43",
                xy=(0, means[0]), xytext=(0, means[0] * 0.55),
                ha="center", va="center",
                fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#7a1010", edgecolor="none"))

    # Per-lesion annotations: above each bar with extra spacing
    annot = {
        2: "d=+0.60\nBF₀₁=1.46",
        3: "d=+0.03\nBF₀₁=2.34",
        4: "d=+0.00\nBF₀₁=2.34",
        5: "d=+0.24\nBF₀₁=2.16",
    }
    upper = ax.get_ylim()[1]
    annot_y = upper * 0.18
    for i in range(2, 6):
        ax.text(i, annot_y, annot[i],
                ha="center", va="bottom",
                fontsize=8.5, style="italic", color="#333",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="#bbb", linewidth=0.4))

    # Optional: light annotation for CONTROL baseline
    ax.text(1, annot_y, "baseline\n(trained)",
            ha="center", va="bottom",
            fontsize=8.5, style="italic", color="#333",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor="#bbb", linewidth=0.4))

    fig.savefig(FIGURES / "fig5_e13_dispensable.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("saved fig5_e13_dispensable.png")


# ------------------------------------------------------------------
# Figure 6 — combined summary panel
# ------------------------------------------------------------------
def fig6_summary():
    rows10 = load_csv(RESULTS / "e10_scaffolded" / "results.csv")
    rows12 = load_csv(RESULTS / "e12_flyem" / "results.csv")
    rows7 = load_csv(RESULTS / "e7v2" / "results.csv")
    rows13 = load_csv(RESULTS / "e13_selective" / "results.csv")

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.0))

    # Panel 1: E10 danger
    c10 = by_condition(rows10)
    if "FULL_LESION_TEST" in c10 and "NAIVE_LESION" in c10:
        ax = axes[0]
        means = [col(c10["NAIVE_LESION"], "post_danger").mean(),
                 col(c10["FULL_LESION_TEST"], "post_danger").mean()]
        sds = [col(c10["NAIVE_LESION"], "post_danger").std(ddof=1),
               col(c10["FULL_LESION_TEST"], "post_danger").std(ddof=1)]
        ax.bar([0, 1], means, yerr=sds, capsize=5,
               color=["#c33", "#39c"], edgecolor="black", linewidth=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Naive", "Trained"], fontsize=9)
        ax.set_title("E10\nrandom wiring", fontsize=10)
        ax.set_ylabel("Danger steps")
        add_padding(ax, top_frac=0.25)
        ax.text(0.5, ax.get_ylim()[1] * 0.85, "d = −6.43",
                ha="center", va="center", fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="#aaa"))

    # Panel 2: E12 FlyEM danger
    c12 = by_condition(rows12)
    ax = axes[1]
    means = [col(c12["NAIVE_LESION"], "post_danger").mean(),
             col(c12["FULL_LESION_TEST"], "post_danger").mean()]
    sds = [col(c12["NAIVE_LESION"], "post_danger").std(ddof=1),
           col(c12["FULL_LESION_TEST"], "post_danger").std(ddof=1)]
    ax.bar([0, 1], means, yerr=sds, capsize=5,
           color=["#c33", "#39c"], edgecolor="black", linewidth=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Naive", "Trained"], fontsize=9)
    ax.set_title("E12\nFlyEM connectome", fontsize=10)
    ax.set_ylabel("Danger steps")
    add_padding(ax, top_frac=0.25)
    ax.text(0.5, ax.get_ylim()[1] * 0.85, "d = −4.76",
            ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="#aaa"))

    # Panel 3: E7v2 eat rate
    c7 = by_condition(rows7)
    ax = axes[2]
    full = col(c7["FULL"], "post_eat_rate") * 100
    les = col(c7["LESION_FULL_MB"], "post_eat_rate") * 100
    ax.boxplot([full, les], tick_labels=["FULL", "LESION"], widths=0.5,
               patch_artist=True, showmeans=True,
               meanprops={"marker": "D", "markerfacecolor": "white",
                          "markeredgecolor": "black", "markersize": 5})
    ax.set_title("E7v2\npre-reg null (n=30)", fontsize=10)
    ax.set_ylabel("Eat rate (%)")
    add_padding(ax, top_frac=0.25,
                bottom=min(full.min(), les.min()) - 0.2)
    ax.text(1.5, ax.get_ylim()[1] * 0.97,
            "BF₀₁ = 3.71\nd = +0.07",
            ha="center", va="top", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="#aaa"))

    # Panel 4: E13 dispensability (CONTROL vs ALL_MB)
    c13 = by_condition(rows13)
    ax = axes[3]
    order = ["CONTROL", "LESION_ALL_MB"]
    means = [col(c13[c], "post_danger").mean() for c in order]
    sds = [col(c13[c], "post_danger").std(ddof=1) for c in order]
    ax.bar([0, 1], means, yerr=sds, capsize=5,
           color=["#39c", "#a30"], edgecolor="black", linewidth=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CONTROL", "ALL MB\nsilenced"], fontsize=9)
    ax.set_title("E13\ndispensability (n=8)", fontsize=10)
    ax.set_ylabel("Danger steps")
    add_padding(ax, top_frac=0.30)
    ax.text(0.5, ax.get_ylim()[1] * 0.85,
            "d = +0.24\nBF₀₁ = 2.16",
            ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="#aaa"))

    fig.suptitle("Figure 6 — Summary across the four experiments",
                 fontsize=11, y=1.04)
    fig.savefig(FIGURES / "fig6_summary_three_paradigms.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("saved fig6_summary_three_paradigms.png")


if __name__ == "__main__":
    fig2_e10()
    fig3_e12_flyem()
    fig4_e7v2()
    fig5_e13()
    fig6_summary()
    print("\nAll figures written to:", FIGURES)
