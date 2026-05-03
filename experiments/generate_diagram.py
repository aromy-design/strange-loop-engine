"""
Generate architecture diagram PNG using matplotlib.
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT = Path(__file__).parent / "plots" / "architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def draw():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # title
    ax.text(7, 9.5, "Metabolic Loop System — Architecture",
            fontsize=16, fontweight="bold", ha="center")
    ax.text(7, 9.1, "5350 neurons (~5150 spiking + 200 dense), no backprop, real-time CPU",
            fontsize=10, ha="center", color="#555")

    # World box
    world = patches.FancyBboxPatch((0.5, 7.0), 3.0, 1.5, boxstyle="round,pad=0.1",
                                    facecolor="#ffe7c2", edgecolor="black", linewidth=1.2)
    ax.add_patch(world)
    ax.text(2.0, 7.95, "WORLD (16x16)", fontsize=11, fontweight="bold", ha="center")
    ax.text(2.0, 7.55, "food · signals · day/night\nshelter · danger · landmarks",
            fontsize=8, ha="center")

    # Sensory box
    sens = patches.FancyBboxPatch((4.5, 7.0), 3.0, 1.5, boxstyle="round,pad=0.1",
                                   facecolor="#cfeaff", edgecolor="black", linewidth=1.2)
    ax.add_patch(sens)
    ax.text(6.0, 7.95, "SENSORY (66 dim)", fontsize=11, fontweight="bold", ha="center")
    ax.text(6.0, 7.55, "5x5 food + 5x5 signals\n+ 10 proprio + last action",
            fontsize=8, ha="center")

    # Action box
    act = patches.FancyBboxPatch((10.5, 7.0), 3.0, 1.5, boxstyle="round,pad=0.1",
                                  facecolor="#ffd6e7", edgecolor="black", linewidth=1.2)
    ax.add_patch(act)
    ax.text(12.0, 7.95, "ACTION (6 dim)", fontsize=11, fontweight="bold", ha="center")
    ax.text(12.0, 7.55, "up/down/left/right\n+ look + speak",
            fontsize=8, ha="center")

    # Insect brain regions
    regions = [
        ("Vision\nLobe", "800", 0.5, 4.5, "#fbe0a1"),
        ("Antennal\nLobe", "200", 2.3, 4.5, "#e0c0e8"),
        ("Lateral\nHorn", "150", 4.1, 4.5, "#d0a0d0"),
        ("Mushroom\nBody", "3000", 6.0, 4.5, "#a87aff"),
        ("Central\nComplex", "500", 8.2, 4.5, "#7ad7ff"),
        ("Subeso-\nphageal", "200", 10.0, 4.5, "#ffd47a"),
        ("Motor\nCenter", "300", 12.0, 4.5, "#ff5d8f"),
    ]
    for name, n, x, y, color in regions:
        box = patches.FancyBboxPatch((x - 0.6, y - 0.5), 1.2, 1.0,
                                      boxstyle="round,pad=0.05",
                                      facecolor=color, edgecolor="black", linewidth=1.0,
                                      alpha=0.85)
        ax.add_patch(box)
        ax.text(x, y + 0.15, name, fontsize=8, fontweight="bold", ha="center")
        ax.text(x, y - 0.3, f"{n}n", fontsize=7, ha="center", color="#444")

    # Insect brain boundary
    bound = patches.FancyBboxPatch((0.2, 3.7), 13.6, 1.7,
                                    boxstyle="round,pad=0.05",
                                    facecolor="none", edgecolor="#a87aff",
                                    linewidth=2, linestyle="--")
    ax.add_patch(bound)
    ax.text(7, 5.7, "INSECT BRAIN (5150 sparse spiking, 4-8% density)",
            fontsize=10, fontweight="bold", ha="center", color="#a87aff")

    # Auxiliary modules
    aux_modules = [
        ("Spatial Map\nTD(λ)", 0.5, 1.8, "#3affc8"),
        ("Behavioral\nModes", 2.3, 1.8, "#3affc8"),
        ("Path\nIntegrator", 4.1, 1.8, "#3affc8"),
        ("Sleep / Dream\nReplay", 5.9, 1.8, "#3affc8"),
        ("Awareness\nDetector", 7.7, 1.8, "#ff5d8f"),
        ("Mirror\nTest", 9.5, 1.8, "#ff5d8f"),
        ("Awakening\nDetector", 11.3, 1.8, "#ff5d8f"),
        ("Continuity\nTracker", 13.0, 1.8, "#ff5d8f"),
    ]
    for name, x, y, color in aux_modules:
        box = patches.FancyBboxPatch((x - 0.6, y - 0.5), 1.2, 1.0,
                                      boxstyle="round,pad=0.05",
                                      facecolor=color, edgecolor="black",
                                      linewidth=1.0, alpha=0.7)
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=7, ha="center")

    ax.text(2.5, 0.9, "BEHAVIORAL CONTROL", fontsize=9, fontweight="bold",
            ha="center", color="#3affc8")
    ax.text(10.5, 0.9, "CONSCIOUSNESS INDICATORS", fontsize=9, fontweight="bold",
            ha="center", color="#ff5d8f")

    # Arrows: world -> sensory -> brain -> action
    ax.annotate("", xy=(4.5, 7.7), xytext=(3.5, 7.7),
                arrowprops=dict(arrowstyle="->", lw=2, color="#444"))
    ax.annotate("", xy=(7.0, 5.7), xytext=(6.0, 7.0),
                arrowprops=dict(arrowstyle="->", lw=2, color="#444"))
    ax.annotate("", xy=(10.5, 7.7), xytext=(12.0, 5.4),
                arrowprops=dict(arrowstyle="->", lw=2, color="#444"))
    ax.annotate("", xy=(2.0, 7.0), xytext=(12.0, 7.0),
                arrowprops=dict(arrowstyle="->", lw=1.5,
                                color="#888", connectionstyle="arc3,rad=-0.3"))
    ax.text(7, 6.45, "world reacts", fontsize=8, color="#888", ha="center", style="italic")

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    draw()
