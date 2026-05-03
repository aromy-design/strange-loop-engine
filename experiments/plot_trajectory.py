"""E5 trajectory plots: time-series of consciousness indicators across seeds."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = Path(__file__).parent / "results" / "trajectory"
PLOT = Path(__file__).parent / "plots"
PLOT.mkdir(parents=True, exist_ok=True)


def load_all():
    rows_per_seed = {}
    for f in DIR.glob("trajectory_seed*.csv"):
        seed = int(f.stem.split("seed")[1])
        rows = []
        with open(f, newline="") as fp:
            for r in csv.DictReader(fp):
                for k, v in list(r.items()):
                    try: r[k] = float(v)
                    except (ValueError, TypeError):
                        try: r[k] = int(v)
                        except: pass
                rows.append(r)
        rows_per_seed[seed] = rows
    return rows_per_seed


def plot_metric(rows_per_seed, key, title, color):
    seeds = sorted(rows_per_seed.keys())
    times = [r["t"] for r in rows_per_seed[seeds[0]]]
    series = np.array([[r[key] for r in rows_per_seed[s]] for s in seeds], dtype=float)
    mean = series.mean(axis=0)
    sd = series.std(axis=0, ddof=1)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, s in enumerate(seeds):
        ax.plot(times, series[i], color=color, alpha=0.20, linewidth=0.8)
    ax.plot(times, mean, color=color, linewidth=2.4, label="mean across seeds")
    ax.fill_between(times, mean - sd, mean + sd, color=color, alpha=0.18, label="±SD")
    ax.set_xlabel("time (steps)")
    ax.set_ylabel(key)
    ax.set_title(f"E5 trajectory — {title} (n={len(seeds)} seeds)")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = PLOT / f"trajectory_{key}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    rows_per_seed = load_all()
    if not rows_per_seed:
        print("no trajectory CSVs.")
        return
    for key, title, color in [
        ("mirror", "self-recognition (mirror)", "#a87aff"),
        ("awareness_idx", "awareness index", "#7ad7ff"),
        ("collapse", "obs-coupling (H3)", "#3affc8"),
        ("presence", "presence (H4)", "#ff5d8f"),
        ("continuity", "self-continuity", "#ffd47a"),
        ("self_pe", "self-prediction error", "#ff5d8f"),
    ]:
        try: plot_metric(rows_per_seed, key, title, color)
        except KeyError: pass


if __name__ == "__main__":
    main()
