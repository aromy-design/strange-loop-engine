"""
E6 long-training analysis: did mushroom body engage over time?

Loads time series from results/long_training/seedN.csv and:
  - aggregates per-bucket eats over seeds (learning curve)
  - tracks MB firing rate and MB→motor weight evolution
  - paired t-test early vs late thirds
"""
import csv, glob
from pathlib import Path
import numpy as np
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = Path(__file__).parent / "results" / "long_training"
PLOT = Path(__file__).parent / "plots"
PLOT.mkdir(parents=True, exist_ok=True)
OUT = DIR / "stats_report.txt"


def load():
    rows_per_seed = {}
    for f in sorted(DIR.glob("seed*.csv")):
        seed = int(f.stem.replace("seed", ""))
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


def aggregate(rows_per_seed, key):
    seeds = sorted(rows_per_seed.keys())
    return np.array([[r[key] for r in rows_per_seed[s]] for s in seeds], dtype=np.float64)


def main():
    rows_per_seed = load()
    if not rows_per_seed:
        print("no long_training CSVs found.")
        return
    seeds = sorted(rows_per_seed.keys())
    n_pts = len(rows_per_seed[seeds[0]])
    times = [r["t"] for r in rows_per_seed[seeds[0]]]
    out = []
    out.append("=" * 76)
    out.append("E6 LONG-TRAINING — Time Series Analysis (V2 81k brain)")
    out.append("=" * 76)
    out.append(f"Seeds: {seeds} ({len(seeds)})  Logs per seed: {n_pts}  Steps total: {times[-1]}")
    out.append("")

    # learning curve: eats per 1000 steps in each bucket
    eats = aggregate(rows_per_seed, "cum_eats")
    eats_per_bucket = np.diff(eats, axis=1)
    out.append(f"Learning curve (eats per log-bucket, mean across seeds):")
    out.append("  bucket  t       eats/bucket")
    for i in range(0, n_pts - 1, max(1, (n_pts - 1) // 8)):
        m = float(eats_per_bucket[:, i].mean())
        s = float(eats_per_bucket[:, i].std(ddof=1)) if eats_per_bucket.shape[0] > 1 else 0
        out.append(f"  {i:3d}     t={int(times[i+1]):6d}  {m:.2f} +- {s:.2f}")
    out.append("")

    # paired t-test: early vs late thirds
    out.append("Paired t-tests (within-seed early vs late thirds):")
    for key in ["cum_eats", "mb_firing_rate", "mb_to_motor_w_mean", "mb_to_motor_w_max",
                "cx_to_motor_w_mean", "lh_to_motor_w_mean", "seg_to_motor_w_mean",
                "vis_to_mush_w_mean", "mb_w_self_mean", "mirror", "awareness_idx",
                "collapse", "presence"]:
        try:
            arr = aggregate(rows_per_seed, key)
        except KeyError:
            continue
        if key == "cum_eats":
            # use diff (rate)
            arr_use = np.diff(arr, axis=1)
        else:
            arr_use = arr
        n = arr_use.shape[1]
        early = arr_use[:, :n // 3].mean(axis=1)
        late = arr_use[:, 2 * n // 3:].mean(axis=1)
        delta = (late.mean() - early.mean()) / max(abs(early.mean()), 1e-9) * 100
        if HAS_SCIPY and len(seeds) > 1:
            t, p = stats.ttest_rel(late, early)
        else:
            t, p = 0.0, 1.0
        out.append(f"  {key:25s} early={early.mean():.4f}  late={late.mean():.4f}  delta={delta:+.1f}%  t={t:+.2f}  p={p:.4f}")
    out.append("")

    # critical question: did cross-projection weights grow?
    out.append("=== H1 (E6): Cross-projection weight evolution ===")
    for w_key, label in [
        ("mb_to_motor_w_mean", "MB->motor"),
        ("cx_to_motor_w_mean", "CX->motor"),
        ("lh_to_motor_w_mean", "LH->motor"),
        ("seg_to_motor_w_mean", "SEG->motor"),
        ("vis_to_mush_w_mean", "Vis->mush"),
    ]:
        try:
            w_arr = aggregate(rows_per_seed, w_key)
        except KeyError:
            continue
        w_initial = w_arr[:, 0]
        w_final = w_arr[:, -1]
        growth_pct = (w_final.mean() - w_initial.mean()) / max(abs(w_initial.mean()), 1e-9) * 100
        if HAS_SCIPY and len(seeds) > 1:
            t, p = stats.ttest_rel(w_final, w_initial)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            out.append(f"  {label:15s}: start={w_initial.mean():.5f}  end={w_final.mean():.5f}  "
                       f"growth={growth_pct:+.1f}%  t={t:+.2f}  p={p:.4f}  {sig}")
        else:
            out.append(f"  {label:15s}: start={w_initial.mean():.5f}  end={w_final.mean():.5f}  growth={growth_pct:+.1f}%")

    text = "\n".join(out)
    print(text)
    with open(OUT, "w", encoding="utf-8") as f: f.write(text)

    # plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.flatten()
    series_to_plot = [
        ("cum_eats", "cumulative eats", "#7ad7ff", axes[0]),
        ("mb_firing_rate", "MB firing rate", "#a87aff", axes[1]),
        ("mb_to_motor_w_mean", "MB->motor weight", "#3affc8", axes[2]),
        ("cx_to_motor_w_mean", "CX->motor weight", "#ffa07a", axes[3]),
        ("lh_to_motor_w_mean", "LH->motor weight", "#ff7ab0", axes[4]),
        ("mirror", "mirror score", "#ffd47a", axes[5]),
    ]
    for key, label, color, ax in series_to_plot:
        try: arr = aggregate(rows_per_seed, key)
        except KeyError:
            ax.set_visible(False)
            continue
        for i, s in enumerate(seeds):
            ax.plot(times, arr[i], color=color, alpha=0.25, linewidth=0.8)
        ax.plot(times, arr.mean(axis=0), color=color, linewidth=2.2, label="mean")
        if arr.shape[0] > 1:
            ax.fill_between(times,
                            arr.mean(axis=0) - arr.std(axis=0, ddof=1),
                            arr.mean(axis=0) + arr.std(axis=0, ddof=1),
                            color=color, alpha=0.15)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.grid(linestyle=":", alpha=0.3)
    fig.suptitle(f"E6 long-training V2 81k (n={len(seeds)} seeds)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_png = PLOT / "long_training_overview.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
