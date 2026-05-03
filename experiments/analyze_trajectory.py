"""E5 trajectory analysis: time series + cross-correlations."""
import csv
from pathlib import Path
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

DIR = Path(__file__).parent / "results" / "trajectory"
OUT = DIR / "stats_report.txt"


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


def aggregate(rows_per_seed, key):
    """Stack values into matrix (seed × time)."""
    seeds = sorted(rows_per_seed.keys())
    series = []
    for s in seeds:
        series.append([r[key] for r in rows_per_seed[s]])
    arr = np.array(series, dtype=np.float64)
    return arr  # shape (n_seeds, n_timesteps)


def main():
    rows_per_seed = load_all()
    if not rows_per_seed:
        print("No trajectory CSVs found.")
        return
    seeds = sorted(rows_per_seed.keys())
    n_steps = len(rows_per_seed[seeds[0]])
    out = []
    out.append("=" * 76)
    out.append("E5 TRAJECTORY — Time Series Analysis")
    out.append("=" * 76)
    out.append(f"Seeds: {len(seeds)}, time points per seed: {n_steps}")
    out.append("")

    # H4 test: do awakening events temporally precede mirror improvements?
    awakes = aggregate(rows_per_seed, "cum_awakenings")  # cumulative awakenings
    mirror = aggregate(rows_per_seed, "mirror")
    # use first-difference (rate) of awakenings vs mirror change
    awake_rate = np.diff(awakes, axis=1)  # awakenings per bucket
    mirror_diff = np.diff(mirror, axis=1)
    # cross-correlation per seed at lag k: corr(awake_rate[t], mirror_diff[t+k])
    out.append("=== H4 TEST: Awakening events precede mirror improvements? ===")
    if HAS_SCIPY:
        for lag in [-3, -2, -1, 0, 1, 2, 3, 5, 10]:
            corrs = []
            for i in range(len(seeds)):
                if lag >= 0:
                    a = awake_rate[i, :len(awake_rate[i]) - lag]
                    b = mirror_diff[i, lag:]
                else:
                    a = awake_rate[i, -lag:]
                    b = mirror_diff[i, :len(mirror_diff[i]) + lag]
                if len(a) > 5 and a.std() > 0 and b.std() > 0:
                    c, _ = stats.pearsonr(a, b)
                    if not np.isnan(c): corrs.append(c)
            if corrs:
                m, sd = np.mean(corrs), np.std(corrs, ddof=1)
                out.append(f"  lag={lag:+3d} (awake leads mirror by {lag} buckets):  r={m:+.3f} +- {sd:.3f}  (n_seeds={len(corrs)})")

    # composite consciousness emergence
    out.append("")
    out.append("=== Consciousness composite emergence over time ===")
    out.append("(per-seed mean across time points, then mean across seeds)")

    metrics_to_plot = ["mirror", "awareness_idx", "continuity", "presence", "collapse", "self_pe"]
    for m in metrics_to_plot:
        try:
            arr = aggregate(rows_per_seed, m)
        except KeyError:
            continue
        # split into early / late thirds
        n = arr.shape[1]
        early = arr[:, :n//3].mean(axis=1)
        late = arr[:, 2*n//3:].mean(axis=1)
        delta = (late.mean() - early.mean()) / max(abs(early.mean()), 1e-9) * 100
        if HAS_SCIPY:
            t, p = stats.ttest_rel(late, early)
            out.append(f"  {m:18s} early={early.mean():.3f}  late={late.mean():.3f}  delta={delta:+.1f}%  paired t={t:+.2f}  p={p:.4f}")
        else:
            out.append(f"  {m:18s} early={early.mean():.3f}  late={late.mean():.3f}  delta={delta:+.1f}%")

    # behavior emergence
    out.append("")
    out.append("=== Behavior emergence over time ===")
    eats = aggregate(rows_per_seed, "cum_eats")
    n = eats.shape[1]
    eats_per_bucket = np.diff(eats, axis=1)
    early_eats = eats_per_bucket[:, :n//3].mean(axis=1)
    late_eats = eats_per_bucket[:, 2*n//3:].mean(axis=1)
    if HAS_SCIPY:
        t, p = stats.ttest_rel(late_eats, early_eats)
        out.append(f"  eats per 200-step bucket  early={early_eats.mean():.2f}  late={late_eats.mean():.2f}  delta={(late_eats.mean()-early_eats.mean())/max(abs(early_eats.mean()),1e-9)*100:+.1f}%  paired t={t:+.2f}  p={p:.4f}")

    # correlation: composite-related metrics vs behavior
    out.append("")
    out.append("=== Mid-time correlation: consciousness vs behavior ===")
    if HAS_SCIPY:
        # at each time point t, correlate over seeds: metric_t vs eats_per_bucket_t
        for m in ["awareness_idx", "presence", "collapse", "mirror"]:
            try: arr = aggregate(rows_per_seed, m)
            except KeyError: continue
            # average correlation over middle 50% of time
            tcorrs = []
            mid_lo = n // 4; mid_hi = 3 * n // 4
            for t_ in range(mid_lo, min(mid_hi, eats_per_bucket.shape[1])):
                if arr[:, t_].std() < 1e-6 or eats_per_bucket[:, t_].std() < 1e-6: continue
                c, _ = stats.pearsonr(arr[:, t_], eats_per_bucket[:, t_])
                if not np.isnan(c): tcorrs.append(c)
            if tcorrs:
                out.append(f"  {m:18s} mean cross-seed correlation with eats: r={np.mean(tcorrs):+.3f}")

    text = "\n".join(out)
    print(text)
    with open(OUT, "w", encoding="utf-8") as f: f.write(text)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
