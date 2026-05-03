"""E3 sparsity sweep analysis: across MB threshold levels."""
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

CSV = Path(__file__).parent / "results" / "sparsity" / "summary.csv"
OUT = Path(__file__).parent / "results" / "sparsity" / "stats_report.txt"


def load(p):
    rows = []
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                try:
                    r[k] = float(v)
                except (ValueError, TypeError):
                    try: r[k] = int(v)
                    except (ValueError, TypeError): pass
            rows.append(r)
    return rows


def main():
    rows = load(CSV)
    by_th = defaultdict(list)
    for r in rows: by_th[r["threshold"]].append(r)
    thresholds = sorted(by_th.keys())

    out = []
    out.append("=" * 76)
    out.append("E3 SPARSITY SWEEP — Statistical Report")
    out.append("=" * 76)
    out.append(f"Thresholds: {thresholds}, n={len(by_th[thresholds[0]])} per level")
    out.append("")
    out.append(f"{'threshold':>10s} {'n':>3s} {'mb_rate':>8s} {'eats':>10s} {'danger':>10s} {'mirror':>8s} {'collapse':>9s}")
    out.append("-" * 76)
    for th in thresholds:
        g = by_th[th]
        out.append(
            f"{th:>10.2f} {len(g):>3d} "
            f"{np.mean([r['mb_firing_mean'] for r in g]):>8.4f} "
            f"{np.mean([r['eats'] for r in g]):>5.1f}+-{np.std([r['eats'] for r in g], ddof=1):>3.1f} "
            f"{np.mean([r['danger'] for r in g]):>5.1f}+-{np.std([r['danger'] for r in g], ddof=1):>3.1f} "
            f"{np.mean([r['final_mirror'] for r in g]):>8.3f} "
            f"{np.mean([r['final_collapse'] for r in g]):>9.3f}"
        )
    out.append("")

    # ANOVA across thresholds
    if HAS_SCIPY:
        for metric in ["eats", "danger", "final_mirror", "final_collapse"]:
            groups = [[r[metric] for r in by_th[th]] for th in thresholds]
            f_stat, p_val = stats.f_oneway(*groups)
            out.append(f"ANOVA {metric:20s}: F={f_stat:.2f}  p={p_val:.4f}")

    # find optimum (max eats)
    best_th = max(thresholds, key=lambda th: np.mean([r['eats'] for r in by_th[th]]))
    best_eats = np.mean([r['eats'] for r in by_th[best_th]])
    out.append("")
    out.append(f"Optimum threshold for eats: {best_th} (mean eats {best_eats:.1f}, mb_rate {np.mean([r['mb_firing_mean'] for r in by_th[best_th]]):.4f})")

    text = "\n".join(out)
    print(text)
    with open(OUT, "w", encoding="utf-8") as f: f.write(text)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
