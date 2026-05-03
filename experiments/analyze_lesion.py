"""
Statistical analysis of E1 lesion study.

Reads experiments/results/lesion/summary.csv and computes per-condition
statistics with Welch's t-test against FULL baseline + Cohen's d.
"""
import csv
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

RESULTS = Path(__file__).parent / "results" / "lesion" / "summary.csv"
OUT_TXT = Path(__file__).parent / "results" / "lesion" / "stats_report.txt"
OUT_CSV = Path(__file__).parent / "results" / "lesion" / "stats_table.csv"


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                try:
                    if "." in str(v):
                        r[k] = float(v)
                    elif str(v).lstrip("-").isdigit():
                        r[k] = int(v)
                except (ValueError, AttributeError):
                    pass
            rows.append(r)
    return rows


def group_by(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return g


def cohen_d(a, b):
    a = np.array(a, dtype=np.float64); b = np.array(b, dtype=np.float64)
    if a.size < 2 or b.size < 2: return 0.0
    pooled = np.sqrt(((a.var(ddof=1) * (a.size - 1)) + (b.var(ddof=1) * (b.size - 1))) / (a.size + b.size - 2))
    if pooled < 1e-12: return 0.0
    return float((a.mean() - b.mean()) / pooled)


def welch_t(a, b):
    if not HAS_SCIPY:
        # manual Welch's t
        a = np.array(a, dtype=np.float64); b = np.array(b, dtype=np.float64)
        if a.size < 2 or b.size < 2: return 0.0, 1.0
        va = a.var(ddof=1) / a.size
        vb = b.var(ddof=1) / b.size
        denom = np.sqrt(va + vb)
        if denom < 1e-12: return 0.0, 1.0
        t = (a.mean() - b.mean()) / denom
        df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
        # rough p-value via 2-sided normal approximation (df > 5)
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        return float(t), float(p)
    res = stats.ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def analyze():
    if not RESULTS.exists():
        print(f"ERROR: {RESULTS} not found. Run lesion_study.py first.")
        return

    rows = load(RESULTS)
    by_cond = group_by(rows, "condition")
    if "FULL" not in by_cond:
        print("ERROR: FULL baseline missing.")
        return

    metrics = ["total_eats", "total_danger", "deaths", "final_mirror",
               "final_awareness_idx", "final_continuity",
               "final_collapse_index", "final_closure_depth"]

    full_data = {m: [r[m] for r in by_cond["FULL"]] for m in metrics}

    report = []
    table_rows = []
    report.append("=" * 76)
    report.append("E1 LESION STUDY — Statistical Report")
    report.append("=" * 76)
    report.append(f"Conditions: {len(by_cond)}, total runs: {len(rows)}")
    n_full = len(by_cond['FULL'])
    report.append(f"FULL baseline n={n_full}")
    report.append("")

    for cond_name, cond_rows in sorted(by_cond.items()):
        n = len(cond_rows)
        if cond_name == "FULL":
            report.append(f"--- BASELINE: FULL (n={n}) ---")
            for m in metrics:
                vals = full_data[m]
                report.append(f"  {m:30s} mean={np.mean(vals):.3f}  sd={np.std(vals, ddof=1) if len(vals)>1 else 0:.3f}")
            report.append("")
            continue

        report.append(f"--- {cond_name} (n={n}) ---")
        for m in metrics:
            vals = [r[m] for r in cond_rows]
            base_vals = full_data[m]
            mean_v = float(np.mean(vals))
            sd_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            mean_b = float(np.mean(base_vals))
            t, p = welch_t(vals, base_vals)
            d = cohen_d(vals, base_vals)
            delta_pct = (mean_v - mean_b) / max(abs(mean_b), 1e-9) * 100
            sig = stars(p)
            report.append(f"  {m:30s} mean={mean_v:.3f} (delta={delta_pct:+.1f}%)  sd={sd_v:.3f}  t={t:+.2f}  p={p:.4f} {sig}  d={d:+.2f}")
            table_rows.append({
                "condition": cond_name, "metric": m, "n": n,
                "mean": round(mean_v, 4), "sd": round(sd_v, 4),
                "baseline_mean": round(mean_b, 4), "delta_pct": round(delta_pct, 2),
                "t": round(t, 3), "p": round(p, 5), "cohen_d": round(d, 3), "sig": sig,
            })
        report.append("")

    report.append("=" * 76)
    report.append("Significance: * p<0.05, ** p<0.01, *** p<0.001")
    report.append("Effect size (Cohen's d): small ~0.2, medium ~0.5, large ~0.8")

    txt = "\n".join(report)
    try:
        print(txt)
    except UnicodeEncodeError:
        print(txt.encode("ascii", "replace").decode())
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"\nSaved report to {OUT_TXT}")

    # save CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        for r in table_rows: w.writerow(r)
    print(f"Saved stats table to {OUT_CSV}")


if __name__ == "__main__":
    analyze()
