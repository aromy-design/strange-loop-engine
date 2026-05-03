"""E2 sleep analysis: WITH-SLEEP vs NO-SLEEP."""
import csv, os
from pathlib import Path
from collections import defaultdict
import numpy as np
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

CSV = Path(__file__).parent / "results" / "sleep" / "summary.csv"
OUT = Path(__file__).parent / "results" / "sleep" / "stats_report.txt"


def load(p):
    rows = []
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                try:
                    if "." in str(v): r[k] = float(v)
                    elif str(v).lstrip("-").isdigit(): r[k] = int(v)
                except: pass
            rows.append(r)
    return rows


def cohen_d(a, b):
    a, b = np.array(a, float), np.array(b, float)
    if a.size < 2 or b.size < 2: return 0.0
    pooled = np.sqrt(((a.var(ddof=1)*(a.size-1)) + (b.var(ddof=1)*(b.size-1))) / (a.size+b.size-2))
    if pooled < 1e-12: return 0.0
    return float((a.mean() - b.mean()) / pooled)


def welch_t(a, b):
    if HAS_SCIPY:
        res = stats.ttest_ind(a, b, equal_var=False)
        return float(res.statistic), float(res.pvalue)
    a, b = np.array(a, float), np.array(b, float)
    va, vb = a.var(ddof=1)/a.size, b.var(ddof=1)/b.size
    denom = np.sqrt(va + vb)
    if denom < 1e-12: return 0.0, 1.0
    t = (a.mean() - b.mean()) / denom
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t)/sqrt(2))))
    return float(t), float(p)


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def main():
    rows = load(CSV)
    by = defaultdict(list)
    for r in rows: by[r["condition"]].append(r)

    metrics = ["total_eats", "total_danger", "deaths", "final_mirror",
               "final_awareness_idx", "final_continuity",
               "final_collapse_index", "final_closure_depth", "awakenings"]

    out = []
    out.append("=" * 76)
    out.append("E2 SLEEP EFFECT — Statistical Report")
    out.append("=" * 76)
    out.append(f"Conditions: {list(by.keys())}, n={len(by[list(by.keys())[0]])} per condition")
    out.append("")
    for m in metrics:
        a = [r[m] for r in by["WITH-SLEEP"]]
        b = [r[m] for r in by["NO-SLEEP"]]
        ma, sa = np.mean(a), np.std(a, ddof=1)
        mb, sb = np.mean(b), np.std(b, ddof=1)
        delta = (mb - ma) / max(abs(ma), 1e-9) * 100
        t, p = welch_t(b, a)
        d = cohen_d(b, a)
        out.append(f"{m:30s} WITH={ma:.3f}+-{sa:.3f}  NO={mb:.3f}+-{sb:.3f}  delta={delta:+.1f}%  t={t:+.2f}  p={p:.4f}{stars(p)}  d={d:+.2f}")
    out.append("")
    out.append("Significance: * p<0.05  ** p<0.01  *** p<0.001")

    text = "\n".join(out)
    print(text)
    with open(OUT, "w", encoding="utf-8") as f: f.write(text)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
