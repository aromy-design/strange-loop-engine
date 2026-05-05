"""
Analysis for E13 (selective MB pathway lesions, n=8 per condition).

Reports for each lesion condition vs CONTROL:
  - Welch's t-test
  - Cohen's d with bootstrap 95% CI (10000 resamples)
  - JZS Bayes Factor BF_01 (r=0.707) for the null

Sanity checks NAIVE_LESION vs CONTROL (the positive control showing training
matters at all — should be hugely positive, not null).

Data: experiments/results/e13_selective/results.csv
"""
import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats as sst
from scipy.integrate import quad


CSV_PATH = Path(__file__).parent / "results" / "e13_selective" / "results.csv"

BOOT_N = 10000
JZS_R = 0.707
NULL_BF_THRESHOLD = 3.0
NULL_D_THRESHOLD = 0.5

LESION_CONDITIONS = [
    "LESION_VIS_TO_MUSH",
    "LESION_MUSH_TO_MOTOR",
    "LESION_MUSH_TO_LH",
    "LESION_ALL_MB",
]


def cohens_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return 0.0 if pooled < 1e-12 else (np.mean(a) - np.mean(b)) / pooled


def bootstrap_d_ci(a, b, n_boot=BOOT_N, seed=20260504):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        ai = rng.choice(a, size=len(a), replace=True)
        bi = rng.choice(b, size=len(b), replace=True)
        ds[i] = cohens_d(ai, bi)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def jzs_bf10_two_sample(t, n1, n2, r=JZS_R):
    nu = n1 + n2 - 2
    n_eff = (n1 * n2) / (n1 + n2)
    def integrand(g):
        return ((1.0 + n_eff * g) ** (-0.5)
                * (1.0 + t * t / (nu * (1.0 + n_eff * g))) ** (-(nu + 1) / 2.0)
                * (r / math.sqrt(2.0 * math.pi)) * g ** (-1.5)
                * math.exp(-r * r / (2.0 * g)))
    numerator, _ = quad(integrand, 0.0, np.inf, limit=200)
    denominator = (1.0 + t * t / nu) ** (-(nu + 1) / 2.0)
    return float(numerator / denominator)


def main():
    rows = list(csv.DictReader(open(CSV_PATH, newline="")))
    cond = {}
    for r in rows:
        cond.setdefault(r["condition"], []).append(float(r["post_danger"]))

    if "CONTROL" not in cond:
        print("ERROR: CONTROL condition missing.")
        return

    ctl = np.array(cond["CONTROL"])
    print("=" * 72)
    print(f"E13 ANALYSIS — n per condition:")
    for k in sorted(cond):
        print(f"  {k:24s} n={len(cond[k]):2d}  mean_danger={np.mean(cond[k]):7.2f}  sd={np.std(cond[k], ddof=1):6.2f}")
    print("=" * 72)

    # Positive control: NAIVE_LESION should be hugely worse than CONTROL
    if "NAIVE_LESION" in cond:
        naive = np.array(cond["NAIVE_LESION"])
        t, p = sst.ttest_ind(ctl, naive, equal_var=False)
        d = cohens_d(ctl, naive)
        print(f"\nPositive control: CONTROL vs NAIVE_LESION")
        print(f"  Welch t={t:.3f}  p={p:.6g}  Cohen d={d:+.3f}  (expect very negative)")

    print("\n" + "-" * 72)
    print("Selective MB lesion tests vs CONTROL (testing dispensability)")
    print("-" * 72)
    for k in LESION_CONDITIONS:
        if k not in cond:
            print(f"  {k}: missing in CSV")
            continue
        x = np.array(cond[k])
        t, p = sst.ttest_ind(ctl, x, equal_var=False)
        d = cohens_d(ctl, x)
        lo, hi = bootstrap_d_ci(ctl, x)
        bf10 = jzs_bf10_two_sample(t, len(ctl), len(x))
        bf01 = 1.0 / bf10
        ok_d = abs(d) < NULL_D_THRESHOLD
        ok_bf = bf01 > NULL_BF_THRESHOLD
        ok_p = p > 0.05
        verdict = "DISPENSABLE" if (ok_d and ok_bf and ok_p) else "evidence inconclusive"
        print(f"\n  {k}")
        print(f"    Welch t={t:.3f}  p={p:.4f}  d={d:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]")
        print(f"    BF_10={bf10:.4f}  BF_01={bf01:.4f}")
        print(f"    Criteria (p>0.05 AND |d|<0.5 AND BF_01>3): {verdict}")

    print("\n" + "=" * 72)
    print("Note: 'DISPENSABLE' criterion is the same triple-criterion used for")
    print("E7v2 (pre-registered): p>0.05, |d|<0.5, BF_01>3. With n=8/cond power")
    print("for d=0.5 is roughly 30%, so this is a stringent test of the null.")


if __name__ == "__main__":
    main()
