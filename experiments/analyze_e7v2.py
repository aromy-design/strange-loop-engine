"""
Pre-registered analysis for E7v2.

Locked on OSF on 2026-05-03 (DOI 10.17605/OSF.IO/397GJ).
Run this ONLY after experiments/results/e7v2/results.csv has all 60 rows.

Pre-registered tests:
  - Welch t-test on eat_rate (FULL vs LESION)
  - Welch t-test on danger_steps (FULL vs LESION)
  - Cohen's d with bootstrap 95% CI (10000 resamples)
  - BF_01 (JZS, r=0.707) for the appetitive null hypothesis
  - Bonferroni correction across the two outcome tests (alpha = 0.025)

H1 (appetitive null) confirmed iff: p > 0.05 AND |d| < 0.5 AND BF_01 > 3
H2 (aversive positive) confirmed iff: p < 0.05 AND d < -1.5

Exploratory (flagged separately): paired analysis, mb_w correlations.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sst
from scipy.integrate import quad


CSV_PATH = Path(__file__).parent / "results" / "e7v2" / "results.csv"

ALPHA_RAW = 0.05
ALPHA_BONF = 0.025
BOOT_N = 10000
JZS_R = 0.707
N_PER_GROUP = 30
NULL_BF_THRESHOLD = 3.0
NULL_D_THRESHOLD = 0.5
POS_D_THRESHOLD = -1.5


def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if pooled < 1e-12:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def bootstrap_d_ci(a, b, n_boot=BOOT_N, seed=20260503):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        ai = rng.choice(a, size=len(a), replace=True)
        bi = rng.choice(b, size=len(b), replace=True)
        ds[i] = cohens_d(ai, bi)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def jzs_bf10_two_sample(t, n1, n2, r=JZS_R):
    """
    JZS BF_10 for an independent two-sample t-test (Rouder et al., 2009).
    Returns the Bayes Factor in favor of H1 over H0.
    BF_01 = 1 / BF_10.
    """
    nu = n1 + n2 - 2
    n_eff = (n1 * n2) / (n1 + n2)

    def integrand(g):
        return (
            (1.0 + n_eff * g) ** (-0.5)
            * (1.0 + t * t / (nu * (1.0 + n_eff * g))) ** (-(nu + 1) / 2.0)
            * (r / math.sqrt(2.0 * math.pi))
            * g ** (-1.5)
            * math.exp(-r * r / (2.0 * g))
        )

    numerator, _ = quad(integrand, 0.0, np.inf, limit=200)
    denominator = (1.0 + t * t / nu) ** (-(nu + 1) / 2.0)
    return float(numerator / denominator)


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def split_by_condition(rows):
    full = [r for r in rows if r["condition"] == "FULL"]
    les = [r for r in rows if r["condition"] == "LESION_FULL_MB"]
    return full, les


def col(rows, key, cast=float):
    return np.array([cast(r[key]) for r in rows])


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run E7v2 first.")
        sys.exit(1)

    rows = load_csv(CSV_PATH)
    if len(rows) != 2 * N_PER_GROUP:
        print(f"WARNING: expected {2*N_PER_GROUP} rows, got {len(rows)}.")
        print("Pre-registered analysis requires all 60 runs complete.")
        sys.exit(1)

    full, lesion = split_by_condition(rows)
    if len(full) != N_PER_GROUP or len(lesion) != N_PER_GROUP:
        print(f"ERROR: condition counts off (FULL={len(full)} LESION={len(lesion)}).")
        sys.exit(1)

    eat_full = col(full, "post_eat_rate") * 100.0  # percent
    eat_les = col(lesion, "post_eat_rate") * 100.0
    dng_full = col(full, "post_danger")
    dng_les = col(lesion, "post_danger")

    print("=" * 72)
    print("E7v2 PRE-REGISTERED ANALYSIS")
    print("OSF DOI: 10.17605/OSF.IO/397GJ")
    print("=" * 72)
    print()
    print(f"  FULL    n={len(eat_full)}  eat={eat_full.mean():.3f}% +- {eat_full.std(ddof=1):.3f}  "
          f"danger={dng_full.mean():.1f} +- {dng_full.std(ddof=1):.1f}")
    print(f"  LESION  n={len(eat_les)}   eat={eat_les.mean():.3f}% +- {eat_les.std(ddof=1):.3f}  "
          f"danger={dng_les.mean():.1f} +- {dng_les.std(ddof=1):.1f}")

    print()
    print("-" * 72)
    print("H1: APPETITIVE (eat rate, two-sided Welch)")
    print("-" * 72)
    t_eat, p_eat = sst.ttest_ind(eat_full, eat_les, equal_var=False)
    d_eat = cohens_d(eat_full, eat_les)
    lo_eat, hi_eat = bootstrap_d_ci(eat_full, eat_les)
    bf10_eat = jzs_bf10_two_sample(t_eat, N_PER_GROUP, N_PER_GROUP)
    bf01_eat = 1.0 / bf10_eat
    p_eat_bonf = min(p_eat * 2, 1.0)

    print(f"  Welch t = {t_eat:.4f}   p = {p_eat:.4f}   p_Bonf = {p_eat_bonf:.4f}")
    print(f"  Cohen's d = {d_eat:+.3f}   bootstrap 95% CI = [{lo_eat:+.3f}, {hi_eat:+.3f}]")
    print(f"  BF_10 = {bf10_eat:.4f}   BF_01 = {bf01_eat:.4f}   (JZS, r=0.707)")

    h1_p_ok = p_eat_bonf > ALPHA_RAW
    h1_d_ok = abs(d_eat) < NULL_D_THRESHOLD
    h1_bf_ok = bf01_eat > NULL_BF_THRESHOLD
    h1_confirmed = h1_p_ok and h1_d_ok and h1_bf_ok
    print(f"  Criteria: p_Bonf>0.05 [{'OK' if h1_p_ok else 'FAIL'}], "
          f"|d|<0.5 [{'OK' if h1_d_ok else 'FAIL'}], "
          f"BF_01>3 [{'OK' if h1_bf_ok else 'FAIL'}]")
    print(f"  H1 (appetitive null): {'CONFIRMED' if h1_confirmed else 'NOT CONFIRMED'}")

    print()
    print("-" * 72)
    print("H2: AVERSIVE positive control (danger steps, two-sided Welch)")
    print("-" * 72)
    t_dng, p_dng = sst.ttest_ind(dng_full, dng_les, equal_var=False)
    d_dng = cohens_d(dng_full, dng_les)
    lo_dng, hi_dng = bootstrap_d_ci(dng_full, dng_les)
    p_dng_bonf = min(p_dng * 2, 1.0)

    print(f"  Welch t = {t_dng:.4f}   p = {p_dng:.4f}   p_Bonf = {p_dng_bonf:.4f}")
    print(f"  Cohen's d = {d_dng:+.3f}   bootstrap 95% CI = [{lo_dng:+.3f}, {hi_dng:+.3f}]")

    h2_p_ok = p_dng_bonf < ALPHA_RAW
    h2_d_ok = d_dng < POS_D_THRESHOLD
    h2_confirmed = h2_p_ok and h2_d_ok
    print(f"  Criteria: p_Bonf<0.05 [{'OK' if h2_p_ok else 'FAIL'}], "
          f"d<-1.5 [{'OK' if h2_d_ok else 'FAIL'}]")
    print(f"  H2 (aversive positive): {'CONFIRMED' if h2_confirmed else 'NOT CONFIRMED'}")

    print()
    print("=" * 72)
    print("EXPLORATORY (flagged in prereg as not part of confirmatory tests)")
    print("=" * 72)

    # Paired by seed
    full_by_seed = {int(r["seed"]): float(r["post_eat_rate"]) * 100 for r in full}
    les_by_seed = {int(r["seed"]): float(r["post_eat_rate"]) * 100 for r in lesion}
    seeds_both = sorted(set(full_by_seed) & set(les_by_seed))
    pa = np.array([full_by_seed[s] for s in seeds_both])
    pb = np.array([les_by_seed[s] for s in seeds_both])
    t_pair, p_pair = sst.ttest_rel(pa, pb)
    print(f"  Paired (eat, n={len(seeds_both)}): t={t_pair:.3f}  p={p_pair:.4f}")

    full_dng = {int(r["seed"]): float(r["post_danger"]) for r in full}
    les_dng = {int(r["seed"]): float(r["post_danger"]) for r in lesion}
    pa_d = np.array([full_dng[s] for s in seeds_both])
    pb_d = np.array([les_dng[s] for s in seeds_both])
    t_pair_d, p_pair_d = sst.ttest_rel(pa_d, pb_d)
    print(f"  Paired (danger, n={len(seeds_both)}): t={t_pair_d:.3f}  p={p_pair_d:.4f}")

    # mb_w correlation
    mb_w = col(full + lesion, "mb_w_after_train")
    eat_all = np.concatenate([eat_full, eat_les])
    if mb_w.std() > 0:
        r_mb, p_mb = sst.pearsonr(mb_w, eat_all)
        print(f"  Pearson r(mb_w_after_train, eat_rate) = {r_mb:+.3f}  p={p_mb:.4f}")

    # Final summary
    print()
    print("=" * 72)
    print("PRE-REGISTERED CONCLUSION")
    print("=" * 72)
    if h1_confirmed and h2_confirmed:
        print("  Both H1 and H2 confirmed: appetitive/aversive dissociation supported.")
    elif h1_confirmed and not h2_confirmed:
        print("  H1 confirmed, H2 not confirmed.")
        print("  Appetitive null replicates with proper power.")
        print("  Aversive positive control did not show predicted effect under shortcuts-active test.")
    elif not h1_confirmed and h2_confirmed:
        print("  H1 not confirmed, H2 confirmed.")
    else:
        print("  Neither H1 nor H2 met all pre-registered criteria.")
        print("  Report descriptively; do not claim confirmation.")


if __name__ == "__main__":
    main()
