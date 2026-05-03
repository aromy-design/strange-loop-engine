"""
Parallel version of E1 lesion study.

Speedup target: 4-8x on consumer CPU. With 30 seeds × 10 conditions × 5000 steps:
  serial: ~50 min
  parallel (8 cores): ~7-8 min
"""
import os, sys, csv
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.parallel import run_parallel
from experiments.lesion_study import CONDITIONS

OUT_DIR = Path(__file__).parent / "results" / "lesion"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_full(seeds_count=30, steps=5000, n_workers=None):
    seeds = list(range(1, seeds_count + 1))
    results = run_parallel(CONDITIONS, seeds, steps, n_workers=n_workers, label="E1 LESION")

    fields = ["condition", "lesion", "seed", "steps", "total_eats", "eat_rate_pct",
              "total_danger", "danger_rate_pct", "deaths", "total_shelter",
              "total_sleep_steps", "awakenings", "final_mirror", "final_awareness_idx",
              "final_continuity", "final_presence", "final_collapse_index",
              "final_closure_depth", "ms_per_step", "elapsed_s"]
    out = OUT_DIR / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"Saved {len(results)} rows to {out}")


if __name__ == "__main__":
    import argparse, multiprocessing as mp
    mp.freeze_support()
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        run_full(seeds_count=4, steps=1500, n_workers=args.workers)
    else:
        run_full(seeds_count=args.seeds, steps=args.steps, n_workers=args.workers)
