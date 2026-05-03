"""E2 sleep study — parallel."""
import os, sys, csv
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.parallel import run_parallel
from experiments.sleep_study import CONDITIONS

OUT = Path(__file__).parent / "results" / "sleep"
OUT.mkdir(parents=True, exist_ok=True)


def run_full(seeds_count=30, steps=8000, n_workers=12):
    seeds = list(range(1, seeds_count + 1))
    results = run_parallel(CONDITIONS, seeds, steps, n_workers=n_workers, label="E2 SLEEP")
    fields = ["condition", "lesion", "seed", "steps", "total_eats", "eat_rate_pct",
              "total_danger", "deaths", "total_sleep_steps", "awakenings",
              "final_mirror", "final_awareness_idx", "final_continuity",
              "final_collapse_index", "final_closure_depth"]
    out = OUT / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"Saved {len(results)} rows to {out}")


if __name__ == "__main__":
    import argparse, multiprocessing as mp
    mp.freeze_support()
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    run_full(args.seeds, args.steps, args.workers)
