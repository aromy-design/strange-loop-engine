"""E3 sparsity sweep — parallel."""
import os, sys, csv
import multiprocessing as mp
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = Path(__file__).parent / "results" / "sparsity"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.20, 0.35, 0.50, 0.70, 0.90, 1.20]


def _worker(args):
    threshold, seed, steps = args
    from experiments.sparsity_sweep import run_one
    return run_one(seed, steps, threshold)


def run_parallel_sparsity(seeds, steps, n_workers=12):
    tasks = [(th, s, steps) for th in THRESHOLDS for s in seeds]
    print(f"E3 SPARSITY parallel: {len(tasks)} tasks, workers={n_workers}")
    import time
    t0 = time.time()
    results = []
    with mp.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
            results.append(r)
            print(f"  [{i+1}/{len(tasks)}] th={r['threshold']} seed={r['seed']} eats={r['eats']} mb_rate={r['mb_firing_mean']:.3f}")
    print(f"Total: {time.time()-t0:.0f}s")
    out = OUT / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"Saved {len(results)} rows to {out}")


if __name__ == "__main__":
    mp.freeze_support()
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=15)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    run_parallel_sparsity(seeds=list(range(1, args.seeds + 1)), steps=args.steps, n_workers=args.workers)
