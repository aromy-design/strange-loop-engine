"""E5 trajectory parallel — long runs with time series logging."""
import os, sys, csv
import multiprocessing as mp
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _worker(args):
    seed, steps, log_every = args
    from experiments.trajectory_study import run_long
    return run_long(seed=seed, steps=steps, log_every=log_every)


if __name__ == "__main__":
    mp.freeze_support()
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    seeds = list(range(1, args.seeds + 1))
    tasks = [(s, args.steps, args.log_every) for s in seeds]
    print(f"E5 TRAJECTORY parallel: {len(tasks)} tasks ({args.steps} steps each)")
    import time
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_worker, tasks):
            pass
    print(f"Total: {time.time()-t0:.0f}s")
