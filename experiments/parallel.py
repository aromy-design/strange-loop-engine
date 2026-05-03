"""
Parallel experiment runner using multiprocessing.Pool.

Each (condition, seed) tuple runs in a worker process. CPU saturated.
4-16x speedup on consumer multi-core CPUs.
"""
import os, sys, csv, time
from pathlib import Path
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _worker(args):
    """Worker function for Pool. Imports inside to be picklable on Windows."""
    from experiments.runner import run_episode
    cond_name, lesion, seed, steps = args
    r = run_episode(seed=seed, steps=steps, lesion=lesion, verbose=False)
    r["condition"] = cond_name
    return r


def run_parallel(conditions, seeds, steps, n_workers=None, label="experiment"):
    """
    conditions: dict {name: lesion_list}
    seeds: list of int
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    tasks = []
    for cname, lesion in conditions.items():
        for s in seeds:
            tasks.append((cname, lesion, s, steps))

    print(f"\n=== {label} (PARALLEL) ===")
    print(f"Conditions: {len(conditions)}, seeds: {len(seeds)}, steps: {steps}")
    print(f"Total tasks: {len(tasks)}, workers: {n_workers}")
    t0 = time.time()

    results = []
    with mp.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
            results.append(r)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(tasks) - i - 1)
            print(f"  [{i+1}/{len(tasks)}] {r['condition']:24s} seed={r['seed']:2d} "
                  f"eats={r['total_eats']:4d} mirror={r['final_mirror']:.2f} "
                  f"deaths={r['deaths']} ETA={eta:.0f}s")

    print(f"\nTotal: {time.time() - t0:.1f}s ({len(tasks)/(time.time()-t0):.2f} runs/sec)")
    return results


if __name__ == "__main__":
    # quick smoke test
    conds = {"FULL": [], "no-spatial-map": ["spatial_map"]}
    rs = run_parallel(conds, seeds=[1, 2, 3, 4], steps=1500, label="smoke")
    print(f"Got {len(rs)} results")
