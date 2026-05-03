"""
Run ALL experiments sequentially. Used for full reproducibility check.

WARNING: this takes hours. Only run when you want a complete dataset.
"""
import os, sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.lesion_study import run_all as lesion_run
from experiments.sleep_study import run_all as sleep_run
from experiments.sparsity_sweep import run_all as sparsity_run
from experiments.trajectory_study import run_long
from experiments.generate_diagram import draw as gen_diagram


def main(quick=False):
    t0 = time.time()
    print("=" * 60)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 60)
    print()

    print(">>> generating architecture diagram")
    gen_diagram()

    if quick:
        seeds_lesion = [1, 2, 3]; steps_lesion = 1500
        seeds_sleep = [1, 2, 3]; steps_sleep = 1500
        seeds_sparsity = [1, 2]; steps_sparsity = 1000
        traj_seeds = [1]; traj_steps = 2000
    else:
        seeds_lesion = list(range(1, 11)); steps_lesion = 5000
        seeds_sleep = list(range(1, 16)); steps_sleep = 8000
        seeds_sparsity = list(range(1, 9)); steps_sparsity = 4000
        traj_seeds = list(range(1, 6)); traj_steps = 15000

    print("\n>>> E1 lesion study")
    lesion_run(seeds=seeds_lesion, steps=steps_lesion)

    print("\n>>> E2 sleep study")
    sleep_run(seeds=seeds_sleep, steps=steps_sleep)

    print("\n>>> E3 sparsity sweep")
    sparsity_run(seeds=seeds_sparsity, steps=steps_sparsity)

    print("\n>>> E5 trajectory study")
    for s in traj_seeds:
        run_long(seed=s, steps=traj_steps)

    print(f"\n=== ALL DONE in {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    main(quick=p.parse_args().quick)
