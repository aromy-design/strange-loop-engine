"""
E1 — Lesion study orchestrator.

Runs N seeds across K conditions, saves CSV for stats analysis.

Conditions:
  FULL                — baseline (no lesion)
  no-spatial-map      — TD value map disabled
  no-behavioral-modes — fixed EXPLORE only
  no-sleep            — no rest cycle
  no-insect-brain     — no spiking circuits
  no-adjacent-food    — no reflex eating
  no-mushroom         — memory circuit silenced
  no-central-complex  — navigation circuit silenced
  no-lateral-horn     — instincts silenced
  no-vision           — visual processing silenced

Output: experiments/results/lesion/seed_*_*.csv + summary.csv
"""
import os, sys, csv, time, json
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.runner import run_episode

OUT_DIR = Path(__file__).parent / "results" / "lesion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Conditions: name -> set of lesion flags
CONDITIONS = {
    "FULL": [],
    "no-spatial-map": ["spatial_map"],
    "no-behavioral-modes": ["behavioral_modes"],
    "no-sleep": ["sleep"],
    "no-insect-brain": ["insect_brain"],
    "no-adjacent-food": ["adjacent_food"],
    "no-mushroom": ["mushroom"],
    "no-central-complex": ["central_complex"],
    "no-lateral-horn": ["lateral_horn"],
    "no-vision": ["vision"],
}

DEFAULT_SEEDS = list(range(1, 11))   # 10 seeds for pilot
DEFAULT_STEPS = 5000


def run_all(seeds=DEFAULT_SEEDS, steps=DEFAULT_STEPS, conditions=None):
    if conditions is None:
        conditions = CONDITIONS

    results = []
    total_runs = len(conditions) * len(seeds)
    completed = 0
    overall_t0 = time.time()

    print(f"\n=== E1 LESION STUDY ===")
    print(f"Conditions: {len(conditions)}, seeds: {len(seeds)}, steps each: {steps}")
    print(f"Total runs: {total_runs}\n")

    for cond_name, lesion in conditions.items():
        print(f"--- {cond_name} ---")
        for seed in seeds:
            t0 = time.time()
            r = run_episode(seed=seed, steps=steps, lesion=lesion, verbose=False)
            r["condition"] = cond_name
            results.append(r)
            completed += 1
            elapsed = time.time() - t0
            remaining = (time.time() - overall_t0) / completed * (total_runs - completed)
            print(f"  seed={seed} eats={r['total_eats']} danger={r['total_danger']} "
                  f"mirror={r['final_mirror']:.2f} deaths={r['deaths']} "
                  f"({elapsed:.0f}s) [done {completed}/{total_runs}, ETA {remaining:.0f}s]")

    # save CSV
    if not results:
        print("No results to save.")
        return

    fields = ["condition", "lesion", "seed", "steps", "total_eats", "eat_rate_pct",
              "total_danger", "danger_rate_pct", "deaths", "total_shelter",
              "total_sleep_steps", "awakenings", "final_mirror", "final_awareness_idx",
              "final_continuity", "final_presence", "final_collapse_index",
              "final_closure_depth", "ms_per_step", "elapsed_s"]
    out_csv = OUT_DIR / "summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nSaved {len(results)} rows to {out_csv}")
    print(f"Total elapsed: {time.time() - overall_t0:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds")
    parser.add_argument("--steps", type=int, default=5000, help="steps per run")
    parser.add_argument("--quick", action="store_true", help="3 seeds × 2000 steps for smoke")
    args = parser.parse_args()
    if args.quick:
        run_all(seeds=[1, 2, 3], steps=2000)
    else:
        run_all(seeds=list(range(1, args.seeds + 1)), steps=args.steps)
