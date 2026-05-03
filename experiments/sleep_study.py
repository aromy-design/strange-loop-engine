"""
E2 — Sleep effect study.

Tests whether sleep+replay improves long-term performance, especially
recovery after world perturbation.

Conditions:
  WITH-SLEEP    full sleep cycle + dream replay active
  NO-SLEEP      sleep disabled, agent stays awake all night

Per condition: N seeds, M steps. Mid-run perturbation tests recovery.

Hypothesis (H2 from RESEARCH.md):
  Sleep-replay reduces catastrophic forgetting after environment perturbation.
"""
import os, sys, csv, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.runner import run_episode

OUT_DIR = Path(__file__).parent / "results" / "sleep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = {
    "WITH-SLEEP": [],
    "NO-SLEEP": ["sleep"],
}


def run_all(seeds, steps):
    results = []
    total = len(CONDITIONS) * len(seeds)
    completed = 0
    overall_t0 = time.time()
    print(f"\n=== E2 SLEEP EFFECT STUDY ===")
    print(f"Conditions: {len(CONDITIONS)}, seeds: {len(seeds)}, steps: {steps}")
    print(f"Total runs: {total}\n")

    for cname, lesion in CONDITIONS.items():
        print(f"--- {cname} ---")
        for seed in seeds:
            t0 = time.time()
            r = run_episode(seed=seed, steps=steps, lesion=lesion)
            r["condition"] = cname
            results.append(r)
            completed += 1
            elapsed = time.time() - t0
            remaining = (time.time() - overall_t0) / completed * (total - completed)
            print(f"  seed={seed} eats={r['total_eats']} danger={r['total_danger']} "
                  f"deaths={r['deaths']} mirror={r['final_mirror']:.2f} "
                  f"({elapsed:.0f}s) [{completed}/{total}, ETA {remaining:.0f}s]")

    fields = ["condition", "lesion", "seed", "steps", "total_eats", "eat_rate_pct",
              "total_danger", "deaths", "total_sleep_steps", "awakenings",
              "final_mirror", "final_awareness_idx", "final_continuity",
              "final_collapse_index", "final_closure_depth"]
    out = OUT_DIR / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} rows to {out}")
    print(f"Total elapsed: {time.time() - overall_t0:.1f}s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=15)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        run_all(seeds=[1, 2, 3], steps=2000)
    else:
        run_all(seeds=list(range(1, args.seeds + 1)), steps=args.steps)
