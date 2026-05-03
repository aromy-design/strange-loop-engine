"""
E3 — Mushroom body sparsity sweep.

Vary firing threshold to produce different sparsity levels in MB Kenyon
cells. Real Drosophila MB has ~5-10% firing. We test 6 levels and measure
behavioral/cognitive outcomes.

Hypothesis (H3): sparse coding (1-5%) maximizes memory capacity vs
interference, yielding higher associative learning + behavioral performance.
"""
import os, sys, csv, time
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.world import GridWorld
from backend.agent import MetabolicAgent

OUT_DIR = Path(__file__).parent / "results" / "sparsity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# threshold -> approximate firing rate (calibrated empirically)
# higher threshold = sparser firing
THRESHOLDS = [0.20, 0.35, 0.50, 0.70, 0.90, 1.20]


def run_one(seed, steps, mushroom_threshold):
    np.random.seed(seed)
    world = GridWorld(seed=seed)
    agent = MetabolicAgent(sensory_dim=world.obs_dim, seed=seed)
    # override mushroom threshold by patching the brain
    agent.insect_brain.mushroom.threshold = mushroom_threshold

    last_ate = False; last_kind = 0; last_dng = False; last_shel = False
    eats = 0; danger = 0; mb_firing_rates = []
    for i in range(steps):
        obs = world.observe(energy=agent.homeostat.E)
        ar, ac = int(world.agent[0]), int(world.agent[1])
        food_pos = np.argwhere(world.grid > 0)
        if len(food_pos) > 0:
            d = food_pos - np.array([ar, ac])
            dists = np.linalg.norm(d, axis=1)
            nearest = food_pos[int(np.argmin(dists))]
            fd = (np.sign(nearest[0] - ar), np.sign(nearest[1] - ac))
        else:
            fd = (0, 0)
        sd = (np.sign(world.shelter[0] - ar), np.sign(world.shelter[1] - ac))
        dangers = list(world.danger_cells)
        if dangers:
            ddists = [np.hypot(dr - ar, dc - ac) for dr, dc in dangers]
            ndr, ndc = dangers[int(np.argmin(ddists))]
            dd = (np.sign(ndr - ar), np.sign(ndc - ac))
        else:
            dd = (0, 0)
        fv = [(int(p[0]), int(p[1])) for p in food_pos]
        sigs = []
        for sr in range(max(0, ar - 2), min(world.SIZE, ar + 3)):
            for sc in range(max(0, ac - 2), min(world.SIZE, ac + 3)):
                k = int(world.signals[sr, sc])
                if k > 0: sigs.append(k)
        out = agent.step(obs, ate_last_action=last_ate, ate_kind=last_kind,
                         in_danger=last_dng, in_shelter=last_shel,
                         light_level=world.light_level,
                         agent_pos=(ar, ac), food_visible=fv,
                         shelter_dir=sd, food_dir=fd, danger_dir=dd,
                         signals_in_view=sigs)
        _, ate_kind, _, in_d, in_s = world.step(out["action"])
        last_ate = ate_kind > 0; last_kind = ate_kind
        last_dng = in_d; last_shel = in_s
        if last_ate:
            eats += 1
            agent.homeostat.E = min(agent.homeostat.max_e,
                                    agent.homeostat.E + (0.6 if ate_kind == 2 else 0.35))
        if in_d: danger += 1
        if i % 50 == 0:
            mb_firing_rates.append(agent.insect_brain.mushroom.firing_rate())

    return {
        "threshold": mushroom_threshold,
        "seed": seed,
        "steps": steps,
        "eats": eats,
        "danger": danger,
        "deaths": agent.homeostat.deaths,
        "mb_firing_mean": float(np.mean(mb_firing_rates)) if mb_firing_rates else 0.0,
        "final_mirror": float(out["mirrorTest"]["recognitionScore"]),
        "final_collapse": float(out["coupling"].get("collapseIndex", 0)),
    }


def run_all(seeds, steps):
    results = []
    total = len(THRESHOLDS) * len(seeds)
    completed = 0
    print(f"\n=== E3 SPARSITY SWEEP ===")
    print(f"Thresholds: {THRESHOLDS}, seeds: {len(seeds)}, steps: {steps}")
    overall_t0 = time.time()
    for th in THRESHOLDS:
        print(f"--- threshold={th} ---")
        for s in seeds:
            t0 = time.time()
            r = run_one(s, steps, th)
            results.append(r)
            completed += 1
            print(f"  seed={s} eats={r['eats']} mb_rate={r['mb_firing_mean']:.3f} "
                  f"mirror={r['final_mirror']:.2f} ({time.time()-t0:.0f}s) [{completed}/{total}]")

    out = OUT_DIR / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} rows to {out}")
    print(f"Total: {time.time()-overall_t0:.1f}s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    seeds = [1, 2] if args.quick else list(range(1, args.seeds + 1))
    steps = 1500 if args.quick else args.steps
    run_all(seeds=seeds, steps=steps)
