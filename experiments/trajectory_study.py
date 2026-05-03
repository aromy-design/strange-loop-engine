"""
E5 — Consciousness emergence trajectory study.

Long-run (high N steps) recording metrics every K steps to see if:
  - composite consciousness rises during learning
  - awakenings precede mirror-test improvement
  - behavioral competence and consciousness indicators correlate over time

Hypothesis (H4): Awakening events temporally PRECEDE mirror-recognition rises.
"""
import os, sys, csv, time
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.world import GridWorld
from backend.agent import MetabolicAgent

OUT_DIR = Path(__file__).parent / "results" / "trajectory"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_long(seed=1, steps=20000, log_every=200):
    """One long run, dump time series of all relevant metrics."""
    np.random.seed(seed)
    world = GridWorld(seed=seed)
    agent = MetabolicAgent(sensory_dim=world.obs_dim, seed=seed)

    last_ate = False; last_kind = 0; last_dng = False; last_shel = False
    cum_eats = 0; cum_danger = 0; cum_awakenings = 0
    rows = []

    t0 = time.time()
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
            cum_eats += 1
            agent.homeostat.E = min(agent.homeostat.max_e,
                                    agent.homeostat.E + (0.6 if ate_kind == 2 else 0.35))
        if in_d: cum_danger += 1
        if out.get("awakening", {}).get("awakened"): cum_awakenings += 1

        if (i + 1) % log_every == 0:
            rows.append({
                "t": i + 1,
                "cum_eats": cum_eats,
                "cum_danger": cum_danger,
                "cum_awakenings": cum_awakenings,
                "energy": agent.homeostat.E,
                "mirror": out["mirrorTest"]["recognitionScore"],
                "awareness_idx": out["awareness"]["awarenessIndex"],
                "continuity": out["continuity"]["continuity"],
                "presence": out["counterfactual"].get("presence", 0),
                "collapse": out["coupling"].get("collapseIndex", 0),
                "closure_depth": out["meta"]["closureDepth"],
                "self_pe": out["selfPredictionError"],
                "world_pe": out["predictionError"],
            })

    out_csv = OUT_DIR / f"trajectory_seed{seed}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"seed={seed} done in {time.time()-t0:.0f}s, eats={cum_eats}, awakenings={cum_awakenings}")
    print(f"  saved {out_csv}")
    return rows


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    seeds = [1, 2] if args.quick else list(range(1, args.seeds + 1))
    steps = 3000 if args.quick else args.steps
    print(f"E5 trajectory: {len(seeds)} seeds x {steps} steps")
    for s in seeds:
        run_long(seed=s, steps=steps)
