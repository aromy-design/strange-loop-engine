"""
E6 — Long training with InsectBrainV2 (81k neurons).

Tests whether mushroom body engages over extended experience. Measures:
  - eat rate per 1000-step bucket (does behavior improve?)
  - MB firing rate (does sparse coding remain?)
  - MB→motor projection weight magnitude (does plasticity strengthen relevant edges?)
  - all 13 consciousness indicators at end

Forces SCALE=large for this experiment.
"""
import os, sys, csv, time
os.environ["SCALE"] = "large"  # use V2 81k brain
import multiprocessing as mp
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _worker(args):
    """Run one long-training agent. Must set SCALE before importing agent."""
    os.environ["SCALE"] = "large"
    seed, steps, log_every = args
    from backend.world import GridWorld
    from backend.agent import MetabolicAgent

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
            mb = agent.insect_brain.mushroom
            ib = agent.insect_brain
            mb_to_motor_w = float(np.abs(ib.W_mush_to_motor.data).mean())
            mb_to_motor_max = float(np.abs(ib.W_mush_to_motor.data).max())
            cx_to_motor_w = float(np.abs(ib.W_cx_to_motor.data).mean())
            lh_to_motor_w = float(np.abs(ib.W_lh_to_motor.data).mean())
            seg_to_motor_w = float(np.abs(ib.W_seg_to_motor.data).mean())
            vis_to_mush_w = float(np.abs(ib.W_vis_to_mush.data).mean())
            rows.append({
                "seed": seed,
                "t": i + 1,
                "cum_eats": cum_eats,
                "cum_danger": cum_danger,
                "cum_awakenings": cum_awakenings,
                "energy": agent.homeostat.E,
                "mb_firing_rate": mb.firing_rate(),
                "mb_w_self_mean": float(np.abs(mb.W.data).mean()),
                "mb_to_motor_w_mean": mb_to_motor_w,
                "mb_to_motor_w_max": mb_to_motor_max,
                "cx_to_motor_w_mean": cx_to_motor_w,
                "lh_to_motor_w_mean": lh_to_motor_w,
                "seg_to_motor_w_mean": seg_to_motor_w,
                "vis_to_mush_w_mean": vis_to_mush_w,
                "mirror": out["mirrorTest"]["recognitionScore"],
                "awareness_idx": out["awareness"]["awarenessIndex"],
                "collapse": out["coupling"].get("collapseIndex", 0),
                "presence": out["counterfactual"].get("presence", 0),
                "self_pe": out["selfPredictionError"],
            })

    elapsed = time.time() - t0
    out_csv = Path(__file__).parent / "results" / "long_training" / f"seed{seed}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"seed={seed} done in {elapsed:.0f}s eats={cum_eats}/{steps} ({cum_eats/steps*100:.2f}%) danger={cum_danger}")
    return {"seed": seed, "elapsed": elapsed, "eats": cum_eats, "steps": steps}


if __name__ == "__main__":
    mp.freeze_support()
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    seeds = list(range(1, args.seeds + 1))
    tasks = [(s, args.steps, args.log_every) for s in seeds]
    print(f"E6 LONG-TRAINING (V2 81k brain): {len(tasks)} agents x {args.steps} steps, {args.workers} workers")
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for _ in pool.imap_unordered(_worker, tasks):
            pass
    print(f"Total: {time.time()-t0:.0f}s")
