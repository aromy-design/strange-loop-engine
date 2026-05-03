"""
Headless agent runner — runs N steps without UI, returns metrics dict.

Used by experiments to run deterministic, multi-seed, multi-condition trials.
"""
import numpy as np
import time
import os
import sys

# allow running as `python -m experiments.runner` from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.world import GridWorld
from backend.agent import MetabolicAgent


def run_episode(seed=0, steps=10000, lesion=None, world_seed=None, log_interval=500, verbose=False, minimal=False):
    """
    minimal=True: skip expensive battery + meta-validation per tick.
    Use for behavior-only experiments. Saves ~50% time.
    """
    """
    Returns dict with summary metrics + time series sampled at log_interval.
    """
    if world_seed is None:
        world_seed = seed
    np.random.seed(seed)

    world = GridWorld(seed=world_seed)
    agent = MetabolicAgent(sensory_dim=world.obs_dim, seed=seed, lesion=lesion or [])

    total_eats = 0
    total_danger = 0
    total_shelter = 0
    total_sleep_steps = 0
    last_ate = False
    last_kind = 0
    last_dng = False
    last_shel = False
    eat_per_bucket = []
    composite_series = []
    mirror_series = []
    awakenings = 0

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
            ddists = [np.hypot(dr_ - ar, dc_ - ac) for dr_, dc_ in dangers]
            ndr, ndc = dangers[int(np.argmin(ddists))]
            dd = (np.sign(ndr - ar), np.sign(ndc - ac))
        else:
            dd = (0, 0)
        fv = [(int(p[0]), int(p[1])) for p in food_pos]
        sigs = []
        for sr in range(max(0, ar - 2), min(world.SIZE, ar + 3)):
            for sc in range(max(0, ac - 2), min(world.SIZE, ac + 3)):
                k = int(world.signals[sr, sc])
                if k > 0:
                    sigs.append(k)

        out = agent.step(
            obs,
            ate_last_action=last_ate,
            ate_kind=last_kind,
            in_danger=last_dng,
            in_shelter=last_shel,
            light_level=world.light_level,
            agent_pos=(ar, ac),
            food_visible=fv,
            shelter_dir=sd,
            food_dir=fd,
            danger_dir=dd,
            signals_in_view=sigs,
        )

        _, ate_kind, _, in_d, in_s = world.step(out["action"])
        last_ate = ate_kind > 0
        last_kind = ate_kind
        last_dng = in_d
        last_shel = in_s

        if last_ate:
            total_eats += 1
            agent.homeostat.E = min(agent.homeostat.max_e, agent.homeostat.E + (0.6 if ate_kind == 2 else 0.35))
        if in_d:
            total_danger += 1
        if in_s:
            total_shelter += 1
        if out.get("sleep", {}).get("isSleeping"):
            total_sleep_steps += 1
        if out.get("awakening", {}).get("awakened"):
            awakenings += 1

        if (i + 1) % log_interval == 0:
            eat_per_bucket.append(total_eats - sum(eat_per_bucket))
            mirror_series.append(out["mirrorTest"]["recognitionScore"])

    elapsed = time.time() - t0

    summary = {
        "steps": steps,
        "elapsed_s": float(elapsed),
        "ms_per_step": float(elapsed * 1000 / max(steps, 1)),
        "total_eats": int(total_eats),
        "eat_rate_pct": float(total_eats / steps * 100),
        "total_danger": int(total_danger),
        "danger_rate_pct": float(total_danger / steps * 100),
        "total_shelter": int(total_shelter),
        "total_sleep_steps": int(total_sleep_steps),
        "deaths": int(agent.homeostat.deaths),
        "awakenings": int(awakenings),
        "final_mirror": float(out["mirrorTest"]["recognitionScore"]),
        "final_awareness_idx": float(out["awareness"]["awarenessIndex"]),
        "final_continuity": float(out["continuity"]["continuity"]),
        "final_presence": float(out["counterfactual"].get("presence", 0)),
        "final_collapse_index": float(out["coupling"].get("collapseIndex", 0)),
        "final_closure_depth": int(out["meta"]["closureDepth"]),
        "lesion": ",".join(sorted(lesion or [])) or "FULL",
        "seed": int(seed),
    }
    summary["eat_per_bucket"] = eat_per_bucket
    summary["mirror_series"] = mirror_series

    if verbose:
        print(f"  [seed={seed} lesion={summary['lesion']}] "
              f"eats={total_eats}/{steps} ({summary['eat_rate_pct']:.2f}%) "
              f"danger={total_danger} mirror={summary['final_mirror']:.3f} "
              f"deaths={agent.homeostat.deaths} {elapsed:.1f}s")

    return summary


if __name__ == "__main__":
    # quick smoke
    s = run_episode(seed=42, steps=2000, verbose=True)
    print(s)
