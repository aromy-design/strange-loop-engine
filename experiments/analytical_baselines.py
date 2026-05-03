"""
Analytical & empirical baselines for foraging task.

Provides closed-form theoretical bounds and Monte Carlo simulations
for three reference policies:

  1. RANDOM_WALK: uniform random over 6 actions
  2. GREEDY_ORACLE: omniscient — moves toward nearest food every step
  3. ANALYTICAL_UNIFORM: closed-form upper bound for any ergodic uniform policy

Used as defensible baselines in PAPER.md results section.

Usage:
  python -m experiments.analytical_baselines --seeds 30 --steps 8000

Output: experiments/results/baselines/results.csv + summary
"""
import os, sys, csv, time
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.world import GridWorld

OUTDIR = Path(__file__).parent / "results" / "baselines"


def closed_form_uniform_upper_bound():
    """
    Analytical upper bound for uniform-coverage policies.

    Grid: 16x16 = 256 cells.
    Food cells: 6 (4 plain + 2 sweet) maintained constant via respawn.
    Danger cells: 8.
    Shelter: 1 cell.
    Reachable empty cells: 256 - 8 = 248 (shelter is reachable; food spawns avoid danger+shelter).

    For an ergodic uniform-coverage policy, in stationary distribution
    P(agent on food cell) = 6 / 256 ≈ 2.34%

    BUT: not every step is a "movement" step. With 6-action policy where
    only 4 actions move (0/1/2/3 = up/down/left/right; 4/5 = look/speak):
      P(movement step) = 4/6 ≈ 0.667

    Eat happens whenever post-action cell has food (regardless of move).
    Since food respawns immediately, cell-occupancy is the binding factor.

    Therefore the closed-form upper bound is:
      eat_rate_uniform = N_food / N_cells = 6/256 = 2.34%

    This is the THEORETICAL CEILING for any ergodic policy with no spatial
    targeting bias. Anything above this implies non-uniform spatial preference
    (i.e., the agent learned where food is).

    Returns: dict with theoretical bounds.
    """
    N_grid = 16 * 16
    N_food = 6  # 4 plain + 2 sweet
    N_danger = 8  # 3x3 minus center corridor
    N_shelter = 1
    N_action = 6
    N_movement = 4

    # Food density on full grid (food can also spawn near shelter; danger excluded for spawn but reachable)
    # The agent CAN walk into danger cells (they don't have food but drain energy)
    # so the relevant denominator for "where am I" is full 256.
    p_on_food_uniform = N_food / N_grid

    # Movement fraction (only matters for movement-based policies; we eat on
    # any tick where post-action cell has food, including 'look' and 'speak'
    # actions which keep position unchanged)
    p_movement = N_movement / N_action

    return {
        "N_grid": N_grid,
        "N_food": N_food,
        "N_danger": N_danger,
        "p_on_food_stationary_uniform": p_on_food_uniform,
        "p_movement_per_action": p_movement,
        "eat_rate_upper_bound_pct": 100 * p_on_food_uniform,
        "note": "Upper bound for ergodic uniform-coverage policies. Above implies learned bias.",
    }


def run_random_walk(seed, steps):
    """Pure uniform random over 6 actions."""
    rng = np.random.default_rng(seed)
    world = GridWorld(seed=seed)
    eats = 0
    danger = 0
    for _ in range(steps):
        a = int(rng.integers(0, world.N_ACTIONS))
        _, ate_kind, _, in_d, _ = world.step(a)
        if ate_kind > 0:
            eats += 1
        if in_d:
            danger += 1
    return eats, danger


def run_greedy_oracle(seed, steps):
    """Omniscient greedy: each step move toward nearest food (no path planning)."""
    world = GridWorld(seed=seed)
    eats = 0
    danger = 0
    for _ in range(steps):
        food_pos = np.argwhere(world.grid > 0)
        if len(food_pos) == 0:
            a = 4  # look
        else:
            ar, ac = world.agent
            d = food_pos - np.array([ar, ac])
            nearest = food_pos[int(np.argmin(np.linalg.norm(d, axis=1)))]
            dr = nearest[0] - ar
            dc = nearest[1] - ac
            # decide axis: move along largest gap
            if abs(dr) >= abs(dc) and dr != 0:
                a = 0 if dr < 0 else 1  # up else down
            elif dc != 0:
                a = 2 if dc < 0 else 3  # left else right
            else:
                a = 4  # already on food (shouldn't happen since food was eaten)
        _, ate_kind, _, in_d, _ = world.step(a)
        if ate_kind > 0:
            eats += 1
        if in_d:
            danger += 1
    return eats, danger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--steps", type=int, default=8000)
    args = p.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=== ANALYTICAL CLOSED-FORM BOUND ===")
    bound = closed_form_uniform_upper_bound()
    for k, v in bound.items():
        print(f"  {k}: {v}")

    print(f"\n=== EMPIRICAL BASELINES ({args.seeds} seeds × {args.steps} steps) ===")
    t0 = time.time()
    results = []

    for policy_name, fn in [("RANDOM_WALK", run_random_walk),
                            ("GREEDY_ORACLE", run_greedy_oracle)]:
        rates = []
        for seed in range(1, args.seeds + 1):
            eats, danger = fn(seed, args.steps)
            rate = 100 * eats / args.steps
            rates.append(rate)
            results.append({
                "policy": policy_name,
                "seed": seed,
                "eats": eats,
                "eat_rate_pct": rate,
                "danger_steps": danger,
            })
        m = np.mean(rates)
        s = np.std(rates, ddof=1)
        print(f"  {policy_name:15s}: {m:.3f}% ± {s:.3f}%  "
              f"(min {min(rates):.2f}, max {max(rates):.2f})")

    # Save
    with open(OUTDIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)

    # Save analytical bound
    with open(OUTDIR / "analytical_bound.txt", "w") as f:
        f.write("=== ANALYTICAL CLOSED-FORM BOUND ===\n")
        for k, v in bound.items():
            f.write(f"{k}: {v}\n")

    print(f"\nElapsed: {time.time()-t0:.1f}s")
    print(f"Saved {OUTDIR}/results.csv")


if __name__ == "__main__":
    main()
