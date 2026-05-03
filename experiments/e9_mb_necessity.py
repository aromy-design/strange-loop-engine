"""
E9 — MB Necessity Test: Can MB drive behavior when shortcuts are removed?

Design: 2×2 factorial
  train_mode × test_mode
  - shortcuts=FULL:  spatial_map + adjacent_food + behavioral_modes active
  - shortcuts=LESION: all three lesioned (MB must do the work)

Conditions:
  TRAIN_FULL_TEST_FULL    : standard training + test (upper bound — shortcuts help)
  TRAIN_FULL_TEST_LESION  : train with shortcuts, test without (does MB carry behavior?)
  TRAIN_LESION_TEST_LESION: train + test without shortcuts (MB forced to learn)
  TRAIN_LESION_TEST_FULL  : train without shortcuts, test with (sanity — shortcuts restore?)
  NAIVE_TEST_LESION       : no training, no shortcuts at test (random baseline)
  NAIVE_TEST_FULL         : no training, shortcuts at test (shortcut baseline)

Key comparisons:
  TRAIN_LESION_TEST_LESION vs NAIVE_TEST_LESION:
    → Does 30k steps of MB-forced training improve behavior over random?
    → If yes: MB CAN learn goal-directed behavior when it's the only driver.

  TRAIN_FULL_TEST_LESION vs NAIVE_TEST_LESION:
    → Does training-with-shortcuts help MB at test-without-shortcuts?
    → If yes: MB learns passively from reward exposure.
    → If no: MB only learns when it's causally responsible.

Note: LESION mode removes spatial_map, adjacent_food, behavioral_modes from agent.
Motor output is: 0.6 * insect_brain + 0.4 * dense_field, softmax action.

CONFOUND WARNING: the legacy 200-neuron dense_field also trains on reward signal
and has access to dr_food/dc_food in the observation. E9 comparisons reflect:
"trained (MB + dense_field) vs untrained (MB + dense_field)" — not pure MB.
To isolate MB: also lesion 'field' (E9 v2, future experiment).
"""
import os, sys, csv, time
os.environ["SCALE"] = "large"
import multiprocessing as mp
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTDIR = Path(__file__).parent / "results" / "e9_mb_necessity"
SHORTCUT_LESIONS = {"spatial_map", "adjacent_food", "behavioral_modes"}


def _set_shortcuts(agent, lesioned: bool):
    if lesioned:
        agent.lesion.update(SHORTCUT_LESIONS)
    else:
        agent.lesion.difference_update(SHORTCUT_LESIONS)


def _run_agent(steps, world, agent, blind=False):
    """
    blind=True: suppress food_dir, shelter_dir, food_visible to agent.
    When True, agent cannot use geometric food cues — only insect brain + world model.
    This isolates MB contribution from EXPLORE-mode food-direction bias.
    """
    last_ate = False; last_kind = 0; last_dng = False; last_shel = False
    cum_eats = 0; cum_danger = 0

    for i in range(steps):
        obs = world.observe(energy=agent.homeostat.E)
        ar, ac = int(world.agent[0]), int(world.agent[1])
        food_pos = np.argwhere(world.grid > 0)
        if not blind:
            fd = (0, 0); sd = (0, 0); fv = []
            if len(food_pos) > 0:
                d = food_pos - np.array([ar, ac])
                nearest = food_pos[int(np.argmin(np.linalg.norm(d, axis=1)))]
                fd = (int(np.sign(nearest[0] - ar)), int(np.sign(nearest[1] - ac)))
                fv = [(int(p[0]), int(p[1])) for p in food_pos]
            sd = (int(np.sign(world.shelter[0] - ar)), int(np.sign(world.shelter[1] - ac)))
        else:
            fd = (0, 0); sd = (0, 0); fv = []  # blind: no directional cues
        sigs = []
        for sr in range(max(0, ar - 2), min(world.SIZE, ar + 3)):
            for sc in range(max(0, ac - 2), min(world.SIZE, ac + 3)):
                k = int(world.signals[sr, sc])
                if k > 0: sigs.append(k)
        out = agent.step(obs, ate_last_action=last_ate, ate_kind=last_kind,
                         in_danger=last_dng, in_shelter=last_shel,
                         light_level=world.light_level, agent_pos=(ar, ac),
                         food_visible=fv, shelter_dir=sd, food_dir=fd,
                         danger_dir=(0, 0), signals_in_view=sigs)
        _, ate_kind, _, in_d, in_s = world.step(out["action"])
        last_ate = ate_kind > 0; last_kind = ate_kind
        last_dng = in_d; last_shel = in_s
        if last_ate:
            cum_eats += 1
            agent.homeostat.E = min(agent.homeostat.max_e,
                                    agent.homeostat.E + (0.6 if ate_kind == 2 else 0.35))
        if in_d:
            cum_danger += 1

    return cum_eats, cum_danger


def _worker(args):
    os.environ["SCALE"] = "large"
    seed, train_steps, test_steps, condition = args
    from backend.world import GridWorld
    from backend.agent import MetabolicAgent

    np.random.seed(seed)
    world = GridWorld(seed=seed)
    agent = MetabolicAgent(sensory_dim=world.obs_dim, seed=seed)

    t0 = time.time()

    # Parse condition
    do_train = "NAIVE" not in condition
    train_lesion = "TRAIN_LESION" in condition
    test_lesion = "TEST_LESION" in condition
    # BLIND conditions suppress food_dir/shelter_dir too (isolation of insect brain)
    test_blind = "BLIND" in condition

    # Phase 1: Training
    if do_train:
        _set_shortcuts(agent, lesioned=train_lesion)
        pre_eats, pre_danger = _run_agent(train_steps, world, agent, blind=train_lesion)
        train_steps_actual = train_steps
    else:
        pre_eats = pre_danger = 0
        train_steps_actual = 0

    mb_w_before = float(np.abs(agent.insect_brain.W_mush_to_motor.data).mean())

    # Phase 2: Test (apply test condition)
    _set_shortcuts(agent, lesioned=test_lesion)
    post_eats, post_danger = _run_agent(test_steps, world, agent, blind=(test_lesion or test_blind))

    elapsed = time.time() - t0
    post_rate = post_eats / test_steps
    pre_rate = pre_eats / train_steps_actual if train_steps_actual > 0 else 0.0

    result = {
        "seed": seed,
        "condition": condition,
        "train_steps": train_steps_actual,
        "test_steps": test_steps,
        "train_lesion": int(train_lesion),
        "test_lesion": int(test_lesion),
        "test_blind": int(test_lesion or test_blind),
        "pre_eats": pre_eats,
        "post_eats": post_eats,
        "pre_eats_rate": pre_rate,
        "post_eats_rate": post_rate,
        "pre_danger": pre_danger,
        "post_danger": post_danger,
        "mb_w_before": mb_w_before,
        "elapsed": elapsed,
    }
    print(f"seed={seed} cond={condition} pre={100*pre_rate:.1f}%  "
          f"post={post_eats}/{test_steps}={100*post_rate:.1f}%  "
          f"mb_w={mb_w_before:.4f}  {elapsed:.0f}s")
    return result


if __name__ == "__main__":
    mp.freeze_support()
    import argparse
    from scipy import stats as sst
    import math

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--train_steps", type=int, default=30000)
    p.add_argument("--test_steps", type=int, default=8000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--conditions", nargs="+", default=[
        "TRAIN_FULL_TEST_FULL",
        "TRAIN_FULL_TEST_LESION",
        "TRAIN_LESION_TEST_LESION",
        "NAIVE_TEST_LESION",
        "NAIVE_TEST_FULL",
    ])
    args = p.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1, args.seeds + 1))
    tasks = [(s, args.train_steps, args.test_steps, c)
             for c in args.conditions for s in seeds]

    print(f"E9 MB NECESSITY: {len(tasks)} runs, train={args.train_steps}, "
          f"test={args.test_steps}, workers={args.workers}")
    print("  Shortcuts lesioned: spatial_map + adjacent_food + behavioral_modes")
    print("  Key Q: does MB-forced training outperform random walk?")

    t0 = time.time()
    results = []
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_worker, tasks):
            results.append(r)
    print(f"\nTotal: {time.time()-t0:.0f}s")

    out_csv = OUTDIR / "results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in sorted(results, key=lambda x: (x["condition"], x["seed"])):
            w.writerow(r)

    conditions = sorted(set(r["condition"] for r in results))
    print("\n=== E9 MB NECESSITY SUMMARY ===")
    post_rates = {}
    for cond in conditions:
        subset = [r for r in results if r["condition"] == cond]
        rates = [r["post_eats_rate"] * 100 for r in subset]
        post_rates[cond] = rates
        print(f"  {cond:30s}: {np.mean(rates):.2f}%±{np.std(rates,ddof=1):.2f}%")

    print()
    print("=== KEY COMPARISON ===")
    comparisons = [
        ("TRAIN_LESION_TEST_LESION", "NAIVE_TEST_LESION",
         "MB-forced training vs random walk (lesion test):"),
        ("TRAIN_FULL_TEST_LESION", "NAIVE_TEST_LESION",
         "Training-with-shortcuts vs random walk (lesion test):"),
        ("TRAIN_LESION_TEST_LESION", "TRAIN_FULL_TEST_LESION",
         "Forced-MB-training vs shortcut-training (same test):"),
    ]
    for cond_a, cond_b, label in comparisons:
        if cond_a not in post_rates or cond_b not in post_rates: continue
        a, b = post_rates[cond_a], post_rates[cond_b]
        t, pv = sst.ttest_ind(a, b)
        d = (np.mean(a) - np.mean(b)) / max(
            math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2), 1e-9)
        sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
        print(f"  {label}")
        print(f"    delta={np.mean(a)-np.mean(b):+.3f}%  t={t:.2f}  p={pv:.4f}  d={d:.2f}  {sig}")

    lines = [f"E9 MB NECESSITY: seeds={args.seeds} train={args.train_steps} test={args.test_steps}", ""]
    for cond in conditions:
        r = post_rates.get(cond, [])
        lines.append(f"{cond}: {np.mean(r):.3f}%±{np.std(r,ddof=1):.3f}% (n={len(r)})")
    with open(OUTDIR / "summary.txt", "w") as f: f.write("\n".join(lines))
    print(f"\nSaved {out_csv}")
