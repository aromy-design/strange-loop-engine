"""
E8 — Toxic World: MB associative memory test.

GridWorldWithToxic has 2 fixed toxic food cells (value 3, distinct visuals).
Toxic food: energy -= 0.4 when eaten (vs +0.35 for plain, +0.6 for sweet).

Test: after 30k training, does MB maintain toxic-food avoidance when
spatial_map is lesioned?

Conditions:
  FULL_FULL      : train + test with all modules
  FULL_LESION_MAP: train with all → lesion spatial_map for test → MB must carry avoidance
  NAIVE_MAP      : no training → test with spatial_map only (spatial map learns from scratch)
  NAIVE_LESION   : no training + no spatial_map → random walk baseline

Key metrics:
  toxic_rate = toxic_eats / total_steps (lower = better avoidance)
  eat_rate   = good_eats / total_steps  (higher = better foraging)
  net_rate   = eat_rate - 2*toxic_rate  (penalized efficiency)

If FULL_LESION_MAP.toxic_rate << NAIVE_LESION.toxic_rate (baseline):
  → MB learned toxic food avoidance and maintains it
If FULL_LESION_MAP.toxic_rate ≈ NAIVE_LESION.toxic_rate:
  → MB did not learn toxic avoidance (only spatial map knows)
"""
import os, sys, csv, time
os.environ["SCALE"] = "large"
import multiprocessing as mp
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTDIR = Path(__file__).parent / "results" / "e8_toxic"


def _run_agent_toxic(steps, world, agent, lesion_spatial_map=False):
    """Run agent for steps steps on a toxic world. Returns metrics."""
    from backend.agent import MetabolicAgent  # import to check lesion
    if lesion_spatial_map:
        agent.lesion.add("spatial_map")
    else:
        agent.lesion.discard("spatial_map")

    last_ate = False; last_kind = 0; last_dng = False; last_shel = False
    cum_good_eats = 0; cum_toxic_eats = 0; cum_danger = 0

    for i in range(steps):
        obs = world.observe(energy=agent.homeostat.E)
        ar, ac = int(world.agent[0]), int(world.agent[1])
        food_pos = np.argwhere(world.grid > 0)
        good_food = np.argwhere((world.grid == 1) | (world.grid == 2))
        fd = (0, 0); sd = (0, 0); fv = []
        if len(good_food) > 0:
            d = good_food - np.array([ar, ac])
            nearest = good_food[int(np.argmin(np.linalg.norm(d, axis=1)))]
            fd = (int(np.sign(nearest[0] - ar)), int(np.sign(nearest[1] - ac)))
            fv = [(int(p[0]), int(p[1])) for p in good_food]
        sd = (int(np.sign(world.shelter[0] - ar)), int(np.sign(world.shelter[1] - ac)))
        sigs = []
        for sr in range(max(0, ar - 2), min(world.SIZE, ar + 3)):
            for sc in range(max(0, ac - 2), min(world.SIZE, ac + 3)):
                k = int(world.signals[sr, sc])
                if k > 0: sigs.append(k)

        out = agent.step(obs, ate_last_action=(last_kind > 0 and last_kind != -1),
                         ate_kind=max(0, last_kind), in_danger=last_dng, in_shelter=last_shel,
                         light_level=world.light_level, agent_pos=(ar, ac), food_visible=fv,
                         shelter_dir=sd, food_dir=fd, danger_dir=(0, 0), signals_in_view=sigs)

        _, ate_kind, _, in_d, in_s = world.step(out["action"])
        last_kind = ate_kind; last_ate = (ate_kind > 0); last_dng = in_d; last_shel = in_s

        if ate_kind > 0:
            cum_good_eats += 1
            agent.homeostat.E = min(agent.homeostat.max_e,
                                    agent.homeostat.E + (0.6 if ate_kind == 2 else 0.35))
        elif ate_kind == -1:
            cum_toxic_eats += 1
            agent.homeostat.E = max(0.01, agent.homeostat.E + world.TOXIC_PENALTY)
            # Octopamine-like aversive signal: anti-Hebbian on vision→MB and MB→motor
            # Weakens connections active when toxic food was consumed
            agent.insect_brain.aversive_conditioning(intensity=abs(world.TOXIC_PENALTY))
        if in_d:
            cum_danger += 1

    return cum_good_eats, cum_toxic_eats, cum_danger


def _worker(args):
    os.environ["SCALE"] = "large"
    seed, train_steps, test_steps, condition = args
    from backend.world_toxic import GridWorldWithToxic
    from backend.agent import MetabolicAgent

    np.random.seed(seed)
    world = GridWorldWithToxic(seed=seed, n_toxic=2)
    agent = MetabolicAgent(sensory_dim=world.obs_dim, seed=seed)

    t0 = time.time()

    do_train = "NAIVE" not in condition
    train_steps_actual = train_steps if do_train else 0

    # Phase 1: Training
    if do_train:
        pre_good, pre_toxic, pre_dng = _run_agent_toxic(
            train_steps, world, agent, lesion_spatial_map=False)
    else:
        pre_good = pre_toxic = pre_dng = 0
        train_steps_actual = 0

    mb_w_before = float(np.abs(agent.insect_brain.W_mush_to_motor.data).mean())

    # Phase 2: Test (apply condition)
    lesion_map_for_test = "LESION_MAP" in condition
    post_good, post_toxic, post_dng = _run_agent_toxic(
        test_steps, world, agent, lesion_spatial_map=lesion_map_for_test)

    elapsed = time.time() - t0
    toxic_rate = post_toxic / test_steps
    eat_rate = post_good / test_steps
    net_rate = eat_rate - 2 * toxic_rate  # penalized efficiency

    result = {
        "seed": seed,
        "condition": condition,
        "train_steps": train_steps_actual,
        "test_steps": test_steps,
        "pre_good_eats": pre_good,
        "pre_toxic_eats": pre_toxic,
        "post_good_eats": post_good,
        "post_toxic_eats": post_toxic,
        "post_eat_rate": eat_rate,
        "post_toxic_rate": toxic_rate,
        "post_net_rate": net_rate,
        "post_danger": post_dng,
        "mb_w_before": mb_w_before,
        "elapsed": elapsed,
    }
    print(f"seed={seed} cond={condition} pre_toxic={pre_toxic}/{train_steps_actual}  "
          f"post_toxic={post_toxic}/{test_steps} ({100*toxic_rate:.2f}%)  "
          f"eat={100*eat_rate:.2f}%  net={100*net_rate:.2f}%  {elapsed:.0f}s", flush=True)
    return result


if __name__ == "__main__":
    mp.freeze_support()
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--train_steps", type=int, default=30000)
    p.add_argument("--test_steps", type=int, default=8000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--conditions", nargs="+",
                   default=["FULL_FULL", "FULL_LESION_MAP", "NAIVE_FULL", "NAIVE_LESION_MAP"])
    args = p.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1, args.seeds + 1))
    tasks = [(s, args.train_steps, args.test_steps, c)
             for c in args.conditions for s in seeds]

    print(f"E8 TOXIC WORLD: {len(tasks)} runs, train={args.train_steps}, "
          f"test={args.test_steps}, workers={args.workers}")
    print(f"  Toxic food = value 3, energy -= 0.4 when eaten")
    print(f"  Question: does trained MB maintain avoidance when spatial_map lesioned?")

    t0 = time.time()
    results = []
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_worker, tasks):
            results.append(r)
    print(f"\nTotal: {time.time()-t0:.0f}s")

    # Save
    out_csv = OUTDIR / "results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in sorted(results, key=lambda x: (x["condition"], x["seed"])):
            w.writerow(r)

    # Quick stats
    try:
        from scipy import stats as sst
        import math
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    conditions = sorted(set(r["condition"] for r in results))
    print("\n=== E8 TOXIC WORLD SUMMARY ===")
    post_toxic = {}; post_eat = {}; post_net = {}
    for cond in conditions:
        subset = [r for r in results if r["condition"] == cond]
        toxic = [r["post_toxic_rate"] * 100 for r in subset]
        eat = [r["post_eat_rate"] * 100 for r in subset]
        net = [r["post_net_rate"] * 100 for r in subset]
        post_toxic[cond] = toxic; post_eat[cond] = eat; post_net[cond] = net
        print(f"  {cond:20s}: toxic={np.mean(toxic):.2f}%±{np.std(toxic,ddof=1):.2f}%  "
              f"eat={np.mean(eat):.2f}%  net={np.mean(net):.2f}%")

    print()
    print("=== KEY COMPARISONS ===")
    if HAS_SCIPY:
        comparisons = [
            ("FULL_LESION_MAP", "NAIVE_LESION_MAP", "toxic_rate",
             "Trained MB helps without spatial map?"),
            ("FULL_FULL", "FULL_LESION_MAP", "toxic_rate",
             "Spatial map ablation effect on trained agent?"),
        ]
        for cond_a, cond_b, metric, question in comparisons:
            if cond_a not in post_toxic or cond_b not in post_toxic: continue
            if metric == "toxic_rate":
                a, b = post_toxic[cond_a], post_toxic[cond_b]
            t, p = sst.ttest_ind(a, b)
            d = (np.mean(a) - np.mean(b)) / max(
                math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2), 1e-9)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            print(f"  {question}")
            print(f"    {cond_a} vs {cond_b}: delta={np.mean(a)-np.mean(b):+.3f}%  "
                  f"t={t:.2f}  p={p:.4f}  d={d:.2f}  {sig}")

    # Save summary
    lines = ["E8 TOXIC WORLD SUMMARY", f"Seeds: {args.seeds}, train={args.train_steps}, "
             f"test={args.test_steps}", ""]
    for cond in conditions:
        t = post_toxic.get(cond, [])
        e = post_eat.get(cond, [])
        lines.append(f"{cond}: toxic={np.mean(t):.3f}%±{np.std(t,ddof=1):.3f}%  "
                     f"eat={np.mean(e):.3f}% (n={len(t)})")
    with open(OUTDIR / "summary.txt", "w") as f: f.write("\n".join(lines))
    print(f"\nSaved {out_csv}")
