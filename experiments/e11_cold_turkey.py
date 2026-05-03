"""
E11 — Cold-Turkey Control: isolate progressive removal vs raw practice without shortcuts.

If E10 finds SCAFFOLDED_4PH > FULL_LESION_TEST, the improvement could come from:
  (a) Progressive removal: each phase teaches MB to substitute one shortcut at a time
  (b) Practice without shortcuts: any time spent training in lesioned state helps,
      regardless of whether the removal was gradual

E11 distinguishes (a) vs (b) with a third condition:

  COLD_TURKEY_45_15: train 45k with full shortcuts, then 15k cold (all removed) → test
  - Same 60k training budget as SCAFFOLDED_4PH
  - Same 15k of "lesioned training" as SCAFFOLDED_4PH phase 4
  - But NO progressive curriculum

Comparisons (all on test_steps with all shortcuts lesioned + blind):
  SCAFFOLDED_4PH > COLD_TURKEY_45_15: progressive removal specifically helps
  COLD_TURKEY_45_15 > FULL_LESION_TEST: any cold practice helps (without curriculum)
  COLD_TURKEY_45_15 ≈ FULL_LESION_TEST: progressive removal is the only effective lever

This pre-registers the 3-way logical decomposition. Run AFTER E10 if E10 is positive.
If E10 is NULL, skip E11 (no effect to decompose).
"""
import os, sys, csv, time
os.environ["SCALE"] = "large"
import multiprocessing as mp
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTDIR = Path(__file__).parent / "results" / "e11_cold_turkey"
SHORTCUT_LESIONS = {"spatial_map", "adjacent_food", "behavioral_modes"}


def _set_lesion(agent, lesions: set):
    agent.lesion = lesions.copy()


def _run_agent_phase(steps, world, agent, blind=False):
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
            fd = (0, 0); sd = (0, 0); fv = []

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
    seed, full_steps, cold_steps, test_steps, condition = args
    from backend.world import GridWorld
    from backend.agent import MetabolicAgent

    np.random.seed(seed)
    world = GridWorld(seed=seed)
    agent = MetabolicAgent(sensory_dim=world.obs_dim, seed=seed)

    t0 = time.time()

    if condition == "NAIVE_LESION":
        _set_lesion(agent, SHORTCUT_LESIONS)
        post_eats, post_danger = _run_agent_phase(test_steps, world, agent, blind=True)
        train_total = 0

    elif condition == "FULL_LESION_TEST":
        # 60k full shortcuts, then test lesioned + blind (matches E9/E10 baseline)
        _set_lesion(agent, set())
        _run_agent_phase(full_steps + cold_steps, world, agent, blind=False)
        train_total = full_steps + cold_steps
        _set_lesion(agent, SHORTCUT_LESIONS)
        post_eats, post_danger = _run_agent_phase(test_steps, world, agent, blind=True)

    elif condition == "COLD_TURKEY":
        # full_steps with shortcuts, then cold_steps WITHOUT shortcuts (training, blind)
        _set_lesion(agent, set())
        _run_agent_phase(full_steps, world, agent, blind=False)
        _set_lesion(agent, SHORTCUT_LESIONS)
        _run_agent_phase(cold_steps, world, agent, blind=True)
        train_total = full_steps + cold_steps
        post_eats, post_danger = _run_agent_phase(test_steps, world, agent, blind=True)

    else:
        raise ValueError(f"Unknown condition: {condition}")

    elapsed = time.time() - t0
    mb_w = float(np.abs(agent.insect_brain.W_mush_to_motor.data).mean())
    post_rate = post_eats / test_steps if test_steps > 0 else 0.0

    result = {
        "seed": seed,
        "condition": condition,
        "train_total": train_total,
        "test_steps": test_steps,
        "post_eats": post_eats,
        "post_eats_rate": post_rate,
        "post_danger": post_danger,
        "mb_w_final": mb_w,
        "elapsed": elapsed,
    }
    print(f"seed={seed} cond={condition} post={post_eats}/{test_steps}={100*post_rate:.2f}%  "
          f"mb_w={mb_w:.4f}  {elapsed:.0f}s", flush=True)
    return result


if __name__ == "__main__":
    mp.freeze_support()
    import argparse
    from scipy import stats as sst
    import math

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--full_steps", type=int, default=45000,
                   help="Steps with full shortcuts (default 45k)")
    p.add_argument("--cold_steps", type=int, default=15000,
                   help="Steps with all shortcuts lesioned (default 15k, matches E10 phase4)")
    p.add_argument("--test_steps", type=int, default=8000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--conditions", nargs="+",
                   default=["COLD_TURKEY", "FULL_LESION_TEST", "NAIVE_LESION"])
    args = p.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1, args.seeds + 1))
    tasks = [(s, args.full_steps, args.cold_steps, args.test_steps, c)
             for c in args.conditions for s in seeds]

    print(f"E11 COLD-TURKEY: {len(tasks)} runs, full={args.full_steps}, "
          f"cold={args.cold_steps}, test={args.test_steps}, workers={args.workers}")
    print(f"  Total training budget per run: {args.full_steps + args.cold_steps}")
    print(f"  Random walk blind baseline (E9): ~0.69%")

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

    # Summary
    print("\n=== E11 COLD-TURKEY SUMMARY ===")
    by_cond = {}
    for cond in sorted(set(r["condition"] for r in results)):
        sub = [r for r in results if r["condition"] == cond]
        rates = [r["post_eats_rate"] * 100 for r in sub]
        by_cond[cond] = rates
        print(f"  {cond:20s}: {np.mean(rates):.3f}% ± {np.std(rates,ddof=1):.3f}%  (n={len(rates)})")

    print("\n=== KEY COMPARISONS ===")
    pairs = [
        ("COLD_TURKEY", "FULL_LESION_TEST", "Does cold practice help vs no practice?"),
        ("COLD_TURKEY", "NAIVE_LESION", "Does cold practice beat random walk?"),
    ]
    for a, b, q in pairs:
        if a not in by_cond or b not in by_cond:
            continue
        ra, rb = by_cond[a], by_cond[b]
        t, pv = sst.ttest_ind(ra, rb)
        d = (np.mean(ra) - np.mean(rb)) / max(
            math.sqrt((np.var(ra, ddof=1) + np.var(rb, ddof=1)) / 2), 1e-9)
        sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
        print(f"  {q}")
        print(f"    {a} vs {b}: delta={np.mean(ra)-np.mean(rb):+.3f}%  "
              f"t={t:.2f}  p={pv:.4f}  d={d:.2f}  {sig}")

    print(f"\nSaved {out_csv}")
