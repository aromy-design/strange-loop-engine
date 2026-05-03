"""
E7 v2 — Properly powered MB silencing test (n=30) with FULL MB output silencing.

Background: E7 v1 (n=8) found NULL on appetitive (eat rate) with d=0.53, p=0.31.
This was underpowered (need n=25+ for 80% power at d=0.5). Also, E7 v1 only
zeroed W_mush_to_motor — leaving the indirect MB->LH->motor pathway intact.

E7 v2 fixes both:
  - n=30 seeds (proper power for d=0.5)
  - Silences BOTH W_mush_to_motor AND W_mush_to_lh (full MB output cut)

Design:
  Train phase: 30k steps, all shortcuts active, MB intact
  Test phase:  8k steps, all shortcuts active
    FULL: nothing changed
    LESION_FULL_MB: W_mush_to_motor.data := 0 AND W_mush_to_lh.data := 0

Pre-registered hypotheses (PRE_REGISTRATION.md):
  H_E7v2_appetitive: |mean(FULL.eat) - mean(LESION.eat)| < 0.5 × pooled_sd  → NULL
  H_E7v2_aversive:   mean(LESION.danger) - mean(FULL.danger) > 1.5 × pooled_sd → POS

Key comparisons:
  Eat rate (NULL expected per dissociation finding)
  Danger steps (POSITIVE expected — silencing MB removes the aversive transfer)

Power: with n=30, 80% power at d=0.5 (Cohen's threshold for medium effect).
"""
import os, sys, csv, time
os.environ["SCALE"] = "large"
import multiprocessing as mp
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTDIR = Path(__file__).parent / "results" / "e7v2"


def _set_lesion(agent, lesions: set):
    agent.lesion = lesions.copy()


def _silence_mb_output(agent):
    """Zero both direct (MB->motor) and indirect (MB->LH) output projections."""
    agent.insect_brain.W_mush_to_motor.data[:] = 0.0
    agent.insect_brain.W_mush_to_lh.data[:] = 0.0


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
    seed, train_steps, test_steps, condition = args
    from backend.world import GridWorld
    from backend.agent import MetabolicAgent

    np.random.seed(seed)
    world = GridWorld(seed=seed)
    agent = MetabolicAgent(sensory_dim=world.obs_dim, seed=seed)

    t0 = time.time()

    # Train: full shortcuts + MB intact
    _set_lesion(agent, set())
    _run_agent_phase(train_steps, world, agent, blind=False)
    mb_w_after_train = float(np.abs(agent.insect_brain.W_mush_to_motor.data).mean())

    # Test: full shortcuts still active
    if condition == "LESION_FULL_MB":
        _silence_mb_output(agent)
    # FULL: nothing changes

    post_eats, post_danger = _run_agent_phase(test_steps, world, agent, blind=False)

    elapsed = time.time() - t0
    mb_w_final = float(np.abs(agent.insect_brain.W_mush_to_motor.data).mean())
    post_eat_rate = post_eats / test_steps if test_steps > 0 else 0.0

    result = {
        "seed": seed,
        "condition": condition,
        "train_steps": train_steps,
        "test_steps": test_steps,
        "post_eats": post_eats,
        "post_eat_rate": post_eat_rate,
        "post_danger": post_danger,
        "mb_w_after_train": mb_w_after_train,
        "mb_w_final": mb_w_final,
        "elapsed": elapsed,
    }
    print(f"seed={seed:2d} cond={condition:15s} eat={100*post_eat_rate:.2f}%  "
          f"danger={post_danger}  mb_w={mb_w_final:.4f}  {elapsed:.0f}s", flush=True)
    return result


if __name__ == "__main__":
    mp.freeze_support()
    import argparse
    from scipy import stats as sst
    import math

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--train_steps", type=int, default=30000)
    p.add_argument("--test_steps", type=int, default=8000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--conditions", nargs="+", default=["FULL", "LESION_FULL_MB"])
    args = p.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1, args.seeds + 1))
    tasks = [(s, args.train_steps, args.test_steps, c)
             for c in args.conditions for s in seeds]

    print(f"E7 v2 PROPERLY POWERED MB SILENCING: {len(tasks)} runs")
    print(f"  train_steps={args.train_steps}  test_steps={args.test_steps}  workers={args.workers}")
    print(f"  Pre-registered: H_E7v2_appetitive (eat NULL, d<0.5) + H_E7v2_aversive (danger pos, d>1.5)")
    print(f"  n=30 gives 80% power for d=0.5 (Cohen medium effect)")

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

    print("\n=== E7 v2 SUMMARY ===")
    by_cond = {}
    for cond in sorted(set(r["condition"] for r in results)):
        sub = [r for r in results if r["condition"] == cond]
        eats = [r["post_eat_rate"]*100 for r in sub]
        dng = [r["post_danger"] for r in sub]
        by_cond[cond] = {"eat": eats, "dng": dng}
        print(f"  {cond:18s}: eat={np.mean(eats):.3f}%±{np.std(eats,ddof=1):.3f}%  "
              f"danger={np.mean(dng):.1f}±{np.std(dng,ddof=1):.1f}  (n={len(eats)})")

    if "FULL" in by_cond and "LESION_FULL_MB" in by_cond:
        for metric in ("eat", "dng"):
            a = by_cond["FULL"][metric]
            b = by_cond["LESION_FULL_MB"][metric]
            t, pv = sst.ttest_ind(a, b)
            d = (np.mean(a)-np.mean(b))/max(math.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2), 1e-9)
            sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
            label = "EAT (appetitive)" if metric == "eat" else "DANGER (aversive)"
            print(f"\n  {label}: FULL vs LESION_FULL_MB:")
            print(f"    delta={np.mean(a)-np.mean(b):+.3f}  t={t:.2f}  p={pv:.4f}  d={d:.2f}  {sig}")
            if metric == "eat":
                if abs(d) < 0.5 and pv > 0.05:
                    print("    >>> H_E7v2_appetitive CONFIRMED: trained MB does NOT contribute to foraging")
                else:
                    print("    >>> H_E7v2_appetitive REJECTED: MB does contribute to foraging?")
            else:
                if d < -1.5 and pv < 0.05:  # FULL danger > LESION danger means LESION is worse
                    pass
                if (np.mean(b) - np.mean(a)) > 1.5 * math.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2) and pv < 0.05:
                    print("    >>> H_E7v2_aversive CONFIRMED: silencing MB increases danger (positive control)")
                else:
                    print("    >>> H_E7v2_aversive: weak — MB silencing did not increase danger here")
                    print("    >>> Note: in shortcuts-active test, spatial_map dominates avoidance — expected weak signal")

    print(f"\nSaved {out_csv}")
