"""
E12 — FlyEM Danger Replication: replicate the appetitive/aversive dissociation
using REAL Drosophila hemibrain connectome instead of random sparse projections.

Background (the finding to replicate):
  E10 (n=8, random wiring): trained MB transfers DANGER AVOIDANCE with d=-6.56
    SCAFFOLDED_4PH: 9.6 danger steps
    FULL_LESION_TEST: 14.0 danger steps
    NAIVE_LESION: 316.4 danger steps

Question: Does this dissociation hold when MB→motor and other inter-circuit
projections come from the actual Drosophila brain (Janelia hemibrain v1.2.1)?

Two possible outcomes (both publishable):
  A) Effect persists with d≥3: dissociation is universal, not random-wiring artifact.
     Implication: appetitive/aversive functional split emerges from any sparse
     connectivity with cross-projection plasticity, not specifically from
     biological wiring.
  B) Effect changes: real connectome reveals what random wiring was hiding.
     Could be amplified (real wiring is better organized) or attenuated
     (real wiring's stereotyped pathways prevent re-purposing).

Conditions (matches E10 for direct comparison):
  FULL_LESION_TEST: 60k full shortcuts → test lesioned + blind
  NAIVE_LESION:     no training → test lesioned + blind

Optionally (longer):
  SCAFFOLDED_4PH:   4-phase progressive removal (60k total)

Setup:
  Sets FLYEM_DATA_DIR in worker env so MetabolicAgent uses FlyEM connectome.
  V2+FlyEM brain is 21,728 neurons, runs at ~5.9ms/step.
  Full run estimate: 68k steps × 5.9ms = 401s/run × 8 runs / 2 workers ≈ 27 min.

Pre-registered hypothesis (locked before running):
  H_E12: FULL_LESION_TEST danger << NAIVE_LESION danger, with d <= -3
  Effect direction must match E10. Effect size criterion: d <= -3 (E10 was -6.43).
"""
import os, sys, csv, time
os.environ["SCALE"] = "large"
# FlyEM data dir is required — must be set BEFORE multiprocessing forks
DEFAULT_FLYEM = str((__import__('pathlib').Path(__file__).parent.parent /
                     "data/flyem/exported-traced-adjacencies-v1.2").resolve())
if "FLYEM_DATA_DIR" not in os.environ:
    os.environ["FLYEM_DATA_DIR"] = DEFAULT_FLYEM
import multiprocessing as mp
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTDIR = Path(__file__).parent / "results" / "e12_flyem"
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
    if "FLYEM_DATA_DIR" not in os.environ:
        os.environ["FLYEM_DATA_DIR"] = DEFAULT_FLYEM
    seed, full_steps, test_steps, condition = args
    from backend.world import GridWorld
    from backend.agent import MetabolicAgent

    np.random.seed(seed)
    world = GridWorld(seed=seed)
    agent = MetabolicAgent(sensory_dim=world.obs_dim, seed=seed)
    brain_version = getattr(agent, "_brain_version", "?")

    t0 = time.time()

    if condition == "NAIVE_LESION":
        _set_lesion(agent, SHORTCUT_LESIONS)
        post_eats, post_danger = _run_agent_phase(test_steps, world, agent, blind=True)
        train_total = 0

    elif condition == "FULL_LESION_TEST":
        _set_lesion(agent, set())
        _run_agent_phase(full_steps, world, agent, blind=False)
        train_total = full_steps
        _set_lesion(agent, SHORTCUT_LESIONS)
        post_eats, post_danger = _run_agent_phase(test_steps, world, agent, blind=True)

    elif condition == "FULL_FULL":
        # control: same training, no lesion at test
        _set_lesion(agent, set())
        _run_agent_phase(full_steps, world, agent, blind=False)
        train_total = full_steps
        post_eats, post_danger = _run_agent_phase(test_steps, world, agent, blind=False)

    else:
        raise ValueError(f"Unknown condition: {condition}")

    elapsed = time.time() - t0
    mb_w = float(np.abs(agent.insect_brain.W_mush_to_motor.data).mean())
    post_rate = post_eats / test_steps if test_steps > 0 else 0.0

    result = {
        "seed": seed,
        "condition": condition,
        "brain_version": brain_version,
        "train_total": train_total,
        "test_steps": test_steps,
        "post_eats": post_eats,
        "post_eats_rate": post_rate,
        "post_danger": post_danger,
        "mb_w_final": mb_w,
        "elapsed": elapsed,
    }
    print(f"seed={seed} cond={condition} brain={brain_version} "
          f"eat={100*post_rate:.2f}%  danger={post_danger}  "
          f"mb_w={mb_w:.4f}  {elapsed:.0f}s", flush=True)
    return result


if __name__ == "__main__":
    mp.freeze_support()
    import argparse
    from scipy import stats as sst
    import math

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--full_steps", type=int, default=60000)
    p.add_argument("--test_steps", type=int, default=8000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--conditions", nargs="+",
                   default=["FULL_LESION_TEST", "NAIVE_LESION"])
    args = p.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1, args.seeds + 1))
    tasks = [(s, args.full_steps, args.test_steps, c)
             for c in args.conditions for s in seeds]

    print(f"E12 FLYEM DANGER REPLICATION: {len(tasks)} runs")
    print(f"  full_steps={args.full_steps}  test_steps={args.test_steps}  workers={args.workers}")
    print(f"  FLYEM_DATA_DIR={os.environ.get('FLYEM_DATA_DIR')}")
    print(f"  Pre-registered: H_E12: FULL_LESION_TEST danger << NAIVE_LESION danger, d <= -3")
    print(f"  E10 baseline (random wiring): d=-6.43")

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

    print("\n=== E12 FLYEM SUMMARY ===")
    by_cond = {}
    for cond in sorted(set(r["condition"] for r in results)):
        sub = [r for r in results if r["condition"] == cond]
        eats = [r["post_eats_rate"] * 100 for r in sub]
        dng = [r["post_danger"] for r in sub]
        by_cond[cond] = {"eat": eats, "dng": dng}
        print(f"  {cond:20s}: eat={np.mean(eats):.3f}%±{np.std(eats,ddof=1):.3f}%  "
              f"danger={np.mean(dng):.1f}±{np.std(dng,ddof=1):.1f}  (n={len(eats)})")

    print("\n=== HYPOTHESIS TEST (DANGER) ===")
    if "FULL_LESION_TEST" in by_cond and "NAIVE_LESION" in by_cond:
        a = by_cond["FULL_LESION_TEST"]["dng"]
        b = by_cond["NAIVE_LESION"]["dng"]
        t, pv = sst.ttest_ind(a, b)
        d = (np.mean(a)-np.mean(b))/max(math.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2), 1e-9)
        sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
        print(f"  FULL_LESION_TEST vs NAIVE_LESION: trained={np.mean(a):.1f} vs naive={np.mean(b):.1f}")
        print(f"  delta={np.mean(a)-np.mean(b):+.1f}  t={t:.2f}  p={pv:.6f}  d={d:.2f}  {sig}")
        print()
        if d <= -3:
            print(f"  *** HYPOTHESIS CONFIRMED: dissociation replicates with REAL Drosophila connectome (d={d:.2f})")
            print(f"  *** E10 (random wiring) d=-6.43, E12 (real wiring) d={d:.2f}")
        else:
            print(f"  *** HYPOTHESIS NOT CONFIRMED at d <= -3 threshold (d={d:.2f})")
            print(f"  *** Real connectome shows different behavior than random — meaningful finding either way")

    print(f"\nSaved {out_csv}")
