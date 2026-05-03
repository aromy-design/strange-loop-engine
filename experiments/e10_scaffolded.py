"""
E10 — Scaffolded Training: Progressive shortcut removal to force MB learning.

Core hypothesis: MB cannot learn goal-directed behavior when shortcuts are removed
all at once (E9 confirmed, p=0.98). But if shortcuts are progressively removed during
training — one at a time — MB may be able to substitute each function.

Design (4-phase scaffolded training):
  Phase 1 (phase1_steps): all shortcuts active — MB hitchhikes on shortcut success
  Phase 2 (phase2_steps): remove adjacent_food reflex — MB must handle food detection
  Phase 3 (phase3_steps): also remove behavioral_modes — MB must drive mode selection
  Phase 4 (phase4_steps): also remove spatial_map — MB must do risk-aversion

Test (test_steps): all shortcuts removed + blind (same as E9 LESION condition)

Conditions:
  SCAFFOLDED_4PH:   4-phase progressive removal
  SCAFFOLDED_2PH:   2-phase (full shortcuts → full lesion, 30k+30k)
  FULL_LESION_TEST: 60k full shortcuts → test lesioned (E9-style baseline)
  NAIVE_LESION:     no training → test lesioned (random baseline)

Key comparison:
  SCAFFOLDED_4PH vs FULL_LESION_TEST: does scaffolding help?
  SCAFFOLDED_4PH vs NAIVE_LESION:     does any training help?

If SCAFFOLDED_4PH >> FULL_LESION_TEST: MB CAN learn to substitute shortcuts. REVOLUTIONARY.
"""
import os, sys, csv, time
os.environ["SCALE"] = "large"
import multiprocessing as mp
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTDIR = Path(__file__).parent / "results" / "e10_scaffolded"
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
    seed, p1, p2, p3, p4, test_steps, condition = args
    from backend.world import GridWorld
    from backend.agent import MetabolicAgent

    np.random.seed(seed)
    world = GridWorld(seed=seed)
    agent = MetabolicAgent(sensory_dim=world.obs_dim, seed=seed)

    t0 = time.time()

    phase_eats = []
    phase_danger = []

    if condition == "NAIVE_LESION":
        # No training, test lesioned + blind
        _set_lesion(agent, SHORTCUT_LESIONS)
        post_eats, post_danger = _run_agent_phase(test_steps, world, agent, blind=True)
        train_total = 0

    elif condition == "FULL_LESION_TEST":
        # 60k full shortcuts, then test lesioned + blind
        _set_lesion(agent, set())
        e, d = _run_agent_phase(p1 + p2 + p3 + p4, world, agent, blind=False)
        phase_eats.append(e); phase_danger.append(d)
        train_total = p1 + p2 + p3 + p4
        _set_lesion(agent, SHORTCUT_LESIONS)
        post_eats, post_danger = _run_agent_phase(test_steps, world, agent, blind=True)

    elif condition == "SCAFFOLDED_4PH":
        # Phase 1: full shortcuts
        _set_lesion(agent, set())
        e, d = _run_agent_phase(p1, world, agent, blind=False)
        phase_eats.append(e); phase_danger.append(d)
        # Phase 2: remove adjacent_food
        _set_lesion(agent, {"adjacent_food"})
        e, d = _run_agent_phase(p2, world, agent, blind=False)
        phase_eats.append(e); phase_danger.append(d)
        # Phase 3: remove adjacent_food + behavioral_modes
        _set_lesion(agent, {"adjacent_food", "behavioral_modes"})
        e, d = _run_agent_phase(p3, world, agent, blind=False)
        phase_eats.append(e); phase_danger.append(d)
        # Phase 4: remove all shortcuts
        _set_lesion(agent, SHORTCUT_LESIONS)
        e, d = _run_agent_phase(p4, world, agent, blind=True)
        phase_eats.append(e); phase_danger.append(d)
        train_total = p1 + p2 + p3 + p4
        post_eats, post_danger = _run_agent_phase(test_steps, world, agent, blind=True)

    elif condition == "SCAFFOLDED_2PH":
        # Phase 1: full shortcuts (30k)
        _set_lesion(agent, set())
        e, d = _run_agent_phase(p1 + p2, world, agent, blind=False)
        phase_eats.append(e); phase_danger.append(d)
        # Phase 2: all shortcuts removed (30k)
        _set_lesion(agent, SHORTCUT_LESIONS)
        e, d = _run_agent_phase(p3 + p4, world, agent, blind=True)
        phase_eats.append(e); phase_danger.append(d)
        train_total = p1 + p2 + p3 + p4
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
        "phase_eats": sum(phase_eats),
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
    p.add_argument("--phase1", type=int, default=15000)
    p.add_argument("--phase2", type=int, default=15000)
    p.add_argument("--phase3", type=int, default=15000)
    p.add_argument("--phase4", type=int, default=15000)
    p.add_argument("--test_steps", type=int, default=8000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--conditions", nargs="+", default=[
        "SCAFFOLDED_4PH",
        "FULL_LESION_TEST",
        "NAIVE_LESION",
    ])
    args = p.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1, args.seeds + 1))
    total = args.phase1 + args.phase2 + args.phase3 + args.phase4
    tasks = [(s, args.phase1, args.phase2, args.phase3, args.phase4, args.test_steps, c)
             for c in args.conditions for s in seeds]

    print(f"E10 SCAFFOLDED TRAINING: {len(tasks)} runs, total_train={total}, "
          f"test={args.test_steps}, workers={args.workers}")
    print(f"  Phase1 (full shortcuts): {args.phase1} steps")
    print(f"  Phase2 (no adjacent_food): {args.phase2} steps")
    print(f"  Phase3 (no adj_food + no modes): {args.phase3} steps")
    print(f"  Phase4 (all lesioned, blind): {args.phase4} steps")
    print("  Key Q: does scaffolded removal let MB take over shortcuts?")
    print(f"  Random walk baseline (blind): ~0.7%")

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
    print("\n=== E10 SCAFFOLDED TRAINING SUMMARY ===")
    post_rates = {}
    for cond in conditions:
        subset = [r for r in results if r["condition"] == cond]
        rates = [r["post_eats_rate"] * 100 for r in subset]
        mb_ws = [r["mb_w_final"] for r in subset]
        post_rates[cond] = rates
        print(f"  {cond:25s}: {np.mean(rates):.3f}%±{np.std(rates,ddof=1):.3f}%  mb_w={np.mean(mb_ws):.4f}")

    print()
    print("  Random walk blind baseline (E9 confirmed): ~0.69%")
    print()
    print("=== KEY COMPARISONS ===")
    comparisons = [
        ("SCAFFOLDED_4PH", "FULL_LESION_TEST", "Scaffolded vs full-then-lesion:"),
        ("SCAFFOLDED_4PH", "NAIVE_LESION",     "Scaffolded vs naive (random baseline):"),
        ("SCAFFOLDED_2PH", "FULL_LESION_TEST", "2-phase scaffold vs full-then-lesion:"),
    ]
    for cond_a, cond_b, label in comparisons:
        if cond_a not in post_rates or cond_b not in post_rates: continue
        a, b = post_rates[cond_a], post_rates[cond_b]
        t, pv = sst.ttest_ind(a, b)
        sp = math.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2)
        d = (np.mean(a)-np.mean(b))/max(sp,1e-9)
        sig = "***" if pv<0.001 else ("**" if pv<0.01 else ("*" if pv<0.05 else "ns"))
        print(f"  {label}")
        print(f"    {np.mean(a):.3f}% vs {np.mean(b):.3f}%  delta={np.mean(a)-np.mean(b):+.3f}%  "
              f"t={t:.2f}  p={pv:.4f}  d={d:.2f}  {sig}")
        if pv < 0.05 and np.mean(a) > np.mean(b):
            if cond_a.startswith("SCAFFOLD") and cond_b == "FULL_LESION_TEST":
                print("  >>> SIGNIFICANT: Scaffolded training beats direct lesion! MB learned shortcuts!")
                if abs(d) > 0.8:
                    print("  >>> REVOLUTIONARY: Large effect — MB CAN substitute for shortcuts with progressive training!")
        print()

    lines = [f"E10 SCAFFOLDED: seeds={args.seeds} total_train={total} test={args.test_steps}", ""]
    for cond in conditions:
        r = post_rates.get(cond, [])
        lines.append(f"{cond}: {np.mean(r):.3f}%±{np.std(r,ddof=1):.3f}% (n={len(r)})")
    with open(OUTDIR / "summary.txt", "w") as f: f.write("\n".join(lines))
    print(f"\nSaved {out_csv}")
