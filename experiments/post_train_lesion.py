"""
E7 — Post-training lesion study.

Tests whether the TRAINED mushroom body causally contributes to behavior.

Design:
  Phase 1 (steps 0..train_steps):  Full training — MB learns via cross-plasticity
  Phase 2 (steps train_steps..):   Lesion applied mid-run
    FULL      : no lesion — MB weights intact
    LESION_MB : zero W_mush_to_motor (MB output blocked, weight zeroed)
    LESION_CX : zero W_cx_to_motor (control: CX lesion)
    LESION_ALL: zero all 4 cross-projections to motor (maximal lesion)

Key question: does FULL outperform LESION_MB in Phase 2?
  If yes  → trained MB causally drives behavior (publishable)
  If no   → MB learns but doesn't yet matter for task at hand

Both conditions train identically (same seed, same code).
At train_steps, the specified weights are zeroed.
"""
import os, sys, csv, time
os.environ["SCALE"] = "large"
import multiprocessing as mp
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTDIR = Path(__file__).parent / "results" / "post_train_lesion"


def _run_agent(seed, steps, world, agent):
    """Run agent for `steps` steps, return (cum_eats, cum_danger)."""
    last_ate = False; last_kind = 0; last_dng = False; last_shel = False
    cum_eats = 0; cum_danger = 0
    for i in range(steps):
        obs = world.observe(energy=agent.homeostat.E)
        ar, ac = int(world.agent[0]), int(world.agent[1])
        food_pos = np.argwhere(world.grid > 0)
        fd = (0, 0); sd = (0, 0); fv = []
        if len(food_pos) > 0:
            d = food_pos - np.array([ar, ac])
            nearest = food_pos[int(np.argmin(np.linalg.norm(d, axis=1)))]
            fd = (int(np.sign(nearest[0] - ar)), int(np.sign(nearest[1] - ac)))
            fv = [(int(p[0]), int(p[1])) for p in food_pos]
        sd = (int(np.sign(world.shelter[0] - ar)), int(np.sign(world.shelter[1] - ac)))
        sigs = []
        for sr in range(max(0, ar - 2), min(world.SIZE, ar + 3)):
            for sc in range(max(0, ac - 2), min(world.SIZE, ac + 3)):
                k = int(world.signals[sr, sc])
                if k > 0: sigs.append(k)
        out = agent.step(obs, ate_last_action=last_ate, ate_kind=last_kind,
                         in_danger=last_dng, in_shelter=last_shel,
                         light_level=world.light_level,
                         agent_pos=(ar, ac), food_visible=fv,
                         shelter_dir=sd, food_dir=fd, danger_dir=(0, 0),
                         signals_in_view=sigs)
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


def _apply_lesion(ib, lesion_type):
    """Zero cross-projection weights based on lesion type."""
    if lesion_type == "FULL":
        pass  # nothing
    elif lesion_type == "LESION_MB":
        ib.W_mush_to_motor.data[:] = 0.0
    elif lesion_type == "LESION_CX":
        ib.W_cx_to_motor.data[:] = 0.0
    elif lesion_type == "LESION_ALL":
        ib.W_mush_to_motor.data[:] = 0.0
        ib.W_cx_to_motor.data[:] = 0.0
        ib.W_lh_to_motor.data[:] = 0.0
        ib.W_seg_to_motor.data[:] = 0.0
    else:
        raise ValueError(f"Unknown lesion: {lesion_type}")


def _worker(args):
    os.environ["SCALE"] = "large"
    seed, train_steps, test_steps, lesion_type = args
    from backend.world import GridWorld
    from backend.agent import MetabolicAgent

    np.random.seed(seed)
    world = GridWorld(seed=seed)
    agent = MetabolicAgent(sensory_dim=world.obs_dim, seed=seed)

    t0 = time.time()

    # Phase 1: Training
    pre_eats, pre_danger = _run_agent(seed, train_steps, world, agent)

    # Record learned weights before lesion
    mb_w_before = float(np.abs(agent.insect_brain.W_mush_to_motor.data).mean())
    cx_w_before = float(np.abs(agent.insect_brain.W_cx_to_motor.data).mean())

    # Apply lesion
    _apply_lesion(agent.insect_brain, lesion_type)

    mb_w_after = float(np.abs(agent.insect_brain.W_mush_to_motor.data).mean())

    # Phase 2: Test
    post_eats, post_danger = _run_agent(seed, test_steps, world, agent)

    elapsed = time.time() - t0
    result = {
        "seed": seed,
        "lesion": lesion_type,
        "train_steps": train_steps,
        "test_steps": test_steps,
        "pre_eats": pre_eats,
        "post_eats": post_eats,
        "pre_eats_rate": pre_eats / train_steps,
        "post_eats_rate": post_eats / test_steps,
        "pre_danger": pre_danger,
        "post_danger": post_danger,
        "mb_w_before": mb_w_before,
        "cx_w_before": cx_w_before,
        "mb_w_after": mb_w_after,
        "elapsed": elapsed,
    }
    print(f"seed={seed} lesion={lesion_type} pre={pre_eats}/{train_steps}={100*pre_eats/train_steps:.1f}% "
          f"post={post_eats}/{test_steps}={100*post_eats/test_steps:.1f}% "
          f"mb_w {mb_w_before:.4f}->{mb_w_after:.4f} elapsed={elapsed:.0f}s")
    return result


if __name__ == "__main__":
    mp.freeze_support()
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--train_steps", type=int, default=30000)
    p.add_argument("--test_steps", type=int, default=8000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--conditions", nargs="+",
                   default=["FULL", "LESION_MB", "LESION_CX", "LESION_ALL"])
    args = p.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1, args.seeds + 1))
    tasks = [(s, args.train_steps, args.test_steps, c)
             for c in args.conditions for s in seeds]
    print(f"E7 POST-TRAINING LESION: {len(tasks)} runs "
          f"({len(args.conditions)} conditions x {args.seeds} seeds), "
          f"train={args.train_steps} test={args.test_steps}, "
          f"workers={args.workers}")

    t0 = time.time()
    results = []
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_worker, tasks):
            results.append(r)

    total = time.time() - t0
    print(f"\nTotal elapsed: {total:.0f}s")

    # Save raw results
    out_csv = OUTDIR / "results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in sorted(results, key=lambda x: (x["lesion"], x["seed"])):
            w.writerow(r)

    # Quick summary
    try:
        from scipy import stats as sst
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    conditions = sorted(set(r["lesion"] for r in results))
    print("\n=== E7 SUMMARY ===")
    post_rates = {}
    for cond in conditions:
        rates = [r["post_eats_rate"] for r in results if r["lesion"] == cond]
        post_rates[cond] = rates
        print(f"  {cond:15s}: post_eats_rate={np.mean(rates)*100:.2f}% ± {np.std(rates,ddof=1)*100:.2f}%")

    if HAS_SCIPY and "FULL" in post_rates and "LESION_MB" in post_rates:
        t, p_val = sst.ttest_ind(post_rates["FULL"], post_rates["LESION_MB"])
        import math
        d = (np.mean(post_rates["FULL"]) - np.mean(post_rates["LESION_MB"])) / max(
            math.sqrt((np.var(post_rates["FULL"], ddof=1) + np.var(post_rates["LESION_MB"], ddof=1)) / 2), 1e-9)
        print(f"\nFULL vs LESION_MB: t={t:.3f} p={p_val:.4f} Cohen's d={d:.3f}")
        if p_val < 0.05 and d > 0.3:
            print(">>> SIGNIFICANT: trained MB causally contributes to behavior!")
        else:
            print(">>> NULL: MB weight growth doesn't translate to behavioral advantage")

    summary = OUTDIR / "summary.txt"
    lines = ["E7 POST-TRAINING LESION SUMMARY", f"Seeds: {args.seeds}, train={args.train_steps}, test={args.test_steps}", ""]
    for cond in conditions:
        rates = post_rates.get(cond, [])
        lines.append(f"{cond}: post_rate={np.mean(rates)*100:.3f}% ± {np.std(rates,ddof=1)*100:.3f}% (n={len(rates)})")
    if HAS_SCIPY and "FULL" in post_rates and "LESION_MB" in post_rates:
        t, p_val = sst.ttest_ind(post_rates["FULL"], post_rates["LESION_MB"])
        d = (np.mean(post_rates["FULL"]) - np.mean(post_rates["LESION_MB"])) / max(
            math.sqrt((np.var(post_rates["FULL"], ddof=1) + np.var(post_rates["LESION_MB"], ddof=1)) / 2), 1e-9)
        lines.append(f"\nFULL vs LESION_MB: t={t:.3f} p={p_val:.4f} d={d:.3f}")
    with open(summary, "w") as f: f.write("\n".join(lines))
    print(f"\nSaved {out_csv} and {summary}")
