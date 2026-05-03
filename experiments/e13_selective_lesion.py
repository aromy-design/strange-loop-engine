"""
E13 — Selective MB Pathway Lesion: localize the aversive transfer signal.

Background (the finding to decompose):
  E8/E10/E12 all show: trained MB transfers DANGER AVOIDANCE when shortcuts removed.
  Effect size d=-3.5 to -6.5 across random and biological wiring.

Question: WHICH pathway carries the aversive signal?
  - W_vis_to_mush: vision → MB (input)
  - W_mush_to_motor: MB → motor (direct output)
  - W_mush_to_lh: MB → lateral horn → motor (indirect output)

Design (5 conditions, all with 60k full-shortcut training, then lesion+blind test):
  CONTROL:                shortcuts only (full MB intact, baseline danger)
  LESION_VIS_TO_MUSH:     + W_vis_to_mush.data = 0   (visual input to MB cut)
  LESION_MUSH_TO_MOTOR:   + W_mush_to_motor.data = 0 (direct output cut)
  LESION_MUSH_TO_LH:      + W_mush_to_lh.data = 0    (indirect output cut)
  LESION_ALL_MB:          + all three above           (full MB output silenced)

Pre-registered hypothesis (locked before running):
  H_E13a: LESION_ALL_MB has danger ≈ NAIVE_LESION (316) — full silencing removes transfer
  H_E13b: At least one single-pathway lesion has danger > CONTROL by d > 1
          (i.e., one specific pathway carries the signal)

Optionally include FlyEM (set FLYEM_DATA_DIR env var). Default uses random wiring
to allow direct comparison with E10 (matched protocol).

Power: with d=-6.43 (E10) effect size, even n=4 detects single-pathway loss easily.
"""
import os, sys, csv, time
os.environ["SCALE"] = "large"
import multiprocessing as mp
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTDIR = Path(__file__).parent / "results" / "e13_selective"
SHORTCUT_LESIONS = {"spatial_map", "adjacent_food", "behavioral_modes"}


def _set_lesion(agent, lesions: set):
    agent.lesion = lesions.copy()


def _silence_projections(agent, projection_names):
    """Zero out specified W_*_to_* matrices in the InsectBrainV2."""
    for attr in projection_names:
        w = getattr(agent.insect_brain, attr)
        # CSR matrix — zero the data array in place
        w.data[:] = 0.0


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


CONDITION_PATHWAY_LESIONS = {
    "CONTROL": [],
    "LESION_VIS_TO_MUSH":   ["W_vis_to_mush"],
    "LESION_MUSH_TO_MOTOR": ["W_mush_to_motor"],
    "LESION_MUSH_TO_LH":    ["W_mush_to_lh"],
    "LESION_ALL_MB":        ["W_vis_to_mush", "W_mush_to_motor", "W_mush_to_lh"],
    "NAIVE_LESION":         [],  # special: skips training
}


def _worker(args):
    os.environ["SCALE"] = "large"
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
    else:
        # Train phase: full shortcuts active, MB intact
        _set_lesion(agent, set())
        _run_agent_phase(full_steps, world, agent, blind=False)
        train_total = full_steps
        # Test phase: lesion shortcuts + blind, AND silence selected MB projections
        _set_lesion(agent, SHORTCUT_LESIONS)
        proj_to_silence = CONDITION_PATHWAY_LESIONS.get(condition, [])
        _silence_projections(agent, proj_to_silence)
        post_eats, post_danger = _run_agent_phase(test_steps, world, agent, blind=True)

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
    print(f"seed={seed} cond={condition:25s} eat={100*post_rate:.2f}%  "
          f"danger={post_danger}  mb_w={mb_w:.4f}  {elapsed:.0f}s", flush=True)
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
    p.add_argument("--conditions", nargs="+", default=[
        "CONTROL",
        "LESION_VIS_TO_MUSH",
        "LESION_MUSH_TO_MOTOR",
        "LESION_MUSH_TO_LH",
        "LESION_ALL_MB",
        "NAIVE_LESION",
    ])
    args = p.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1, args.seeds + 1))
    tasks = [(s, args.full_steps, args.test_steps, c)
             for c in args.conditions for s in seeds]

    print(f"E13 SELECTIVE MB PATHWAY LESION: {len(tasks)} runs")
    print(f"  full_steps={args.full_steps}  test_steps={args.test_steps}  workers={args.workers}")
    print(f"  Pre-registered: H_E13a (LESION_ALL_MB ~ NAIVE), H_E13b (one pathway dominant)")
    print(f"  E10/E12 baseline: trained MB -> danger ~10-14, naive MB -> danger ~316")

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

    print("\n=== E13 SUMMARY ===")
    by_cond = {}
    for cond in sorted(set(r["condition"] for r in results)):
        sub = [r for r in results if r["condition"] == cond]
        dng = [r["post_danger"] for r in sub]
        by_cond[cond] = dng
        print(f"  {cond:25s}: danger={np.mean(dng):.1f}±{np.std(dng,ddof=1):.1f}  (n={len(dng)})")

    if "CONTROL" in by_cond and "NAIVE_LESION" in by_cond:
        ctrl = by_cond["CONTROL"]
        naive = by_cond["NAIVE_LESION"]
        print(f"\n=== PATHWAY DECOMPOSITION (vs CONTROL trained) ===")
        for cond, dng in by_cond.items():
            if cond in ("CONTROL", "NAIVE_LESION"):
                continue
            t, pv = sst.ttest_ind(dng, ctrl)
            d = (np.mean(dng)-np.mean(ctrl))/max(
                math.sqrt((np.var(dng,ddof=1)+np.var(ctrl,ddof=1))/2), 1e-9)
            sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
            recovered = (np.mean(dng) - np.mean(ctrl)) / max(np.mean(naive) - np.mean(ctrl), 1e-9) * 100
            print(f"  {cond:25s}: delta_vs_ctrl={np.mean(dng)-np.mean(ctrl):+.1f}  "
                  f"d={d:.2f}  p={pv:.4f}  {sig}  ({recovered:.0f}% of naive deficit)")

    print(f"\nSaved {out_csv}")
