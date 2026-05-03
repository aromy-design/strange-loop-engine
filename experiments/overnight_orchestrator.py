"""
Overnight autonomous orchestrator.

Runs sequentially:
  1. Wait for E7 (post_train_lesion) to finish
  2. Analyze E7 → print results
  3. Run E6 v3 (long training 100k steps)
  4. Analyze E6 v3 → print results
  5. Run E8 (harder world or longer seeds if triggered)
  6. Print summary for user in morning

Usage:
  python -m experiments.overnight_orchestrator
  (can also be called after E7 already running)
"""
import os, sys, time, subprocess
os.environ["SCALE"] = "large"
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "experiments" / "results"
PLOTS = ROOT / "experiments" / "plots"


def wait_for_file(path, timeout=7200, poll=30):
    path = Path(path)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 100:
            return True
        time.sleep(poll)
    return False


def run_py(args, env_extra=None):
    env = os.environ.copy()
    env["SCALE"] = "large"
    if env_extra:
        env.update(env_extra)
    cmd = [sys.executable] + args
    print(f"\n[ORCHESTRATOR] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=str(ROOT))
    return result.returncode == 0


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


if __name__ == "__main__":
    t_start = time.time()
    PLOTS.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Wait for E7 ---
    section("Step 1: Waiting for E7 post-training lesion to finish...")
    e7_results = RESULTS / "post_train_lesion" / "results.csv"
    if e7_results.exists():
        print("E7 results already present. Proceeding to analysis.")
    else:
        print(f"Waiting for {e7_results}  (up to 2h)")
        ok = wait_for_file(e7_results, timeout=7200, poll=30)
        if not ok:
            print("ERROR: E7 didn't finish in 2 hours. Skipping.")
        else:
            print("E7 finished.")

    # --- Step 2: Analyze E7 ---
    section("Step 2: Analyzing E7 post-training lesion")
    if e7_results.exists():
        run_py(["-m", "experiments.analyze_post_train_lesion"])
    else:
        print("Skipping E7 analysis (no results file).")

    # --- Step 3: Run E6 v3 (100k steps) ---
    section("Step 3: E6 v3 extended training (100k steps, 4 seeds)")
    # backup previous long_training results if they exist
    lt_dir = RESULTS / "long_training"
    lt_v3_dir = RESULTS / "long_training_v3"
    lt_v3_dir.mkdir(parents=True, exist_ok=True)
    # Run E6 v3 directly (blocking)
    ok = run_py(["-m", "experiments.long_training",
                 "--seeds", "4",
                 "--steps", "100000",
                 "--log_every", "1000",
                 "--workers", "4"])
    if ok:
        # backup
        import shutil
        for f in lt_dir.glob("seed*.csv"):
            shutil.copy(f, lt_v3_dir / f.name)
        print("E6 v3 complete. Results backed up to long_training_v3/")
    else:
        print("E6 v3 failed or was interrupted.")

    # --- Step 4: Analyze E6 v3 ---
    section("Step 4: Analyzing E6 v3 (100k steps)")
    if ok:
        run_py(["-m", "experiments.analyze_long_training"])

    # --- Step 5: Run E8 toxic world ---
    section("Step 5: E8 toxic world (MB associative memory test)")
    e8_dir = RESULTS / "e8_toxic"
    e8_dir.mkdir(parents=True, exist_ok=True)
    print("  Toxic food: 2 fixed cells, distinct visual (1.5 vs 0.5/1.0)")
    print("  Question: does trained MB maintain avoidance when spatial_map lesioned?")
    ok_e8 = run_py(["-m", "experiments.e8_toxic_world",
                    "--seeds", "8",
                    "--train_steps", "30000",
                    "--test_steps", "8000",
                    "--workers", "8",
                    "--conditions", "FULL_FULL", "FULL_LESION_MAP",
                    "NAIVE_FULL", "NAIVE_LESION_MAP"])
    if ok_e8:
        print("E8 complete.")
    else:
        print("E8 failed or interrupted.")

    # --- Step 6: Run E9 MB necessity test ---
    section("Step 6: E9 MB necessity (train/test with shortcuts lesioned)")
    e9_dir = RESULTS / "e9_mb_necessity"
    e9_dir.mkdir(parents=True, exist_ok=True)
    ok_e9 = run_py(["-m", "experiments.e9_mb_necessity",
                    "--seeds", "8",
                    "--train_steps", "30000",
                    "--test_steps", "8000",
                    "--workers", "8"])
    if ok_e9:
        print("E9 complete.")
    else:
        print("E9 failed or interrupted.")

    # --- Final summary ---
    elapsed = (time.time() - t_start) / 60
    section(f"OVERNIGHT COMPLETE — {elapsed:.1f} minutes elapsed")
    print("Files generated:")
    for d in [RESULTS / "post_train_lesion", lt_v3_dir, RESULTS / "e8_toxic",
              RESULTS / "e9_mb_necessity", PLOTS]:
        for f in d.glob("*"):
            print(f"  {f}")
    print("\nCheck analysis.txt in post_train_lesion/ and stats_report.txt in long_training/")
    print("User brief: see CONTINUE.md for updated findings.")
