"""
E9 launcher — waits for E8 to finish, then runs E9 automatically.
Run this independently after the overnight orchestrator is already active.
"""
import os, sys, time, subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "experiments" / "results"
E8_RESULTS = RESULTS / "e8_toxic" / "results.csv"
E9_RESULTS = RESULTS / "e9_mb_necessity"


def wait_for_file(path, timeout=7200, poll=60):
    path = Path(path)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 100:
            return True
        elapsed = (deadline - time.time()) / 60
        print(f"  waiting... {elapsed:.0f}min remaining")
        time.sleep(poll)
    return False


if __name__ == "__main__":
    print("E9 LAUNCHER — waiting for E8 to complete...")
    print(f"Watching: {E8_RESULTS}")

    if E8_RESULTS.exists() and E8_RESULTS.stat().st_size > 100:
        print("E8 already complete. Launching E9 now.")
    else:
        ok = wait_for_file(E8_RESULTS, timeout=7200, poll=60)
        if not ok:
            print("ERROR: E8 didn't finish in 2 hours. Launching E9 anyway.")

    E9_RESULTS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SCALE"] = "large"
    cmd = [sys.executable, "-m", "experiments.e9_mb_necessity",
           "--seeds", "8",
           "--train_steps", "30000",
           "--test_steps", "8000",
           "--workers", "8"]
    print(f"\nLaunching E9: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=str(ROOT))
    if result.returncode == 0:
        print("\nE9 complete.")
    else:
        print(f"\nE9 failed (exit code {result.returncode}).")
