"""
Active mirror mark test — operationalizes self-recognition.

Method:
  Periodically the test "shows" the creature an injected pattern in its
  observer neurons. Two conditions are randomized:
    OWN: the pattern is its own activity from N steps ago
    STRANGER: the pattern is a shuffled / scrambled version

  We measure how much the creature's motor neurons (its action propensity)
  shifts in response. Crucially, we measure the difference between:
      shift_own = || motor_after_own - motor_baseline ||
      shift_stranger = || motor_after_stranger - motor_baseline ||

  If the creature is self-recognizing, OWN should produce smaller shift
  (the creature recognizes the pattern as "self" and integrates smoothly)
  while STRANGER produces larger shift (foreign signal disrupts).

  Recognition score = 1 - shift_own / (shift_own + shift_stranger)
  Score > 0.5 means own patterns are more familiar than stranger patterns.

This is an *active probe* — closer in spirit to the actual mirror test
(where an animal sees itself), unlike the passive cosine-similarity version.
"""
from collections import deque
import numpy as np


class MirrorTest:
    def __init__(self, period=200, snapshot_lag=40):
        self.period = period
        self.snapshot_lag = snapshot_lag
        self.snapshots = deque(maxlen=80)
        self.last_score = 0.5
        self.last_t = 0
        self.runs = 0
        self.t = 0
        self._probe_phase = 0  # 0=baseline, 1=own probe, 2=stranger probe
        self._baseline_motor = None
        self._own_motor = None
        self._stranger_motor = None

    def record(self, phi):
        self.snapshots.append(np.asarray(phi, dtype=np.float32).copy())

    def step(self, field):
        """
        Returns probe signal to inject into field this tick (or None).
        Captures motor outputs across the 3-tick probe.
        """
        self.t += 1
        n = field.n
        # update last_score occasionally with stable read
        scheduled = (self.t % self.period == 0) and len(self.snapshots) >= self.snapshot_lag + 5

        if not scheduled and self._probe_phase == 0:
            return None

        if scheduled and self._probe_phase == 0:
            self._probe_phase = 1
            self._baseline_motor = field.motor_logits().copy()
            return None  # baseline tick: no probe

        if self._probe_phase == 1:
            # inject own past pattern into observer neurons
            past = self.snapshots[-self.snapshot_lag]
            probe = past * 0.5  # gentle injection
            field.x[field.idx_obs] += probe[field.idx_obs]
            self._own_motor_pending = True
            self._probe_phase = 2
            return ("own", probe)

        if self._probe_phase == 2:
            # capture motor response after own probe
            self._own_motor = field.motor_logits().copy()
            # now inject stranger pattern (shuffled own)
            past = self.snapshots[np.random.randint(0, len(self.snapshots))]
            shuffled = past[np.random.permutation(n)]
            field.x[field.idx_obs] += shuffled[field.idx_obs] * 0.5
            self._probe_phase = 3
            return ("stranger", shuffled)

        if self._probe_phase == 3:
            self._stranger_motor = field.motor_logits().copy()
            # compute recognition
            shift_own = float(np.linalg.norm(self._own_motor - self._baseline_motor))
            shift_str = float(np.linalg.norm(self._stranger_motor - self._baseline_motor))
            denom = shift_own + shift_str + 1e-6
            # recognition: smaller shift on own = recognized = score > 0.5
            score = float(np.clip(shift_str / denom, 0.0, 1.0))
            self.last_score = 0.7 * self.last_score + 0.3 * score
            self.runs += 1
            self.last_t = self.t
            self._probe_phase = 0
            self._baseline_motor = self._own_motor = self._stranger_motor = None
            return None

        return None

    def snapshot(self):
        return {
            "recognitionScore": float(self.last_score),
            "lastRunT": int(self.last_t),
            "runs": int(self.runs),
        }
