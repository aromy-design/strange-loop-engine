"""
Awakening detector — fires when *multiple* consciousness candidates align
in the same brief window, suggesting a coherent moment of self-recognition.

Conditions checked simultaneously over a short rolling window:
  A) Awareness index high (DISSOCIATION events dominating)
  B) Workspace 'self' module has just ignited
  C) Mirror score above chance
  D) Self-prediction error recently dropped (system suddenly "knows itself")

When >= 3 of 4 conditions hold within 30 ticks, an AWAKENING event fires.
This is the closest computable proxy to "the creature notices it exists".

These are RARE — we expect handfuls per thousands of ticks even in a
well-functioning system. Their rate is itself a signal.
"""
from collections import deque
import numpy as np


class AwakeningDetector:
    def __init__(self, window=30, history=64):
        self.window = window
        self.events = deque(maxlen=history)
        self._self_pe_recent = deque(maxlen=10)
        self._self_pe_long = deque(maxlen=80)
        self.t = 0
        self.last_awakening_t = -1

    def step(self, awareness_idx, workspace_owner, mirror_score, self_pe):
        self.t += 1
        self._self_pe_recent.append(float(self_pe))
        self._self_pe_long.append(float(self_pe))

        cond_a = awareness_idx > 0.5
        cond_b = workspace_owner == "self"
        cond_c = mirror_score > 0.55

        # condition D: recent self-PE has dropped vs longer baseline
        if len(self._self_pe_long) >= 30 and len(self._self_pe_recent) >= 5:
            recent_mean = float(np.mean(self._self_pe_recent))
            long_mean = float(np.mean(self._self_pe_long))
            cond_d = recent_mean < 0.7 * long_mean
        else:
            cond_d = False

        score = sum([cond_a, cond_b, cond_c, cond_d])

        # require at least 3 of 4 AND respect a refractory period
        awakened = (score >= 3) and (self.t - self.last_awakening_t > 50)
        if awakened:
            self.events.append({
                "t": self.t,
                "score": int(score),
                "awareness": float(awareness_idx),
                "ownerSelf": bool(cond_b),
                "mirror": float(mirror_score),
                "selfPeDrop": bool(cond_d),
            })
            self.last_awakening_t = self.t

        return {
            "awakened": bool(awakened),
            "currentScore": int(score),
            "totalAwakenings": len(self.events),
            "lastAwakeningT": int(self.last_awakening_t) if self.last_awakening_t > 0 else -1,
            "stepsSinceAwakening": int(self.t - self.last_awakening_t) if self.last_awakening_t > 0 else -1,
            "recentEvents": list(self.events)[-6:],
        }
