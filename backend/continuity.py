"""
Self-continuity tracker.

Recent literature (2024-2025 recursive self-modeling work) emphasizes
that consciousness is what *persists across changes* — the ongoing
modeling of one's own continuity through time.

This module maintains a slowly-updating "identity vector" — the
exponential moving average of the agent's neural state — and reports:
  - similarity of current state to identity (am I still myself?)
  - how stable that identity has been (is my self-vector itself stable?)

If continuity = 1: state matches identity strongly (steady self).
If continuity = 0: state is foreign to identity (acute disruption,
                    or too-young agent).

We also detect IDENTITY_BREAK events when continuity drops abruptly.
"""
from collections import deque
import numpy as np


class ContinuityTracker:
    def __init__(self, n_neurons, alpha=0.005, history=64):
        self.n = n_neurons
        self.alpha = alpha
        self.identity = np.zeros(n_neurons, dtype=np.float32)
        self.identity_norm = 0.0
        self.warmup = 0
        self.cont_hist = deque(maxlen=history)
        self.events = deque(maxlen=32)
        self.t = 0
        self.last_break_t = -1

    def step(self, phi):
        self.t += 1
        phi = np.asarray(phi, dtype=np.float32)
        # update identity vector slowly
        self.identity = (1 - self.alpha) * self.identity + self.alpha * phi
        self.identity_norm = float(np.linalg.norm(self.identity))

        if self.warmup < 100:
            self.warmup += 1
            self.cont_hist.append(0.5)
            return self._snapshot()

        # cosine similarity
        ni = self.identity_norm + 1e-9
        np_ = float(np.linalg.norm(phi)) + 1e-9
        cos = float(np.dot(self.identity, phi) / (ni * np_))
        # map [-1,1] -> [0,1]
        cont = (cos + 1) / 2
        self.cont_hist.append(cont)

        # detect identity break: sudden drop in continuity
        if len(self.cont_hist) >= 20:
            recent = float(np.mean(list(self.cont_hist)[-5:]))
            baseline = float(np.mean(list(self.cont_hist)[-20:-5]))
            if (baseline - recent) > 0.15 and (self.t - self.last_break_t > 60):
                self.events.append({
                    "t": self.t,
                    "drop": float(baseline - recent),
                    "recent": float(recent),
                })
                self.last_break_t = self.t

        return self._snapshot()

    def _snapshot(self):
        cur = float(self.cont_hist[-1]) if self.cont_hist else 0.5
        # stability = 1 - std of recent continuity values
        if len(self.cont_hist) >= 10:
            stab = float(np.clip(1.0 - np.std(list(self.cont_hist)[-30:]) * 4, 0, 1))
        else:
            stab = 0.5
        return {
            "continuity": cur,
            "identityStability": stab,
            "totalBreaks": len(self.events),
            "recentBreaks": list(self.events)[-4:],
        }
