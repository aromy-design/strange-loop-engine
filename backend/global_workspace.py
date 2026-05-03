"""
Global Workspace with Bayesian binding — incarnates the "Beautiful Loop"
theory of consciousness (Laukkonen & Slagter 2024 / Hohwy active inference).

Key conditions:
  1. Generative world model — provided by NeuralField predictive coding
  2. Bayesian binding — winner-take-all competition: only the inference that
     reduces long-term uncertainty most enters the workspace and becomes
     "broadcast" to other modules
  3. Epistemic depth — workspace contents fed back to all modules,
     creating recursive sharing of beliefs

This module: takes K candidate inferences from specialized modules,
runs ignition competition, broadcasts winner, tracks ignition events.

Modules competing here:
  - perception: current sensory state
  - prediction:  predicted next sensory
  - self-model: predicted next field state
  - memory: best-matching past episode
  - drive: dominant homeostatic need (hunger, etc.)
"""
from collections import deque
import numpy as np


class GlobalWorkspace:
    def __init__(self, dim=64, n_modules=5, capacity=1, ignition_threshold=0.05, history=64):
        self.dim = dim
        self.n_modules = n_modules
        self.capacity = capacity
        self.ignition_threshold = ignition_threshold
        self.module_names = ["perception", "prediction", "self", "memory", "drive"]

        # workspace state: the "broadcast" content (current contents of consciousness)
        self.contents = np.zeros(dim, dtype=np.float32)
        self.contents_owner = -1  # which module currently owns workspace
        self.confidence = 0.0  # how strongly the winner won

        # ignition events (bayesian binding moments)
        self.ignitions = deque(maxlen=history)
        self.last_ignition_t = 0
        self.t = 0

        # running "salience" estimate per module (helps avoid one module always winning)
        self.salience_baseline = np.zeros(n_modules, dtype=np.float32)

    def step(self, candidates):
        """
        candidates: list of dicts {vector, salience, name}
        salience: how informative / urgent this candidate is (higher = more
        likely to win competition). Driven by:
          - perception: surprise (prediction error)
          - prediction: confidence
          - self-model: prediction-of-self error
          - memory: similarity to current state
          - drive: urgency of need (hunger, low energy)
        Bayesian binding: pick winner by salience, but require beating
        ignition threshold + recent baseline.
        """
        self.t += 1
        if not candidates:
            return self.snapshot(ignited=False)

        n = len(candidates)
        sal = np.array([c["salience"] for c in candidates], dtype=np.float32)

        # update running baseline (slow)
        for i in range(min(n, self.n_modules)):
            self.salience_baseline[i] = 0.98 * self.salience_baseline[i] + 0.02 * sal[i]

        # subtract baseline so a module that's always salient doesn't dominate
        adjusted = sal - 0.3 * self.salience_baseline[:n]

        winner_idx = int(np.argmax(adjusted))
        winner_sal = float(adjusted[winner_idx])

        # ignition: only enter workspace if winner clears threshold
        ignited = winner_sal > self.ignition_threshold

        if ignited:
            v = candidates[winner_idx]["vector"]
            v = np.asarray(v, dtype=np.float32)
            if v.shape[0] != self.dim:
                # project (zero-pad or truncate to dim)
                if v.shape[0] < self.dim:
                    pad = np.zeros(self.dim - v.shape[0], dtype=np.float32)
                    v = np.concatenate([v, pad])
                else:
                    v = v[: self.dim]
            self.contents = v
            self.contents_owner = winner_idx
            self.confidence = float(winner_sal)
            self.ignitions.append({
                "t": self.t,
                "module": candidates[winner_idx]["name"],
                "salience": float(sal[winner_idx]),
                "confidence": float(winner_sal),
            })
            self.last_ignition_t = self.t
        else:
            # workspace contents fade slowly when nothing ignites
            self.contents *= 0.95
            self.confidence *= 0.95

        return self.snapshot(ignited=ignited)

    def snapshot(self, ignited=False):
        return {
            "contents": self.contents.tolist(),
            "ownerIdx": int(self.contents_owner),
            "ownerName": self.module_names[self.contents_owner] if 0 <= self.contents_owner < len(self.module_names) else "—",
            "confidence": float(self.confidence),
            "ignited": bool(ignited),
            "stepsSinceIgnition": int(self.t - self.last_ignition_t),
            "recentIgnitions": list(self.ignitions)[-12:],
        }
