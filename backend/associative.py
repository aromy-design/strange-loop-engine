"""
Associative learning between local signals and food/danger outcomes.

Insects classically conditioned: "yellow flower = nectar". They learn that
specific stimuli predict outcomes.

We track for each signal kind (1..4):
  - association[kind] = expected reward when this signal is in view

Updated by simple delta rule: when food eaten while signal-kind in view,
boost association[kind]. When in danger near signal-kind, decrease.

Used to bias mode_select (forage harder near food-associated signals).
"""
import numpy as np


class AssociativeLearner:
    def __init__(self, n_signal_kinds=4, lr=0.05):
        self.n_kinds = n_signal_kinds
        self.lr = lr
        self.assoc = np.zeros(n_signal_kinds + 1, dtype=np.float32)  # index 0 unused
        self.exposure_count = np.zeros(n_signal_kinds + 1, dtype=np.float32)

    def step(self, signals_in_view, ate_kind, in_danger):
        """
        signals_in_view: list of int signal kinds visible.
        Updates associations.
        """
        reward = 0.0
        if ate_kind == 2: reward = 1.0
        elif ate_kind == 1: reward = 0.5
        if in_danger: reward -= 0.4

        for k in signals_in_view:
            if 1 <= k <= self.n_kinds:
                self.assoc[k] += self.lr * (reward - self.assoc[k])
                self.exposure_count[k] += 1

    def best_associated(self):
        """Return signal-kind with strongest positive association."""
        if self.exposure_count[1:].sum() == 0:
            return None, 0.0
        idx = int(np.argmax(self.assoc[1:])) + 1
        return idx, float(self.assoc[idx])

    def snapshot(self):
        return {
            "associations": [float(self.assoc[i]) for i in range(self.n_kinds + 1)],
            "exposureCounts": [int(self.exposure_count[i]) for i in range(self.n_kinds + 1)],
            "bestKind": self.best_associated()[0],
            "bestValue": self.best_associated()[1],
        }
