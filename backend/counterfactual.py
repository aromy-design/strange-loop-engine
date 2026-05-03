"""
Counterfactual self-modeling — implements a key intuition from Higher-Order
Theory and Cleeremans' radical plasticity thesis: a system that *knows it
exists* can simulate "what would happen if I were not acting?" and notice
that its presence changes outcomes.

Implementation:
  - A "ghost field" runs in parallel with the real field, receiving the
    same sensory input but no recurrent self-influence (we zero out the
    reflexive observer feedback in the ghost copy).
  - The difference between real-field trajectory and ghost-field
    trajectory measures how much the system's own activity is driving its
    own state — its "presence signature".
  - Higher presence signature = more self-driven = more "I am here doing
    this".

Cheap version: track variance attributable to recurrent self-input vs
external sensory input via running covariance.
"""
from collections import deque
import numpy as np


class CounterfactualSelf:
    def __init__(self, n_neurons, history=120):
        self.n = n_neurons
        self.history = history
        self.real_traj = deque(maxlen=history)
        self.input_traj = deque(maxlen=history)
        self.presence = 0.0

    def step(self, field_phi, input_drive):
        """
        input_drive: the sensory + instinct projection BEFORE recurrent
        contribution — so we compare real (with self-loop) vs input-only.
        """
        self.real_traj.append(np.asarray(field_phi, dtype=np.float32).copy())
        self.input_traj.append(np.asarray(input_drive, dtype=np.float32).copy())

        if len(self.real_traj) < 30:
            return {"presence": float(self.presence), "ready": False}

        R = np.stack(self.real_traj)
        I = np.stack(self.input_traj)

        # variance not explained by input alone = self-driven variance
        # use running covariance of (R) decomposed into input-projection + residual
        # cheap: per neuron, ratio of var(residual) / var(R)
        input_dim = I.shape[1]
        if input_dim < 1:
            return {"presence": float(self.presence), "ready": True}

        # ridge regression: how much of R can input alone predict?
        # XtX small (~input_dim x input_dim)
        ratios = []
        n_neurons = R.shape[1]
        # sample a few neurons to keep cost low
        sample = list(range(0, n_neurons, 8))
        XtX = I.T @ I + 1.0 * np.eye(input_dim)
        for k in sample:
            y = R[:, k]
            try:
                w = np.linalg.solve(XtX, I.T @ y)
            except np.linalg.LinAlgError:
                continue
            yp = I @ w
            v_total = float(np.var(y)) + 1e-6
            v_resid = float(np.var(y - yp)) + 1e-6
            # presence = fraction of variance NOT explained by input alone
            # = self-driven activity
            ratios.append(min(1.0, v_resid / v_total))

        if not ratios:
            return {"presence": float(self.presence), "ready": True}

        cur = float(np.mean(ratios))
        self.presence = 0.9 * self.presence + 0.1 * cur
        return {"presence": float(self.presence), "ready": True, "current": cur}
