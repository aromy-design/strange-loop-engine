from collections import deque
import numpy as np


class WeightTurnover:
    """
    Incarnates Hypothesis 2: identity persists through substrate change.

    Periodically replaces a small fraction of synaptic weights with random
    values (simulating neuron / synapse turnover, like real brains where
    synapses are constantly created and pruned).

    Identity persistence is measured by tracking the activity pattern
    (a low-dim signature of field state) across turnover events.
    If the pattern recovers despite turnover -> identity is in the dynamics,
    not in the weights -> H2 supported.
    """

    def __init__(self, period=500, fraction=0.01, signature_dim=32, history=8):
        self.period = period
        self.fraction = fraction
        self.signature_dim = signature_dim
        self.t = 0
        self.signatures = deque(maxlen=history)
        self.persistence = 0.5  # initialized neutral
        self.last_turnover_step = 0
        self.last_turnover_replaced = 0

    def _signature(self, phi):
        # low-dim summary of activity (first signature_dim coords averaged in chunks)
        n = phi.shape[0]
        chunks = np.array_split(phi, self.signature_dim)
        return np.array([float(c.mean()) for c in chunks], dtype=np.float32)

    def step(self, field):
        sig = self._signature(field.last_phi)
        self.signatures.append(sig)
        self.t += 1
        replaced = 0

        if self.t % self.period == 0 and self.t > 0:
            # remember signature just before turnover for comparison after
            sig_before = sig.copy()

            # replace `fraction` of weights with random values
            n = field.W.shape[0]
            n_replace = int(n * n * self.fraction)
            ii = np.random.randint(0, n, size=n_replace)
            jj = np.random.randint(0, n, size=n_replace)
            field.W[ii, jj] = np.random.normal(0, 0.05, size=n_replace).astype(np.float32)
            replaced = n_replace
            self.last_turnover_step = self.t
            self.last_turnover_replaced = replaced

            # do not measure persistence at this step; it builds across following steps
            self._sig_before = sig_before
            self._steps_since = 0
            self._measuring = True

        # if currently measuring, accumulate similarity over a few steps
        if hasattr(self, "_measuring") and self._measuring:
            self._steps_since += 1
            if self._steps_since >= 30:
                # compare current signature to pre-turnover signature
                a = sig
                b = self._sig_before
                cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                # cos in [-1, 1]; map to [0, 1]
                p = (cos + 1) / 2
                self.persistence = 0.7 * self.persistence + 0.3 * p
                self._measuring = False

        return {
            "turnoverStep": self.last_turnover_step,
            "turnoverReplaced": int(replaced),
            "identityPersistence": float(self.persistence),
        }
