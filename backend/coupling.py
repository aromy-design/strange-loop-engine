from collections import deque
import numpy as np


class BidirectionalCoupling:
    """
    Incarnates Hypothesis 3: observer and observed are the same physical units.

    For a sample of neuron pairs, measure how much knowing neuron i predicts
    neuron j's *future* state, AND how much knowing j predicts i's future
    state. When BOTH directions carry information, the system has true
    bidirectional coupling — the boundary between observer and observed
    has collapsed.

    Metric: "collapse index" = mean of min(MI(i->j), MI(j->i)) across pairs.
    High collapse index -> observer-observed asymmetry has dissolved.

    Fast approximation: use sliding correlations with a 1-step lag for both
    directions. Real MI is expensive; correlation captures the essential
    bidirectional structure cheaply.
    """

    def __init__(self, n_neurons, n_pairs=32, history=60, seed=2):
        rng = np.random.default_rng(seed)
        self.n = n_neurons
        self.n_pairs = n_pairs
        # sample random pairs (i != j)
        pairs = []
        while len(pairs) < n_pairs:
            i, j = int(rng.integers(0, n_neurons)), int(rng.integers(0, n_neurons))
            if i != j:
                pairs.append((i, j))
        self.pairs = pairs
        self.history = deque(maxlen=history)
        self.collapse_index = 0.0
        self._step_counter = 0
        self._cached_result = {"collapseIndex": 0.0, "ready": False}

    def step(self, phi):
        self.history.append(np.asarray(phi, dtype=np.float32).copy())
        self._step_counter += 1
        # only recompute every 20 ticks (expensive)
        if len(self.history) < 20 or self._step_counter % 20 != 0:
            return self._cached_result

        H = np.stack(list(self.history))  # (T, n)
        # we measure: corr(i_t, j_{t+1}) and corr(j_t, i_{t+1})
        # both must be high for true bidirectionality
        corr_forward = []
        corr_back = []
        with np.errstate(invalid="ignore", divide="ignore"):
            for (i, j) in self.pairs:
                xi = H[:-1, i]
                xj = H[:-1, j]
                yi = H[1:, i]
                yj = H[1:, j]
                if xi.std() < 1e-6 or xj.std() < 1e-6 or yi.std() < 1e-6 or yj.std() < 1e-6:
                    continue
                a = float(np.corrcoef(xi, yj)[0, 1])
                b = float(np.corrcoef(xj, yi)[0, 1])
                if not np.isnan(a) and not np.isnan(b):
                    corr_forward.append(abs(a))
                    corr_back.append(abs(b))
        if not corr_forward:
            return self._cached_result
        per_pair_min = np.minimum(corr_forward, corr_back)
        ci = float(np.mean(per_pair_min))
        self.collapse_index = 0.9 * self.collapse_index + 0.1 * ci
        self._cached_result = {
            "collapseIndex": float(self.collapse_index),
            "ready": True,
            "forwardMean": float(np.mean(corr_forward)),
            "backwardMean": float(np.mean(corr_back)),
        }
        return self._cached_result
