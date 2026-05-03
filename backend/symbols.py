import numpy as np


class EmergentSymbols:
    """
    Online clustering of neural state into discrete symbols.
    No supervision. No softmax head. Symbols are basins of dynamics.

    Implementation: streaming mini-batch k-means with adaptive learning rate.
    Each step:
      - find nearest cluster to current state
      - move that cluster slightly toward state
      - emit cluster index as symbol

    Symbols become meaningful because clusters carve up trajectory of
    actual lived states. They represent recurring patterns of being.
    """

    def __init__(self, dim, n_symbols=16, lr=0.05, smooth=0.7, seed=1):
        rng = np.random.default_rng(seed)
        self.k = n_symbols
        self.dim = dim
        self.lr = lr
        self.smooth = smooth
        self.centers = rng.normal(0, 0.5, size=(n_symbols, dim)).astype(np.float32)
        self.usage = np.ones(n_symbols, dtype=np.float32)
        self.last_symbol = 0
        self._smoothed_state = np.zeros(dim, dtype=np.float32)

    def step(self, state):
        s = state.astype(np.float32)
        # exponential smoothing — clusters now track stable patterns of being,
        # not instantaneous fluctuations
        self._smoothed_state = self.smooth * self._smoothed_state + (1 - self.smooth) * s
        s = self._smoothed_state
        d = self.centers - s
        dists = np.einsum("ij,ij->i", d, d)
        usage_norm = self.usage / (self.usage.sum() + 1e-9)
        bias = 5.0 * usage_norm * float(np.mean(dists))
        idx = int(np.argmin(dists + bias))

        # move winner toward state
        self.centers[idx] += self.lr * (s - self.centers[idx])

        # decay usage and increment winner
        self.usage *= 0.995
        self.usage[idx] += 1.0

        self.last_symbol = idx
        return idx

    def diversity(self):
        u = self.usage / (self.usage.sum() + 1e-9)
        # entropy of usage distribution, normalized to [0,1]
        ent = -np.sum(u * np.log(u + 1e-9))
        return float(ent / np.log(self.k))
