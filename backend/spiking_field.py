"""
Sparse spiking neural circuit — scalable to thousands of neurons.

Drosophila brain has ~130k neurons; ant ~250k; honeybee ~960k. Our previous
field had 200 with dense connectivity (40k weights). To match insect scale
we need:
  - sparse connectivity (~50-100 connections per neuron, not all-to-all)
  - binary spikes (1 bit, not float)
  - vectorized scipy.sparse operations
  - leaky integrate-and-fire dynamics

A single SpikingCircuit can hold 1000-10000 neurons in real-time on CPU.
Multiple circuits compose into an insect-brain architecture.
"""
import numpy as np
from scipy import sparse


class SpikingCircuit:
    """
    Leaky Integrate-and-Fire neurons with sparse Hebbian-eligible connectivity.

    State per neuron:
      v           membrane potential (float32)
      spike       binary 0/1 last tick

    Dynamics:
      v[t+1] = (1 - dt/tau) * v[t] + dt/tau * (W @ spike[t] + I_ext[t]) + noise
      if v >= threshold: spike=1, v=v_reset
      else: spike=0
    """

    def __init__(self, n, density=None, conn_per_neuron=None, name="circuit", seed=0,
                 tau=8.0, threshold=0.3, v_reset=0.0, refractory=2,
                 alpha=2e-4, decay=1e-4, e_decay=0.92):
        """
        Connectivity:
          - if `conn_per_neuron` given: each neuron has ~K outgoing connections
            (biological: insect neurons have ~100-1000 synapses regardless of
            brain size). Linear scaling in N. Recommended for large N.
          - else if `density` given: nnz = density * N^2 (quadratic in N).
            Backward compatible.
        """
        self.n = n
        self.name = name
        self.tau = tau
        self.threshold = threshold
        self.v_reset = v_reset
        self.refractory = refractory
        self.alpha = alpha
        self.decay = decay
        self.e_decay = e_decay

        rng = np.random.default_rng(seed)
        self.rng = rng

        # sparse connectivity
        if conn_per_neuron is not None:
            nnz = int(n * conn_per_neuron)
            effective_density = conn_per_neuron / n
        else:
            if density is None: density = 0.05
            nnz = int(n * n * density)
            effective_density = density
        density = effective_density  # for downstream computations
        rows = rng.integers(0, n, size=nnz)
        cols = rng.integers(0, n, size=nnz)
        # remove diagonal
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]
        vals = rng.normal(0, 1.0 / np.sqrt(n * density), size=len(rows)).astype(np.float32)
        self.W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)

        self.v = np.zeros(n, dtype=np.float32)
        self.spike = np.zeros(n, dtype=np.int8)
        self.refrac_ctr = np.zeros(n, dtype=np.int8)

        # eligibility traces — but we keep it as DENSE only for bins of recent
        # co-activation, so we don't need full N×N. Use a small running matrix
        # via outer-product on indices that just spiked.
        self.spike_history = []  # short list of recent spike vectors for plasticity
        self.history_len = 4

        self.noise_sigma = 0.15

    def step(self, I_ext):
        """One tick. I_ext is (n,) external input current."""
        # decrement refractory counters
        self.refrac_ctr = np.maximum(0, self.refrac_ctr - 1)

        # synaptic input from previous spikes
        syn = self.W @ self.spike.astype(np.float32)

        # leak + inputs
        v_new = (1 - 1.0 / self.tau) * self.v + (1.0 / self.tau) * (syn + I_ext)
        v_new += self.rng.normal(0, self.noise_sigma, size=self.n).astype(np.float32)

        # neurons in refractory don't update v
        in_refrac = self.refrac_ctr > 0
        v_new[in_refrac] = self.v_reset

        # threshold check
        new_spikes = (v_new >= self.threshold).astype(np.int8)
        # reset after spike
        v_new[new_spikes == 1] = self.v_reset
        self.refrac_ctr[new_spikes == 1] = self.refractory

        self.v = v_new
        self.spike = new_spikes

        # store for plasticity window
        self.spike_history.append(self.spike.copy())
        if len(self.spike_history) > self.history_len:
            self.spike_history.pop(0)

        return self.spike

    def hebbian_update(self, modulator=0.0):
        """
        STDP-lite: strengthen existing W edges between recent co-firing neurons.
        Operates directly on CSR data array (no lil conversion) for scalability.

        For each (j -> i) edge in W where i fired now and j fired previously:
            W[i,j] += alpha * (1 + modulator)
        Plus uniform decay.
        """
        if len(self.spike_history) < 2 or self.W.nnz == 0:
            return

        recent_spikes = self.spike_history[-1]
        prev_spikes = self.spike_history[-2]

        active_i = np.nonzero(recent_spikes)[0]
        active_j_set = set(np.nonzero(prev_spikes)[0].tolist())
        if len(active_i) == 0 or len(active_j_set) == 0:
            # decay only
            self.W.data *= (1 - self.decay)
            return

        # operate on CSR directly: for each row i in active_i, scan that row's
        # column-indices; if column j is in active_j_set, increment data
        lr = self.alpha * (1.0 + float(modulator))
        indptr = self.W.indptr
        indices = self.W.indices
        data = self.W.data

        # cap recent activity to keep per-tick cost bounded
        for i in active_i[:128]:
            row_start = indptr[i]
            row_end = indptr[i + 1]
            for k in range(row_start, row_end):
                j = indices[k]
                if j in active_j_set and j != i:
                    data[k] = float(np.clip(data[k] + lr, -1.5, 1.5))

        # uniform decay
        data *= (1 - self.decay)

    def firing_rate(self):
        """Mean firing rate over recent history."""
        if not self.spike_history:
            return 0.0
        return float(np.mean([s.mean() for s in self.spike_history]))

    def population_activity(self):
        """Returns (n,) average activity over short window."""
        if not self.spike_history:
            return np.zeros(self.n, dtype=np.float32)
        return np.mean(self.spike_history, axis=0).astype(np.float32)

    def snapshot(self, k=50):
        """Light-weight summary for transmission."""
        return {
            "name": self.name,
            "n": int(self.n),
            "firingRate": float(self.firing_rate()),
            "activeNow": int(self.spike.sum()),
            "wDensity": float(self.W.nnz / (self.n * self.n)),
            "wMagnitude": float(np.abs(self.W.data).mean()) if self.W.nnz > 0 else 0.0,
            "sample": self.population_activity()[:k].tolist(),
        }
