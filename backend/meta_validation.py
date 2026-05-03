from collections import deque
import numpy as np


class MetaValidation:
    """
    Rigorous validity checks for the consciousness battery.

    Strategy: maintain a *parallel*, dummy battery that consumes shuffled
    or random versions of the same neural states. After enough samples,
    compare the real battery's outputs against the controls. If real
    scores aren't consistently higher than controls, the tests are noise.

    Stability: low variance of recent real outputs.

    Output:
        per-component validity:    real - control gap, normalized
        composite trust = mean of validity * stability
    """

    def __init__(self, history=120, n_neurons=200):
        self.history = history
        self.n_neurons = n_neurons
        self._real_outputs = deque(maxlen=history)
        # cheap proxy controls: collect summary stats of shuffled / random phi
        self._shuffled_collapse = deque(maxlen=history)
        self._random_collapse = deque(maxlen=history)
        self._shuffled_phi = deque(maxlen=history)
        self._random_phi = deque(maxlen=history)
        self._shuffled_temporal = deque(maxlen=history)
        self._random_temporal = deque(maxlen=history)
        self.last_validity = {"trust": 0.0}

    def _shuffle(self, phi):
        idx = np.random.permutation(len(phi))
        return phi[idx]

    def _random(self, phi):
        return np.random.normal(0, np.std(phi) + 1e-3, size=phi.shape).astype(np.float32)

    def _spectral_phi(self, history_arr):
        # mirror of _phi_subset but on first 8 indices of provided history
        if history_arr.shape[0] < 30:
            return 0.0
        H = history_arr[:, :8]
        std = H.std(axis=0)
        if (std < 1e-5).any():
            return 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(H.T)
        if not np.isfinite(corr).all():
            return 0.0
        eigs = np.linalg.eigvalsh(corr)
        eigs = np.clip(eigs, 1e-6, None)
        ratio = float(eigs.min() / eigs.max())
        return float(np.clip(1.0 - ratio, 0.0, 1.0))

    def _collapse_proxy(self, history_arr):
        # bidirectional lag-1 correlation across a small set of pairs
        if history_arr.shape[0] < 20:
            return 0.0
        n = history_arr.shape[1]
        # sample 24 pairs (not all C(N, 2)) — cheaper, still informative
        rng_i = np.random.randint(0, n, size=24)
        rng_j = np.random.randint(0, n, size=24)
        with np.errstate(invalid="ignore", divide="ignore"):
            mins = []
            for i, j in zip(rng_i, rng_j):
                if i == j:
                    continue
                xi = history_arr[:-1, i]; yj = history_arr[1:, j]
                xj = history_arr[:-1, j]; yi = history_arr[1:, i]
                if xi.std() < 1e-6 or xj.std() < 1e-6 or yi.std() < 1e-6 or yj.std() < 1e-6:
                    continue
                a = float(abs(np.corrcoef(xi, yj)[0, 1]))
                b = float(abs(np.corrcoef(xj, yi)[0, 1]))
                if not np.isnan(a) and not np.isnan(b):
                    mins.append(min(a, b))
        return float(np.mean(mins)) if mins else 0.0

    def _temporal_proxy(self, history_arr):
        if history_arr.shape[0] < 30:
            return 0.0
        H = history_arr[-30:, :32]
        smoothed = np.array([H[i:i+5].mean(axis=0) for i in range(0, 25)])
        if smoothed.size == 0:
            return 0.0
        var_total = float(np.var(H, axis=0).mean())
        var_low = float(np.var(smoothed, axis=0).mean())
        if var_total < 1e-9:
            return 0.0
        return float(np.clip(var_low / var_total, 0.0, 1.0))

    def attach_battery(self, battery):
        self.battery = battery

    def step(self, phi, sensory, real_components):
        self._real_outputs.append(dict(real_components))

        # update shuffled / random history streams (one tick each)
        self._shuffled_phi.append(self._shuffle(phi))
        self._random_phi.append(self._random(phi))

        # compute control proxies on accumulated control-history
        if len(self._shuffled_phi) >= 30:
            sh_arr = np.stack(self._shuffled_phi)
            rd_arr = np.stack(self._random_phi)

            sh_collapse = self._collapse_proxy(sh_arr)
            rd_collapse = self._collapse_proxy(rd_arr)
            sh_phi_score = self._spectral_phi(sh_arr)
            rd_phi_score = self._spectral_phi(rd_arr)
            sh_temp = self._temporal_proxy(sh_arr)
            rd_temp = self._temporal_proxy(rd_arr)

            self._shuffled_collapse.append(sh_collapse)
            self._random_collapse.append(rd_collapse)
            self._shuffled_phi_score = sh_phi_score
            self._random_phi_score = rd_phi_score
            self._shuffled_temporal.append(sh_temp)
            self._random_temporal.append(rd_temp)

        if len(self._real_outputs) < 30:
            return self.last_validity

        # gather component arrays
        recent = self._real_outputs
        def real_mean(k):
            return float(np.mean([o.get(k, 0.0) for o in recent]))

        sh_collapse_m = float(np.mean(self._shuffled_collapse)) if self._shuffled_collapse else 0.0
        rd_collapse_m = float(np.mean(self._random_collapse)) if self._random_collapse else 0.0
        sh_temp_m = float(np.mean(self._shuffled_temporal)) if self._shuffled_temporal else 0.0
        rd_temp_m = float(np.mean(self._random_temporal)) if self._random_temporal else 0.0
        sh_phi_m = getattr(self, "_shuffled_phi_score", 0.0)
        rd_phi_m = getattr(self, "_random_phi_score", 0.0)

        per_test = {
            "phiSubset": float(np.clip((real_mean("phiSubset") - max(sh_phi_m, rd_phi_m)) * 2.0, 0.0, 1.0)),
            "collapseIndex": float(np.clip((real_mean("collapseIndex") - max(sh_collapse_m, rd_collapse_m)) * 2.0, 0.0, 1.0)),
            "temporalBinding": float(np.clip((real_mean("temporalBinding") - max(sh_temp_m, rd_temp_m)) * 2.0, 0.0, 1.0)),
        }

        # stability: variance of real outputs (low variance = test is stable)
        keys = list(real_components.keys())
        stds = [float(np.std([o.get(k, 0.0) for o in recent])) for k in keys]
        stability = float(np.clip(1.0 - np.mean(stds) * 4.0, 0.0, 1.0))

        # composite trust: mean of validity scores * stability
        validity_mean = float(np.mean(list(per_test.values())))
        trust = float(np.clip(0.7 * validity_mean + 0.3 * stability, 0.0, 1.0))

        self.last_validity = {
            "perTest": per_test,
            "stability": stability,
            "shuffledMean": {
                "collapse": sh_collapse_m,
                "phi": sh_phi_m,
                "temporal": sh_temp_m,
            },
            "randomMean": {
                "collapse": rd_collapse_m,
                "phi": rd_phi_m,
                "temporal": rd_temp_m,
            },
            "trust": trust,
        }
        return self.last_validity
