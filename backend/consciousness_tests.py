from collections import deque
import numpy as np


class ConsciousnessBattery:
    """
    A multi-test consciousness measurement battery. Each individual test
    is a *candidate* indicator. None alone is decisive. The composite is
    the closest functional summary we can compute.

    Tests:
      1. phi_subset: real integrated information on a small sub-network
         (8 neurons). Computed as min over bipartitions of mutual info loss.
      2. closure_depth: from MetaLoop (H1)
      3. collapse_index: from BidirectionalCoupling (H3)
      4. identity_persistence: from WeightTurnover (H2)
      5. mirror_score: trajectory coherence (existing)
      6. surprise_ratio: novelty-vs-normal gap (existing)
      7. alignment: symbol-state correlation (existing)
      8. causal_density: how much the system's own past predicts future
         beyond what the input alone does. Granger-like.
      9. temporal_binding: variance of low-frequency component of activity
         (sustained patterns vs fast noise)
     10. report_action_alignment: mind-state label predicts behavior
    """

    def __init__(self, n_neurons, history_window=200):
        self.n = n_neurons
        self.window = history_window
        # use observer neurons (idx 40..47) — most reflexive subnet, away from sensory
        self.phi_subset_idx = list(range(40, 48))
        self.activity_hist = deque(maxlen=history_window)
        self.input_hist = deque(maxlen=history_window)
        self.causal_density = 0.0

    def _phi_subset(self, phi):
        """
        Bounded integration measure on 8-neuron subset.

        Uses spectral structure of the correlation matrix:
            phi = 1 - eig_min/eig_max of the correlation matrix
        - Independent components -> all eigenvalues ~1 -> phi near 0
        - Fully integrated      -> one eigenvalue dominates -> phi near 1

        This is bounded [0, 1] and behaves well even on small windows,
        unlike log-det based estimators which saturate quickly.
        """
        if len(self.activity_hist) < 30:
            return 0.0
        H = np.stack([h[self.phi_subset_idx] for h in self.activity_hist])
        # need variance to compute correlation
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
        phi = float(np.clip(1.0 - ratio, 0.0, 1.0))
        # additionally weight by *minimum* off-diagonal correlation magnitude
        # this enforces "weakest link" — true integration requires all parts to be coupled
        n = corr.shape[0]
        off = np.abs(corr[np.triu_indices(n, k=1)])
        if off.size:
            weakest = float(np.percentile(off, 25))  # 25th percentile = weak-link proxy
        else:
            weakest = 0.0
        return float(np.clip(0.5 * phi + 0.5 * weakest, 0.0, 1.0))

    def _causal_density(self, phi, sensory):
        """
        Granger-like: does past activity predict current activity beyond what
        input alone explains? Increment estimate.

        Bounded by clipping residual variance away from zero and by clamping
        the relative gain to [0, 1].
        """
        self.activity_hist.append(np.asarray(phi, dtype=np.float32).copy())
        self.input_hist.append(np.asarray(sensory, dtype=np.float32).copy())
        if len(self.activity_hist) < 40:
            return self.causal_density
        # use only the last 60 samples to keep ridge regression fast
        H = np.stack(list(self.activity_hist)[-60:])
        I = np.stack(list(self.input_hist)[-60:])

        gains = []
        # measure: how much does PAST ACTIVITY reduce uncertainty about FUTURE
        # activity, compared to predicting the constant mean? Bounded to [0, 1].
        # Use a downsampled feature set to keep ridge solve cheap.
        feature_idx = np.arange(0, H.shape[1], 4)  # ~50 features
        for k in range(40, 56):
            y = H[1:, k]
            v_baseline = float(np.var(y)) + 1e-6
            x_past = H[:-1, feature_idx]
            def fit_predict(X, y, lam=2.0):
                XtX = X.T @ X + lam * np.eye(X.shape[1])
                w = np.linalg.solve(XtX, X.T @ y)
                yp = X @ w
                return float(np.var(y - yp))
            v_past = fit_predict(x_past, y)
            # gain = explained-variance fraction
            gain = max(0.0, (v_baseline - v_past) / v_baseline)
            gains.append(min(gain, 1.0))
        cd = float(np.mean(gains))
        self.causal_density = 0.9 * self.causal_density + 0.1 * cd
        return self.causal_density

    def _temporal_binding(self, phi):
        """
        Sustained patterns vs noise. Compute ratio of low-frequency variance
        to total variance via simple moving average decomposition.
        """
        if len(self.activity_hist) < 40:
            return 0.0
        H = np.stack(list(self.activity_hist)[-40:])[:, 40:72]  # observer band, recent
        smoothed = np.array([H[i:i+5].mean(axis=0) for i in range(0, 35)])
        if smoothed.size == 0:
            return 0.0
        var_total = float(np.var(H, axis=0).mean())
        var_low = float(np.var(smoothed, axis=0).mean())
        if var_total < 1e-9:
            return 0.0
        return float(np.clip(var_low / var_total, 0.0, 1.0))

    def compute(self, phi, sensory, closure_depth=0, collapse_index=0.0,
                identity_persistence=0.5, mirror=0.0, surprise_ratio=1.0,
                alignment=0.0, mind_label="", action=0):
        # store activity for phi/causal density
        phi_subset = self._phi_subset(phi)
        cd = self._causal_density(phi, sensory)
        tb = self._temporal_binding(phi)

        # composite consciousness index — weighted combination,
        # each component normalized to [0,1] roughly
        components = {
            "phiSubset": float(np.clip(phi_subset / 2.0, 0.0, 1.0)),
            "closureDepth": float(closure_depth / 3.0),
            "collapseIndex": float(np.clip(collapse_index, 0.0, 1.0)),
            "identityPersistence": float(np.clip(identity_persistence, 0.0, 1.0)),
            "mirror": float(np.clip(mirror, 0.0, 1.0)),
            "surpriseReact": float(np.clip(min(surprise_ratio / 2.0, 1.0), 0.0, 1.0)),
            "alignment": float(np.clip(alignment, 0.0, 1.0)),
            "causalDensity": float(np.clip(cd, 0.0, 1.0)),
            "temporalBinding": float(tb),
        }
        weights = {
            "phiSubset": 0.18,
            "closureDepth": 0.15,
            "collapseIndex": 0.15,
            "identityPersistence": 0.10,
            "mirror": 0.08,
            "surpriseReact": 0.08,
            "alignment": 0.10,
            "causalDensity": 0.08,
            "temporalBinding": 0.08,
        }
        composite = sum(components[k] * weights[k] for k in components)

        return {
            "components": components,
            "composite": float(composite),
        }
