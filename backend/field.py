import numpy as np


class NeuralField:
    """
    Continuous-time recurrent neural field with Hebbian plasticity.

    Dynamics (Euler-integrated, dt small):
        tau * dx/dt = -x + g * W @ phi(x) + I_ext + sigma * eta(t)

    Plasticity (online, local):
        dW/dt = alpha * (phi(x) phi(x)^T - decay * W) - lambda * W (norm pressure)

    No backprop. No loss. Weights evolve from co-activation only.
    Roles in field:
        - predictors: receive sensory input directly
        - observers: receive aggregate stats of other neurons
        - motors: read out for action
        - general: bulk recurrent population
    """

    def __init__(
        self,
        n=200,
        n_predictors=40,
        n_observers=32,
        n_motors=6,
        sensory_dim=59,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n = n
        self.n_predictors = n_predictors
        self.n_observers = n_observers
        self.n_motors = n_motors

        self.idx_pred = np.arange(0, n_predictors)
        self.idx_obs = np.arange(n_predictors, n_predictors + n_observers)
        self.idx_motor = np.arange(
            n_predictors + n_observers,
            n_predictors + n_observers + n_motors,
        )
        self.idx_general = np.arange(
            n_predictors + n_observers + n_motors, n
        )

        self.x = rng.normal(0, 0.1, size=n).astype(np.float32)

        # sparse random initialization, near critical regime
        density = 0.15
        scale = 1.0 / np.sqrt(n * density)
        mask = rng.random((n, n)) < density
        np.fill_diagonal(mask, False)
        self.W = (rng.normal(0, 1.0, size=(n, n)) * mask * scale).astype(np.float32)

        # input projection: sensory -> predictors
        self.W_in = rng.normal(0, 1.0 / np.sqrt(sensory_dim), size=(n_predictors, sensory_dim)).astype(np.float32)

        # observer input weights: observer receives stats from non-observer pool
        self.W_obs = rng.normal(0, 0.3, size=(n_observers, 4)).astype(np.float32)

        # innate motor instinct: small fixed projection from sensory to motors
        # creates a faint pull toward food when hungry; learning can override.
        # Sensory layout (new world):
        #   food_view(25) + sig_view(25) +
        #   [energy, hunger, ar, ac, dr_food, dc_food, food_dist, dr_shelter, dc_shelter, light] +
        #   action_oh(6)
        self.W_motor_in = np.zeros((n_motors, sensory_dim), dtype=np.float32)
        IDX_HUNGER = 25 + 25 + 1
        IDX_DR_FOOD = 25 + 25 + 4
        IDX_DC_FOOD = 25 + 25 + 5
        IDX_DR_SHELTER = 25 + 25 + 7
        IDX_DC_SHELTER = 25 + 25 + 8
        IDX_LIGHT = 25 + 25 + 9
        if sensory_dim > IDX_LIGHT and n_motors >= 4:
            # bias toward food when hungry
            instinct_food = 2.0
            self.W_motor_in[0, IDX_DR_FOOD] = -instinct_food
            self.W_motor_in[1, IDX_DR_FOOD] = +instinct_food
            self.W_motor_in[2, IDX_DC_FOOD] = -instinct_food
            self.W_motor_in[3, IDX_DC_FOOD] = +instinct_food
            # hunger as multiplier-like baseline drive
            self.W_motor_in[:4, IDX_HUNGER] = 0.3
            # mild bias toward shelter at night (low light)
            instinct_shelter = 0.6
            self.W_motor_in[0, IDX_DR_SHELTER] = -instinct_shelter
            self.W_motor_in[1, IDX_DR_SHELTER] = +instinct_shelter
            self.W_motor_in[2, IDX_DC_SHELTER] = -instinct_shelter
            self.W_motor_in[3, IDX_DC_SHELTER] = +instinct_shelter
            # darkness amplifies shelter pull
            self.W_motor_in[:4, IDX_LIGHT] = -0.2

        # parameters
        self.tau = 5.0
        self.dt = 0.5
        self.gain = 1.0
        self.noise_sigma = 0.15

        # plasticity (3-factor: pre x post x neuromodulator)
        self.alpha = 8e-4
        self.decay = 2e-3
        self.norm_lambda = 5e-4
        self.eligibility = np.zeros((n, n), dtype=np.float32)
        self.elig_decay = 0.9

        # readouts
        self.last_phi = np.tanh(self.x)

    def reset(self):
        self.x = self.rng.normal(0, 0.1, size=self.n).astype(np.float32)
        self.last_phi = np.tanh(self.x)
        self.eligibility *= 0.0

    def step(self, sensory):
        # construct input current
        I = np.zeros(self.n, dtype=np.float32)

        # 1. sensory drive into predictors
        sensory_f = sensory.astype(np.float32)
        I[self.idx_pred] += self.W_in @ sensory_f
        # 1b. innate instinct drive into motors (hunger-gated direction-to-food)
        I[self.idx_motor] += self.W_motor_in @ sensory_f

        # 2. reflexive observation: observers receive stats of non-observer neurons
        non_obs_mask = np.ones(self.n, dtype=bool)
        non_obs_mask[self.idx_obs] = False
        x_other = self.x[non_obs_mask]
        stats = np.array(
            [
                float(np.mean(x_other)),
                float(np.std(x_other)),
                float(np.mean(np.abs(x_other))),
                float(np.max(np.abs(x_other))),
            ],
            dtype=np.float32,
        )
        I[self.idx_obs] += self.W_obs @ stats

        # 3. recurrent drive (motors get scaled-down recurrent input
        # so reflex/instinct can dominate over random initial connectivity)
        phi = np.tanh(self.x)
        rec = self.gain * (self.W @ phi)
        rec[self.idx_motor] *= 0.2

        # 4. noise (criticality requires it)
        eta = self.rng.normal(0, self.noise_sigma, size=self.n).astype(np.float32)

        # Euler step
        dx = (-self.x + rec + I + eta) / self.tau
        self.x = self.x + self.dt * dx
        self.last_phi = np.tanh(self.x)

        return self.last_phi

    def hebbian_update(self, dopamine=0.0, baseline_plasticity=0.15):
        """
        3-factor learning rule:
            eligibility_ij accumulates phi_i * phi_j over time
            dW = alpha * (baseline + dopamine) * eligibility - decay - normalization

        Without dopamine: only baseline plasticity (slow, stabilizing)
        With dopamine: recent co-activations get strongly reinforced
            (REINFORCEMENT learning emerges from local rule + global signal)
        """
        phi = self.last_phi
        # update eligibility trace (decaying memory of recent coactivation)
        outer = np.outer(phi, phi)
        np.fill_diagonal(outer, 0.0)
        self.eligibility = self.elig_decay * self.eligibility + outer.astype(np.float32)

        # 3-factor update
        modulator = baseline_plasticity + float(dopamine)
        dW = self.alpha * (modulator * self.eligibility - self.decay * self.W)
        norm = np.linalg.norm(self.W) + 1e-9
        dW -= self.norm_lambda * self.W * norm
        self.W += dW.astype(np.float32)
        np.clip(self.W, -1.5, 1.5, out=self.W)

    def motor_logits(self):
        # use raw state x (unsaturated) for sharper action differentiation
        return self.x[self.idx_motor].copy()

    def predictor_state(self):
        return self.last_phi[self.idx_pred].copy()

    def observer_state(self):
        return self.last_phi[self.idx_obs].copy()

    def variance(self):
        return float(np.var(self.last_phi))

    def activity(self):
        return float(np.mean(np.abs(self.last_phi)))

    def weight_summary(self, blocks=16):
        """
        Coarse view of W matrix for visualization.
        Returns blocks x blocks downsampled mean magnitude.
        """
        n = self.n
        b = max(1, n // blocks)
        m = blocks
        out = np.zeros((m, m), dtype=np.float32)
        absW = np.abs(self.W)
        for i in range(m):
            for j in range(m):
                out[i, j] = float(
                    absW[i * b : (i + 1) * b, j * b : (j + 1) * b].mean()
                )
        return out
