import numpy as np


class MetaLoop:
    """
    Hierarchical self-modeling stack — incarnates Hypothesis 1 (closed
    topological loop). The system models itself, models its own modeling,
    models *that*, and so on. The depth at which the loop "grounds out"
    (prediction quality stops improving) is the closure depth.

    Levels:
        M0: NeuralField state (provided externally; not owned here)
        M1: predicts compressed M0 state
        M2: predicts M1 output
        M3: predicts M2 output

    All levels learned with simple delta rule (local, no backprop through
    the field). Each level's prediction error contributes to the closure
    metric.

    Closure depth = highest level where prediction error is below threshold.
    Higher closure depth = system models itself at deeper recursive levels
    = more "trapped inside its own loop".
    """

    def __init__(self, m0_dim, m1_dim=64, m2_dim=32, m3_dim=16, lr=0.02, lookahead=3, seed=0):
        from collections import deque
        rng = np.random.default_rng(seed)
        self.m0_dim = m0_dim
        self.m1_dim = m1_dim
        self.m2_dim = m2_dim
        self.m3_dim = m3_dim
        self.lookahead = lookahead

        self.E0 = rng.normal(0, 1.0 / np.sqrt(m0_dim), size=(m1_dim, m0_dim)).astype(np.float32)
        self.E1 = rng.normal(0, 1.0 / np.sqrt(m1_dim), size=(m2_dim, m1_dim)).astype(np.float32)
        self.E2 = rng.normal(0, 1.0 / np.sqrt(m2_dim), size=(m3_dim, m2_dim)).astype(np.float32)

        self.P1 = np.zeros((m1_dim, m1_dim), dtype=np.float32)
        self.P2 = np.zeros((m2_dim, m2_dim), dtype=np.float32)
        self.P3 = np.zeros((m3_dim, m3_dim), dtype=np.float32)

        self.lr = lr

        # ring buffers of past representations: predict from K steps ago to NOW
        # forces models to capture real dynamics, not local identity
        self.buf1 = deque(maxlen=lookahead + 1)
        self.buf2 = deque(maxlen=lookahead + 1)
        self.buf3 = deque(maxlen=lookahead + 1)

        # baselines: variance of representations (to normalize errors against signal scale)
        self.var1 = 1.0
        self.var2 = 1.0
        self.var3 = 1.0

        self.err1 = 1.0
        self.err2 = 1.0
        self.err3 = 1.0

    def step(self, m0_state):
        m0 = m0_state.astype(np.float32)
        z1 = np.tanh(self.E0 @ m0)
        z2 = np.tanh(self.E1 @ z1)
        z3 = np.tanh(self.E2 @ z2)

        self.buf1.append(z1.copy())
        self.buf2.append(z2.copy())
        self.buf3.append(z3.copy())

        # update running variance of representations (signal scale)
        self.var1 = 0.99 * self.var1 + 0.01 * float(np.var(z1))
        self.var2 = 0.99 * self.var2 + 0.01 * float(np.var(z2))
        self.var3 = 0.99 * self.var3 + 0.01 * float(np.var(z3))

        # predict NOW from K steps ago (lookahead-step prediction)
        if len(self.buf1) >= self.lookahead + 1:
            past_z1 = self.buf1[0]
            past_z2 = self.buf2[0]
            past_z3 = self.buf3[0]

            p1 = self.P1 @ past_z1
            p2 = self.P2 @ past_z2
            p3 = self.P3 @ past_z3

            e1 = z1 - p1
            e2 = z2 - p2
            e3 = z3 - p3

            self.P1 += self.lr * np.outer(e1, past_z1)
            self.P2 += self.lr * np.outer(e2, past_z2)
            self.P3 += self.lr * np.outer(e3, past_z3)

            # baseline: identity prediction error (just copy past as prediction)
            # a real self-model must beat this to count as closure
            id_e1 = z1 - past_z1
            id_e2 = z2 - past_z2
            id_e3 = z3 - past_z3

            # ratio: model error / identity error. <1 = model beats trivial copy.
            # clip to [0, 2] to prevent numerical blowup on stable representations.
            r1 = float(np.clip(np.var(e1) / max(1e-4, np.var(id_e1)), 0.0, 2.0))
            r2 = float(np.clip(np.var(e2) / max(1e-4, np.var(id_e2)), 0.0, 2.0))
            r3 = float(np.clip(np.var(e3) / max(1e-4, np.var(id_e3)), 0.0, 2.0))

            self.err1 = 0.95 * self.err1 + 0.05 * r1
            self.err2 = 0.95 * self.err2 + 0.05 * r2
            self.err3 = 0.95 * self.err3 + 0.05 * r3

        # closure depth: levels whose model meaningfully beats the trivial identity baseline
        # err < 0.85 means model error <85% of identity-copy error — measurable structure
        depth = sum(1 for e in (self.err1, self.err2, self.err3) if e < 0.85)

        return {
            "z1": z1,
            "z2": z2,
            "z3": z3,
            "err1": self.err1,
            "err2": self.err2,
            "err3": self.err3,
            "closureDepth": depth,
        }
