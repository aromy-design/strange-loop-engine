import numpy as np


class Neuromodulators:
    """
    Three global neuromodulator signals modulating plasticity, gain, and exploration.
    No supervised learning. Modulators are computed from intrinsic events.

    Dopamine (DA):
        Phasic spike when reward (ate). Otherwise decays toward baseline.
        Multiplies eligibility trace -> reinforces recent activity patterns
        that preceded reward.

    Serotonin (5HT):
        Slow signal tracking long-term wellbeing (mean energy).
        High = satisfied, calm. Low = stressed, restless.
        Modulates exploration: low 5HT -> more exploratory noise.

    Norepinephrine (NE):
        Spikes on surprise (large prediction error or novelty).
        Boosts gain transiently -> attentional alertness.
    """

    def __init__(self):
        self.DA = 0.0
        self.SHT = 0.5
        self.NE = 0.0

        self.DA_decay = 0.85
        self.NE_decay = 0.6  # faster decay so NE only fires on transients
        self.SHT_lr = 0.005

        # tracker for PE baseline (NE responds to novelty, not absolute level)
        self._pe_running = 1.0

    def update(self, ate, prediction_error, prev_pe, energy):
        # dopamine: phasic on reward, plus prediction-error-improvement bonus
        self.DA *= self.DA_decay
        if ate:
            self.DA += 1.0
        pe_drop = max(0.0, prev_pe - prediction_error)
        self.DA += 0.2 * pe_drop
        self.DA = float(np.clip(self.DA, 0.0, 3.0))

        # norepinephrine: fires on UNEXPECTED prediction error (novelty)
        # tracks running baseline of PE, NE rises when PE deviates from it
        self._pe_running = 0.97 * self._pe_running + 0.03 * float(prediction_error)
        deviation = max(0.0, float(prediction_error) - self._pe_running)
        self.NE *= self.NE_decay
        self.NE += 0.4 * deviation
        self.NE = float(np.clip(self.NE, 0.0, 2.5))

        # serotonin: slow tracker of energy
        self.SHT += self.SHT_lr * (float(energy) - self.SHT)
        self.SHT = float(np.clip(self.SHT, 0.0, 1.0))

        return self.snapshot()

    def snapshot(self):
        return {
            "dopamine": float(np.clip(self.DA, 0.0, 5.0)),
            "serotonin": float(self.SHT),
            "norepinephrine": float(np.clip(self.NE, 0.0, 5.0)),
        }
