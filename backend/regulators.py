import numpy as np


class CriticalityRegulator:
    """
    Keeps the neural field at the edge of chaos.

    Tracks running variance of activity. Adjusts global gain so variance
    stays near a target band. Too low gain -> silent network. Too high ->
    seizure. Regulator slowly nudges gain.
    """

    def __init__(self, target_var=0.05, lr=0.01, gain_min=0.1, gain_max=2.0):
        self.target = target_var
        self.lr = lr
        self.gain_min = gain_min
        self.gain_max = gain_max
        self._smoothed_var = target_var

    def update(self, field):
        v = field.variance()
        self._smoothed_var = 0.95 * self._smoothed_var + 0.05 * v
        # if variance too high, lower gain; too low, raise gain
        err = self.target - self._smoothed_var
        field.gain = float(np.clip(field.gain + self.lr * err, self.gain_min, self.gain_max))
        return {
            "variance": v,
            "smoothedVar": self._smoothed_var,
            "gain": field.gain,
        }


class Homeostat:
    """
    Energy budget. Activity costs energy. Eating restores.
    Energy <= 0 forces death (field reset).
    """

    def __init__(self, e_init=1.0, cost_per_activity=0.0003, food_value=0.5, max_e=1.0):
        self.E = e_init
        self.cost = cost_per_activity
        self.food_value = food_value
        self.max_e = max_e
        self.alive_steps = 0
        self.deaths = 0

    def reset(self):
        self.E = 1.0
        self.alive_steps = 0

    def update(self, field, ate):
        a = field.activity()
        self.E -= self.cost * (1.0 + 4.0 * a)  # high activity costs more
        if ate:
            self.E = min(self.max_e, self.E + self.food_value)
        self.alive_steps += 1

        died = False
        if self.E <= 0.0:
            self.deaths += 1
            self.alive_steps = 0
            self.E = 1.0
            died = True
        return {"energy": float(self.E), "died": died, "aliveSteps": self.alive_steps}
