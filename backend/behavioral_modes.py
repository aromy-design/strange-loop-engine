"""
Behavioral mode switcher.

Insects display distinct behavioral programs that activate based on internal
state. Examples: foraging, fleeing, resting, courting. Each program biases
motor output differently.

Modes here:
  FORAGE   — actively seek food (default when hungry, daytime, safe)
  FLEE     — exit danger zone fast (when in danger)
  REST     — return to shelter (when energy low + safe to rest)
  EXPLORE  — wander to find new area (when bored / sated / curious)
  SLEEP    — minimal motor activity (delegated to sleep cycle)

Mode switch is a simple priority hierarchy with hysteresis (avoid rapid
flipping). The chosen mode produces a bias vector over actions which
multiplies action probabilities.
"""
import numpy as np


class BehavioralModes:
    MODES = ["FORAGE", "FLEE", "REST", "EXPLORE", "SLEEP"]

    def __init__(self):
        self.current = "EXPLORE"
        self.last_switch_t = 0
        self.t = 0
        self.history = []
        self.hysteresis = 8  # steps before allowing another switch

    def select(self, energy, in_danger, light_level, last_eat_recent, hunger):
        self.t += 1
        # hard priority: flee danger
        if in_danger:
            new = "FLEE"
        elif light_level < 0.25 and energy > 0.25:
            new = "SLEEP"
        elif energy < 0.30:
            new = "REST"
        elif hunger > 0.15 or last_eat_recent is False:
            new = "FORAGE"
        else:
            new = "EXPLORE"

        # hysteresis: don't switch too fast unless danger
        if new != self.current and (new == "FLEE" or self.t - self.last_switch_t > self.hysteresis):
            self.current = new
            self.last_switch_t = self.t
            self.history.append({"t": self.t, "mode": new})
            self.history = self.history[-32:]

        return self.current

    def action_bias(self, mode, spatial_map_dir, shelter_dir, food_dir, danger_dir):
        """
        Return per-action prior bias (length 6: up,down,left,right,look,speak).
        Each mode shapes preferences differently.
        spatial_map_dir, shelter_dir, food_dir, danger_dir are (dr, dc) tuples.
        """
        bias = np.ones(6, dtype=np.float32) * 0.05  # baseline
        # convert (dr, dc) to action index preference
        def dir_to_action_pref(d):
            dr, dc = d
            v = np.zeros(6, dtype=np.float32)
            if dr < 0: v[0] = -dr
            elif dr > 0: v[1] = dr
            if dc < 0: v[2] = -dc
            elif dc > 0: v[3] = dc
            return v

        if mode == "FORAGE":
            # toward food + spatial-map direction
            bias += 3.0 * dir_to_action_pref(food_dir)
            bias += 1.5 * dir_to_action_pref(spatial_map_dir)
            bias[4] = 0.05
            bias[5] = 0.03
        elif mode == "FLEE":
            # AWAY from danger
            ddr, ddc = danger_dir
            bias += 2.5 * dir_to_action_pref((-ddr, -ddc))
            bias[4] = 0.0
            bias[5] = 0.0
        elif mode == "REST":
            # toward shelter
            bias += 2.0 * dir_to_action_pref(shelter_dir)
            bias[4] = 0.3
            bias[5] = 0.05
        elif mode == "EXPLORE":
            # explore but still bias toward food / good cells
            bias += 0.6 * dir_to_action_pref(food_dir)
            bias += 0.8 * dir_to_action_pref(spatial_map_dir)
            bias[:4] += 0.4
            bias[4] = 0.3
            bias[5] = 0.15
        elif mode == "SLEEP":
            bias[:4] = 0.04
            bias[4] = 0.85
            bias[5] = 0.02

        bias = np.clip(bias, 0.0, 5.0)
        return bias

    def snapshot(self):
        return {
            "current": self.current,
            "lastSwitchT": int(self.last_switch_t),
            "stepsInMode": int(self.t - self.last_switch_t),
            "recentSwitches": self.history[-8:],
        }
