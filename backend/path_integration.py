"""
Path integration / dead reckoning.

Insects (especially desert ants) maintain an internal vector pointing back
to home from anywhere they've been, by integrating their movements over
time. This lets them return directly even when displaced or after a
zigzag foraging trip.

We track:
  - displacement_from_shelter: cumulative (dr, dc) from last shelter visit
  - displacement_from_last_food: cumulative (dr, dc) from last meal
  - so the agent can compute "go home" direction or "return to food spot"
"""
import numpy as np


class PathIntegrator:
    def __init__(self):
        self.dr_shelter = 0
        self.dc_shelter = 0
        self.dr_lastfood = 0
        self.dc_lastfood = 0
        self.steps_since_shelter = 0
        self.steps_since_food = 0

    def step(self, action, in_shelter=False, ate=False):
        # actions: 0=up 1=down 2=left 3=right
        if action == 0:
            self.dr_shelter += 1
            self.dr_lastfood += 1
        elif action == 1:
            self.dr_shelter -= 1
            self.dr_lastfood -= 1
        elif action == 2:
            self.dc_shelter += 1
            self.dc_lastfood += 1
        elif action == 3:
            self.dc_shelter -= 1
            self.dc_lastfood -= 1

        if in_shelter:
            self.dr_shelter = 0
            self.dc_shelter = 0
            self.steps_since_shelter = 0
        else:
            self.steps_since_shelter += 1

        if ate:
            self.dr_lastfood = 0
            self.dc_lastfood = 0
            self.steps_since_food = 0
        else:
            self.steps_since_food += 1

    def shelter_direction(self):
        """Direction back to shelter (unit vector)."""
        return (np.sign(self.dr_shelter), np.sign(self.dc_shelter))

    def lastfood_direction(self):
        return (np.sign(self.dr_lastfood), np.sign(self.dc_lastfood))

    def snapshot(self):
        return {
            "shelterDR": int(self.dr_shelter),
            "shelterDC": int(self.dc_shelter),
            "stepsSinceShelter": int(self.steps_since_shelter),
            "lastFoodDR": int(self.dr_lastfood),
            "lastFoodDC": int(self.dc_lastfood),
            "stepsSinceFood": int(self.steps_since_food),
        }
