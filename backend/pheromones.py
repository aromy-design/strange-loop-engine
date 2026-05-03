"""
Pheromone trail — external memory in the environment.

Real ants and bees leave chemical trails: a "memory in the world" that
guides future foraging. We simulate this with a 16x16 scalar grid that
gets boosted at cells where food was found, and decays slowly. The agent
senses pheromone in its 5x5 view and is biased toward stronger
pheromone gradients.

This couples insect-style trail-following to our agent.
"""
import numpy as np


class PheromoneField:
    def __init__(self, size=16, decay=0.995, deposit=1.0, sense_radius=2):
        self.size = size
        self.decay = decay
        self.deposit = deposit
        self.sense_radius = sense_radius
        self.field = np.zeros((size, size), dtype=np.float32)

    def step(self, r, c, found_food):
        # decay everywhere
        self.field *= self.decay
        # deposit at current cell when food found
        if found_food:
            self.field[int(r), int(c)] += self.deposit
            np.clip(self.field, 0.0, 5.0, out=self.field)

    def gradient_direction(self, r, c):
        """
        Returns (dr, dc) direction toward strongest local pheromone, ignoring own cell.
        """
        r = int(r); c = int(c)
        best_v = 0.0
        best_d = (0, 0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if self.field[nr, nc] > best_v:
                    best_v = float(self.field[nr, nc])
                    best_d = (np.sign(dr), np.sign(dc))
        return best_d, best_v

    def heatmap(self):
        return self.field.tolist()

    def snapshot(self):
        return {
            "max": float(self.field.max()),
            "mean": float(self.field.mean()),
            "totalActive": int((self.field > 0.05).sum()),
        }
