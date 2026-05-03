"""
Spatial value map — a 16x16 grid where each cell tracks an expected value
based on outcomes experienced there. Mirrors the role of insect mushroom
body and central complex: a spatial cognitive map.

Updated with TD(λ) eligibility traces. When agent eats, value of eaten
cell goes up, AND a decaying trail of eligibility back-propagates value
to cells visited just before. This lets the agent learn "which paths lead
to food".

Provides:
  - V(r,c)            scalar value per cell
  - V_food(r,c)       remembered food likelihood
  - V_danger(r,c)     remembered danger
  - lastSeenFood      coordinate of most recent food sighting
  - lastSeenShelter   coordinate of last shelter visit

This is what gives insect-level robustness: stable place memory, learned
goal-direction, value-driven approach.
"""
import numpy as np


class SpatialMap:
    def __init__(self, size=16, lr=0.15, gamma=0.9, lam=0.85):
        self.size = size
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.V = np.zeros((size, size), dtype=np.float32)
        self.V_food = np.zeros((size, size), dtype=np.float32)
        self.V_danger = np.zeros((size, size), dtype=np.float32)
        self.elig = np.zeros((size, size), dtype=np.float32)
        self.visit_count = np.zeros((size, size), dtype=np.float32)
        self.last_food_seen = None
        self.last_shelter_seen = None
        self.last_pos = None
        self.last_value = 0.0

    def update(self, r, c, ate_kind, in_danger, in_shelter, food_visible_cells):
        """
        Called every step. Updates value at (r,c) using TD update with
        eligibility trace.
        """
        r = int(r); c = int(c)
        self.visit_count[r, c] += 1

        # immediate reward signal
        reward = 0.0
        if ate_kind == 2: reward += 1.0   # sweet
        elif ate_kind == 1: reward += 0.5 # plain
        if in_danger: reward -= 0.3
        if in_shelter: reward += 0.05

        # TD error
        v_now = float(self.V[r, c])
        td = reward + self.gamma * v_now - self.last_value

        # decay eligibility trace, set current cell to 1
        self.elig *= self.gamma * self.lam
        self.elig[r, c] = 1.0

        # apply update across all cells weighted by eligibility
        self.V += self.lr * td * self.elig
        # clip to keep stable, slow decay to prevent unbounded growth
        np.clip(self.V, -2.0, 2.0, out=self.V)
        self.V *= 0.9999

        # update food/danger memory at this cell
        if ate_kind > 0:
            self.V_food[r, c] = 0.95 * self.V_food[r, c] + 0.05 * 1.0
        else:
            self.V_food[r, c] *= 0.999  # slow decay if no food found

        if in_danger:
            self.V_danger[r, c] = 0.9 * self.V_danger[r, c] + 0.1 * 1.0
        else:
            self.V_danger[r, c] *= 0.998

        # remember sightings of food in current view
        for (fr, fc) in food_visible_cells:
            self.V_food[fr, fc] = max(self.V_food[fr, fc], 0.6)
            self.last_food_seen = (int(fr), int(fc))

        if in_shelter:
            self.last_shelter_seen = (r, c)

        self.last_pos = (r, c)
        self.last_value = float(self.V[r, c])

    def best_neighbor_direction(self, r, c):
        """
        Returns dr, dc towards highest-value neighbor (4-direction).
        Used to bias motor neurons toward profitable directions.
        """
        r = int(r); c = int(c)
        best_v = -1e9
        best_d = (0, 0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                v = (self.V[nr, nc]
                     + 0.5 * self.V_food[nr, nc]
                     - 0.6 * self.V_danger[nr, nc])
                if v > best_v:
                    best_v = v
                    best_d = (dr, dc)
        return best_d, float(best_v)

    def best_remembered_food(self):
        """Coordinate of cell with highest food memory."""
        if self.V_food.max() < 0.1:
            return None
        idx = int(np.argmax(self.V_food))
        r = idx // self.size
        c = idx % self.size
        return (r, c, float(self.V_food[r, c]))

    def direction_to_remembered_food(self, r, c):
        """Unit-vector direction to highest-memory food cell."""
        rem = self.best_remembered_food()
        if rem is None:
            return (0, 0)
        fr, fc, _ = rem
        return (int(np.sign(fr - r)), int(np.sign(fc - c)))

    def explore_bonus(self, r, c):
        """Negative-correlated with visit count — encourages exploration."""
        return float(1.0 / (1.0 + self.visit_count[r, c]))

    def heatmap(self):
        """Combined heatmap for viz."""
        m = self.V + 0.5 * self.V_food - 0.6 * self.V_danger
        return m.tolist()

    def snapshot(self):
        return {
            "lastFoodSeen": list(self.last_food_seen) if self.last_food_seen else None,
            "lastShelterSeen": list(self.last_shelter_seen) if self.last_shelter_seen else None,
            "valueRange": [float(self.V.min()), float(self.V.max())],
            "foodMemoryMax": float(self.V_food.max()),
            "dangerMemoryMax": float(self.V_danger.max()),
            "totalVisits": int(self.visit_count.sum()),
            "uniqueCellsVisited": int((self.visit_count > 0).sum()),
        }
