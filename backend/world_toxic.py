"""
GridWorldWithToxic — extends GridWorld with toxic food cells.

Toxic food (FOOD_TOXIC=3) appears at 2 fixed locations.
Visual: food_view encodes as 1.5 (= 3/2.0) vs 0.5 (plain) vs 1.0 (sweet).
Effect: stepping on toxic food returns ate_kind=-1; experiment runner applies energy=-0.4.

MB associative memory test: can MB learn to avoid toxic food visually,
so that when spatial_map is lesioned, avoidance persists?
"""
import numpy as np
from .world import GridWorld


class GridWorldWithToxic(GridWorld):
    FOOD_TOXIC = 3
    TOXIC_PENALTY = -0.4    # energy cost per toxic-eating event

    def __init__(self, seed=0, n_toxic=2):
        self.n_toxic = n_toxic
        # Will be set in _setup_static_world
        self.toxic_food_locs = []
        super().__init__(seed=seed)

    def _setup_static_world(self):
        super()._setup_static_world()
        # Fixed toxic patches — near landmarks, away from shelter/danger
        locs = [(12, 4), (4, 12), (11, 11), (3, 11)]
        self.toxic_food_locs = locs[:self.n_toxic]

    def _empty_cell(self):
        """Override to also exclude toxic food locations."""
        for _ in range(64):
            r = self.rng.integers(0, self.SIZE)
            c = self.rng.integers(0, self.SIZE)
            if (self.grid[r, c] == 0
                    and self.signals[r, c] == 0
                    and (r, c) != tuple(self.agent)
                    and (r, c) not in self.danger_cells
                    and (r, c) not in set(self.toxic_food_locs)):
                return r, c
        return None

    def reset(self):
        super().reset()
        # Place toxic food at fixed locations (parent may have left them empty)
        for loc in self.toxic_food_locs:
            r, c = loc
            if self.grid[r, c] == 0:  # don't overwrite if somehow occupied
                self.grid[r, c] = self.FOOD_TOXIC

    def step(self, action):
        # Parent handles: move + good-food consumption + signals + last_reward
        # Toxic food (value 3) is NOT consumed by parent (not in FOOD_PLAIN/FOOD_SWEET)
        obs, ate_kind, spoke, in_danger, in_shelter = super().step(action)

        r, c = int(self.agent[0]), int(self.agent[1])

        # Check if agent landed on toxic food (parent left it on grid)
        if self.grid[r, c] == self.FOOD_TOXIC:
            self.grid[r, c] = 0   # consume it
            ate_kind = -1         # flag: toxic eaten (not a positive reward)

        # Ensure all toxic locations are populated (respawn if consumed)
        for loc in self.toxic_food_locs:
            lr, lc = loc
            if self.grid[lr, lc] == 0 and (lr, lc) not in self.danger_cells:
                self.grid[lr, lc] = self.FOOD_TOXIC

        return obs, ate_kind, spoke, in_danger, in_shelter
