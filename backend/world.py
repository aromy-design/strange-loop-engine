"""
Structured world for the metabolic agent.

Features:
  - 16x16 grid
  - Day/night cycle (visible to agent via 'light' sensor); night drains energy faster
  - Food: two types — sweet (high reward, sparse) and plain (lower reward, common)
  - Shelter: a fixed safe cell where agent regenerates faster and metabolism slows
  - Predator zone: marked cells; entering them drains energy fast (danger)
  - Persistent landmarks: the world has stable structure so the agent can
    learn it (not random every reset)

Sensory:
  - local 5x5 view of food
  - local 5x5 view of signals (now signal type encodes role: shelter/danger/landmark)
  - hunger, energy
  - position
  - direction to nearest food
  - direction to shelter
  - light level (day/night)
  - last action
"""
import numpy as np


class GridWorld:
    SIZE = 16
    VIEW = 5
    N_ACTIONS = 6  # up down left right look speak

    # cell encodings
    EMPTY = 0
    FOOD_PLAIN = 1
    FOOD_SWEET = 2

    # signal kinds
    SIG_SHELTER = 1
    SIG_DANGER = 2
    SIG_LANDMARK_A = 3
    SIG_LANDMARK_B = 4

    DAY_LENGTH = 600  # ticks per full day (light cycles)

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self._setup_static_world()
        self.reset()

    def _setup_static_world(self):
        """Persistent landmarks that don't change between episodes."""
        self.shelter = (2, 2)
        self.landmarks = [(13, 13), (2, 13), (13, 2)]
        # danger cells — fixed band
        self.danger_cells = set()
        for r in range(7, 10):
            for c in range(7, 10):
                if (r, c) != (8, 8):  # leave passable corridor center
                    self.danger_cells.add((r, c))

    def reset(self):
        self.grid = np.zeros((self.SIZE, self.SIZE), dtype=np.int8)
        self.signals = np.zeros((self.SIZE, self.SIZE), dtype=np.int8)

        # place persistent signals
        self.signals[self.shelter] = self.SIG_SHELTER
        for (r, c) in self.landmarks[:2]:
            self.signals[r, c] = self.SIG_LANDMARK_A
        for (r, c) in self.landmarks[2:]:
            self.signals[r, c] = self.SIG_LANDMARK_B
        for (r, c) in self.danger_cells:
            self.signals[r, c] = self.SIG_DANGER

        # spawn agent at shelter
        self.agent = np.array(list(self.shelter))
        self.steps = 0
        self.last_action = 0
        self.last_reward = 0.0

        # food: 4 plain + 2 sweet, random non-danger non-shelter cells
        for _ in range(4):
            self._spawn_food(self.FOOD_PLAIN)
        for _ in range(2):
            self._spawn_food(self.FOOD_SWEET)

    def _empty_cell(self):
        for _ in range(64):
            r = self.rng.integers(0, self.SIZE)
            c = self.rng.integers(0, self.SIZE)
            if (
                self.grid[r, c] == 0
                and self.signals[r, c] == 0
                and (r, c) != tuple(self.agent)
                and (r, c) not in self.danger_cells
            ):
                return r, c
        return None

    def _spawn_food(self, kind):
        cell = self._empty_cell()
        if cell:
            self.grid[cell] = kind

    @property
    def is_night(self):
        phase = (self.steps % self.DAY_LENGTH) / self.DAY_LENGTH
        return phase > 0.5

    @property
    def light_level(self):
        # smooth 0..1 light, peaking at midday
        phase = (self.steps % self.DAY_LENGTH) / self.DAY_LENGTH
        # 0..0.5 = day (light high), 0.5..1 = night
        return float(np.clip(0.5 + 0.5 * np.cos(2 * np.pi * phase), 0.0, 1.0))

    def step(self, action):
        action = int(action) % self.N_ACTIONS
        self.last_action = action
        spoke = False
        ate_kind = 0
        in_danger = False
        in_shelter = False

        if action == 0:
            self.agent[0] = max(0, self.agent[0] - 1)
        elif action == 1:
            self.agent[0] = min(self.SIZE - 1, self.agent[0] + 1)
        elif action == 2:
            self.agent[1] = max(0, self.agent[1] - 1)
        elif action == 3:
            self.agent[1] = min(self.SIZE - 1, self.agent[1] + 1)
        elif action == 5:
            spoke = True

        r, c = self.agent
        kind = int(self.grid[r, c])
        if kind in (self.FOOD_PLAIN, self.FOOD_SWEET):
            ate_kind = kind
            self.grid[r, c] = 0
            self._spawn_food(kind)

        in_danger = (r, c) in self.danger_cells
        in_shelter = (r, c) == self.shelter

        self.steps += 1
        self.last_reward = float(ate_kind) / 2.0  # 0.5 plain, 1.0 sweet

        return self.observe(), ate_kind, spoke, in_danger, in_shelter

    def observe(self, energy=0.5):
        pad = self.VIEW // 2
        # food layer
        food_padded = np.pad(self.grid, pad, mode="constant", constant_values=0).astype(np.float32)
        # signal layer (encode kind into [0..1])
        sig_padded = np.pad(self.signals, pad, mode="constant", constant_values=0).astype(np.float32) / 4.0
        r, c = self.agent
        food_view = food_padded[r : r + self.VIEW, c : c + self.VIEW] / 2.0
        sig_view = sig_padded[r : r + self.VIEW, c : c + self.VIEW]

        action_oh = np.zeros(self.N_ACTIONS, dtype=np.float32)
        action_oh[self.last_action] = 1.0

        hunger = float(np.clip(1.0 - energy, 0.0, 1.0))

        food_pos = np.argwhere(self.grid > 0)
        if len(food_pos) > 0:
            d = food_pos - np.array([r, c])
            dists = np.linalg.norm(d, axis=1)
            nearest = food_pos[int(np.argmin(dists))]
            dr_food = float((nearest[0] - r) / self.SIZE)
            dc_food = float((nearest[1] - c) / self.SIZE)
            food_dist = float(np.min(dists)) / self.SIZE
        else:
            dr_food = dc_food = food_dist = 0.0

        # direction to shelter
        sr, sc = self.shelter
        dr_shelter = float((sr - r) / self.SIZE)
        dc_shelter = float((sc - c) / self.SIZE)

        light = self.light_level

        obs = np.concatenate(
            [
                food_view.flatten(),
                sig_view.flatten(),
                np.array([
                    float(energy),
                    hunger,
                    float(r) / self.SIZE,
                    float(c) / self.SIZE,
                    dr_food,
                    dc_food,
                    food_dist,
                    dr_shelter,
                    dc_shelter,
                    light,
                ], dtype=np.float32),
                action_oh,
            ]
        )
        return obs

    @property
    def obs_dim(self):
        return self.VIEW * self.VIEW * 2 + 10 + self.N_ACTIONS

    def snapshot(self):
        food_cells = []
        for (r, c) in np.argwhere(self.grid > 0).tolist():
            food_cells.append({"r": int(r), "c": int(c), "kind": int(self.grid[r, c])})
        signals = []
        for (r, c) in np.argwhere(self.signals > 0).tolist():
            signals.append({"r": int(r), "c": int(c), "kind": int(self.signals[r, c])})
        return {
            "size": self.SIZE,
            "agent": [int(self.agent[0]), int(self.agent[1])],
            "food": food_cells,
            "signals": signals,
            "steps": int(self.steps),
            "lastAction": int(self.last_action),
            "lastReward": float(self.last_reward),
            "lightLevel": float(self.light_level),
            "isNight": bool(self.is_night),
            "shelter": list(self.shelter),
            "landmarks": [list(l) for l in self.landmarks],
        }
