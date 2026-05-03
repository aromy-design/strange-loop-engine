"""
Insect-brain architecture composed from sparse spiking circuits.

Inspired by Drosophila / honeybee neuroanatomy. Each circuit specializes
in a function and connects to others through projection paths.

  VisionLobe        ~800   retinotopic visual processing
  AntennalLobe      ~200   olfactory glomeruli (here: signals)
  LateralHorn       ~150   innate drives, fast reflexes
  MushroomBody      ~3000  sparse Kenyon cells, associative memory
  CentralComplex    ~500   heading + path integration
  SubesophagealG    ~200   feeding / taste
  MotorCenter       ~300   action pattern generation

Total: ~5150 neurons. Each connected sparsely to others. Sparsity 4-8%.

Inputs:
  vision_features    flat 5x5x2 = 50 (food + signal channels)
  olfaction          signal-kind in view (4 kinds)
  proprioception     position, energy, hunger, light, last action
  reward             dopamine-like, gates plasticity

Output:
  motor_logits       6-dim action vector
  thought_vector     compressed mushroom-body state for monologue
"""
import numpy as np
from .spiking_field import SpikingCircuit


class InsectBrain:
    def __init__(self, seed=0, mushroom_threshold=0.6):
        rng = np.random.default_rng(seed)
        self.rng = rng

        # specialized circuits
        self.vision = SpikingCircuit(n=800, density=0.06, name="vision", seed=seed + 1)
        self.antennal = SpikingCircuit(n=200, density=0.10, name="antennal", seed=seed + 2)
        self.lateral_horn = SpikingCircuit(n=150, density=0.12, name="lateral_horn", seed=seed + 3)
        self.mushroom = SpikingCircuit(n=3000, density=0.02, name="mushroom", seed=seed + 4,
                                       threshold=mushroom_threshold)
        self.central_complex = SpikingCircuit(n=500, density=0.08, name="central_complex", seed=seed + 5)
        self.subesophageal = SpikingCircuit(n=200, density=0.10, name="subesophageal", seed=seed + 6)
        self.motor = SpikingCircuit(n=300, density=0.10, name="motor", seed=seed + 7)

        # projection matrices (sparse) between circuits
        self.W_vis_to_mush = self._sparse_proj(self.vision.n, self.mushroom.n, density=0.01, rng=rng, scale=0.4)
        self.W_ant_to_mush = self._sparse_proj(self.antennal.n, self.mushroom.n, density=0.05, rng=rng, scale=0.5)
        self.W_mush_to_lh = self._sparse_proj(self.mushroom.n, self.lateral_horn.n, density=0.05, rng=rng, scale=0.3)
        self.W_mush_to_motor = self._sparse_proj(self.mushroom.n, self.motor.n, density=0.04, rng=rng, scale=0.4)
        self.W_cx_to_motor = self._sparse_proj(self.central_complex.n, self.motor.n, density=0.10, rng=rng, scale=0.6)
        self.W_lh_to_motor = self._sparse_proj(self.lateral_horn.n, self.motor.n, density=0.15, rng=rng, scale=0.7)
        self.W_seg_to_motor = self._sparse_proj(self.subesophageal.n, self.motor.n, density=0.10, rng=rng, scale=0.4)

        # input projections
        self.W_food_to_vision = rng.normal(0, 0.4, size=(self.vision.n, 25)).astype(np.float32)
        self.W_signal_to_vision = rng.normal(0, 0.4, size=(self.vision.n, 25)).astype(np.float32)
        self.W_signal_to_antennal = rng.normal(0, 0.5, size=(self.antennal.n, 4)).astype(np.float32)
        # proprioception channels: hunger, energy, light, dr_food, dc_food, dr_shelter, dc_shelter
        self.W_proprio_to_lh = rng.normal(0, 0.6, size=(self.lateral_horn.n, 7)).astype(np.float32)
        self.W_proprio_to_cx = rng.normal(0, 0.6, size=(self.central_complex.n, 7)).astype(np.float32)
        # reward to subesophageal (taste pathway)
        self.W_reward_to_seg = rng.normal(0, 0.7, size=(self.subesophageal.n, 1)).astype(np.float32)

        # readout: motor neurons -> 6 actions
        self.motor_to_action = rng.normal(0, 0.5, size=(6, self.motor.n)).astype(np.float32)
        # innate motor primitives (hard-coded direction toward food/shelter)
        # we leave learning to refine but seed with strong prior on a small subset
        # action neurons (first 30 of motor, partition into 6 groups of 5)
        # action group g (5 neurons) gets +1 bias when food/shelter is in direction g
        self.action_groups = []
        for a in range(6):
            self.action_groups.append(slice(a * 5, (a + 1) * 5))

        self.t = 0
        self.total_neurons = (self.vision.n + self.antennal.n + self.lateral_horn.n
                              + self.mushroom.n + self.central_complex.n
                              + self.subesophageal.n + self.motor.n)

    @staticmethod
    def _sparse_proj(n_in, n_out, density, rng, scale):
        from scipy import sparse
        nnz = max(1, int(n_in * n_out * density))
        rows = rng.integers(0, n_out, size=nnz)
        cols = rng.integers(0, n_in, size=nnz)
        vals = rng.normal(0, scale / np.sqrt(n_in * density), size=nnz).astype(np.float32)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(n_out, n_in), dtype=np.float32)

    def step(self, food_view, sig_view, signal_kinds_in_view, proprioception, reward):
        """
        food_view: (25,) flattened 5x5 food
        sig_view: (25,) flattened 5x5 signals
        signal_kinds_in_view: (4,) one-hot count of kinds nearby
        proprioception: (7,) hunger, energy, light, dr_food, dc_food, dr_shelter, dc_shelter
        reward: scalar
        """
        food_view = np.asarray(food_view, dtype=np.float32)
        sig_view = np.asarray(sig_view, dtype=np.float32)
        signal_kinds_in_view = np.asarray(signal_kinds_in_view, dtype=np.float32)
        proprioception = np.asarray(proprioception, dtype=np.float32)
        reward_arr = np.array([reward], dtype=np.float32)

        # 1. Vision
        I_vision = (self.W_food_to_vision @ food_view + self.W_signal_to_vision @ sig_view)
        self.vision.step(I_vision)

        # 2. Antennal lobe (signal "smell")
        I_ant = self.W_signal_to_antennal @ signal_kinds_in_view
        self.antennal.step(I_ant)

        # 3. Lateral horn (drives + innate)
        I_lh = self.W_proprio_to_lh @ proprioception
        # add fast pathway from antennal (smell triggers lh reflex)
        I_lh += 0.3 * self._project(self.antennal.spike[:self.lateral_horn.n], self.lateral_horn.n)
        self.lateral_horn.step(I_lh)

        # 4. Mushroom body (associative)
        I_mush = (self.W_vis_to_mush @ self.vision.spike.astype(np.float32)
                  + self.W_ant_to_mush @ self.antennal.spike.astype(np.float32))
        self.mushroom.step(I_mush)

        # 5. Central complex (heading)
        I_cx = self.W_proprio_to_cx @ proprioception
        self.central_complex.step(I_cx)

        # 6. Subesophageal (taste/feeding)
        I_seg = self.W_reward_to_seg @ reward_arr
        self.subesophageal.step(I_seg)

        # 7. Motor center
        mush_spikes = self.mushroom.spike.astype(np.float32)
        cx_spikes = self.central_complex.spike.astype(np.float32)
        lh_spikes = self.lateral_horn.spike.astype(np.float32)
        seg_spikes = self.subesophageal.spike.astype(np.float32)
        I_motor = (self.W_mush_to_motor @ mush_spikes
                   + self.W_cx_to_motor @ cx_spikes
                   + self.W_lh_to_motor @ lh_spikes
                   + self.W_seg_to_motor @ seg_spikes)
        self.motor.step(I_motor)

        # 8. Plasticity (gated by reward as dopamine signal)
        if reward > 0.05:
            # only do plasticity when reward to keep CPU low
            self.mushroom.hebbian_update(modulator=reward)
            self.motor.hebbian_update(modulator=reward)

        # 9. Action readout: aggregate motor neuron activity into 6-dim
        motor_act = self.motor.population_activity()
        action_logits = self.motor_to_action @ motor_act

        # also use the action_groups direct readout (sum of group spikes)
        group_activity = np.zeros(6, dtype=np.float32)
        for a, sl in enumerate(self.action_groups):
            group_activity[a] = float(self.motor.spike[sl].sum())

        # combine: continuous logits + discrete group activity
        combined = 0.4 * action_logits + 0.6 * group_activity

        self.t += 1
        return combined

    def _project(self, source_spikes, target_size):
        """Random projection helper: source spikes onto target dimension."""
        s = np.asarray(source_spikes, dtype=np.float32)
        if len(s) >= target_size:
            return s[:target_size]
        out = np.zeros(target_size, dtype=np.float32)
        out[:len(s)] = s
        return out

    def snapshot(self):
        return {
            "totalNeurons": int(self.total_neurons),
            "circuits": {
                "vision": self.vision.snapshot(k=20),
                "antennal": self.antennal.snapshot(k=20),
                "lateral_horn": self.lateral_horn.snapshot(k=20),
                "mushroom": self.mushroom.snapshot(k=40),
                "central_complex": self.central_complex.snapshot(k=20),
                "subesophageal": self.subesophageal.snapshot(k=20),
                "motor": self.motor.snapshot(k=20),
            },
        }

    def reset(self):
        for c in [self.vision, self.antennal, self.lateral_horn, self.mushroom,
                  self.central_complex, self.subesophageal, self.motor]:
            c.v.fill(0.0)
            c.spike.fill(0)
            c.refrac_ctr.fill(0)
            c.spike_history.clear()
