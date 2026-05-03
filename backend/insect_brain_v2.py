"""
InsectBrainV2 — biologically-scaled (~90k neurons) version.

7 specialized circuits with sizes inspired by Drosophila proportions.
All sparse spiking, ~100 connections per neuron (biological constraint),
linear scaling in N.

Total ~91,000 neurons. Real-time CPU achievable.
"""
import numpy as np
from scipy import sparse
from .spiking_field import SpikingCircuit


CIRCUIT_SIZES = {
    "vision":         15000,
    "antennal":        3000,
    "lateral_horn":    2000,
    "mushroom":       50000,  # Drosophila MB ~5k Kenyon, we exceed for capacity
    "central_complex": 5000,
    "subesophageal":   3000,
    "motor":           3000,
}


def _sparse_proj(n_in, n_out, conn_per_target, rng, scale):
    """Sparse projection: each target receives ~conn_per_target inputs."""
    nnz = max(1, n_out * conn_per_target)
    rows = rng.integers(0, n_out, size=nnz)
    cols = rng.integers(0, n_in, size=nnz)
    vals = rng.normal(0, scale / np.sqrt(conn_per_target), size=nnz).astype(np.float32)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_out, n_in), dtype=np.float32)


class InsectBrainV2:
    def __init__(self, sizes=None, conn_per_neuron=100, seed=0, mushroom_threshold=0.6):
        sizes = sizes or CIRCUIT_SIZES
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.sizes = sizes

        # circuits
        self.vision = SpikingCircuit(
            n=sizes["vision"], conn_per_neuron=conn_per_neuron, name="vision", seed=seed + 1)
        self.antennal = SpikingCircuit(
            n=sizes["antennal"], conn_per_neuron=conn_per_neuron, name="antennal", seed=seed + 2)
        self.lateral_horn = SpikingCircuit(
            n=sizes["lateral_horn"], conn_per_neuron=conn_per_neuron, name="lateral_horn", seed=seed + 3)
        self.mushroom = SpikingCircuit(
            n=sizes["mushroom"], conn_per_neuron=conn_per_neuron, name="mushroom",
            seed=seed + 4, threshold=mushroom_threshold)
        self.central_complex = SpikingCircuit(
            n=sizes["central_complex"], conn_per_neuron=conn_per_neuron, name="central_complex", seed=seed + 5)
        self.subesophageal = SpikingCircuit(
            n=sizes["subesophageal"], conn_per_neuron=conn_per_neuron, name="subesophageal", seed=seed + 6)
        self.motor = SpikingCircuit(
            n=sizes["motor"], conn_per_neuron=conn_per_neuron, name="motor", seed=seed + 7)

        # inter-circuit projections (each target neuron receives ~50 inputs from source circuit)
        proj_conn = 50
        self.W_vis_to_mush = _sparse_proj(sizes["vision"], sizes["mushroom"], proj_conn, rng, 0.4)
        self.W_ant_to_mush = _sparse_proj(sizes["antennal"], sizes["mushroom"], proj_conn, rng, 0.5)
        self.W_mush_to_lh = _sparse_proj(sizes["mushroom"], sizes["lateral_horn"], proj_conn, rng, 0.3)
        self.W_mush_to_motor = _sparse_proj(sizes["mushroom"], sizes["motor"], proj_conn, rng, 0.4)
        self.W_cx_to_motor = _sparse_proj(sizes["central_complex"], sizes["motor"], proj_conn, rng, 0.6)
        self.W_lh_to_motor = _sparse_proj(sizes["lateral_horn"], sizes["motor"], proj_conn, rng, 0.7)
        self.W_seg_to_motor = _sparse_proj(sizes["subesophageal"], sizes["motor"], proj_conn, rng, 0.4)

        # input projections (dense in -> first K neurons of circuit)
        # vision receives food_view(25) + sig_view(25) = 50-d input
        self.W_food_to_vision = rng.normal(0, 0.4, size=(sizes["vision"], 25)).astype(np.float32)
        self.W_signal_to_vision = rng.normal(0, 0.4, size=(sizes["vision"], 25)).astype(np.float32)
        self.W_signal_to_antennal = rng.normal(0, 0.5, size=(sizes["antennal"], 4)).astype(np.float32)
        self.W_proprio_to_lh = rng.normal(0, 0.6, size=(sizes["lateral_horn"], 7)).astype(np.float32)
        self.W_proprio_to_cx = rng.normal(0, 0.6, size=(sizes["central_complex"], 7)).astype(np.float32)
        self.W_reward_to_seg = rng.normal(0, 0.7, size=(sizes["subesophageal"], 1)).astype(np.float32)

        # action readout: aggregate motor neurons -> 6 actions
        # use grouped readout: split motor into 6 groups
        self.n_actions = 6
        self.motor_to_action_groups = []
        gsize = sizes["motor"] // 6
        for a in range(6):
            self.motor_to_action_groups.append(slice(a * gsize, (a + 1) * gsize))
        self.motor_to_action = rng.normal(0, 0.5 / np.sqrt(sizes["motor"]),
                                          size=(6, sizes["motor"])).astype(np.float32)

        self.t = 0
        self.total_neurons = sum(sizes.values())

    def step(self, food_view, sig_view, signal_kinds_in_view, proprioception, reward):
        food_view = np.asarray(food_view, dtype=np.float32)
        sig_view = np.asarray(sig_view, dtype=np.float32)
        signal_kinds_in_view = np.asarray(signal_kinds_in_view, dtype=np.float32)
        proprioception = np.asarray(proprioception, dtype=np.float32)
        reward_arr = np.array([reward], dtype=np.float32)

        # 1. Vision
        I_vision = self.W_food_to_vision @ food_view + self.W_signal_to_vision @ sig_view
        self.vision.step(I_vision)

        # 2. Antennal
        I_ant = self.W_signal_to_antennal @ signal_kinds_in_view
        self.antennal.step(I_ant)

        # 3. Lateral horn (proprioception only — antennal feedforward removed for size)
        I_lh = self.W_proprio_to_lh @ proprioception
        self.lateral_horn.step(I_lh)

        # 4. Mushroom body (vision + antennal projections)
        I_mush = (self.W_vis_to_mush @ self.vision.spike.astype(np.float32)
                  + self.W_ant_to_mush @ self.antennal.spike.astype(np.float32))
        self.mushroom.step(I_mush)

        # 5. Central complex (proprioception)
        I_cx = self.W_proprio_to_cx @ proprioception
        self.central_complex.step(I_cx)

        # 6. Subesophageal (taste/reward)
        I_seg = self.W_reward_to_seg @ reward_arr
        self.subesophageal.step(I_seg)

        # 7. Motor center (sum from MB, CX, LH, SEG)
        I_motor = (self.W_mush_to_motor @ self.mushroom.spike.astype(np.float32)
                   + self.W_cx_to_motor @ self.central_complex.spike.astype(np.float32)
                   + self.W_lh_to_motor @ self.lateral_horn.spike.astype(np.float32)
                   + self.W_seg_to_motor @ self.subesophageal.spike.astype(np.float32))
        self.motor.step(I_motor)

        # 8. Plasticity (gated by reward / dopamine)
        if reward > 0.05:
            self.mushroom.hebbian_update(modulator=reward)
            self.motor.hebbian_update(modulator=reward)
            self._cross_proj_plasticity(reward)

        # 9. Action readout: grouped sums + linear projection blend
        motor_act = self.motor.population_activity()
        action_logits = self.motor_to_action @ motor_act
        group_activity = np.zeros(self.n_actions, dtype=np.float32)
        for a, sl in enumerate(self.motor_to_action_groups):
            group_activity[a] = float(self.motor.spike[sl].sum())
        # normalize group activity to compete properly
        if group_activity.max() > 0:
            group_activity = group_activity / max(group_activity.max(), 1.0)
        combined = 0.4 * action_logits + 0.6 * group_activity

        self.t += 1
        return combined

    def _cross_proj_plasticity(self, reward):
        """
        STDP-lite on inter-circuit projections (target<-source). For each existing
        edge (i,j) where target i fired now and source j fired previously,
        strengthen W[i,j]. Vectorized via boolean mask on source indices.
        """
        lr = np.float32(2e-2 * (1.0 + float(reward)))

        def _update(W, src_recent, tgt_now):
            if W.nnz == 0: return
            active_src = np.nonzero(src_recent)[0]
            active_tgt = np.nonzero(tgt_now)[0]
            if len(active_src) == 0 or len(active_tgt) == 0:
                W.data *= np.float32(0.999995)
                return
            # boolean mask over source neurons (fast numpy indexing vs Python set)
            src_mask = np.zeros(W.shape[1], dtype=bool)
            src_mask[active_src] = True
            indptr = W.indptr; indices = W.indices; data = W.data
            for i in active_tgt[:256]:
                rs, re = indptr[i], indptr[i + 1]
                if rs == re: continue
                cols = indices[rs:re]
                hit = src_mask[cols]  # vectorized lookup — no Python loop
                if hit.any():
                    data[rs:re][hit] = np.clip(data[rs:re][hit] + lr,
                                               np.float32(-2.0), np.float32(2.0))
            data *= np.float32(0.999995)

        if len(self.mushroom.spike_history) >= 2:
            _update(self.W_mush_to_motor, self.mushroom.spike_history[-2], self.motor.spike)
        if len(self.central_complex.spike_history) >= 2:
            _update(self.W_cx_to_motor, self.central_complex.spike_history[-2], self.motor.spike)
        if len(self.lateral_horn.spike_history) >= 2:
            _update(self.W_lh_to_motor, self.lateral_horn.spike_history[-2], self.motor.spike)
        if len(self.subesophageal.spike_history) >= 2:
            _update(self.W_seg_to_motor, self.subesophageal.spike_history[-2], self.motor.spike)
        if len(self.vision.spike_history) >= 2:
            _update(self.W_vis_to_mush, self.vision.spike_history[-2], self.mushroom.spike)
        if len(self.antennal.spike_history) >= 2:
            _update(self.W_ant_to_mush, self.antennal.spike_history[-2], self.mushroom.spike)

    def aversive_conditioning(self, intensity=0.4):
        """
        Anti-Hebbian on vision→MB and MB→motor after aversive event (e.g. toxic food).
        Weakens connections that were active when the bad outcome occurred.
        Biologically: octopamine-mediated punishment signal in insect MB.
        """
        lr = np.float32(-2e-2 * float(intensity))

        def _anti_update(W, src_recent, tgt_now):
            if W.nnz == 0: return
            active_src = np.nonzero(src_recent)[0]
            active_tgt = np.nonzero(tgt_now)[0]
            if len(active_src) == 0 or len(active_tgt) == 0: return
            src_mask = np.zeros(W.shape[1], dtype=bool)
            src_mask[active_src] = True
            indptr = W.indptr; indices = W.indices; data = W.data
            for i in active_tgt[:256]:
                rs, re = indptr[i], indptr[i + 1]
                if rs == re: continue
                cols = indices[rs:re]
                hit = src_mask[cols]
                if hit.any():
                    data[rs:re][hit] = np.clip(data[rs:re][hit] + lr,
                                               np.float32(-2.0), np.float32(2.0))

        if len(self.vision.spike_history) >= 2:
            _anti_update(self.W_vis_to_mush, self.vision.spike_history[-2], self.mushroom.spike)
        if len(self.mushroom.spike_history) >= 2:
            _anti_update(self.W_mush_to_motor, self.mushroom.spike_history[-2], self.motor.spike)

    def snapshot(self):
        circs = {}
        for name in ["vision", "antennal", "lateral_horn", "mushroom",
                    "central_complex", "subesophageal", "motor"]:
            c = getattr(self, name)
            circs[name] = {
                "name": c.name,
                "n": c.n,
                "firingRate": c.firing_rate(),
                "activeNow": int(c.spike.sum()),
                "wDensity": float(c.W.nnz / (c.n * c.n)),
                "sample": c.population_activity()[:30].tolist(),
            }
        return {"totalNeurons": int(self.total_neurons), "circuits": circs}

    def reset(self):
        for c in [self.vision, self.antennal, self.lateral_horn, self.mushroom,
                  self.central_complex, self.subesophageal, self.motor]:
            c.v.fill(0.0)
            c.spike.fill(0)
            c.refrac_ctr.fill(0)
            c.spike_history.clear()
