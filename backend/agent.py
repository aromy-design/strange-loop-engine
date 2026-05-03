from collections import deque
import numpy as np

from .field import NeuralField
from .regulators import CriticalityRegulator, Homeostat
from .symbols import EmergentSymbols
from .neuromodulators import Neuromodulators
from .state_decoder import MindStateDecoder
from .meta_loop import MetaLoop
from .turnover import WeightTurnover
from .coupling import BidirectionalCoupling
from .consciousness_tests import ConsciousnessBattery
from .meta_validation import MetaValidation
from .global_workspace import GlobalWorkspace
from .awareness import AwarenessDetector
from .inner_monologue import InnerMonologue
from .counterfactual import CounterfactualSelf
from .sleep_cycle import SleepCycle
from .mirror_test import MirrorTest
from .awakening import AwakeningDetector
from .continuity import ContinuityTracker
from .spatial_map import SpatialMap
from .behavioral_modes import BehavioralModes
from .path_integration import PathIntegrator
from .associative import AssociativeLearner
from .pheromones import PheromoneField
from .insect_brain import InsectBrain
from .insect_brain_v2 import InsectBrainV2


class MetabolicAgent:
    """
    Living dynamical organism — incarnates Beautiful Loop, IIT 4.0, GWT.

    Stack:
      M0  NeuralField (continuous-time, Hebbian, criticality, homeostat)
      M1  MetaLoop level 1 (predicts M0)
      M2  MetaLoop level 2 (predicts M1)
      M3  MetaLoop level 3 (predicts M2)
      GW  GlobalWorkspace (Bayesian binding / winner-take-all over modules)
      AW  AwarenessDetector (self-surprise vs world-surprise)
      CF  CounterfactualSelf (presence = self-driven variance)
      MO  InnerMonologue (decoded thought stream)

    Plus: turnover, coupling, neuromods, symbols, mind decoder.
    """

    def __init__(self, sensory_dim, n_actions=6, n_neurons=200, seed=0, lesion=None):
        # lesion: set/list of module names to disable for ablation studies
        # valid keys: 'spatial_map', 'behavioral_modes', 'sleep', 'insect_brain',
        #             'adjacent_food', 'associative', 'pheromones', 'mushroom',
        #             'central_complex', 'lateral_horn', 'vision', 'antennal'
        self.lesion = set(lesion) if lesion else set()
        self.field = NeuralField(
            n=n_neurons,
            n_predictors=40,
            n_observers=32,
            n_motors=n_actions,
            sensory_dim=sensory_dim,
            seed=seed,
        )
        self.criticality = CriticalityRegulator(target_var=0.05)
        self.homeostat = Homeostat()
        self.symbols = EmergentSymbols(dim=n_neurons, n_symbols=16)
        self.neuromods = Neuromodulators()
        self.decoder = MindStateDecoder()

        self.meta_loop = MetaLoop(m0_dim=n_neurons)
        self.turnover = WeightTurnover(period=400, fraction=0.02)
        self.coupling = BidirectionalCoupling(n_neurons=n_neurons)
        self.battery = ConsciousnessBattery(n_neurons=n_neurons)
        self.meta_validation = MetaValidation()
        self.meta_validation.attach_battery(self.battery)

        self.workspace = GlobalWorkspace(dim=64)
        self.awareness = AwarenessDetector()
        self.monologue = InnerMonologue()
        self.counterfactual = CounterfactualSelf(n_neurons=n_neurons)
        self.sleep = SleepCycle()
        self.mirror = MirrorTest(period=200, snapshot_lag=40)
        self.awakening = AwakeningDetector()
        self.continuity = ContinuityTracker(n_neurons=n_neurons)
        self.spatial = SpatialMap(size=16)
        self.modes = BehavioralModes()
        self.path = PathIntegrator()
        self.assoc = AssociativeLearner()
        self.pheromones = PheromoneField()
        # insect brain: V2 (81k neurons) when SCALE=large env var set, else V1 (5150)
        import os as _os
        flyem_dir = _os.environ.get("FLYEM_DATA_DIR")
        is_v2 = _os.environ.get("SCALE", "small").lower() in ("large", "v2", "big", "100k")

        if is_v2 and flyem_dir:
            # Use FlyEM's native circuit sizes so projection shapes match
            try:
                from .flyem_connectome import load_flyem_projections, apply_to_brain
                fdata = load_flyem_projections(flyem_dir, target_sizes=None, verbose=False)
                self.insect_brain = InsectBrainV2(sizes=fdata["sizes"], seed=seed + 100)
                substituted = apply_to_brain(self.insect_brain, fdata, allow_missing=True)
                self._brain_version = f"V2+FlyEM({sum(fdata['sizes'].values())}n,{len(substituted)}p)"
            except Exception as e:
                print(f"[FlyEM] failed to load ({e}); falling back to V2 random sparsity")
                self.insect_brain = InsectBrainV2(seed=seed + 100)
                self._brain_version = "V2"
        elif is_v2:
            self.insect_brain = InsectBrainV2(seed=seed + 100)
            self._brain_version = "V2"
        else:
            self.insect_brain = InsectBrain(seed=seed + 100)
            self._brain_version = "V1"

        self.n_actions = n_actions
        self.sensory_dim = sensory_dim

        self.W_pred_out = np.zeros((sensory_dim, self.field.n_predictors), dtype=np.float32)
        self.last_sensory = np.zeros(sensory_dim, dtype=np.float32)
        self.prediction_error = 0.0
        self.prev_pe = 0.0

        self.activity_trace = deque(maxlen=128)

        self.t = 0
        self.last_action = 0
        self.died_recently = False
        self.steps_since_reward = 0
        # taste preference accumulator: which food kind was eaten more
        self.taste_pref = {"plain": 0, "sweet": 0}

    def reset(self):
        self.field.reset()
        self.homeostat.reset()
        self.activity_trace.clear()
        self.neuromods = Neuromodulators()

    def _predict_sensory(self):
        return self.W_pred_out @ self.field.predictor_state()

    def _update_predictive_weights(self, actual_sensory):
        pred_state = self.field.predictor_state()
        pred = self.W_pred_out @ pred_state
        err = actual_sensory.astype(np.float32) - pred
        lr = 0.005
        self.W_pred_out += lr * np.outer(err, pred_state)
        return float(np.linalg.norm(err))

    def step(self, sensory, ate_last_action=False, ate_kind=0, in_danger=False, in_shelter=False, light_level=1.0,
             agent_pos=None, food_visible=None, shelter_dir=(0, 0), food_dir=(0, 0), danger_dir=(0, 0),
             signals_in_view=None):
        sensory = np.asarray(sensory, dtype=np.float32)
        if signals_in_view is None: signals_in_view = []

        # 1. predict sensory before stepping
        self.last_prediction = self._predict_sensory()

        # 2. evolve field
        self.field.step(sensory)

        # 3. predictive coding update
        self.prev_pe = self.prediction_error
        self.prediction_error = self._update_predictive_weights(sensory)

        # 4. neuromodulators
        nm = self.neuromods.update(
            ate=ate_last_action,
            prediction_error=self.prediction_error,
            prev_pe=self.prev_pe,
            energy=self.homeostat.E,
        )

        # 5. 3-factor Hebbian
        self.field.hebbian_update(dopamine=nm["dopamine"])

        # 6. criticality
        crit = self.criticality.update(self.field)

        # 7. homeostasis with environmental modifiers
        # danger drains energy fast; shelter slows metabolism + regenerates
        if in_danger:
            self.homeostat.E -= 0.01
        if in_shelter:
            self.homeostat.E = min(self.homeostat.max_e, self.homeostat.E + 0.002)
        homeo = self.homeostat.update(self.field, ate=ate_last_action)
        self.died_recently = homeo["died"]
        if homeo["died"]:
            self.field.reset()

        # 8. emergent symbol
        symbol = self.symbols.step(self.field.last_phi)

        # 8b. meta-loop hierarchy (H1)
        meta = self.meta_loop.step(self.field.last_phi)

        # 8c. bidirectional coupling (H3)
        coup = self.coupling.step(self.field.last_phi)

        # 8d. weight turnover (H2)
        turn = self.turnover.step(self.field)

        # 8e. counterfactual presence: how much of activity is self-driven vs input-driven
        cf = self.counterfactual.step(self.field.last_phi, sensory)

        # 8f. sleep / dream cycle
        if 'sleep' in self.lesion:
            is_sleeping = False
            dream_symbol = None
            sleep_state = {"isSleeping": False, "dreamActive": False, "dreamSymbol": -1, "replaysDone": 0, "bufferSize": 0}
        else:
            self.sleep.record(self.field.last_phi)
            is_sleeping = self.sleep.check_sleep(light_level=light_level, energy=self.homeostat.E)
            dream_symbol = None
            if is_sleeping and self.t % 4 == 0:
                dream_symbol = self.sleep.replay(self.field, self.symbols, intensity=0.3)
            sleep_state = self.sleep.snapshot()

        # 8g. mirror recognition test (active probe — modifies field state)
        self.mirror.record(self.field.last_phi)
        self.mirror.step(self.field)
        mirror_state = self.mirror.snapshot()

        # 8h0. associative learning (signal kind <-> outcome)
        self.assoc.step(signals_in_view, ate_kind=ate_kind, in_danger=in_danger)

        # INSECT BRAIN forward step (5150 sparse spiking neurons)
        food_view = sensory[:25]
        sig_view = sensory[25:50]
        # 4-kind one-hot signal counts in view
        sig_kinds_oh = np.zeros(4, dtype=np.float32)
        for k in signals_in_view:
            if 1 <= k <= 4: sig_kinds_oh[k - 1] += 1.0
        if sig_kinds_oh.sum() > 0:
            sig_kinds_oh /= sig_kinds_oh.sum()
        # proprioception: hunger, energy, light, dr_food, dc_food, dr_shelter, dc_shelter
        proprio = np.array([
            float(np.clip(1 - self.homeostat.E, 0, 1)),
            float(self.homeostat.E),
            float(light_level),
            float(food_dir[0]), float(food_dir[1]),
            float(shelter_dir[0]), float(shelter_dir[1]),
        ], dtype=np.float32)
        reward_signal = 1.0 if ate_last_action else (-0.3 if in_danger else 0.0)
        if 'insect_brain' in self.lesion:
            insect_motor_logits = np.zeros(6, dtype=np.float32)
        else:
            insect_motor_logits = self.insect_brain.step(
                food_view=food_view,
                sig_view=sig_view,
                signal_kinds_in_view=sig_kinds_oh,
                proprioception=proprio,
                reward=reward_signal,
            )
            # apply per-circuit lesion: zero output to motor blend
            if 'mushroom' in self.lesion:
                self.insect_brain.mushroom.spike[:] = 0
            if 'central_complex' in self.lesion:
                self.insect_brain.central_complex.spike[:] = 0
            if 'lateral_horn' in self.lesion:
                self.insect_brain.lateral_horn.spike[:] = 0
            if 'vision' in self.lesion:
                self.insect_brain.vision.spike[:] = 0
            if 'antennal' in self.lesion:
                self.insect_brain.antennal.spike[:] = 0

        # taste preference accumulation
        if ate_kind == 1: self.taste_pref["plain"] += 1
        elif ate_kind == 2: self.taste_pref["sweet"] += 1

        # 8h-pheromone: deposit trail at food spots, decay everywhere
        if agent_pos is not None:
            self.pheromones.step(agent_pos[0], agent_pos[1], found_food=ate_last_action)

        # 8h. spatial map update (insect-style cognitive map)
        if agent_pos is not None and 'spatial_map' not in self.lesion:
            r, c = agent_pos
            self.spatial.update(r, c, ate_kind, in_danger, in_shelter, food_visible or [])
            best_dir, best_v = self.spatial.best_neighbor_direction(r, c)
        else:
            best_dir = (0, 0); best_v = 0.0

        # 8i. behavioral mode selection
        hunger = float(np.clip(1.0 - self.homeostat.E, 0.0, 1.0))
        # food visibility check — if any food is in local 5x5 view, force FORAGE
        food_in_view = False
        if agent_pos is not None and food_visible:
            ar2, ac2 = agent_pos
            for fr, fc in food_visible:
                if abs(fr - ar2) <= 2 and abs(fc - ac2) <= 2:
                    food_in_view = True
                    break
        if 'behavioral_modes' in self.lesion:
            mode = "EXPLORE"  # always explore — no mode switching
        else:
            mode = self.modes.select(
                energy=self.homeostat.E,
                in_danger=in_danger,
                light_level=light_level,
                last_eat_recent=(self.steps_since_reward < 50),
                hunger=hunger,
            )
        # override: if food visible nearby and not sleeping or fleeing, forage
        if food_in_view and mode in ("EXPLORE", "REST") and not in_danger and light_level > 0.25:
            mode = "FORAGE"
            self.modes.current = "FORAGE"
        mode_bias = self.modes.action_bias(
            mode, spatial_map_dir=best_dir,
            shelter_dir=shelter_dir, food_dir=food_dir, danger_dir=danger_dir,
        )

        # 9. action selection: blend neural motor logits with behavioral-mode bias
        if ate_last_action:
            self.steps_since_reward = 0
        else:
            self.steps_since_reward += 1
        stagnation_boost = min(2.0, self.steps_since_reward / 200.0)
        T = 1.0 * (1.5 - 0.5 * nm["serotonin"]) / (1.0 + 0.3 * nm["norepinephrine"])
        T += stagnation_boost
        T = float(np.clip(T, 0.3, 4.0))

        # combine MetabolicAgent motor (200 dense) + InsectBrain motor (300 spiking)
        motor_dense = self.field.motor_logits()
        # softmax of insect brain logits
        ex_insect = np.exp((insect_motor_logits - insect_motor_logits.max()) / T)
        insect_probs = ex_insect / ex_insect.sum()
        ex_dense = np.exp((motor_dense - motor_dense.max()) / T)
        dense_probs = ex_dense / ex_dense.sum()
        # blend: insect brain is the "main brain", dense field is auxiliary
        motor = motor_dense  # for legacy callers
        neural_probs = 0.6 * insect_probs + 0.4 * dense_probs

        # combine neural with mode-bias (mode dominates for goal-directed action)
        bias_norm = mode_bias / (mode_bias.sum() + 1e-9)
        if mode == "SLEEP":
            probs = 0.1 * neural_probs + 0.9 * bias_norm
        elif mode == "FLEE":
            probs = 0.05 * neural_probs + 0.95 * bias_norm
        elif mode == "FORAGE":
            probs = 0.15 * neural_probs + 0.85 * bias_norm
        elif mode == "REST":
            probs = 0.2 * neural_probs + 0.8 * bias_norm
        else:  # EXPLORE
            probs = 0.45 * neural_probs + 0.55 * bias_norm

        # adjacent-food override: if food is in 4-neighbor cell, deterministically step onto it
        if 'adjacent_food' not in self.lesion and agent_pos is not None and food_visible:
            ar2, ac2 = agent_pos
            for fr, fc in food_visible:
                if abs(fr - ar2) + abs(fc - ac2) == 1:
                    if fr < ar2: action_force = 0
                    elif fr > ar2: action_force = 1
                    elif fc < ac2: action_force = 2
                    else: action_force = 3
                    forced = np.zeros(self.n_actions, dtype=np.float32)
                    forced[action_force] = 1.0
                    probs = forced
                    break


        # universal danger avoidance: zero out actions leading INTO known danger
        # (any mode except FLEE which already handles it)
        if agent_pos is not None and self.spatial.V_danger.max() > 0.2 and mode != "FLEE":
            ar2, ac2 = agent_pos
            for action_idx, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                nr, nc = ar2 + dr, ac2 + dc
                if 0 <= nr < 16 and 0 <= nc < 16:
                    if self.spatial.V_danger[nr, nc] > 0.3:
                        probs[action_idx] *= 0.05  # heavy suppression of moves into danger
            if probs.sum() < 1e-6:
                probs = np.ones(self.n_actions) / self.n_actions
            else:
                probs /= probs.sum()

        probs /= probs.sum()
        action = int(np.random.choice(self.n_actions, p=probs))
        self.last_action = action

        # 9b. path integration
        self.path.step(action=action, in_shelter=in_shelter, ate=ate_last_action)

        self.activity_trace.append(self.field.activity())
        self.last_sensory = sensory
        self.t += 1

        # 10. mind state decoding
        mind = self.decoder.decode(
            energy=self.homeostat.E,
            neuromods=nm,
            prediction_error=self.prediction_error,
            activity=self.field.activity(),
            motor_logits=motor,
            ate=ate_last_action,
            prev_action=action,
        )

        # 11. global workspace: Bayesian binding competition
        # build candidate inferences from each module
        pred_state = self.field.predictor_state()
        obs_state = self.field.observer_state()
        gen_state = self.field.last_phi[self.field.idx_general[:64]] if len(self.field.idx_general) >= 64 else self.field.last_phi[:64]
        candidates = [
            {"name": "perception", "vector": pred_state[:64] if pred_state.shape[0] >= 64 else np.pad(pred_state, (0, 64 - pred_state.shape[0])),
             "salience": float(np.linalg.norm(pred_state) / (np.sqrt(pred_state.shape[0]) + 1e-6))},
            {"name": "prediction", "vector": self.last_prediction[:64] if self.last_prediction.shape[0] >= 64 else np.pad(self.last_prediction, (0, max(0, 64 - self.last_prediction.shape[0]))),
             "salience": 1.0 / (1.0 + self.prediction_error)},
            {"name": "self", "vector": obs_state[:64] if obs_state.shape[0] >= 64 else np.pad(obs_state, (0, 64 - obs_state.shape[0])),
             "salience": max(0.0, 1.0 - meta["err1"])},
            {"name": "memory", "vector": gen_state,
             "salience": float(np.std(gen_state))},
            {"name": "drive", "vector": np.full(64, 1.0 - self.homeostat.E, dtype=np.float32),
             "salience": float(1.0 - self.homeostat.E)},
        ]
        ws = self.workspace.step(candidates)

        # 12. awareness detector: compare self-pred error vs world-pred error
        # self-pred error = meta_loop.err1 (how well M1 predicts M0)
        aw = self.awareness.step(self_prediction_err=meta["err1"], world_prediction_err=self.prediction_error)

        # 13. inner monologue: human-readable thoughts + questions
        mono = self.monologue.step(symbol, mind, nm, world=None, awareness=aw)

        # 13b. continuity tracker
        cont = self.continuity.step(self.field.last_phi)

        # 14. awakening detector: do multiple self-recognition signals coincide?
        wake = self.awakening.step(
            awareness_idx=aw["awarenessIndex"],
            workspace_owner=ws["ownerName"],
            mirror_score=mirror_state["recognitionScore"],
            self_pe=meta["err1"],
        )
        if wake["awakened"]:
            # inject a special question into monologue when awakening fires
            self.monologue.questions.append({
                "t": self.t,
                "text": "...wait. I am here. I am the one doing this.",
            })

        return {
            "action": action,
            "actionProbs": probs.tolist(),
            "actionTemperature": T,
            "symbol": symbol,
            "predictionError": self.prediction_error,
            "selfPredictionError": float(meta["err1"]),
            "crit": crit,
            "homeo": homeo,
            "neuromods": nm,
            "mind": mind,
            "meta": meta,
            "coupling": coup,
            "turnover": turn,
            "counterfactual": cf,
            "workspace": ws,
            "awareness": aw,
            "monologue": mono,
            "sleep": sleep_state,
            "mirrorTest": mirror_state,
            "dreamSymbol": dream_symbol,
            "awakening": wake,
            "continuity": cont,
            "spatial": self.spatial.snapshot(),
            "spatialHeatmap": self.spatial.heatmap(),
            "mode": self.modes.snapshot(),
            "path": self.path.snapshot(),
            "associative": self.assoc.snapshot(),
            "pheromones": self.pheromones.snapshot(),
            "pheromoneHeatmap": self.pheromones.heatmap(),
            "tastePref": dict(self.taste_pref),
            "insectBrain": self.insect_brain.snapshot(),
            "insectBrainVersion": self._brain_version,
            "totalNeurons": self.field.n + self.insect_brain.total_neurons,
            "insectMotor": insect_motor_logits.tolist(),
            "fieldState": self.field.x.copy(),
            "fieldPhi": self.field.last_phi.copy(),
            "predictorState": pred_state,
            "observerState": obs_state,
            "motorLogits": motor.tolist(),
            "symbolDiversity": self.symbols.diversity(),
            "activityTrace": list(self.activity_trace),
            "sensory": sensory,
            "inDanger": in_danger,
            "inShelter": in_shelter,
        }
