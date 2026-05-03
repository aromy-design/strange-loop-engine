import asyncio
import json
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agent import MetabolicAgent
from .world import GridWorld
from .introspection import IntrospectionTests
from .learning_curve import LearningCurve


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
if EXPERIMENTS_DIR.exists():
    app.mount("/experiments", StaticFiles(directory=str(EXPERIMENTS_DIR)), name="experiments")


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


class EngineState:
    def __init__(self):
        self.world = GridWorld(seed=7)
        self.agent = MetabolicAgent(sensory_dim=self.world.obs_dim, n_actions=self.world.N_ACTIONS, n_neurons=200)
        self.introspect = IntrospectionTests(
            hidden_size=self.agent.field.n, n_symbols=16, window=200
        )
        self.symbol_stream = deque(maxlen=64)
        self.novelty_pending = 0
        self.t = 0
        self.last_ate_kind = 0
        self.last_in_danger = False
        self.last_in_shelter = False
        self.learning = LearningCurve(bucket=200)
        self.lock = asyncio.Lock()
        self._battery_cache = {"components": {}, "composite": 0.0}
        self._validation_cache = {"trust": 0.0}

    def perturb(self):
        # randomize food positions (creates "world surprise")
        plain_count = int(np.sum(self.world.grid == self.world.FOOD_PLAIN))
        sweet_count = int(np.sum(self.world.grid == self.world.FOOD_SWEET))
        self.world.grid[:, :] = 0
        for _ in range(plain_count):
            self.world._spawn_food(self.world.FOOD_PLAIN)
        for _ in range(sweet_count):
            self.world._spawn_food(self.world.FOOD_SWEET)
        self.novelty_pending = 12

    def reset(self):
        self.world.reset()
        self.agent.reset()
        self.symbol_stream.clear()
        self.t = 0
        self.last_ate_kind = 0

    def tick(self):
        novelty = self.novelty_pending > 0
        if novelty:
            self.novelty_pending -= 1

        obs = self.world.observe(energy=self.agent.homeostat.E)
        ate_bool = self.last_ate_kind > 0

        # build spatial helpers for the agent
        ar, ac = int(self.world.agent[0]), int(self.world.agent[1])
        # nearest food direction
        import numpy as _np
        food_pos = _np.argwhere(self.world.grid > 0)
        if len(food_pos) > 0:
            d = food_pos - _np.array([ar, ac])
            dists = _np.linalg.norm(d, axis=1)
            nearest = food_pos[int(_np.argmin(dists))]
            food_dir = (
                _np.sign(int(nearest[0] - ar)),
                _np.sign(int(nearest[1] - ac)),
            )
        else:
            food_dir = (0, 0)
        # shelter direction
        sr, sc = self.world.shelter
        shelter_dir = (_np.sign(sr - ar), _np.sign(sc - ac))
        # nearest danger direction
        dangers = list(self.world.danger_cells)
        if dangers:
            ddists = [_np.hypot(dr - ar, dc - ac) for dr, dc in dangers]
            ndr, ndc = dangers[int(_np.argmin(ddists))]
            danger_dir = (_np.sign(ndr - ar), _np.sign(ndc - ac))
        else:
            danger_dir = (0, 0)
        food_visible = [(int(p[0]), int(p[1])) for p in food_pos]
        # signals visible in 5x5 view around agent
        signals_in_view = []
        for srow in range(max(0, ar - 2), min(self.world.SIZE, ar + 3)):
            for scol in range(max(0, ac - 2), min(self.world.SIZE, ac + 3)):
                k = int(self.world.signals[srow, scol])
                if k > 0:
                    signals_in_view.append(k)

        result = self.agent.step(
            obs,
            ate_last_action=ate_bool,
            ate_kind=int(self.last_ate_kind),
            in_danger=self.last_in_danger,
            in_shelter=self.last_in_shelter,
            light_level=self.world.light_level,
            agent_pos=(ar, ac),
            food_visible=food_visible,
            shelter_dir=shelter_dir,
            food_dir=food_dir,
            danger_dir=danger_dir,
            signals_in_view=signals_in_view,
        )

        next_obs, ate_kind, spoke, in_danger, in_shelter = self.world.step(result["action"])
        self.last_ate_kind = ate_kind
        self.last_in_danger = in_danger
        self.last_in_shelter = in_shelter

        if ate_kind > 0:
            bump = 0.6 if ate_kind == self.world.FOOD_SWEET else 0.35
            self.agent.homeostat.E = min(self.agent.homeostat.max_e, self.agent.homeostat.E + bump)

        # learning curve tracker
        self.learning.step(
            ate=ate_kind > 0,
            in_danger=in_danger,
            in_shelter=in_shelter,
            died=bool(result["homeo"].get("died", False)),
            energy=result["homeo"]["energy"],
            pe=result["predictionError"],
        )

        self.symbol_stream.append(int(result["symbol"]))
        self.introspect.record(
            hidden=result["fieldPhi"],
            gap=result["predictionError"],
            symbol=result["symbol"],
            novelty=novelty,
        )

        intro = self.introspect.compute()
        self.t += 1

        if self.t % 20 == 0:
            self._battery_cache = self.agent.battery.compute(
                phi=result["fieldPhi"],
                sensory=result["sensory"],
                closure_depth=result["meta"]["closureDepth"],
                collapse_index=result["coupling"].get("collapseIndex", 0.0),
                identity_persistence=result["turnover"]["identityPersistence"],
                mirror=intro["mirror"],
                surprise_ratio=intro["surpriseRatio"],
                alignment=intro["alignment"],
            )
            # extend battery with new components from H4..H7 (not yet in core battery)
            extra = {
                "presence": float(result["counterfactual"].get("presence", 0.0)),
                "awareness": float(result["awareness"].get("awarenessIndex", 0.0)),
                "ignitionRate": min(1.0, len(result["workspace"].get("recentIgnitions", [])) / 12.0),
                "selfRecognition": float(result["mirrorTest"].get("recognitionScore", 0.0)),
            }
            for k, v in extra.items():
                self._battery_cache["components"][k] = v
            # update composite to include new components
            comp = self._battery_cache["composite"]
            self._battery_cache["composite"] = float(
                0.7 * comp + 0.3 * np.mean(list(extra.values()))
            )
            self._validation_cache = self.agent.meta_validation.step(
                phi=result["fieldPhi"],
                sensory=result["sensory"],
                real_components=self._battery_cache["components"],
            )

        battery = self._battery_cache
        validation = self._validation_cache

        field_phi = result["fieldPhi"]
        bucket = np.array_split(field_phi, 20)
        neuron_activations = [float(b.mean()) for b in bucket]

        W_view = self.agent.field.weight_summary(blocks=16)
        obs_state = result["observerState"][:16].tolist() if result["observerState"].shape[0] >= 16 else result["observerState"].tolist()

        return {
            "t": self.t,
            "world": self.world.snapshot(),
            "neuronActivations": neuron_activations,
            "selfPrediction": obs_state,
            "symbol": result["symbol"],
            "symbolStream": list(self.symbol_stream),
            "metrics": {
                "predictionError": result["predictionError"],
                "selfPredictionError": result["selfPredictionError"],
                "energy": result["homeo"]["energy"],
                "aliveSteps": result["homeo"]["aliveSteps"],
                "deaths": self.agent.homeostat.deaths,
                "variance": result["crit"]["variance"],
                "smoothedVar": result["crit"]["smoothedVar"],
                "gain": result["crit"]["gain"],
                "symbolDiversity": result["symbolDiversity"],
                "activity": float(np.mean(np.abs(field_phi))),
                "actionTemperature": result.get("actionTemperature", 1.0),
                "presence": float(result["counterfactual"].get("presence", 0.0)),
            },
            "neuromods": result["neuromods"],
            "mind": result["mind"],
            "motorLogits": result["motorLogits"],
            "tests": intro,
            "meta": {
                "err1": result["meta"]["err1"],
                "err2": result["meta"]["err2"],
                "err3": result["meta"]["err3"],
                "closureDepth": result["meta"]["closureDepth"],
            },
            "coupling": result["coupling"],
            "turnover": result["turnover"],
            "battery": battery,
            "validation": validation,
            "workspace": result["workspace"],
            "awareness": result["awareness"],
            "monologue": result["monologue"],
            "sleep": result["sleep"],
            "mirrorTest": result["mirrorTest"],
            "awakening": result["awakening"],
            "continuity": result["continuity"],
            "spatial": result["spatial"],
            "spatialHeatmap": result["spatialHeatmap"],
            "mode": result["mode"],
            "path": result["path"],
            "associative": result["associative"],
            "pheromones": result["pheromones"],
            "pheromoneHeatmap": result["pheromoneHeatmap"],
            "tastePref": result["tastePref"],
            "totalNeurons": result.get("totalNeurons", 5350),
            "insectBrainVersion": result.get("insectBrainVersion", "V1"),
            "learning": self.learning.snapshot(),
            "spoke": spoke,
            "ate": bool(ate_kind > 0),
            "ateKind": int(ate_kind),
            "inDanger": bool(in_danger),
            "inShelter": bool(in_shelter),
            "novelty": novelty,
            "weightHeatmap": W_view.tolist(),
            "activityTrace": result["activityTrace"],
        }


engine = EngineState()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        send_task = asyncio.create_task(_stream(ws))
        recv_task = asyncio.create_task(_recv(ws))
        done, pending = await asyncio.wait(
            {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for p in pending:
            p.cancel()
    except WebSocketDisconnect:
        pass


async def _stream(ws: WebSocket):
    while True:
        async with engine.lock:
            payload = engine.tick()
        await ws.send_text(json.dumps(payload))
        await asyncio.sleep(1 / 20)


async def _recv(ws: WebSocket):
    while True:
        msg = await ws.receive_text()
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            continue
        kind = data.get("type")
        async with engine.lock:
            if kind == "perturb":
                engine.perturb()
            elif kind == "reset":
                engine.reset()
