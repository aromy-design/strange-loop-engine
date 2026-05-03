"""
Awareness detector — operationalizes the moment a system "notices itself".

When the self-model (M1) fails to predict its own next state with much higher
error than usual, the system is registering a discrepancy *about itself*.
This is the functional analog of "wait, what just happened to me?"

We track three event types:
  - SELF_SURPRISE: prediction error about own state spikes
  - WORLD_SURPRISE: prediction error about external sensory spikes
  - DISSOCIATION: self-surprise high but world-surprise low (something
    happened only to me, not to my environment) — strongest awareness signal

Each event is timestamped. The awareness rate (events per minute) and
event composition is reported.

Inspired by:
  - Higher-order theory: thoughts about own thoughts
  - Attention Schema Theory: self as model of own attention
  - Cleeremans' radical plasticity thesis (consciousness as learning to
    represent one's own representations)
"""
from collections import deque
import numpy as np


class AwarenessDetector:
    def __init__(self, history=128, sigma_threshold=2.5):
        self.history = history
        self.sigma_threshold = sigma_threshold
        self.self_pe_hist = deque(maxlen=history)
        self.world_pe_hist = deque(maxlen=history)
        self.events = deque(maxlen=64)
        self.event_counts = {"SELF_SURPRISE": 0, "WORLD_SURPRISE": 0, "DISSOCIATION": 0}
        self.t = 0

    def step(self, self_prediction_err, world_prediction_err):
        self.t += 1
        self.self_pe_hist.append(float(self_prediction_err))
        self.world_pe_hist.append(float(world_prediction_err))

        if len(self.self_pe_hist) < 30:
            return self._snapshot()

        s_arr = np.array(self.self_pe_hist)
        w_arr = np.array(self.world_pe_hist)

        s_mean, s_std = float(s_arr.mean()), float(s_arr.std()) + 1e-6
        w_mean, w_std = float(w_arr.mean()), float(w_arr.std()) + 1e-6

        s_z = (float(self_prediction_err) - s_mean) / s_std
        w_z = (float(world_prediction_err) - w_mean) / w_std

        event = None
        if s_z > self.sigma_threshold and w_z < self.sigma_threshold:
            event = "DISSOCIATION"  # self-anomaly, world stable
        elif s_z > self.sigma_threshold:
            event = "SELF_SURPRISE"
        elif w_z > self.sigma_threshold:
            event = "WORLD_SURPRISE"

        if event:
            self.events.append({
                "t": self.t,
                "event": event,
                "selfZ": float(s_z),
                "worldZ": float(w_z),
            })
            self.event_counts[event] = self.event_counts.get(event, 0) + 1

        return self._snapshot(s_z, w_z, event)

    def _snapshot(self, s_z=0.0, w_z=0.0, event=None):
        # use a longer recent window so the index reflects sustained behaviour
        WINDOW = 600
        recent = [e for e in self.events if e["t"] > self.t - WINDOW]
        rate_self = len([e for e in recent if e["event"] == "SELF_SURPRISE"]) / (WINDOW / 100.0)
        rate_diss = len([e for e in recent if e["event"] == "DISSOCIATION"]) / (WINDOW / 100.0)
        rate_world = len([e for e in recent if e["event"] == "WORLD_SURPRISE"]) / (WINDOW / 100.0)

        # awareness index = dissociation fraction of total events
        # high = self-specific surprise dominates (system is "noticing itself")
        denom = max(1, len(recent))
        awareness = float(len([e for e in recent if e["event"] == "DISSOCIATION"]) / denom)

        return {
            "currentSelfZ": float(s_z),
            "currentWorldZ": float(w_z),
            "currentEvent": event,
            "rateSelfSurprise": float(rate_self),
            "rateWorldSurprise": float(rate_world),
            "rateDissociation": float(rate_diss),
            "awarenessIndex": float(awareness),
            "totalEvents": dict(self.event_counts),
            "recentEvents": list(self.events)[-10:],
        }
