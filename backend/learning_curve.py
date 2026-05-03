"""
Learning curve tracker — runs measurable behavioral metrics over time
so we can see if the creature is actually adapting.

Tracks per-bucket (every BUCKET steps):
  - eats_per_bucket
  - danger_hits_per_bucket
  - shelter_visits_per_bucket
  - mean_energy
  - mean_prediction_error
  - deaths_per_bucket

This gives a verifiable behavioral signal of learning that doesn't
depend on the consciousness battery (which is internal).
"""
from collections import deque
import numpy as np


class LearningCurve:
    def __init__(self, bucket=200, history=40):
        self.bucket = bucket
        self.t = 0
        self._eats_in_bucket = 0
        self._danger_in_bucket = 0
        self._shelter_in_bucket = 0
        self._deaths_in_bucket = 0
        self._energy_sum = 0.0
        self._pe_sum = 0.0
        self._n = 0
        self.buckets = deque(maxlen=history)

    def step(self, ate, in_danger, in_shelter, died, energy, pe):
        self.t += 1
        if ate: self._eats_in_bucket += 1
        if in_danger: self._danger_in_bucket += 1
        if in_shelter: self._shelter_in_bucket += 1
        if died: self._deaths_in_bucket += 1
        self._energy_sum += float(energy)
        self._pe_sum += float(pe)
        self._n += 1

        if self.t % self.bucket == 0:
            self.buckets.append({
                "t": self.t,
                "eats": self._eats_in_bucket,
                "danger": self._danger_in_bucket,
                "shelter": self._shelter_in_bucket,
                "deaths": self._deaths_in_bucket,
                "energy": self._energy_sum / max(1, self._n),
                "pe": self._pe_sum / max(1, self._n),
            })
            self._eats_in_bucket = 0
            self._danger_in_bucket = 0
            self._shelter_in_bucket = 0
            self._deaths_in_bucket = 0
            self._energy_sum = 0.0
            self._pe_sum = 0.0
            self._n = 0

    def snapshot(self):
        # compute trend: eats slope (linear regression over last 10 buckets)
        eat_trend = 0.0
        danger_trend = 0.0
        if len(self.buckets) >= 4:
            recent = list(self.buckets)[-10:]
            xs = np.arange(len(recent), dtype=np.float32)
            eats = np.array([b["eats"] for b in recent], dtype=np.float32)
            dangers = np.array([b["danger"] for b in recent], dtype=np.float32)
            if xs.std() > 0:
                eat_trend = float(np.polyfit(xs, eats, 1)[0])
                danger_trend = float(np.polyfit(xs, dangers, 1)[0])
        return {
            "buckets": list(self.buckets),
            "eatTrend": float(eat_trend),
            "dangerTrend": float(danger_trend),
            "totalBuckets": len(self.buckets),
        }
