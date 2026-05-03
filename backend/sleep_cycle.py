"""
Sleep / dream cycle — consolidation through offline replay.

Inspired by hippocampal replay during slow-wave sleep: during night, the
agent reduces external interaction and replays recent neural state
sequences, applying Hebbian plasticity again on memorized patterns.
This reinforces frequently-co-active synapses in the absence of fresh
stimulation, simulating consolidation.

Effects during sleep:
  - reduced motor exploration (creature stays put)
  - reduced sensory drive
  - replay-driven Hebbian updates on past phi snapshots
  - dream symbols emerge from replay (visible to user)
"""
from collections import deque
import numpy as np


class SleepCycle:
    def __init__(self, replay_size=80):
        self.replay_buffer = deque(maxlen=replay_size)
        self.is_sleeping = False
        self.dream_active = False
        self.dream_symbol = -1
        self.dreams_count = 0
        self.replays_done = 0

    def record(self, phi):
        self.replay_buffer.append(np.asarray(phi, dtype=np.float32).copy())

    def check_sleep(self, light_level, energy):
        # sleep only deep into night and if energy comfortable; insects sleep less
        self.is_sleeping = bool(light_level < 0.15 and energy > 0.3)
        return self.is_sleeping

    def replay(self, field, symbols, intensity=0.3):
        """
        Pick a recent state sample and re-apply Hebbian learning on it
        without external sensory input. Generates a "dream symbol".
        """
        if not self.replay_buffer:
            self.dream_active = False
            return None
        idx = np.random.randint(0, len(self.replay_buffer))
        sample_phi = self.replay_buffer[idx]
        # weak replay: nudge field state toward sampled state
        # without overwriting it
        field.x = (1 - intensity) * field.x + intensity * sample_phi
        field.last_phi = np.tanh(field.x)
        # apply gentle hebbian update on this replayed pattern
        outer = np.outer(sample_phi, sample_phi)
        np.fill_diagonal(outer, 0.0)
        # reuse field's plasticity scale
        dW = field.alpha * 0.3 * outer
        field.W += dW.astype(np.float32)
        np.clip(field.W, -1.5, 1.5, out=field.W)
        # produce a dream symbol from the replayed state
        self.dream_symbol = symbols.step(sample_phi)
        self.dream_active = True
        self.replays_done += 1
        return self.dream_symbol

    def snapshot(self):
        return {
            "isSleeping": bool(self.is_sleeping),
            "dreamActive": bool(self.dream_active),
            "dreamSymbol": int(self.dream_symbol) if self.dream_symbol >= 0 else -1,
            "replaysDone": int(self.replays_done),
            "bufferSize": len(self.replay_buffer),
        }
