from collections import deque
import numpy as np


class MindStateDecoder:
    """
    Translates raw internals (energy, neuromodulators, prediction error,
    activity, recent eats) into human-readable mind state with labels.

    NOT a measurement of subjective experience (no one can do that).
    It is a *consistent reading frame*: same internal pattern -> same label,
    so the user can learn to recognize what the creature is doing.

    Outputs:
        valence (-1 .. 1): pleasant vs unpleasant
        arousal (0 .. 1): calm vs alert
        hunger (0 .. 1): satiated vs starving
        curiosity (0 .. 1): bored vs curious
        intent: dominant motor direction or "still"
        label: short text describing dominant state
    """

    INTENT_NAMES = ["up", "down", "left", "right", "look", "speak", "still"]

    def __init__(self, window=60):
        self.eat_history = deque(maxlen=window)
        self.pe_history = deque(maxlen=window)

    def decode(self, energy, neuromods, prediction_error, activity, motor_logits, ate, prev_action):
        self.eat_history.append(1 if ate else 0)
        self.pe_history.append(float(prediction_error))

        hunger = float(np.clip(1.0 - energy, 0.0, 1.0))

        # arousal: norepinephrine + activity
        arousal = float(np.clip(0.5 * neuromods["norepinephrine"] + 0.5 * activity, 0.0, 1.0))

        # valence: serotonin centered + recent eats
        recent_eats = sum(self.eat_history) / max(1, len(self.eat_history))
        valence = float(np.clip(2.0 * (neuromods["serotonin"] - 0.5) + 2.0 * recent_eats - 0.5 * hunger, -1.0, 1.0))

        # curiosity: prediction error trend (rising or sustained high = curious)
        if len(self.pe_history) >= 5:
            recent_pe = float(np.mean(list(self.pe_history)[-5:]))
        else:
            recent_pe = float(prediction_error)
        curiosity = float(np.clip(recent_pe / 3.0, 0.0, 1.0))

        # intent: dominant motor neuron
        ml = np.asarray(motor_logits, dtype=np.float32)
        if float(ml.std()) < 0.01:
            intent_idx = 6
        else:
            intent_idx = int(np.argmax(ml))
        intent = self.INTENT_NAMES[intent_idx]

        # label: priority-based dominant state
        labels = []
        if hunger > 0.7:
            labels.append("starving")
        elif hunger > 0.4:
            labels.append("hungry")
        elif valence > 0.5:
            labels.append("satisfied")

        if neuromods["dopamine"] > 0.6:
            labels.append("rewarded")
        if neuromods["norepinephrine"] > 1.0:
            labels.append("alert")
        if curiosity > 0.5:
            labels.append("exploring")
        if neuromods["serotonin"] < 0.25:
            labels.append("stressed")
        if not labels:
            labels.append("calm")

        return {
            "hunger": hunger,
            "valence": valence,
            "arousal": arousal,
            "curiosity": curiosity,
            "intent": intent,
            "intentIdx": intent_idx,
            "labels": labels,
            "recentEats": recent_eats,
        }
