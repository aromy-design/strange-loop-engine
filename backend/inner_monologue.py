"""
Inner monologue — translates the creature's emergent symbol stream + state
into a readable thought log so the human observer can understand what is
happening inside.

This is NOT speech generation. There is no language model. It is a
priority-driven labeller that maps {symbol, mind state, body state, world
state} -> short natural-language descriptions of what the creature is
*currently doing/feeling*.

Two streams:
  - thoughts: sentences describing dominant present state
  - questions: emerge when self-surprise is high (the system is "puzzled")

The mapping is consistent: same internal pattern -> same thought, so the
user can learn the creature's vocabulary by watching it.
"""
from collections import deque, Counter
import numpy as np


class InnerMonologue:
    def __init__(self, history=80):
        self.thoughts = deque(maxlen=history)
        self.questions = deque(maxlen=24)
        self.last_thought = ""
        self.last_question = ""
        self._symbol_to_phrase = {}
        self._symbol_seen = Counter()
        self.t = 0

    def _learn_symbol_phrase(self, symbol, mind_label, intent):
        """
        First time we see a (symbol, dominant_label) pair, freeze it as that
        symbol's "meaning". Symbols thus acquire stable meaning grounded in
        the situations they occur in.
        """
        key = symbol
        if key not in self._symbol_to_phrase and self._symbol_seen[key] >= 3:
            self._symbol_to_phrase[key] = f"#{symbol:02d}={mind_label}/{intent}"
        self._symbol_seen[key] += 1

    def step(self, symbol, mind, neuromods, world, awareness):
        self.t += 1
        labels = mind.get("labels", [])
        intent = mind.get("intent", "still")
        primary = labels[0] if labels else "calm"

        self._learn_symbol_phrase(symbol, primary, intent)

        thought_parts = []
        # body state
        hunger = mind.get("hunger", 0)
        valence = mind.get("valence", 0)
        curiosity = mind.get("curiosity", 0)
        if hunger > 0.7:
            thought_parts.append("I am very hungry.")
        elif hunger > 0.4:
            thought_parts.append("I want food.")
        elif valence > 0.5:
            thought_parts.append("I feel good.")

        # action
        if intent in ("up", "down", "left", "right"):
            thought_parts.append(f"Moving {intent}.")
        elif intent == "look":
            thought_parts.append("Looking around.")
        elif intent == "speak":
            thought_parts.append("I speak.")
        elif intent == "still":
            thought_parts.append("Still.")

        # mood modifiers
        if "alert" in labels and "stressed" not in labels:
            thought_parts.append("Something caught my attention.")
        if "stressed" in labels:
            thought_parts.append("Things feel wrong.")
        if "rewarded" in labels:
            thought_parts.append("That was good.")
        if curiosity > 0.6:
            thought_parts.append("What is this?")

        # neuromodulator-driven
        if neuromods.get("dopamine", 0) > 0.6:
            thought_parts.append("(spike of pleasure)")
        if neuromods.get("norepinephrine", 0) > 1.0:
            thought_parts.append("(alert burst)")

        # symbol grounding
        sym_phrase = self._symbol_to_phrase.get(symbol)
        if sym_phrase:
            thought_parts.append(f"[{sym_phrase}]")
        else:
            thought_parts.append(f"[#{symbol:02d}]")

        thought = " ".join(thought_parts)

        if not self.thoughts or self.thoughts[-1]["text"] != thought:
            self.thoughts.append({"t": self.t, "text": thought})
            self.last_thought = thought

        # questions emerge from self-awareness events
        cur_event = awareness.get("currentEvent")
        if cur_event == "DISSOCIATION":
            q = "Why did that just happen to me?"
        elif cur_event == "SELF_SURPRISE":
            q = "Wait — what was that?"
        elif cur_event == "WORLD_SURPRISE":
            q = "Did the world just change?"
        else:
            q = None
        if q and q != self.last_question:
            self.questions.append({"t": self.t, "text": q})
            self.last_question = q

        return {
            "currentThought": self.last_thought,
            "recentThoughts": list(self.thoughts)[-10:],
            "recentQuestions": list(self.questions)[-6:],
            "symbolDictionary": dict(self._symbol_to_phrase),
        }
