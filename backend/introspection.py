from collections import deque
import numpy as np


class IntrospectionTests:
    """
    Three online tests probing self-knowledge of the agent.

      1. Mirror: stores the agent's own hidden states. Periodically asks:
         given a hidden state (own or random shuffle), can a simple match
         function distinguish? Score = 1.0 means own states have higher
         similarity to recent trajectory than shuffles.

      2. Surprise: tracks self-model gap on "expected" steps vs steps
         marked as "novel" (after a perturbation). Reports ratio.
         A self-aware system should react more to novelty.

      3. Alignment: correlation between emitted symbol and gap level.
         If certain symbols cluster around high-gap states, the agent
         is reporting its own uncertainty consistently.
    """

    def __init__(self, hidden_size, n_symbols=16, window=200):
        self.window = window
        self.hidden_size = hidden_size
        self.n_symbols = n_symbols

        self.hidden_history = deque(maxlen=window)
        self.gap_history = deque(maxlen=window)
        self.symbol_history = deque(maxlen=window)
        self.novelty_flags = deque(maxlen=window)

        self.mirror_score = 0.0
        self.surprise_ratio = 1.0
        self.alignment_score = 0.0

    def record(self, hidden, gap, symbol, novelty=False):
        self.hidden_history.append(np.asarray(hidden, dtype=np.float32))
        self.gap_history.append(float(gap))
        self.symbol_history.append(int(symbol))
        self.novelty_flags.append(bool(novelty))

    def _mirror(self):
        if len(self.hidden_history) < 30:
            return self.mirror_score
        H = np.stack(list(self.hidden_history))
        # similarity of consecutive states (real trajectory)
        diffs_real = np.linalg.norm(H[1:] - H[:-1], axis=1)
        # shuffled baseline
        idx = np.random.permutation(H.shape[0])
        Hs = H[idx]
        diffs_shuf = np.linalg.norm(Hs[1:] - Hs[:-1], axis=1)
        # score: how much smaller real diffs are vs shuffle
        a, b = float(np.mean(diffs_real)), float(np.mean(diffs_shuf))
        if b < 1e-9:
            return self.mirror_score
        score = float(np.clip(1.0 - a / b, 0.0, 1.0))
        self.mirror_score = 0.9 * self.mirror_score + 0.1 * score
        return self.mirror_score

    def _surprise(self):
        if len(self.gap_history) < 30:
            return self.surprise_ratio
        gaps = np.array(self.gap_history)
        nov = np.array(self.novelty_flags)
        if nov.sum() < 3 or (~nov).sum() < 3:
            return self.surprise_ratio
        nov_mean = float(np.mean(gaps[nov]))
        norm_mean = float(np.mean(gaps[~nov]))
        if norm_mean < 1e-9:
            return self.surprise_ratio
        ratio = nov_mean / norm_mean
        self.surprise_ratio = 0.9 * self.surprise_ratio + 0.1 * ratio
        return self.surprise_ratio

    def _alignment(self):
        """
        Alignment = how predictable is the next symbol from the current one
        normalized by entropy. If symbols form deterministic transitions
        (cycles, fixed sequences), they carry stable information about
        internal trajectory -> high alignment. Random symbols -> alignment 0.
        """
        if len(self.symbol_history) < 30:
            return self.alignment_score
        syms = np.array(self.symbol_history)
        # transition matrix
        T = np.zeros((self.n_symbols, self.n_symbols), dtype=np.float32)
        for i in range(len(syms) - 1):
            T[syms[i], syms[i + 1]] += 1
        row_sum = T.sum(axis=1, keepdims=True) + 1e-9
        P = T / row_sum

        # per-source entropy: low entropy = predictable transitions
        entropies = []
        for r in range(self.n_symbols):
            if T[r].sum() < 3:
                continue
            p = P[r]
            p = p[p > 0]
            h = float(-np.sum(p * np.log(p))) / float(np.log(self.n_symbols) + 1e-9)
            entropies.append(h)
        if not entropies:
            return self.alignment_score
        # alignment = 1 - mean normalized entropy. high if transitions deterministic
        score = float(np.clip(1.0 - np.mean(entropies), 0.0, 1.0))
        self.alignment_score = 0.9 * self.alignment_score + 0.1 * score
        return self.alignment_score

    def compute(self):
        return {
            "mirror": self._mirror(),
            "surpriseRatio": self._surprise(),
            "alignment": self._alignment(),
        }
