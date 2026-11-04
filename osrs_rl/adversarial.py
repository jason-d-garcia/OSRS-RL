"""Detector-in-the-loop reward (GAIL-style) for PPO.

At the end of every episode the agent's session is cut into 15-minute windows, the current
detectors score each one, and the agent gets ``weight * log((1 - p_bot) / p_bot)`` (clipped),
averaged over its windows, on top of its normal reward. Every ``retrain_every`` PPO updates the
detectors are retrained from scratch on the human windows vs the agent's most recent windows,
so the agent can't settle into one detector's blind spot.

Both learned detectors are used (the gradient-boosted stats model and the GRU over the action
sequence); their log-odds are averaged. The final evaluation in run_real.py always trains fresh
detectors after the agent is frozen, so a low score here is not taken on trust.
"""

from __future__ import annotations

import numpy as np

from .detector import FeatureDetector, SequenceDetector

WINDOW_MS = 15 * 60 * 1000


def split_windows(session: dict, window_ms: int = WINDOW_MS) -> list[dict]:
    """Cut a simulator session into full windows; progress is shared out by time."""
    n = int(session["ms"] // window_ms)
    out = []
    for k in range(n):
        lo, hi = k * window_ms, (k + 1) * window_ms
        rows = [dict(r) for r in session["log"] if lo <= r["t_ms"] < hi]
        for r in rows:
            r["t_ms"] -= lo
        out.append({"log": rows, "progress": session["progress"] * window_ms / max(session["ms"], 1.0),
                    "ms": float(window_ms), "activity": session.get("activity", "")})
    return out


class DetectorReward:
    def __init__(self, human_windows, weight: float = 1.0, clip: float = 4.0, retrain_every: int = 10,
                 buffer_windows: int = 160, seed: int = 0, log=print, project=None):
        # ``project`` maps every window before the detectors see it (e.g. bystanders.observable,
        # when the human side includes other players whose tab checks and camera moves are unseen).
        self.project = project or (lambda w: w)
        self.human = [self.project(w) for w in human_windows]
        self.weight, self.clip = weight, clip
        self.retrain_every, self.buffer_windows = retrain_every, buffer_windows
        self.seed, self.log = seed, log
        self.buffer: list[dict] = []
        self.fd = self.sd = None
        self.history: list[dict] = []

    def fit(self, bot_windows=None) -> None:
        bots = [self.project(w) for w in (bot_windows if bot_windows is not None else self.buffer[-self.buffer_windows:])]
        if len(bots) < 8:
            return
        sess = self.human + bots
        y = [0] * len(self.human) + [1] * len(bots)
        self.fd = FeatureDetector().fit(sess, y)
        self.sd = SequenceDetector(epochs=20, seed=self.seed).fit(sess, y)

    def p_bot(self, windows) -> np.ndarray:
        if self.fd is None or not windows:
            return np.full(len(windows), 0.5)
        windows = [self.project(w) for w in windows]
        return 0.5 * (self.fd.predict(windows) + self.sd.predict(windows))

    def bonus(self, session: dict) -> float:
        wins = split_windows(session)
        self.buffer += wins
        if self.fd is None or not wins:
            return 0.0
        p = np.clip(self.p_bot(wins), 1e-4, 1 - 1e-4)
        logit = np.clip(np.log((1 - p) / p), -self.clip, self.clip)
        return float(self.weight * logit.mean())

    def maybe_retrain(self, update: int) -> None:
        if update > 0 and update % self.retrain_every == 0 and len(self.buffer) >= 8:
            recent = self.buffer[-self.buffer_windows:]
            p_before = float(np.mean(self.p_bot(recent[-32:])))
            self.fit(recent)
            p_after = float(np.mean(self.p_bot(recent[-32:])))
            self.history.append({"update": update, "p_bot_before_retrain": p_before, "p_bot_after_retrain": p_after})
            self.log(f"  detector retrained at update {update}: agent p(bot) {p_before:.2f} -> {p_after:.2f}")
