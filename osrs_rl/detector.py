"""Bot detectors: the scorecard for humanness.

Three detectors, all trained and tested with held-out players and held-out bot
sessions so they cannot memorise individuals:

* ``rules_floor``: fixed checks a lazy anti-cheat would run. A bot that fails
  these is not worth training further.
* ``FeatureDetector``: gradient-boosted trees on session summary features.
* ``SequenceDetector``: a GRU over the per-action token stream.

The reported humanness score is the best of the learned detectors on the test
split: balanced accuracy (0.5 is chance) and ROC AUC.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .metrics import FEATURE_NAMES, gaps, reaction_delays, session_features, session_tokens
from .env import CAMERA, CHECK_TAB


def rules_floor(sess) -> bool:
    """True means flagged as a bot. Two or more tells are needed."""
    log = sess["log"]
    g = gaps(log)
    rd = reaction_delays(log)
    tells = 0
    if len(g) > 5 and g.std() / max(g.mean(), 1) < 0.35:
        tells += 1  # metronome timing
    if not any(r["intent"] in (CAMERA, CHECK_TAB) for r in log):
        tells += 1  # never touches the camera or a tab in 30 minutes
    if len(g) > 5 and g.max() < 20_000:
        tells += 1  # never once looks away
    if len(rd) > 3 and np.percentile(rd, 10) < 250:
        tells += 1  # reacts faster than people do
    return tells >= 2


def _scores(y, p):
    return {"balanced_acc": float(balanced_accuracy_score(y, p > 0.5)), "auc": float(roc_auc_score(y, p))}


class FeatureDetector:
    def fit(self, sessions, labels):
        X = np.stack([session_features(s) for s in sessions])
        y = np.asarray(labels)
        w = np.where(y == 1, 0.5 / max(y.mean(), 1e-6), 0.5 / max(1 - y.mean(), 1e-6))
        self.m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
        self.m.fit(X, y, sample_weight=w)
        return self

    def predict(self, sessions):
        X = np.stack([session_features(s) for s in sessions])
        return self.m.predict_proba(X)[:, 1]


class _GRU(nn.Module):
    def __init__(self, d_in, h=64):
        super().__init__()
        self.gru = nn.GRU(d_in, h, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, hN = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (torch.arange(out.shape[1])[None, :] < lengths[:, None]).float().unsqueeze(-1)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(torch.cat([mean, hN[-1]], -1)).squeeze(-1)


def _pad(seqs):
    L = torch.tensor([len(s) for s in seqs])
    x = torch.zeros(len(seqs), int(L.max()), seqs[0].shape[1])
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.from_numpy(s)
    return x, L


class SequenceDetector:
    def __init__(self, epochs: int = 40, seed: int = 0):
        self.epochs, self.seed = epochs, seed

    def fit(self, sessions, labels):
        torch.manual_seed(self.seed)
        seqs = [session_tokens(s) for s in sessions]
        y = torch.tensor(labels, dtype=torch.float32)
        self.m = _GRU(seqs[0].shape[1])
        opt = torch.optim.Adam(self.m.parameters(), lr=2e-3)
        pos_w = (1 - y.mean()) / y.mean().clamp(min=1e-6)
        n = len(seqs)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 32):
                idx = perm[i : i + 32]
                x, L = _pad([seqs[j] for j in idx])
                logit = self.m(x, L)
                loss = nn.functional.binary_cross_entropy_with_logits(logit, y[idx], pos_weight=pos_w)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, sessions):
        x, L = _pad([session_tokens(s) for s in sessions])
        return torch.sigmoid(self.m(x, L)).numpy()


def evaluate_detectors(human_train, human_test, bot_train, bot_test) -> dict:
    """Train each detector on (human_train vs bot_train), score on the test split."""
    tr = human_train + bot_train
    ytr = [0] * len(human_train) + [1] * len(bot_train)
    te = human_test + bot_test
    yte = np.array([0] * len(human_test) + [1] * len(bot_test))
    res = {}
    rf = np.array([rules_floor(s) for s in te], dtype=float)
    res["rules_floor"] = {
        "balanced_acc": float(balanced_accuracy_score(yte, rf > 0.5)),
        "bot_flag_rate": float(rf[yte == 1].mean()),
        "human_false_flag_rate": float(rf[yte == 0].mean()),
    }
    fd = FeatureDetector().fit(tr, ytr)
    res["feature_gbt"] = _scores(yte, fd.predict(te))
    sd = SequenceDetector().fit(tr, ytr)
    res["sequence_gru"] = _scores(yte, sd.predict(te))
    best = max(("feature_gbt", "sequence_gru"), key=lambda k: res[k]["auc"])
    res["best"] = {"name": best, **res[best]}
    res["_feature_detector"] = fd
    return res


__all__ = ["rules_floor", "FeatureDetector", "SequenceDetector", "evaluate_detectors", "FEATURE_NAMES"]
