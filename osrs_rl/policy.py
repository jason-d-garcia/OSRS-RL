"""Policy network with the plan's three heads.

* intent head: categorical over ``INTENTS``
* target head: a pointer over the visible-entity slots (scores each entity
  against the global context), so the action space does not change with the scene
* timing head: per-intent mixture of log-normals over the delay in ms. The mixture
  is what lets one policy produce quick routine clicks, slower reactions after a
  change and the long right tail of attention lapses.

The network sees the current decision point only plus a short history in the
observation (last three delays, actions per minute, time since break). The plan's
32-tick transformer is a later upgrade; this first version keeps the policy small
so it trains on CPU in minutes.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import ENT_FEATS, GLOBAL_DIM, INTERACT, MAX_ENTITIES, MAX_DELAY_MS, MIN_DELAY_MS, N_INTENTS

N_MIX = 3
LOG_MIN, LOG_MAX = math.log(MIN_DELAY_MS), math.log(MAX_DELAY_MS)


class Policy(nn.Module):
    def __init__(self, hidden: int = 256, ent_hidden: int = 64):
        super().__init__()
        self.ent = nn.Sequential(nn.Linear(ENT_FEATS, ent_hidden), nn.Tanh(), nn.Linear(ent_hidden, ent_hidden), nn.Tanh())
        self.glob = nn.Sequential(
            nn.Linear(GLOBAL_DIM + ent_hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.intent = nn.Linear(hidden, N_INTENTS)
        self.query = nn.Linear(hidden, ent_hidden)
        self.delay = nn.Linear(hidden, N_INTENTS * N_MIX * 3)
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, obs: torch.Tensor):
        g = obs[:, :GLOBAL_DIM]
        E = obs[:, GLOBAL_DIM:].view(-1, MAX_ENTITIES, ENT_FEATS)
        mask = E[:, :, -1] > 0.5
        e = self.ent(E)
        pooled = (e * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        h = self.glob(torch.cat([g, pooled], -1))
        intent_logits = self.intent(h)
        tgt = torch.einsum("bd,bkd->bk", self.query(h), e) / math.sqrt(e.shape[-1])
        tgt = tgt.masked_fill(~mask, -1e9)
        dp = self.delay(h).view(-1, N_INTENTS, N_MIX, 3)
        mix_logits = dp[..., 0]
        mu = 7.0 + 2.0 * dp[..., 1]
        log_std = dp[..., 2].clamp(-3.0, 1.0) - 0.5
        return {"intent": intent_logits, "target": tgt, "mix": mix_logits, "mu": mu, "log_std": log_std,
                "value": self.value(h).squeeze(-1)}

    # ------------------------------------------------------------------
    @staticmethod
    def log_prob(out, intent, target, log_delay):
        lp_i = F.log_softmax(out["intent"], -1).gather(1, intent[:, None]).squeeze(1)
        lp_t = F.log_softmax(out["target"], -1).gather(1, target[:, None]).squeeze(1)
        lp_t = torch.where(intent == INTERACT, lp_t, torch.zeros_like(lp_t))
        idx = intent[:, None, None].expand(-1, 1, N_MIX)
        mix = out["mix"].gather(1, idx).squeeze(1)
        mu = out["mu"].gather(1, idx).squeeze(1)
        ls = out["log_std"].gather(1, idx).squeeze(1)
        x = log_delay[:, None]
        comp = -0.5 * ((x - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)
        lp_d = torch.logsumexp(F.log_softmax(mix, -1) + comp, -1)
        return lp_i + lp_t + lp_d, (lp_i, lp_t, lp_d)

    @staticmethod
    def entropy_proxy(out):
        pi = F.softmax(out["intent"], -1)
        return -(pi * F.log_softmax(out["intent"], -1)).sum(-1)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, greedy: bool = False):
        out = self(obs)
        intent = torch.distributions.Categorical(logits=out["intent"]).sample()
        target = torch.distributions.Categorical(logits=out["target"]).sample()
        idx = intent[:, None, None].expand(-1, 1, N_MIX)
        mix = out["mix"].gather(1, idx).squeeze(1)
        mu = out["mu"].gather(1, idx).squeeze(1)
        ls = out["log_std"].gather(1, idx).squeeze(1)
        k = torch.distributions.Categorical(logits=mix).sample()
        m = mu.gather(1, k[:, None]).squeeze(1)
        s = ls.gather(1, k[:, None]).squeeze(1).exp()
        log_delay = (m + s * torch.randn_like(m)).clamp(LOG_MIN, LOG_MAX)
        lp, _ = self.log_prob(out, intent, target, log_delay)
        return intent, target, log_delay, lp, out["value"]


def to_actions(intent, target, log_delay):
    return [
        {"intent": int(i), "target": int(t), "delay_ms": float(np.exp(d))}
        for i, t, d in zip(intent.tolist(), target.tolist(), log_delay.tolist())
    ]
