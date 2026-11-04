"""Policy network with the plan's three heads.

* intent head: categorical over ``INTENTS``
* target head: a pointer over the visible-entity slots (scores each entity
  against the global context), so the action space does not change with the scene
* timing head: per-intent mixture of log-normals over the delay in ms. The mixture
  is what lets one policy produce quick routine clicks, slower reactions after a
  change and the long right tail of attention lapses.

By default the network sees the current decision point only, plus a short history in the
observation (last three delays, actions per minute, time since break). Options, all off by
default so old checkpoints load and behave as before (``PolicyConfig``):

* ``n_mix``: more mixture components in the timing head.
* ``wide_tail``: let each component be wider (log-sd up to e^1.5 instead of e^0.5), for the
  long attention-lapse tail of Jason's reaction times.
* ``situated_delay``: feed the trigger event, focus flag and time-since-last-input straight into
  the timing head as well as through the trunk, so timing can differ sharply by situation
  (crab burrowed, gems landed, coming back to an unfocused window).
* ``recurrent``: a GRU cell after the trunk carries state from one decision to the next within
  an episode, so the policy can have streaks (focused spells, distracted spells) that a
  memoryless policy can only fake from the observation's timers.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import ENT_FEATS, EVENTS, GLOBAL_DIM, MAX_ENTITIES, MAX_DELAY_MS, MIN_DELAY_MS, N_INTENTS, TARGETED
from .sim.base import N_SELF_EXTRA

N_MIX = 3  # the default; see PolicyConfig.n_mix
LOG_MIN, LOG_MAX = math.log(MIN_DELAY_MS), math.log(MAX_DELAY_MS)

# Slices of the global observation the situated timing head reads (see OsrsEnv._obs).
_EV0 = 4 + N_SELF_EXTRA
_CTX0 = _EV0 + len(EVENTS)
_SITUATION = list(range(_EV0, _CTX0)) + [_CTX0 + 6, _CTX0 + 7, _CTX0 + 8]  # trigger, since-input, focus


@dataclass
class PolicyConfig:
    n_mix: int = N_MIX
    wide_tail: bool = False
    situated_delay: bool = False
    recurrent: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


class Policy(nn.Module):
    def __init__(self, hidden: int = 256, ent_hidden: int = 64, config: PolicyConfig | dict | None = None):
        super().__init__()
        if isinstance(config, dict):
            config = PolicyConfig(**config)
        self.config = config or PolicyConfig()
        c = self.config
        self.hidden = hidden
        self.ent = nn.Sequential(nn.Linear(ENT_FEATS, ent_hidden), nn.Tanh(), nn.Linear(ent_hidden, ent_hidden), nn.Tanh())
        self.glob = nn.Sequential(
            nn.Linear(GLOBAL_DIM + ent_hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        if c.recurrent:
            self.rnn = nn.GRUCell(hidden, hidden)
        self.intent = nn.Linear(hidden, N_INTENTS)
        self.query = nn.Linear(hidden, ent_hidden)
        delay_in = hidden + (len(_SITUATION) if c.situated_delay else 0)
        self.delay = nn.Linear(delay_in, N_INTENTS * c.n_mix * 3)
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    @property
    def recurrent(self) -> bool:
        return self.config.recurrent

    def initial_state(self, batch: int) -> torch.Tensor | None:
        return torch.zeros(batch, self.hidden) if self.recurrent else None

    def forward(self, obs: torch.Tensor, hx: torch.Tensor | None = None):
        g = obs[:, :GLOBAL_DIM]
        E = obs[:, GLOBAL_DIM:].view(-1, MAX_ENTITIES, ENT_FEATS)
        mask = E[:, :, -1] > 0.5
        e = self.ent(E)
        pooled = (e * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        h = self.glob(torch.cat([g, pooled], -1))
        if self.recurrent:
            if hx is None:
                hx = torch.zeros(obs.shape[0], self.hidden)
            h = self.rnn(h, hx)
        intent_logits = self.intent(h)
        tgt = torch.einsum("bd,bkd->bk", self.query(h), e) / math.sqrt(e.shape[-1])
        tgt = tgt.masked_fill(~mask, -1e9)
        dh = torch.cat([h, g[:, _SITUATION]], -1) if self.config.situated_delay else h
        dp = self.delay(dh).view(-1, N_INTENTS, self.config.n_mix, 3)
        mix_logits = dp[..., 0]
        mu = 7.0 + 2.0 * dp[..., 1]
        if self.config.wide_tail:
            log_std = dp[..., 2].clamp(-3.0, 1.5)
        else:
            log_std = dp[..., 2].clamp(-3.0, 1.0) - 0.5
        return {"intent": intent_logits, "target": tgt, "mix": mix_logits, "mu": mu, "log_std": log_std,
                "value": self.value(h).squeeze(-1), "hx": h if self.recurrent else None}

    def forward_seq(self, obs: torch.Tensor, hx0: torch.Tensor | None, starts: torch.Tensor):
        """Run a [T, B] block of decisions. ``starts[t, b]`` resets the state before step t (a new
        episode). Returns the per-step outputs concatenated in [T*B] order (time-major)."""
        T, B = starts.shape
        if not self.recurrent:
            return self(obs.reshape(T * B, -1))
        hx = hx0 if hx0 is not None else torch.zeros(B, self.hidden)
        outs = []
        for t in range(T):
            hx = hx * (1.0 - starts[t].float())[:, None]
            out = self(obs[t], hx)
            hx = out["hx"]
            outs.append(out)
        return {k: (torch.cat([o[k] for o in outs]) if outs[0][k] is not None else None) for k in outs[0]}

    # ------------------------------------------------------------------
    @staticmethod
    def _delay_params(out, intent):
        n_mix = out["mix"].shape[-1]
        idx = intent[:, None, None].expand(-1, 1, n_mix)
        mix = out["mix"].gather(1, idx).squeeze(1)
        mu = out["mu"].gather(1, idx).squeeze(1)
        ls = out["log_std"].gather(1, idx).squeeze(1)
        return mix, mu, ls

    @staticmethod
    def log_prob(out, intent, target, log_delay):
        lp_i = F.log_softmax(out["intent"], -1).gather(1, intent[:, None]).squeeze(1)
        lp_t = F.log_softmax(out["target"], -1).gather(1, target[:, None]).squeeze(1)
        targeted = torch.zeros_like(intent, dtype=torch.bool)
        for k in TARGETED:
            targeted |= intent == k
        lp_t = torch.where(targeted, lp_t, torch.zeros_like(lp_t))
        mix, mu, ls = Policy._delay_params(out, intent)
        x = log_delay[:, None]
        comp = -0.5 * ((x - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)
        lp_d = torch.logsumexp(F.log_softmax(mix, -1) + comp, -1)
        return lp_i + lp_t + lp_d, (lp_i, lp_t, lp_d)

    @staticmethod
    def delay_moments(out, intent):
        """Mean and variance of log-delay under the timing mixture for the chosen intent (analytic,
        so a distribution-matching loss can be differentiated)."""
        mix, mu, ls = Policy._delay_params(out, intent)
        w = F.softmax(mix, -1)
        m = (w * mu).sum(-1)
        v = (w * (ls.exp() ** 2 + mu**2)).sum(-1) - m**2
        return m, v.clamp(min=1e-6)

    @staticmethod
    def entropy_proxy(out):
        pi = F.softmax(out["intent"], -1)
        return -(pi * F.log_softmax(out["intent"], -1)).sum(-1)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, greedy: bool = False, hx: torch.Tensor | None = None):
        """Returns (intent, target, log_delay, log_prob, value); with ``hx`` given (recurrent
        policies) also the next state as a sixth element."""
        out = self(obs, hx)
        intent = torch.distributions.Categorical(logits=out["intent"]).sample()
        target = torch.distributions.Categorical(logits=out["target"]).sample()
        mix, mu, ls = Policy._delay_params(out, intent)
        k = torch.distributions.Categorical(logits=mix).sample()
        m = mu.gather(1, k[:, None]).squeeze(1)
        s = ls.gather(1, k[:, None]).squeeze(1).exp()
        log_delay = (m + s * torch.randn_like(m)).clamp(LOG_MIN, LOG_MAX)
        lp, _ = self.log_prob(out, intent, target, log_delay)
        if hx is not None:
            return intent, target, log_delay, lp, out["value"], out["hx"]
        return intent, target, log_delay, lp, out["value"]


def timing_targets(transitions) -> dict[int, tuple[float, float]]:
    """Per-intent (mean, sd) of log-delay in recorded play, for ``timing_match_loss``."""
    by = {}
    for _, a in transitions:
        by.setdefault(int(a["intent"]), []).append(math.log(min(max(a["delay_ms"], 120.0), 300_000.0)))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in by.items() if len(v) >= 20}


def timing_match_loss(out, intent, targets: dict[int, tuple[float, float]]) -> torch.Tensor:
    """Moment matching on log-delay, per intent: squared gap between the batch's mixture mean and
    sd and the recorded ones, weighted by how often each intent occurs in the batch."""
    m, v = Policy.delay_moments(out, intent)
    loss = torch.zeros(())
    n = len(intent)
    for k, (mt, st) in targets.items():
        sel = intent == k
        cnt = int(sel.sum())
        if cnt < 8:
            continue
        mk = m[sel].mean()
        sk = (v[sel].mean() + m[sel].var(unbiased=False)).sqrt()
        loss = loss + (cnt / n) * ((mk - mt) ** 2 + (sk - st) ** 2)
    return loss


def policy_from_checkpoint(ckpt: dict, which: str) -> Policy:
    pol = Policy(config=ckpt.get(f"config_{which}", ckpt.get("config")))
    pol.load_state_dict(ckpt[which])
    return pol


def to_actions(intent, target, log_delay):
    return [
        {"intent": int(i), "target": int(t), "delay_ms": float(np.exp(d))}
        for i, t, d in zip(intent.tolist(), target.tolist(), log_delay.tolist())
    ]
