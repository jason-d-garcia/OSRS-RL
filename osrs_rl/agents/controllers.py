"""Hand-written controllers that read the simulator's privileged state.

* ``ScriptedBot`` is the classic bot: always the optimal next step, with a
  uniformly random sleep between 600 and 1800 ms. A detector should catch it.
* ``SyntheticHuman`` is a *stand-in for human recordings*. Each instance is one
  made-up player with their own reaction speed, attention lapses, fatigue, camera
  and tab-checking habits, redundant clicks and imperfect target choice. It lets
  the whole pipeline (behaviour cloning, detector, RL) run before any real
  recordings exist. Every humanness number measured against it measures
  closeness to this model, not to real people. Replace it with Jason's RuneLite
  recordings as soon as they exist.
"""

from __future__ import annotations

import math

import numpy as np

from ..env import CAMERA, CHECK_TAB, CLOSE, DEPOSIT, INTERACT, WAIT
from ..sim import params as P
from ..sim.base import cheb

CHANGE_EVENTS = {"start", "inv_full", "target_gone", "bank_open", "shell_ready", "arrived", "invalid"}


def act(intent, target=0, delay=800.0):
    return {"intent": intent, "target": max(int(target), 0), "delay_ms": float(delay)}


class Controller:
    def reset(self, env) -> None:
        pass

    def __call__(self, env) -> dict:
        raise NotImplementedError


# ----------------------------------------------------------------------
class ScriptedBot(Controller):
    def __init__(self, seed: int = 0, lo: float = 600.0, hi: float = 1800.0):
        self.rng = np.random.default_rng(seed)
        self.lo, self.hi = lo, hi

    def delay(self):
        return self.rng.uniform(self.lo, self.hi)

    def __call__(self, env):
        sim = env.sim
        d = self.delay()
        if env.activity_name == "willows":
            if sim.logs >= P.INVENTORY_SLOTS:
                if sim.bank_open:
                    return act(DEPOSIT, 0, d)
                return act(INTERACT, env.slot_of(sim.nearest("bank")), d)
            if sim.busy or sim.moving:
                return act(WAIT, 0, d)
            trees = sim.trees_by_distance()
            if trees and env.slot_of(trees[0]) >= 0:
                return act(INTERACT, env.slot_of(trees[0]), d)
            return act(WAIT, 0, d)
        # gem crab
        if sim.transit_until > sim.tick_count or sim.moving or sim.busy:
            return act(WAIT, 0, d)
        if sim.crab is not None:
            return act(INTERACT, env.slot_of(sim.crab), d)
        if sim.shell is not None and sim.eligible and not sim.mined_this_shell:
            return act(INTERACT, env.slot_of(sim.shell), d)
        return act(INTERACT, env.slot_of(sim.cave), d)


# ----------------------------------------------------------------------
class SyntheticHuman(Controller):
    """One made-up player. ``profile`` fixes their habits across sessions."""

    def __init__(self, player_id: int, session_seed: int = 0):
        prof = np.random.default_rng(10_000 + player_id)
        self.player_id = player_id
        self.mu = math.log(prof.uniform(450, 1000))  # median reaction, ms
        self.sigma = prof.uniform(0.35, 0.6)
        self.change_mult = prof.uniform(1.3, 2.0)  # slower after something changes
        self.repeat_mult = prof.uniform(0.75, 0.95)  # faster on a repeated action
        self.rho = prof.uniform(0.1, 0.45)  # autocorrelation of log-delays
        self.fatigue = prof.uniform(0.0, 0.35)  # slow-down by the end of 30 min
        self.p_afk = prof.uniform(0.02, 0.08)
        self.afk_scale = prof.uniform(3000, 9000)
        self.glance_mu = math.log(prof.uniform(4000, 15000))  # time between glances while busy
        self.crab_glance_mu = math.log(prof.uniform(20000, 90000))
        self.p_camera = prof.uniform(0.02, 0.09)
        self.p_tab = prof.uniform(0.01, 0.06)
        self.p_redundant = prof.uniform(0.05, 0.3)
        self.tree_temp = prof.uniform(1.0, 5.0)
        self.p_close_bank = prof.uniform(0.1, 0.8)
        self.p_misclick = prof.uniform(0.005, 0.03)
        self.break_rate = prof.uniform(1 / 3600, 1 / 900)  # breaks per second of play
        # Gem crab habits, set from Jason's recordings (calibrate.py, 9.4 h): after the crab
        # burrows or the target is lost, the next input comes a median 26-37 s later (p90 ~3-5 min),
        # because the game window is unfocused 61% of the time; arriving at a new mine is answered
        # in ~1.4 s; ~12 tab clicks an hour; and the shell is almost never mined.
        self.p_crab_afk = prof.uniform(0.4, 0.8)
        self.crab_afk_mu = math.log(prof.uniform(8_000, 35_000))
        self.p_crab_tab = prof.uniform(0.03, 0.3)
        self.p_burst = prof.uniform(0.5, 0.8)  # REAL: tab clicks come in quick runs (median gap 2.8 s)
        self.burst_mu = math.log(prof.uniform(700, 2000))
        self.p_try_shell = prof.uniform(0.0, 0.15) if prof.random() < 0.4 else prof.uniform(0.5, 0.98)
        self.session_seed = session_seed
        self.rng = np.random.default_rng(hash((player_id, session_seed)) % (2**32))

    def reset(self, env):
        self.burst_left = 0
        self.prev_z = 0.0
        self.prev_intent = -1
        self.prev_uid = -1
        self.pending = []

    # --- timing ---------------------------------------------------------
    def reaction(self, env, intent, uid=-1) -> float:
        z = self.rho * self.prev_z + math.sqrt(1 - self.rho**2) * self.rng.standard_normal()
        self.prev_z = z
        d = math.exp(self.mu + self.sigma * z)
        if env.trigger in CHANGE_EVENTS:
            d *= self.change_mult
        if intent == self.prev_intent and uid == self.prev_uid:
            d *= self.repeat_mult
        d *= 1.0 + self.fatigue * env.t_ms / 1_800_000.0
        if self.rng.random() < self.p_afk:
            d += self.afk_scale * (self.rng.pareto(1.6) + 1.0)
        if (env.activity_name == "gemcrab" and env.trigger in ("target_gone", "shell_ready")
                and self.rng.random() < self.p_crab_afk):
            d += math.exp(self.crab_afk_mu + 1.1 * self.rng.standard_normal())  # tabbed out
        return d

    def glance(self, env, crab=False) -> float:
        mu = self.crab_glance_mu if crab else self.glance_mu
        d = math.exp(mu + 0.6 * self.rng.standard_normal())
        return d * (1.0 + self.fatigue * env.t_ms / 1_800_000.0)

    def _act(self, env, intent, ent=None):
        uid = ent.uid if ent is not None else -1
        slot = env.slot_of(ent) if ent is not None else 0
        a = act(intent, slot, self.reaction(env, intent, uid))
        self.prev_intent, self.prev_uid = intent, uid
        return a

    # --- policy -----------------------------------------------------------
    def __call__(self, env):
        sim = env.sim
        # a break: a long wait, then carry on
        if self.rng.random() < self.break_rate * 20:
            return act(WAIT, 0, self.rng.uniform(60_000, 240_000))
        if self.rng.random() < self.p_camera:
            return self._act(env, CAMERA)
        if self.rng.random() < self.p_tab:
            return self._act(env, CHECK_TAB)
        if env.activity_name == "willows":
            return self._willows(env, sim)
        return self._crab(env, sim)

    def _misclick(self, env):
        vis = env.visible()
        if vis and self.rng.random() < self.p_misclick:
            return vis[int(self.rng.integers(len(vis)))]
        return None

    def _willows(self, env, sim):
        if sim.logs >= P.INVENTORY_SLOTS:
            if sim.bank_open:
                return self._act(env, DEPOSIT)
            if sim.moving:
                return act(WAIT, 0, self.glance(env) * 0.3)
            booths = sorted([e for e in env.visible() if e.kind == "bank"], key=lambda e: cheb(e.x, e.y, sim.px, sim.py))
            if not booths:
                return act(WAIT, 0, 1000)
            pick = booths[min(int(self.rng.geometric(0.6)) - 1, len(booths) - 1)]
            return self._act(env, INTERACT, self._misclick(env) or pick)
        if sim.bank_open and sim.logs == 0 and self.rng.random() < self.p_close_bank and self.prev_intent != CLOSE:
            return self._act(env, CLOSE)
        if sim.busy:
            if self.rng.random() < self.p_redundant:
                return self._act(env, INTERACT, sim.chop_tree)
            return act(WAIT, 0, self.glance(env))
        if sim.moving:
            return act(WAIT, 0, self.glance(env) * 0.3)
        trees = [e for e in env.visible() if e.kind == "tree"]
        if not trees:
            return act(WAIT, 0, self.reaction(env, WAIT))
        d = np.array([cheb(t.x, t.y, sim.px, sim.py) for t in trees], dtype=float)
        p = np.exp(-(d - d.min()) / self.tree_temp)
        p /= p.sum()
        pick = trees[int(self.rng.choice(len(trees), p=p))]
        return self._act(env, INTERACT, self._misclick(env) or pick)

    def _crab(self, env, sim):
        if sim.transit_until > sim.tick_count:
            return act(WAIT, 0, self.reaction(env, WAIT))
        if sim.attacking:
            if self.rng.random() < self.p_redundant * 0.3:
                return self._act(env, INTERACT, sim.crab)
            if self.burst_left > 0:
                self.burst_left -= 1
                return act(CHECK_TAB, 0, math.exp(self.burst_mu + 0.5 * self.rng.standard_normal()))
            if self.rng.random() < self.p_crab_tab:
                self.burst_left = int(self.rng.geometric(1 - self.p_burst)) - 1
                return self._act(env, CHECK_TAB)
            return act(WAIT, 0, self.glance(env, crab=True))
        if sim.busy or sim.moving:
            return act(WAIT, 0, self.glance(env) * 0.3)
        if sim.crab is not None:
            return self._act(env, INTERACT, sim.crab)
        if sim.shell is not None and not sim.mined_this_shell and env.trigger != "invalid" and self.rng.random() < self.p_try_shell:
            return self._act(env, INTERACT, sim.shell)
        return self._act(env, INTERACT, sim.cave)


def run_episode(env, controller, seed=None, record=False):
    """Roll out one session. Returns (log, progress, transitions)."""
    obs, _ = env.reset(seed=seed)
    controller.reset(env)
    transitions = []
    done = False
    while not done:
        a = controller(env)
        if record:
            transitions.append((obs, a))
        obs, r, done, _, info = env.step(a)
    return {"log": env.log, "progress": env.sim.progress, "ms": env.t_ms, "activity": env.activity_name}, transitions
