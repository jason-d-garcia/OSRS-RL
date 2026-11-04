"""Gymnasium environment over the tick simulator.

Decision points, not ticks
--------------------------
A human does not decide something every 600 ms tick. They click, watch, and
click again when something changes or when they get bored. The env therefore
hands control to the policy at *decision points*: when an action finishes or
fails, when something happens (inventory full, tree gone, crab burrowed), when
the player has been busy for a while (a chance to do something else, such as a
redundant click or a camera move), or when a chosen wait ends.

At each decision point the policy picks
  * an intent (wait, interact, deposit all, close interface, camera, check tab, walk),
  * a target, as a pointer into the visible-entity list (used by interact and walk),
  * a delay in milliseconds before the intent is carried out.

There is no fixed clock: after an input the policy immediately picks the delay to
its next one, so decision points only come from the policy's own delays and from
the world's events. (An earlier version woke the policy every 15 s during a fight,
and that cadence showed up in the clone's gap distribution.)

The world keeps ticking during the delay. Waits are invisible to the server and
to the detector; every other intent is an observable input event and is logged
with its timestamp so sessions can be scored for humanness.

A delay is "do this unless something happens": if the world raises an event that
needs a response (the crab burrows, a shell appears, you arrive at a mine, the
inventory fills, the bank opens) before the delay is up, the pending intent is
dropped and the policy decides again with that event as the trigger. So a long
delay means "nothing until something changes", which is what a person tabbed
out is doing, and the recorded human waits can be labelled the same way
(``real_bc.py``) without cutting them into fixed chunks.

The server-side agent plugin for RSMod is meant to speak this same
observation/action schema, so the trainer will not change when the simulator is
swapped for the real server.
"""

from __future__ import annotations

import math
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .sim.base import EVENT_INDEX, EVENTS, KIND_INDEX, KINDS, N_SELF_EXTRA, TICK_MS, cheb
from .sim.gemcrab import GemCrab
from .sim.willows import Willows

INTENTS = ("wait", "interact", "deposit_all", "close_interface", "camera", "check_tab", "walk")
WAIT, INTERACT, DEPOSIT, CLOSE, CAMERA, CHECK_TAB, WALK = range(len(INTENTS))
N_INTENTS = len(INTENTS)
TARGETED = (INTERACT, WALK)  # intents that use the target pointer

# Events that cut a pending delay short (see the module docstring). "idle" is the end of a walk
# or of mining the shell: the player is standing still with nothing in progress.
INTERRUPTS = {"inv_full", "target_gone", "bank_open", "shell_ready", "arrived", "idle"}

MAX_ENTITIES = 12
ENT_FEATS = len(KINDS) + 6
# Timing context: time since break, inputs in the last minute, last three delays, time since
# a camera move, time since the last input (linear and log), whether the game window is
# focused, and the last input's intent.
N_CTX = 9 + N_INTENTS
GLOBAL_DIM = 4 + N_SELF_EXTRA + len(EVENTS) + N_CTX
OBS_DIM = GLOBAL_DIM + MAX_ENTITIES * ENT_FEATS

MIN_DELAY_MS = 120.0
MAX_DELAY_MS = 300_000.0
BUSY_CHECK_TICKS = 25  # 15 s
EPISODE_TICKS = 3000  # 30 minutes of game time

ACTIVITIES = {"willows": Willows, "gemcrab": GemCrab}

# Attention. Jason's game window is unfocused about two thirds of the time and he never clicks
# while it is: he clicks, watches for a while, then switches to something else until a
# notification or a glance brings him back. The env models that as a focus flag the policy can
# see: an input focuses the window, and it stays focused for a log-normal time after the last
# input before going unfocused. run_real.py sets the median and spread from the recordings.
FOCUS_MODEL = {"median_ms": 20_000.0, "sigma": 1.0}


def set_focus_model(median_ms: float, sigma: float) -> None:
    FOCUS_MODEL["median_ms"], FOCUS_MODEL["sigma"] = float(median_ms), float(sigma)


class OsrsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, activity: str = "gemcrab", seed: int | None = None, episode_ticks: int = EPISODE_TICKS,
                 reward_cfg: dict | None = None, randomize: bool = True):
        self.activity_name = activity
        self.rng = np.random.default_rng(seed)
        self.sim = ACTIVITIES[activity](self.rng, randomize=randomize)
        self.episode_ticks = episode_ticks
        self.reward_cfg = reward_cfg or {}
        self.observation_space = spaces.Box(-5, 5, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Dict(
            {
                "intent": spaces.Discrete(N_INTENTS),
                "target": spaces.Discrete(MAX_ENTITIES),
                "delay_ms": spaces.Box(MIN_DELAY_MS, MAX_DELAY_MS, shape=(), dtype=np.float32),
            }
        )

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.sim.rng = self.rng
        self.sim.reset()
        self.t_ms = 0.0
        self.trigger = "start"
        self.last_event_ms = 0.0
        self.last_event = "start"
        self.last_obs_action_ms = 0.0
        self.last_camera_ms = 0.0
        self.last_break_ms = 0.0
        self.recent_actions: deque = deque()
        self.last_delays = deque([800.0, 800.0, 800.0], maxlen=3)
        self.last_intent = -1
        self.focused = True
        self._schedule_unfocus(0.0)
        self.log: list[dict] = []
        self.events_log: list[tuple[float, str]] = []
        self.cum_progress = 0.0
        self.capped_progress = 0.0
        self.n_decisions = 0
        self.ticks_since_camera_penalty = 0.0
        self._visible = []
        return self._obs(), {}

    # ------------------------------------------------------------------
    def _advance_to(self, t_ms: float, interruptible: bool = False):
        """Tick the world until the clock reaches t_ms.

        Returns ``(done, interrupt)``: ``done`` at episode end, ``interrupt`` the event that cut
        the advance short (the clock then stands at that tick), or None.
        """
        target_tick = int(t_ms // TICK_MS)
        while self.sim.tick_count < target_tick:
            if self.sim.tick_count >= self.episode_ticks:
                return True, None
            evs = self.sim.tick()
            if self.focused and self.sim.tick_count * TICK_MS >= self.unfocus_at:
                self.focused = False
            for e in evs:
                self._note_event(e)
            hit = [e for e in evs if e in INTERRUPTS] if interruptible else []
            if hit:
                return self.sim.tick_count >= self.episode_ticks, hit[-1]
        return self.sim.tick_count >= self.episode_ticks, None

    def _schedule_unfocus(self, t_ms: float) -> None:
        m = FOCUS_MODEL
        self.unfocus_at = t_ms + m["median_ms"] * math.exp(m["sigma"] * self.rng.standard_normal())

    def _note_event(self, e: str) -> None:
        self.last_event = e
        self.last_event_ms = self.sim.tick_count * TICK_MS
        self.events_log.append((self.last_event_ms, e))

    def step(self, action):
        intent = int(action["intent"])
        target = int(action["target"])
        delay = float(np.clip(action["delay_ms"], MIN_DELAY_MS, MAX_DELAY_MS))
        self.n_decisions += 1
        progress_before = self.sim.progress
        decision_trigger = self.trigger
        planned = self.t_ms + delay
        done, interrupt = self._advance_to(planned, interruptible=True)
        if interrupt is not None:
            self.t_ms = self.sim.tick_count * TICK_MS  # the world moved on; decide again
        else:
            self.t_ms = planned
        self.last_delays.append(self.t_ms - (planned - delay))
        shaping = 0.0
        if interrupt is not None:
            self.trigger = interrupt
        elif not done:
            if intent == WAIT:
                self.trigger = "wait_done"
            else:
                valid, ent = self._execute(intent, target)
                shaping += self._log_action(intent, ent, valid, decision_trigger)
                if not valid:
                    self.trigger = "invalid"
                elif intent in TARGETED:
                    # The click is in. Decide the next input straight away; events interrupt it.
                    self.trigger = "busy_check"
                else:
                    self.trigger = "idle" if self.sim.idle else "busy_check"
        reward, info = self._reward(progress_before, shaping)
        if done:
            info["episode_progress"] = self.sim.progress
            info["episode_ms"] = self.t_ms
        return self._obs(), reward, done, False, info

    def _execute(self, intent: int, target: int):
        sim = self.sim
        ent = None
        if intent == INTERACT:
            if target < len(self._visible):
                ent = self._visible[target]
                return sim.interact(ent), ent
            return False, None
        if intent == WALK:
            # Walk towards something without interacting with it ("Walk here" in the client).
            if target < len(self._visible):
                return sim.walk_near(self._visible[target]), None
            return False, None
        if intent == DEPOSIT:
            return sim.deposit_all(), None
        if intent == CLOSE:
            return sim.close_interface(), None
        if intent == CAMERA:
            self.last_camera_ms = self.t_ms
            return True, None
        if intent == CHECK_TAB:
            return True, None
        return False, None

    def _log_action(self, intent, ent, valid, decision_trigger) -> float:
        t = self.t_ms
        gap = t - self.last_obs_action_ms
        if gap > 60_000:
            self.last_break_ms = t
        rec = {
            "t_ms": t,
            "intent": intent,
            "kind": ent.kind if ent is not None else "",
            "uid": ent.uid if ent is not None else -1,
            "valid": bool(valid),
            "trigger": decision_trigger,
            "since_event_ms": t - self.last_event_ms,
            "last_event": self.last_event,
            "gap_ms": gap,
            "redundant": bool(ent is not None and self.sim.busy and valid and self._is_current_target(ent)),
            "rank": self._distance_rank(ent),
        }
        self.log.append(rec)
        self.last_obs_action_ms = t
        self.last_intent = intent
        self.focused = True  # nobody clicks into an unfocused window
        self._schedule_unfocus(t)
        self.recent_actions.append(t)
        # humanness shaping: reacting faster than the human floor for this context
        pen = 0.0
        cfg = self.reward_cfg
        floor = cfg.get("reaction_floor_ms")
        if floor is not None and rec["since_event_ms"] < floor and rec["last_event"] not in ("busy_check", "idle"):
            pen -= cfg.get("shaping_weight", 0.0)
        return pen

    def _is_current_target(self, ent) -> bool:
        sim = self.sim
        return getattr(sim, "chop_tree", None) is ent or (getattr(sim, "attacking", False) and ent.kind == "crab")

    def _distance_rank(self, ent) -> int:
        if ent is None:
            return -1
        same = [e for e in self.sim.entities_list if e.kind == ent.kind and e.interactable]
        d = cheb(ent.x, ent.y, self.sim.px, self.sim.py)
        return sum(1 for e in same if cheb(e.x, e.y, self.sim.px, self.sim.py) < d)

    # ------------------------------------------------------------------
    def _reward(self, progress_before: float, shaping: float):
        cfg = self.reward_cfg
        delta = self.sim.progress - progress_before
        self.cum_progress += delta
        median_per_ms = cfg.get("human_median_rate_per_hour", 1.0) / 3_600_000.0
        cap_rate = cfg.get("cap_rate_per_hour")
        new_capped = self.cum_progress
        if cap_rate is not None:
            new_capped = min(self.cum_progress, cap_rate / 3_600_000.0 * max(self.t_ms, 1.0))
        r = (new_capped - self.capped_progress) / (median_per_ms * 3_600_000.0)
        self.capped_progress = new_capped
        # camera: a person moves the camera now and then; penalise 10 min without
        cam_w = cfg.get("camera_weight", 0.0)
        if cam_w and self.t_ms - self.last_camera_ms > 600_000:
            r -= cam_w
            self.last_camera_ms = self.t_ms
        r *= cfg.get("reward_scale", 1.0)
        return r + shaping, {"progress_delta": delta}

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        sim = self.sim
        g = np.zeros(GLOBAL_DIM, dtype=np.float32)
        g[0] = sim.px / 20.0
        g[1] = sim.py / 20.0
        g[2] = sim.tick_count / self.episode_ticks
        g[3] = float(sim.idle)
        g[4 : 4 + N_SELF_EXTRA] = sim.self_features()
        o = 4 + N_SELF_EXTRA
        g[o + EVENT_INDEX[self.trigger]] = 1.0
        o += len(EVENTS)
        while self.recent_actions and self.t_ms - self.recent_actions[0] > 60_000:
            self.recent_actions.popleft()
        g[o + 0] = min((self.t_ms - self.last_break_ms) / 600_000.0, 3.0)
        g[o + 1] = len(self.recent_actions) / 20.0
        for i, d in enumerate(self.last_delays):
            g[o + 2 + i] = (math.log(d) - 7.0) / 2.0
        g[o + 5] = min((self.t_ms - self.last_camera_ms) / 600_000.0, 3.0)
        g[o + 6] = min((self.t_ms - self.last_obs_action_ms) / 60_000.0, 5.0)
        g[o + 7] = (math.log(max(self.t_ms - self.last_obs_action_ms, 1.0)) - 7.0) / 2.0
        g[o + 8] = float(self.focused)
        if self.last_intent >= 0:
            g[o + 9 + self.last_intent] = 1.0

        # Things you can click first, then bystanders, each nearest first. Large NPCs make raw tile
        # distance a poor guide, and at a busy crab other players would otherwise crowd it out.
        ents = sorted(sim.visible_entities(),
                      key=lambda e: (not e.interactable, cheb(e.x, e.y, sim.px, sim.py), e.uid))[:MAX_ENTITIES]
        self._visible = ents
        E = np.zeros((MAX_ENTITIES, ENT_FEATS), dtype=np.float32)
        for i, e in enumerate(ents):
            E[i, KIND_INDEX[e.kind]] = 1.0
            b = len(KINDS)
            E[i, b + 0] = (e.x - sim.px) / 15.0
            E[i, b + 1] = (e.y - sim.py) / 15.0
            E[i, b + 2] = cheb(e.x, e.y, sim.px, sim.py) / 15.0
            E[i, b + 3] = float(e.interactable)
            E[i, b + 4] = float(self._is_current_target(e))
            E[i, b + 5] = 1.0  # valid slot
        return np.concatenate([g, E.ravel()])

    # helpers for controllers -------------------------------------------
    def visible(self):
        return self._visible

    def slot_of(self, ent) -> int:
        for i, e in enumerate(self._visible):
            if e is ent:
                return i
        return -1
