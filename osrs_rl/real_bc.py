"""Behaviour-cloning data from real recordings: (observation, action) pairs in the simulator's format.

The policy sees the world the way ``OsrsEnv._obs`` builds it and acts at the env's decision
points. This module replays a gem crab recording through the same decision rules:

* a decision happens at the start, after every input, when the crab burrows or a shell appears
  (``target_gone`` / ``shell_ready``), on arriving at a mine after a cave hop (``arrived``), and
  when a walk ends (``idle``);
* if Jason's next input comes before the next such event (and within ``MAX_DELAY_MS``), the label
  is that input with ``delay_ms`` = the time from the decision to the input;
* if an event comes first, the label is WAIT until the event: the env cuts a pending delay short
  at the same events, so "wait, then decide again at the burrow" is exactly what the clone does;
* only a stretch longer than ``MAX_DELAY_MS`` with nothing happening is cut, into WAITs of that
  length followed by a ``wait_done`` decision.

The observation is rebuilt from the tick snapshot at the decision: the player's tile relative
to the nearest mine's crab tile (the simulator's local frame), the crab, its shell, the cave and
up to six other players as entities, the crab's HP bar, and the same timing context the env
keeps (last three delays, inputs in the last minute, time since a camera move, a break, the last
input, and the last input's intent). "Walk here" clicks become WALK towards the nearest clickable
thing, the simulator's stand-in for repositioning.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .env import (CAMERA, CHECK_TAB, ENT_FEATS, GLOBAL_DIM, INTERACT, MAX_ENTITIES, MAX_DELAY_MS, MIN_DELAY_MS,
                  N_INTENTS, WAIT, WALK)
from .recordings import IGNORED_OPTIONS, in_crab_area, load
from .sim.base import EVENT_INDEX, EVENTS, KIND_INDEX, KINDS, N_SELF_EXTRA

TICK_MS = 600
EPISODE_MS = 3000 * TICK_MS
ATTACK_LVL = 80  # Jason's Attack level

# Crab tiles of the three mines (rsmod-content/gemstone-crab GemstoneCrabRules.MINES).
MINES = [(1273, 3173), (1353, 3112), (1239, 3043)]


def _name(n):
    import re
    return re.sub(r"<[^>]+>", "", n or "").lower()


def _nearest_mine(x, y):
    return min(range(len(MINES)), key=lambda i: max(abs(x - MINES[i][0]), abs(y - MINES[i][1])))


class _Ent:
    __slots__ = ("kind", "dx", "dy", "interactable", "target")

    def __init__(self, kind, dx, dy, interactable=True, target=False):
        self.kind, self.dx, self.dy, self.interactable, self.target = kind, dx, dy, interactable, target


def _entities(tick, cave_ids):
    inter = tick.get("interacting")
    attacking = isinstance(inter, dict) and "gemstone crab" in _name(inter.get("name")) and "shell" not in _name(inter.get("name"))
    ents = []
    for n in tick.get("npcs", []):
        nm = _name(n.get("name"))
        if "gemstone crab" in nm:
            kind = "shell" if "shell" in nm else "crab"
            ents.append(_Ent(kind, n["dx"], n["dy"], True, kind == "crab" and attacking))
    caves = [o for o in tick.get("objects", []) if o.get("id") in cave_ids]
    if caves:
        c = min(caves, key=lambda o: max(abs(o["dx"]), abs(o["dy"])))
        ents.append(_Ent("cave", c["dx"], c["dy"]))
    others = sorted(tick.get("players", []), key=lambda p: max(abs(p.get("dx", 0)), abs(p.get("dy", 0))))
    for p in others[:6]:  # the simulator shows at most six bystanders
        ents.append(_Ent("player", p.get("dx", 0), p.get("dy", 0), False))
    ents = [e for e in ents if max(abs(e.dx), abs(e.dy)) <= 20]
    # Same order as OsrsEnv._obs: clickable things first, then bystanders, nearest first.
    ents.sort(key=lambda e: (not e.interactable, max(abs(e.dx), abs(e.dy)), KIND_INDEX[e.kind]))
    return ents[:MAX_ENTITIES], attacking


def recording_transitions(path) -> list[tuple[np.ndarray, dict]]:
    recs = list(load(path))
    if not recs:
        return []
    cave_ids = {r.get("id") for r in recs if r["type"] == "menu_click" and (r.get("option") or "").lower() == "crawl-through"}
    t0 = recs[0]["t"]

    # Ticks in the crab area, events, and inputs, all in ms from the start.
    ticks, events, inputs = [], [], []
    prev, prev_xy = None, None
    for r in recs:
        t = r["t"] - t0
        if r["type"] == "tick":
            if not in_crab_area(r):
                prev, prev_xy = None, None
                continue
            ents, attacking = _entities(r, cave_ids)
            r["_ents"], r["_att"] = ents, attacking
            xy = (r["x"], r["y"])
            r["_moving"] = prev_xy is not None and xy != prev_xy
            if prev is not None:
                if prev["_att"] and not attacking:
                    events.append((t, "target_gone"))
                had_shell = any(e.kind == "shell" for e in prev["_ents"])
                if not had_shell and any(e.kind == "shell" for e in ents):
                    events.append((t, "shell_ready"))
                if max(abs(xy[0] - prev_xy[0]), abs(xy[1] - prev_xy[1])) > 10:
                    events.append((t, "arrived"))
                elif prev["_moving"] and not r["_moving"] and not attacking:
                    events.append((t, "idle"))  # a walk ended with nothing in progress
            ticks.append((t, r))
            prev, prev_xy = r, xy
            if r.get("camera_moved"):
                inputs.append((t, CAMERA, None))
        elif r["type"] == "menu_click":
            opt = (r.get("option") or "").strip().lower()
            tgt = _name(r.get("target"))
            if opt in IGNORED_OPTIONS:
                continue
            if opt == "walk here":
                inputs.append((t, WALK, None))
            elif not tgt:
                inputs.append((t, CHECK_TAB, None))
            elif "gemstone crab" in tgt or tgt == "cave" or opt == "crawl-through":
                kind = "shell" if "shell" in tgt else "crab" if "gemstone crab" in tgt else "cave"
                inputs.append((t, INTERACT, kind))
    if not ticks:
        return []
    inputs.sort(key=lambda x: x[0])
    tick_times = np.array([t for t, _ in ticks])

    def tick_at(t):
        i = int(np.searchsorted(tick_times, t, side="right")) - 1
        return ticks[max(i, 0)][1]

    # Decision replay.
    out = []
    t_d, trigger = float(ticks[0][0]), "start"
    last_delays = deque([800.0, 800.0, 800.0], maxlen=3)
    recent = deque()
    last_input = last_camera = last_break = t_d
    last_intent = -1
    ii = int(np.searchsorted([x[0] for x in inputs], t_d))
    ev_times = [e[0] for e in events]
    end_t = ticks[-1][0]
    session_start = t_d
    while t_d < end_t:
        tick = tick_at(t_d)
        if not in_crab_area(tick):
            break
        ents = tick["_ents"]
        obs = _obs(tick, ents, trigger, t_d - session_start, recent, last_delays, last_input, last_camera, last_break,
                   last_intent, t_d)
        nxt = inputs[ii] if ii < len(inputs) else None
        k = int(np.searchsorted(ev_times, t_d, side="right"))
        nxt_ev = events[k] if k < len(events) else None  # first event strictly after the decision
        if nxt is not None and nxt[0] - t_d <= MAX_DELAY_MS and (nxt_ev is None or nxt[0] <= nxt_ev[0]):
            t_a, intent, kind = nxt
            ii += 1
            slot = 0
            if intent == INTERACT:
                slots = [i for i, e in enumerate(ents) if e.kind == kind]
                if not slots:
                    t_d, trigger = t_a, "busy_check"  # clicked something outside the snapshot
                    continue
                slot = slots[0]
            elif intent == WALK and not ents:
                t_d, trigger = t_a, "busy_check"
                continue
            delay = float(np.clip(t_a - t_d, MIN_DELAY_MS, MAX_DELAY_MS))
            out.append((obs, {"intent": intent, "target": slot, "delay_ms": delay}))
            last_delays.append(delay)
            if t_a - last_input > 60_000:
                last_break = t_a
            last_input = t_a
            last_intent = intent
            recent.append(t_a)
            if intent == CAMERA:
                last_camera = t_a
            if intent in (INTERACT, WALK):
                t_d, trigger = t_a, "busy_check"  # the click is in; decide the next input now
            else:
                t_d, trigger = t_a, ("busy_check" if tick["_att"] else "idle")
        elif nxt_ev is not None and nxt_ev[0] - t_d <= MAX_DELAY_MS:
            # Nothing from Jason before the next event: wait until it, as the env would.
            delay = float(np.clip(nxt_ev[0] - t_d, MIN_DELAY_MS, MAX_DELAY_MS))
            out.append((obs, {"intent": WAIT, "target": 0, "delay_ms": delay}))
            last_delays.append(delay)
            t_d, trigger = float(nxt_ev[0]), nxt_ev[1]
        else:
            out.append((obs, {"intent": WAIT, "target": 0, "delay_ms": MAX_DELAY_MS}))
            last_delays.append(MAX_DELAY_MS)
            t_d, trigger = t_d + MAX_DELAY_MS, "wait_done"
        while recent and t_d - recent[0] > 60_000:
            recent.popleft()
    return out


def _obs(tick, ents, trigger, t_rel, recent, last_delays, last_input, last_camera, last_break, last_intent, t):
    mi = _nearest_mine(tick["x"], tick["y"])
    cx, cy = MINES[mi]
    px, py = tick["x"] - cx, tick["y"] - cy
    g = np.zeros(GLOBAL_DIM, dtype=np.float32)
    g[0], g[1] = px / 20.0, py / 20.0
    g[2] = (t_rel % EPISODE_MS) / EPISODE_MS  # position within a 30-minute episode, as in the sim
    g[3] = float(not tick["_att"] and not tick["_moving"])
    crab = next((e for e in ents if e.kind == "crab"), None)
    hp = 0.0
    if crab is not None:
        n = next((n for n in tick.get("npcs", []) if "gemstone crab" in _name(n.get("name"))
                  and "shell" not in _name(n.get("name"))), None)
        if n and n.get("hp_scale", 0) > 0 and n.get("hp_ratio", -1) >= 0:
            hp = n["hp_ratio"] / n["hp_scale"]
    self_f = [
        (tick.get("run") or 0) / 100.0,
        ATTACK_LVL / 99.0,
        float(tick["_att"]),
        float(tick["_moving"]),
        hp,
        float(any(e.kind == "shell" for e in ents)),
        0.0,  # in cave transit: the recorder only sees ticks on either side
        float(crab is not None),
    ]
    g[4 : 4 + N_SELF_EXTRA] = self_f[:N_SELF_EXTRA]
    o = 4 + N_SELF_EXTRA
    g[o + EVENT_INDEX[trigger]] = 1.0
    o += len(EVENTS)
    g[o + 0] = min((t - last_break) / 600_000.0, 3.0)
    g[o + 1] = len(recent) / 20.0
    for i, d in enumerate(last_delays):
        g[o + 2 + i] = (math.log(d) - 7.0) / 2.0
    g[o + 5] = min((t - last_camera) / 600_000.0, 3.0)
    g[o + 6] = min((t - last_input) / 60_000.0, 5.0)
    g[o + 7] = (math.log(max(t - last_input, 1.0)) - 7.0) / 2.0
    g[o + 8] = float(tick.get("focused", True))
    if last_intent >= 0:
        g[o + 9 + last_intent] = 1.0
    E = np.zeros((MAX_ENTITIES, ENT_FEATS), dtype=np.float32)
    for i, e in enumerate(ents):
        E[i, KIND_INDEX[e.kind]] = 1.0
        b = len(KINDS)
        E[i, b + 0] = e.dx / 15.0
        E[i, b + 1] = e.dy / 15.0
        E[i, b + 2] = max(abs(e.dx), abs(e.dy)) / 15.0
        E[i, b + 3] = float(e.interactable)
        E[i, b + 4] = float(e.target)
        E[i, b + 5] = 1.0
    return np.concatenate([g, E.ravel()])
