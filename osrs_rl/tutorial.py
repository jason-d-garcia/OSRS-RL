"""Tutorial Island on the RSMod server: a client for the bridge's tutorial scene and a scripted bot.

The server side is ``rsmod-content/tutorial-island`` (the island from the Gielinor Guide to the
gate out of the Survival Expert's area) and the agent bridge's ``scene: "tutorial"``. Progress is
the ``tutorial`` varp with the live game's values, so a bot run and a recording of a person on the
live game are measured the same way.

The island is a fixed sequence, so *what* to do next is not learned: ``next_action`` reads it off
the stage. What a policy has to get right is *how*: when it clicks, how long it sits on each line of
dialogue, how it reacts after something finishes. ``HumanTiming`` holds those delays, fitted to
Jason's live run (4 Nov 2026, see docs/tutorial-island.md), and the scripted bot uses it, so it is
the baseline a learned policy has to beat on the detectors.

    python run_tutorial.py --episodes 4        # server running with RSMOD_TICK_MS=0
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rsmod_bridge import BridgeClient, DEFAULT_PORT

TICK_MS = 600

# Stages (values of the `tutorial` varp), as in TutorialRules.kt.
TALK_TO_GUIDE = 2
OPEN_SETTINGS = 3
TALK_TO_GUIDE_AGAIN = 7
OPEN_DOOR = 10
TALK_TO_SURVIVAL = 20
OPEN_INVENTORY = 30
CATCH_SHRIMP = 40
OPEN_SKILLS = 50
TALK_TO_SURVIVAL_AGAIN = 60
CHOP_TREE = 70
LIGHT_FIRE = 80
COOK_SHRIMP = 90
LEAVE_SURVIVAL = 120
MEET_CHEF = 130

STAGES = (TALK_TO_GUIDE, OPEN_SETTINGS, TALK_TO_GUIDE_AGAIN, OPEN_DOOR, TALK_TO_SURVIVAL,
          OPEN_INVENTORY, CATCH_SHRIMP, OPEN_SKILLS, TALK_TO_SURVIVAL_AGAIN, CHOP_TREE,
          LIGHT_FIRE, COOK_SHRIMP, LEAVE_SURVIVAL, MEET_CHEF)

# Sections as in docs/tutorial-island.md: (name, first stage, stage that ends it).
SECTIONS = (("Gielinor Guide", TALK_TO_GUIDE, TALK_TO_SURVIVAL),
            ("Survival Expert", TALK_TO_SURVIVAL, MEET_CHEF))

# Cache ids (the island's "newbie" types).
NPC_GUIDE = 3308
NPC_SURVIVAL = 8503
NPC_FISHING_SPOT = 3317
LOC_GUIDE_DOOR = 9398
LOC_TREE = 9730
LOC_FIRE = 26185
LOC_GATES = (9470, 9708)
OBJ_TINDERBOX = 590
OBJ_LOGS = 2511
OBJ_RAW_SHRIMPS = 2514


@dataclass
class HumanTiming:
    """Delays, in ms, drawn from lognormals fitted to Jason's live run.

    ``continue_*``: from a dialogue line appearing to moving past it (99 lines, median 317 ms; he
    skips with the space bar rather than reading). ``option_*``: picking a dialogue option (5,
    median 594 ms). ``react_*``: from something finishing (a stage change or a dialogue closing)
    to the next click (66, median 1.16 s).
    """

    continue_mu: float = 5.83
    continue_sd: float = 0.48
    option_mu: float = 6.40
    option_sd: float = 0.45
    react_mu: float = 7.06
    react_sd: float = 0.62
    #: give up on an action that changed nothing after this long and try again
    retry_ms: float = 9000.0

    def draw(self, rng: np.random.Generator, kind: str) -> float:
        mu, sd = {"continue": (self.continue_mu, self.continue_sd),
                  "option": (self.option_mu, self.option_sd)}.get(kind, (self.react_mu, self.react_sd))
        return float(rng.lognormal(mu, sd))


def next_action(s: dict, fire_blocked: int = 0) -> tuple[str, dict] | None:
    """What a player on the island does next, from one bridge tick. None when nothing is needed.

    Returns ``(kind, action)``, where kind is "continue", "option" or "act" (for timing).
    ``fire_blocked``: how many times lighting a fire on this tile was refused because something
    (usually an earlier fire) is on it. Step to a neighbouring tile first, as a player would,
    trying another side each time in case one is blocked.
    """
    d = s.get("dialogue")
    if d:
        if d["kind"] == "options":
            # The guide asks how experienced you are; Jason picks "experienced", the last option.
            n = max(1, len(d.get("options", [])))
            return "option", {"type": "option", "n": n}
        return "continue", {"type": "continue"}
    stage = s.get("tutorial", {}).get("stage", 0)
    if fire_blocked and stage == LIGHT_FIRE:
        dx, dz = ((1, 0), (0, -1), (-1, 0), (0, 1))[(fire_blocked - 1) % 4]
        return "act", {"type": "walk", "x": s["x"] + dx, "z": s["z"] + dz}
    npcs = s.get("npcs", [])
    locs = s.get("locs", [])

    def npc(npc_id):
        near = [n for n in npcs if n["id"] == npc_id]
        if not near:
            return None
        n = min(near, key=lambda n: max(abs(n["x"] - s["x"]), abs(n["z"] - s["z"])))
        return {"type": "npc", "slot": n["slot"], "op": 1}

    def loc(*ids):
        near = [lc for lc in locs if lc["id"] in ids]
        if not near:
            return None
        lc = min(near, key=lambda lc: max(abs(lc["x"] - s["x"]), abs(lc["z"] - s["z"])))
        return {"type": "loc", "id": lc["id"], "op": 1}

    act = {
        TALK_TO_GUIDE: lambda: npc(NPC_GUIDE),
        OPEN_SETTINGS: lambda: {"type": "tab", "name": "settings"},
        TALK_TO_GUIDE_AGAIN: lambda: npc(NPC_GUIDE),
        OPEN_DOOR: lambda: loc(LOC_GUIDE_DOOR),
        TALK_TO_SURVIVAL: lambda: npc(NPC_SURVIVAL),
        OPEN_INVENTORY: lambda: {"type": "tab", "name": "inventory"},
        CATCH_SHRIMP: lambda: npc(NPC_FISHING_SPOT),
        OPEN_SKILLS: lambda: {"type": "tab", "name": "skills"},
        TALK_TO_SURVIVAL_AGAIN: lambda: npc(NPC_SURVIVAL),
        CHOP_TREE: lambda: loc(LOC_TREE),
        LIGHT_FIRE: lambda: {"type": "use_on_item", "item": OBJ_TINDERBOX, "target": OBJ_LOGS},
        COOK_SHRIMP: lambda: {"type": "use_on_loc", "item": OBJ_RAW_SHRIMPS, "loc": LOC_FIRE},
        LEAVE_SURVIVAL: lambda: loc(*LOC_GATES),
    }.get(stage)
    if act is None:
        return None
    a = act()
    return ("act", a) if a else None


def _signature(s: dict) -> tuple:
    """What changes when an action took effect: stage, dialogue, inventory, position, interaction."""
    d = s.get("dialogue")
    return (s.get("tutorial", {}).get("stage"), d and d["kind"], tuple(map(tuple, s["inv"])),
            s["x"], s["z"], bool(s.get("interaction")), bool(s["delayed"]))


@dataclass
class EpisodeLog:
    ticks: int = 0
    stage_ticks: dict = field(default_factory=dict)  # stage -> first tick it was seen
    actions: list = field(default_factory=list)  # (t_ms, kind, action, ok)
    dialogue_delays: list = field(default_factory=list)
    finished: bool = False

    def section_seconds(self) -> dict:
        out = {}
        for name, start, end in SECTIONS:
            if start in self.stage_ticks and end in self.stage_ticks:
                out[name] = (self.stage_ticks[end] - self.stage_ticks[start]) * TICK_MS / 1000
        return out


class TutorialBot:
    """Plays the island on the server with human-fitted delays. One connection, one bot."""

    def __init__(self, name: str = "tut1", *, lockstep: bool = True, host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT, timing: HumanTiming | None = None, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.timing = timing or HumanTiming()
        self.client = BridgeClient(host, port)
        self.client.send({"op": "hello"})
        self.client.wait_for("hello")
        self.client.send({"op": "spawn", "name": name, "lockstep": lockstep, "scene": "tutorial"})
        self.client.wait_for("spawned")
        self.state = self.client.wait_for("tick")

    def step(self, actions: list[dict] | None = None) -> dict:
        self.client.send({"op": "step", "actions": actions or []})
        self.state = self.client.wait_for("tick")
        return self.state

    def reset(self, stage: int = TALK_TO_GUIDE) -> None:
        self.client.send({"op": "reset", "scene": "tutorial", "stage": stage})
        self.client.wait_for("reset_done")
        self.step()
        self.step()

    def play(self, max_ticks: int = 3000, stage: int = TALK_TO_GUIDE) -> EpisodeLog:
        """Plays from ``stage`` until through the Survival Expert's gate or ``max_ticks``."""
        self.reset(stage)
        log = EpisodeLog()
        t_ms = 0.0  # human clock; ticks advance it by 600 ms
        due_ms = None  # when the next input is due
        planned = None
        planned_at = 0.0
        sent_sig = None
        sent_at = None
        refused = {}  # tile -> times lighting a fire there was refused
        for tick in range(max_ticks):
            s = self.state
            st = s.get("tutorial", {}).get("stage", 0)
            log.stage_ticks.setdefault(st, tick)
            if st >= MEET_CHEF:
                log.finished = True
                break
            sig = _signature(s)
            waiting = sent_sig is not None and sig == sent_sig and t_ms - sent_at < self.timing.retry_ms
            busy = s["delayed"] or s["moving"] or bool(s.get("interaction"))
            actions = []
            if not waiting and (not busy or s.get("dialogue")):
                nxt = next_action(s, refused.get((s["x"], s["z"]), 0))
                if nxt is None:
                    planned = due_ms = None
                elif planned is None or planned[1] != nxt[1]:
                    planned = nxt
                    planned_at = t_ms
                    due_ms = t_ms + self.timing.draw(self.rng, nxt[0])
                if planned is not None and t_ms + TICK_MS >= due_ms:
                    kind, action = planned
                    actions.append(action)
                    if kind in ("continue", "option"):
                        log.dialogue_delays.append(due_ms - planned_at)
                    log.actions.append((due_ms, kind, action))
                    if action["type"] == "walk":  # stepping off a refused tile; next time another side
                        refused[(s["x"], s["z"])] += 1
                    # Two dialogue lines of the same kind look alike in the signature, and the
                    # server answers a continue within the tick, so only world actions wait.
                    if kind == "act":
                        sent_sig, sent_at = sig, t_ms
                    planned = due_ms = None
            elif not waiting:
                planned = due_ms = None
            if sent_sig is not None and sig != sent_sig:
                sent_sig = None
            self.step(actions)
            t_ms += TICK_MS
            if any("can't light a fire here" in m.lower() for m in self.state.get("messages", [])):
                tile = (self.state["x"], self.state["z"])
                refused[tile] = refused.get(tile, 0) + 1
            results = self.state.get("results", [])
            if results and log.actions:
                log.actions[-1] = log.actions[-1][:3] + (all(r["ok"] for r in results),)
        log.ticks = tick + 1
        return log

    def close(self) -> None:
        try:
            self.client.send({"op": "logout"})
        finally:
            self.client.close()


def jason_sections() -> dict:
    """Section times from Jason's live run (docs/tutorial-island.md), in seconds."""
    return {"Gielinor Guide": 23.0, "Survival Expert": 49.0}


__all__ = ["TutorialBot", "HumanTiming", "EpisodeLog", "next_action", "STAGES", "SECTIONS",
           "jason_sections"]
