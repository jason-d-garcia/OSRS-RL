"""Shared pieces of the tick simulator.

The simulator is a stand-in for the real RSMod server while the server-side agent
plugin does not exist yet. It keeps the server's rules that matter to an agent:
the 600 ms tick, tile movement, skilling rolls and timers. It does not render
anything; the policy only ever sees structured state.

Every number that comes from the game is in ``params.py`` so it can be checked
against tick traces from the real server later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TICK_MS = 600

# Entity kinds. The order is the one-hot order in the observation.
KINDS = ("tree", "stump", "bank", "crab", "shell", "cave", "player")
KIND_INDEX = {k: i for i, k in enumerate(KINDS)}

# Events that end an action and hand control back to the policy.
EVENTS = (
    "start",
    "idle",  # the player stopped doing anything
    "inv_full",
    "target_gone",  # tree depleted, crab burrowed, shell sank
    "bank_open",
    "shell_ready",
    "arrived",  # finished a cave transit
    "busy_check",  # still busy; periodic chance to do something else
    "wait_done",
    "invalid",  # the last action did nothing
)
EVENT_INDEX = {e: i for i, e in enumerate(EVENTS)}


@dataclass
class Entity:
    kind: str
    x: int
    y: int
    uid: int
    interactable: bool = True
    meta: dict = field(default_factory=dict)


def stat_random(rng: np.random.Generator, level: int, low: int, high: int) -> bool:
    """OSRS skilling success roll (the formula RSMod's ``statRandom`` uses)."""
    value = (low * (99 - level) + high * (level - 1)) // 98
    chance = (1 + value) / 256.0
    return rng.random() < min(chance, 1.0)


# Standard OSRS experience table, levels 1..99.
_XP_TABLE = [0]
_points = 0
for _lvl in range(1, 99):
    _points += int(_lvl + 300 * 2 ** (_lvl / 7.0))
    _XP_TABLE.append(_points // 4)


def level_for_xp(xp: float) -> int:
    lvl = 1
    for i, need in enumerate(_XP_TABLE):
        if xp >= need:
            lvl = i + 1
    return min(lvl, 99)


def xp_for_level(level: int) -> int:
    return _XP_TABLE[level - 1]


def cheb(ax: int, ay: int, bx: int, by: int) -> int:
    return max(abs(ax - bx), abs(ay - by))


class Activity:
    """One activity loop in a small tile world.

    Subclasses implement the rules. The base class owns the player position,
    walking with run energy, and the "walk to an entity, then do something when
    adjacent" pattern that almost every OSRS interaction follows.
    """

    name = "base"
    #: progress units are XP plus a value for items, defined per activity
    progress_label = "xp"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.tick_count = 0
        self.progress = 0.0
        self.px = 0
        self.py = 0
        self.run_energy = 100.0
        self.dest: Entity | None = None  # entity we are walking to
        self.on_arrive = None  # callback when adjacent to dest
        self.busy = False  # animating a skilling / combat action
        self.bank_open = False
        self.entities_list: list[Entity] = []
        self._uid = 0

    # --- helpers -------------------------------------------------------
    def new_uid(self) -> int:
        self._uid += 1
        return self._uid

    def walk_to(self, ent: Entity, on_arrive) -> None:
        self.dest = ent
        self.on_arrive = on_arrive
        self.busy = False
        self.bank_open = False  # walking closes interfaces

    def _step_movement(self) -> list[str]:
        if self.dest is None:
            if self.run_energy < 100:
                self.run_energy = min(100.0, self.run_energy + 0.45)
            return []
        steps = 2 if self.run_energy > 1 else 1
        for _ in range(steps):
            if cheb(self.px, self.py, self.dest.x, self.dest.y) <= 1:
                break
            self.px += int(np.sign(self.dest.x - self.px))
            self.py += int(np.sign(self.dest.y - self.py))
        if steps == 2:
            self.run_energy = max(0.0, self.run_energy - 0.67)
        if cheb(self.px, self.py, self.dest.x, self.dest.y) <= 1:
            dest, cb = self.dest, self.on_arrive
            self.dest, self.on_arrive = None, None
            return cb(dest) or []
        return []

    @property
    def moving(self) -> bool:
        return self.dest is not None

    @property
    def idle(self) -> bool:
        return not self.moving and not self.busy

    # --- interface for the env ----------------------------------------
    def reset(self) -> None:
        raise NotImplementedError

    def tick(self) -> list[str]:
        """Advance one game tick and return the events that happened."""
        raise NotImplementedError

    def interact(self, ent: Entity) -> bool:
        """Left-click an entity. Returns False when it does nothing."""
        raise NotImplementedError

    def walk_near(self, ent: Entity) -> bool:
        """Walk to a tile next to ``ent`` without interacting with it ("Walk here")."""
        self.walk_to(ent, lambda dest: ["idle"])
        return True

    def deposit_all(self) -> bool:
        return False

    def close_interface(self) -> bool:
        if self.bank_open:
            self.bank_open = False
            return True
        return False

    def self_features(self) -> list[float]:
        """Activity-specific self features, fixed length ``N_SELF_EXTRA``."""
        raise NotImplementedError

    def visible_entities(self) -> list[Entity]:
        return [e for e in self.entities_list if cheb(e.x, e.y, self.px, self.py) <= 20]

    # privileged helpers for scripted controllers
    def nearest(self, kind: str, interactable_only: bool = True) -> Entity | None:
        best, bd = None, 1e9
        for e in self.entities_list:
            if e.kind == kind and (e.interactable or not interactable_only):
                d = cheb(e.x, e.y, self.px, self.py)
                if d < bd:
                    best, bd = e, d
        return best


N_SELF_EXTRA = 8
