"""Draynor willows: cut willows until the inventory is full, bank, repeat."""

from __future__ import annotations

import numpy as np

from . import params as P
from .base import Activity, Entity, cheb, level_for_xp, stat_random, xp_for_level


class Willows(Activity):
    name = "willows"
    progress_label = "woodcutting xp"

    def __init__(self, rng: np.random.Generator, randomize: bool = True):
        super().__init__(rng)
        self.randomize = randomize

    def reset(self) -> None:
        rng = self.rng
        self.tick_count = 0
        self.progress = 0.0
        if self.randomize:
            self.level = int(rng.integers(30, 61))
            self.axe = str(rng.choice(["steel", "mithril", "adamant", "rune"]))
            self.px = P.DRAYNOR_START[0] + int(rng.integers(-2, 3))
            self.py = P.DRAYNOR_START[1] + int(rng.integers(-1, 2))
            n_others = int(rng.integers(0, 4))
            self.logs = int(rng.integers(0, 10))
        else:
            self.level, self.axe = 40, "rune"
            self.px, self.py = P.DRAYNOR_START
            n_others, self.logs = 0, 0
        self.xp = float(xp_for_level(self.level))
        self.run_energy = 100.0
        self.dest = self.on_arrive = None
        self.busy = False
        self.bank_open = False
        self.chop_tree: Entity | None = None
        self.next_roll = 0
        self.entities_list = []
        for x, y in P.DRAYNOR_BOOTHS:
            self.entities_list.append(Entity("bank", x, y, self.new_uid()))
        self.trees = []
        for x, y in P.DRAYNOR_WILLOWS:
            t = Entity("tree", x, y, self.new_uid(), meta={"cut": 0, "respawn": 0, "cutters": 0})
            self.trees.append(t)
            self.entities_list.append(t)
        # other woodcutters, modelled only by which tree they are on
        self.others = []
        for _ in range(n_others):
            t = self.trees[int(rng.integers(len(self.trees)))]
            o = Entity("player", t.x + 1, t.y, self.new_uid(), interactable=False, meta={"tree": t})
            self.others.append(o)
            self.entities_list.append(o)

    # --- actions -------------------------------------------------------
    def interact(self, ent: Entity) -> bool:
        if ent.kind == "tree" and ent.interactable:
            if self.logs >= P.INVENTORY_SLOTS:
                return False  # "Your inventory is too full"
            if self.busy and self.chop_tree is ent:
                return True  # redundant click on the tree you are cutting; harmless
            self.walk_to(ent, self._start_chop)
            return True
        if ent.kind == "bank":
            self.walk_to(ent, self._open_bank)
            return True
        return False  # stumps, players

    def deposit_all(self) -> bool:
        if not self.bank_open or self.logs == 0:
            return False
        self.logs = 0
        return True

    def _start_chop(self, tree: Entity):
        if not tree.interactable:
            return ["target_gone"]
        self.busy = True
        self.chop_tree = tree
        self.next_roll = self.tick_count + P.WC_FIRST_ROLL_DELAY
        return []

    def _open_bank(self, booth: Entity):
        self.bank_open = True
        return ["bank_open"]

    # --- tick ------------------------------------------------------------
    def tick(self) -> list[str]:
        self.tick_count += 1
        events = self._step_movement()
        # trees: active-cut counters and respawns
        for t in self.trees:
            cutters = sum(1 for o in self.others if o.meta["tree"] is t)
            if self.busy and self.chop_tree is t:
                cutters += 1
            t.meta["cutters"] = cutters
            if t.interactable:
                t.meta["cut"] = t.meta["cut"] + 1 if cutters else max(0, t.meta["cut"] - 1)
            else:
                t.meta["respawn"] -= 1
                if t.meta["respawn"] <= 0:
                    t.interactable, t.kind, t.meta["cut"] = True, "tree", 0
        # other players cut logs too, which can deplete the tree under us
        for o in self.others:
            t = o.meta["tree"]
            if t.interactable and self.tick_count % 4 == 0 and stat_random(self.rng, 50, 48, 150):
                self._maybe_deplete(t)
            if not t.interactable and self.rng.random() < 0.2:
                live = [x for x in self.trees if x.interactable]
                if live:
                    nt = live[int(self.rng.integers(len(live)))]
                    o.meta["tree"], o.x, o.y = nt, nt.x + 1, nt.y
        # our chopping
        if self.busy and self.chop_tree is not None:
            t = self.chop_tree
            if not t.interactable:
                self.busy, self.chop_tree = False, None
                events.append("target_gone")
            elif self.tick_count >= self.next_roll:
                self.next_roll = self.tick_count + P.WC_ROLL_INTERVAL
                low, high = P.WILLOW_AXE_RATES[self.axe]
                if stat_random(self.rng, self.level, low, high):
                    self.logs += 1
                    self.xp += P.WILLOW_XP
                    self.progress += P.WILLOW_XP
                    self.level = level_for_xp(self.xp)
                    self._maybe_deplete(t)
                    if self.logs >= P.INVENTORY_SLOTS:
                        self.busy, self.chop_tree = False, None
                        events.append("inv_full")
                    elif not t.interactable:
                        self.busy, self.chop_tree = False, None
                        events.append("target_gone")
        return events

    def _maybe_deplete(self, t: Entity) -> None:
        if t.meta["cut"] >= P.WILLOW_DESPAWN_TICKS:
            t.interactable, t.kind = False, "stump"
            t.meta["respawn"] = P.WILLOW_RESPAWN_TICKS

    def self_features(self) -> list[float]:
        return [
            self.logs / P.INVENTORY_SLOTS,
            self.run_energy / 100.0,
            self.level / 99.0,
            float(self.bank_open),
            float(self.busy),
            float(self.moving),
            float(self.logs >= P.INVENTORY_SLOTS),
            0.0,
        ]

    # privileged helpers for scripted controllers
    def trees_by_distance(self):
        live = [t for t in self.trees if t.interactable]
        return sorted(live, key=lambda t: cheb(t.x, t.y, self.px, self.py))
