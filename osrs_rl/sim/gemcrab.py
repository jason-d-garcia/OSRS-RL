"""Gemstone crab: attack the crab until it burrows, mine its shell if you were in
the top 16 for damage, then take the cave to the crab's next mine.

The crab's health bar is a timer: damage does not kill it. Players keep
attacking for about 10 minutes, then it burrows, leaves a shell for 90 seconds
and resurfaces at another of three mines.
"""

from __future__ import annotations

import numpy as np

from . import params as P
from .base import Activity, Entity


class GemCrab(Activity):
    name = "gemcrab"
    progress_label = "combat xp + gem value"

    def __init__(self, rng: np.random.Generator, randomize: bool = True):
        super().__init__(rng)
        self.randomize = randomize

    def reset(self) -> None:
        rng = self.rng
        self.tick_count = 0
        self.progress = 0.0
        self.gems = 0
        if self.randomize:
            self.attack_lvl = int(rng.integers(60, 100))
            # REAL: Jason (98 Str, dragon scimitar) hits 0-26, median 12.75 per hit; a spread of
            # mid-to-high level melee setups around that.
            self.max_hit = int(rng.integers(18, 29))
            phase = int(rng.integers(0, P.CRAB_ALIVE_TICKS // 2))
        else:
            self.attack_lvl, self.max_hit, phase = 70, 16, 0
        self.mine = int(rng.integers(P.CRAB_N_MINES))  # where the crab is
        self.player_mine = self.mine
        self.crab_timer = P.CRAB_ALIVE_TICKS - phase
        self.shell_timer = 0
        self.shell_mine = -1
        self.eligible = False
        self.mined_this_shell = False
        self.my_damage = 0.0
        self.others_damage = self._spawn_others(phase)
        self.attacking = False
        self.next_attack = 0
        self.mining_until = -1
        self.transit_until = -1
        self.run_energy = 100.0
        self.dest = self.on_arrive = None
        self.busy = False
        self.bank_open = False
        cx, cy = P.CRAB_MINES[self.player_mine]["crab"]
        self.px = cx + int(rng.integers(-6, 7))
        self.py = cy + int(rng.integers(-6, 7))
        self._rebuild_entities()

    def _spawn_others(self, phase: int) -> np.ndarray:
        n = int(self.rng.integers(*P.CRAB_OTHER_PLAYERS))
        # damage per tick of each other player, and when they started
        self.others_rate = self.rng.uniform(0.6, 2.6, size=n)
        self.others_start = self.rng.integers(0, 300, size=n)
        return np.maximum(0, phase - self.others_start) * self.others_rate

    def _rebuild_entities(self) -> None:
        self.entities_list = []
        m = P.CRAB_MINES[self.player_mine]
        if self.mine == self.player_mine and self.crab_timer > 0:
            self.crab = Entity("crab", *m["crab"], uid=1000 + self.mine)
            self.entities_list.append(self.crab)
        else:
            self.crab = None
        if self.shell_timer > 0 and self.shell_mine == self.player_mine:
            self.shell = Entity("shell", *m["crab"], uid=2000 + self.shell_mine)
            self.entities_list.append(self.shell)
        else:
            self.shell = None
        self.cave = Entity("cave", *m["cave"], uid=3000 + self.player_mine)
        self.entities_list.append(self.cave)
        # a few visible other players standing around the crab
        cx, cy = m["crab"]
        rng = np.random.default_rng(self.player_mine * 7919 + self.tick_count // 100)
        for i in range(min(6, len(self.others_rate))):
            ang = rng.uniform(0, 2 * np.pi)
            self.entities_list.append(
                Entity("player", cx + int(round(2 * np.cos(ang))), cy + int(round(2 * np.sin(ang))), 4000 + i, interactable=False)
            )

    # --- actions -------------------------------------------------------
    def interact(self, ent: Entity) -> bool:
        if self.transit_until > self.tick_count:
            return False
        if ent.kind == "crab" and self.crab is not None:
            if self.attacking:
                return True  # redundant click while already attacking
            self.walk_to(ent, self._start_attack)
            return True
        if ent.kind == "shell" and self.shell is not None:
            self.walk_to(ent, self._start_mine)
            return True
        if ent.kind == "cave":
            self.walk_to(ent, self._enter_cave)
            return True
        return False

    def walk_to(self, ent, on_arrive) -> None:
        super().walk_to(ent, on_arrive)
        self.attacking = False
        self.mining_until = -1

    def walk_near(self, ent) -> bool:
        if self.transit_until > self.tick_count:
            return False
        return super().walk_near(ent)

    def _start_attack(self, crab):
        if self.crab is None:
            return ["target_gone"]
        self.attacking = self.busy = True
        self.next_attack = self.tick_count + 1
        return []

    def _start_mine(self, shell):
        if self.shell is None:
            return ["target_gone"]
        if not self.eligible or self.mined_this_shell:
            return ["invalid"]  # "You don't have enough damage to mine this"
        self.busy = True
        self.mining_until = self.tick_count + int(self.rng.integers(*P.SHELL_MINE_TICKS))
        return []

    def _enter_cave(self, cave):
        self.busy = True
        self.transit_until = self.tick_count + P.CAVE_TRANSIT_TICKS
        return []

    # --- tick ------------------------------------------------------------
    def tick(self) -> list[str]:
        self.tick_count += 1
        events = []
        if self.transit_until == self.tick_count:
            self.busy = False
            self.player_mine = self.mine
            cx, cy = P.CRAB_MINES[self.player_mine]["cave"]
            self.px, self.py = cx, cy
            self._rebuild_entities()
            events.append("arrived")
        events += self._step_movement()
        # crab timer and the other players' damage
        if self.crab_timer > 0:
            self.crab_timer -= 1
            age = P.CRAB_ALIVE_TICKS - self.crab_timer
            self.others_damage += np.where(age > self.others_start, self.others_rate, 0.0)
            if self.attacking and self.tick_count >= self.next_attack:
                self.next_attack = self.tick_count + P.ATTACK_SPEED_TICKS
                if self.rng.random() < P.CRAB_ACCURACY:
                    dmg = int(self.rng.integers(0, self.max_hit + 1))
                    self.my_damage += dmg
                    self.progress += dmg * P.XP_PER_DAMAGE
            if self.crab_timer == 0:
                was_here = self.player_mine == self.mine
                rank = 1 + int(np.sum(self.others_damage > self.my_damage))
                self.eligible = self.my_damage > 0 and rank <= P.CRAB_TOP_N
                self.mined_this_shell = False
                self.shell_mine, self.shell_timer = self.mine, P.CRAB_SHELL_TICKS
                # the crab resurfaces at another mine straight away
                self.mine = int((self.mine + 1 + self.rng.integers(P.CRAB_N_MINES - 1)) % P.CRAB_N_MINES)
                self.crab_timer = P.CRAB_ALIVE_TICKS
                self.my_damage = 0.0
                self.others_damage = self._spawn_others(0)
                if self.attacking:
                    self.attacking = self.busy = False
                    events.append("target_gone")
                if was_here:
                    events.append("shell_ready")
                self._rebuild_entities()
        if self.shell_timer > 0:
            self.shell_timer -= 1
            if self.shell_timer == 0:
                if self.mining_until > 0:
                    self.mining_until = -1
                    self.busy = False
                    events.append("target_gone")
                self._rebuild_entities()
        if self.mining_until == self.tick_count:
            self.mining_until = -1
            self.busy = False
            self.mined_this_shell = True
            self.gems += P.CRAB_GEMS_PER_SHELL
            self.progress += P.CRAB_GEMS_PER_SHELL * P.GEM_VALUE
            events.append("idle")
        return events

    def self_features(self) -> list[float]:
        on_crab_mine = float(self.player_mine == self.mine)
        return [
            self.run_energy / 100.0,
            self.attack_lvl / 99.0,
            float(self.attacking),
            float(self.moving),
            self.crab_timer / P.CRAB_ALIVE_TICKS if on_crab_mine else 0.0,  # visible HP bar
            float(self.shell is not None),
            float(self.transit_until > self.tick_count),
            on_crab_mine,
        ]
