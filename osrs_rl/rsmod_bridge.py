"""Run the gem crab env against the real RSMod server instead of the simulator.

The RSMod ``agent-bridge`` module (``rsmod-content/agent-bridge``) streams one player's state
every game cycle over a local socket and takes inputs back. ``RemoteGemCrab`` is a drop-in for
``sim.gemcrab.GemCrab``: it has the same attributes the env, the scripted controllers and the
policy read (``px``, ``entities_list``, ``self_features()``, ``attacking``, ``transit_until``...),
but every ``tick()`` is one real server cycle. So the trainer and the detectors don't change:

    sim = RemoteGemCrab(name="agent1", lockstep=True)
    env = OsrsEnv("gemcrab", sim=sim)

The world is the server's, not ours: ``reset()`` puts the bot back near the crab but the crab
keeps its own timer, and other players (or other bots) are whoever is logged in.

With ``lockstep=True`` the server waits for every step, so with the tick patch
(``RSMOD_TICK_MS=0``, see ``rsmod-content/README.md``) an episode runs as fast as the policy;
about 500 cycles a second on one core. With ``lockstep=False`` the server runs at its own pace
and ``tick()`` simply waits for the next cycle, which is what you want with a real client
watching at 600 ms.

``RecordingWriter`` writes the bot's session in the RuneLite recorder's format (``tick`` and
``menu_click`` records), so ``recordings.to_session`` and ``crab_windows`` read a bot run exactly
as they read your own recordings.
"""

from __future__ import annotations

import gzip
import json
import socket
import time
from pathlib import Path

import numpy as np

from .env import CAMERA, CHECK_TAB, INTERACT, WALK
from .sim import params as P
from .sim.base import Activity, Entity, cheb

DEFAULT_PORT = 45600
COMBAT_SKILLS = ("attack", "strength", "defence", "hitpoints", "ranged", "magic")
VIEW = 20

# What a human's client would have shown for each click, for the recording writer. The words
# match recordings.KIND_WORDS so a written session maps back to the same kinds.
CLICK_TEXT = {
    "crab": ("Attack", "Gemstone Crab"),
    "shell": ("Mine", "Gemstone Crab shell"),
    "cave": ("Crawl-through", "Tunnel"),
}

MSG_NOT_ELIGIBLE = "not great enough to mine"
MSG_ALREADY_MINED = "already"
MSG_SAME_MINE = "already at this mine"


class BridgeError(RuntimeError):
    pass


class BridgeClient:
    """Newline-delimited JSON over TCP to the RSMod agent bridge."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float = 60.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        # Without this, Nagle plus delayed ACKs cap lockstep at ~25 cycles a second.
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.f = self.sock.makefile("rw", encoding="utf-8", newline="\n")

    def send(self, obj: dict) -> None:
        self.f.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self.f.flush()

    def recv(self) -> dict:
        line = self.f.readline()
        if not line:
            raise BridgeError("the server closed the connection")
        return json.loads(line)

    def wait_for(self, *types: str) -> dict:
        """Read until a message of one of ``types`` (errors raise)."""
        while True:
            m = self.recv()
            if m["type"] in types:
                return m
            if m["type"] == "error":
                raise BridgeError(m.get("error", "error"))

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class RemoteGemCrab(Activity):
    """The gem crab activity, played on the server by a headless bot or an attached player."""

    name = "gemcrab"
    progress_label = "combat xp + gem value"

    def __init__(self, rng: np.random.Generator | None = None, randomize: bool = True, *,
                 name: str = "agent1", attach: str | None = None, lockstep: bool = True,
                 host: str = "127.0.0.1", port: int = DEFAULT_PORT, loadout: dict | None = None,
                 spread: int = 6, recorder: "RecordingWriter | None" = None):
        super().__init__(rng if rng is not None else np.random.default_rng())
        self.randomize = randomize
        self.client = BridgeClient(host, port)
        self.client.send({"op": "hello"})
        hello = self.client.wait_for("hello")
        self.mines = hello["mines"]
        self.alive_ticks = hello["alive_ticks"]
        self.tunnel_ticks = hello["tunnel_ticks"]
        self.lockstep = lockstep
        self.spread = spread
        self.recorder = recorder
        self.next_loadout: dict | None = None
        cmd = {"lockstep": lockstep, "spread": spread, **(loadout or {})}
        if attach:
            self.client.send({"op": "attach", "name": attach, "lockstep": lockstep,
                              **({"loadout": loadout} if loadout else {})})
        else:
            self.client.send({"op": "spawn", "name": name, **cmd})
        self.spawned = self.client.wait_for("spawned")
        self.state = self.client.wait_for("tick")
        self._pending: list[dict] = []
        self.tick_count = 0
        self.progress = 0.0

    # ---- lifecycle ------------------------------------------------------------------------
    def reset(self) -> None:
        cmd = {"op": "reset", "spread": self.spread}
        if self.next_loadout:  # applied on the server before the teleport (stats, gear, stance)
            cmd["loadout"] = self.next_loadout
            self.next_loadout = None
        self.client.send(cmd)
        self.client.wait_for("reset_done")
        self._pending = []
        self._recording = False
        self._step()  # the teleport lands on the next cycle
        self._step()
        self._recording = True
        s = self.state
        self.tick_count = 0
        self.progress = 0.0
        self.gems = 0
        self._xp0 = self._combat_xp(s)
        self._gems_seen = s["gems"]
        self.attack_lvl = s["levels"]["attack"]
        self.transit_until = -1
        self.mining_until = -1
        self.walking = False
        self._clicked_tick = -9  # tick_count restarts at 0, so a click from last episode must not linger
        self.dest = self.on_arrive = None
        self.bank_open = False
        self.busy = False
        self._camera_pending = False
        self._derive(s, first=True)

    def close(self) -> None:
        self.client.send({"op": "logout"})
        self.client.close()

    # ---- the env's interface ----------------------------------------------------------------
    def tick(self) -> list[str]:
        prev = self.state
        self._step()
        self.tick_count += 1
        return self._events(prev, self.state)

    def interact(self, ent: Entity) -> bool:
        if self.transit_until > self.tick_count:
            return False
        if ent.kind == "crab" and self.crab is not None:
            self._queue({"type": "attack_crab"}, "crab")
            return True
        if ent.kind == "shell" and self.shell is not None:
            self._queue({"type": "mine_shell"}, "shell")
            self.mining_until = self.tick_count + 20  # cleared by gems, a refusal, or the shell sinking
            return True
        if ent.kind == "cave":
            self._queue({"type": "tunnel"}, "cave")
            self.transit_until = self.tick_count + 40  # cleared on arrival or "already at this mine"
            return True
        return False

    def walk_near(self, ent: Entity) -> bool:
        if self.transit_until > self.tick_count:
            return False
        # A tile next to the target, on the player's side of it.
        x = ent.x - int(np.sign(ent.x - self.px))
        y = ent.y - int(np.sign(ent.y - self.py))
        cx, cz = self._origin
        self._queue({"type": "walk", "x": x + cx, "z": y + cz}, "walk")
        self.walking = True
        self.mining_until = -1
        return True

    def note_input(self, intent: int, option: str = "Skills") -> None:
        """Inputs the server never sees (camera, tabs): only the recording gets them."""
        if intent == CAMERA:
            self._camera_pending = True
        self._pending.append({"type": "camera" if intent == CAMERA else "check_tab"})
        if self.recorder is not None:
            self.recorder.input(intent, option)

    def set_focus(self, focused: bool) -> None:
        """The game window gained or lost focus: the real client sends this to the server."""
        self._pending.append({"type": "focus", "on": bool(focused)})
        if self.recorder is not None:
            self.recorder.focus(focused)

    def set_clock(self, t_ms: float) -> None:
        """The env's clock at the input it is about to make, for the recording's timestamps."""
        if self.recorder is not None:
            self.recorder.t_ms = t_ms

    @property
    def moving(self) -> bool:
        return bool(self.state["moving"]) or getattr(self, "_clicked_tick", -9) >= self.tick_count

    @property
    def idle(self) -> bool:
        return not self.moving and not self.busy

    def self_features(self) -> list[float]:
        s = self.state
        crab = s.get("crab")
        on_crab_mine = float(crab is not None and crab["mine"] == self.player_mine)
        timer = (self.alive_ticks - crab["age"]) / self.alive_ticks if crab and on_crab_mine else 0.0
        return [
            s["run_energy"] / 100.0,
            self.attack_lvl / 99.0,
            float(self.attacking),
            float(self.moving),
            max(0.0, timer),
            float(self.shell is not None),
            float(self.transit_until > self.tick_count),
            on_crab_mine,
        ]

    # ---- internals ----------------------------------------------------------------------------
    def _queue(self, action: dict, kind: str) -> None:
        self._pending.append(action)
        # The server answers on the next cycle; until then the click is "in progress", as in the
        # simulator, where walk_to starts moving at once.
        self._clicked_tick = self.tick_count
        if self.recorder is not None:
            self.recorder.click(kind, self.state)

    def _step(self) -> None:
        self.client.send({"op": "step", "actions": self._pending})
        self._pending = []
        self.state = self.client.wait_for("tick")
        if self.recorder is not None and getattr(self, "_recording", False):
            self.recorder.tick(self.state, camera_moved=getattr(self, "_camera_pending", False))
            self._camera_pending = False

    @staticmethod
    def _combat_xp(s) -> float:
        return float(sum(s["xp"][k] for k in COMBAT_SKILLS))

    def _nearest_mine(self, x: int, z: int) -> int:
        return int(np.argmin([cheb(x, z, m["crab_x"], m["crab_z"]) for m in self.mines]))

    def _derive(self, s: dict, first: bool = False) -> None:
        """Rebuild the simulator-shaped view (local frame, entity list, flags) from a tick."""
        self.player_mine = self._nearest_mine(s["x"], s["z"])
        m = self.mines[self.player_mine]
        self._origin = (m["crab_x"], m["crab_z"])
        cx, cz = self._origin
        self.px, self.py = s["x"] - cx, s["z"] - cz
        crab = s.get("crab")
        self.mine = crab["mine"] if crab else self.player_mine
        self.entities_list = []
        self.crab = None
        if crab and crab["mine"] == self.player_mine and crab["visible"]:
            self.crab = Entity("crab", crab["x"] - cx, crab["z"] - cz, uid=1000 + crab["mine"],
                               meta={"slot": crab["slot"]})
            self.entities_list.append(self.crab)
        self.shell = None
        self.eligible = False
        self.mined_this_shell = False
        for sh in s["shells"]:
            if sh["mine"] == self.player_mine and sh["visible"]:
                self.shell = Entity("shell", sh["x"] - cx, sh["z"] - cz, uid=2000 + sh["mine"],
                                    meta={"slot": sh["slot"]})
                self.eligible = bool(sh["eligible"])
                self.mined_this_shell = bool(sh["mined"])
                self.entities_list.append(self.shell)
                break
        self.cave = Entity("cave", m["tunnel_x"] - cx, m["tunnel_z"] - cz, uid=3000 + self.player_mine)
        self.entities_list.append(self.cave)
        for p in s["players"]:
            self.entities_list.append(Entity("player", p["x"] - cx, p["z"] - cz, 4000 + p["slot"],
                                             interactable=False))
        self.attacking = bool(s["attacking"])
        self.my_damage = crab["my_damage"] if crab else 0
        mining = self.mining_until > self.tick_count
        self.busy = self.attacking or mining or self.transit_until > self.tick_count

    def _events(self, prev: dict, s: dict) -> list[str]:
        """The simulator's events, inferred from two consecutive ticks."""
        events = []
        was_attacking = self.attacking
        had_shell = self.shell is not None
        was_moving = bool(prev["moving"])
        if cheb(prev["x"], prev["z"], s["x"], s["z"]) > 10:
            self.transit_until = -1
            self.walking = False
            events.append("arrived")
        msgs = " ".join(s.get("messages", [])).lower()
        if MSG_SAME_MINE in msgs:
            self.transit_until = -1
            events.append("idle")
        refused = MSG_NOT_ELIGIBLE in msgs or (MSG_ALREADY_MINED in msgs and "mine" in msgs and MSG_SAME_MINE not in msgs)
        if refused and self.mining_until > self.tick_count:
            self.mining_until = -1
            events.append("invalid")
        if any(not r["ok"] for r in s.get("results", [])):
            events.append("invalid")
        if s["gems"] > self._gems_seen:
            gained = s["gems"] - self._gems_seen
            self.gems += gained
            self.progress += gained * P.GEM_VALUE
            if self.mining_until > 0:
                self.mining_until = -1
                events.append("idle")
        self._gems_seen = s["gems"]
        xp = self._combat_xp(s)
        self.progress += xp - self._xp0
        self._xp0 = xp
        self._derive(s)
        if was_attacking and not self.attacking and self.crab is None:
            events.append("target_gone")
        if had_shell and self.shell is None and self.mining_until > self.tick_count:
            self.mining_until = -1
            events.append("target_gone")
        if self.shell is not None and not had_shell:
            events.append("shell_ready")
        if self.walking and was_moving and not self.moving:
            self.walking = False
            events.append("idle")
        self.busy = self.attacking or self.mining_until > self.tick_count or self.transit_until > self.tick_count
        return events


class RecordingWriter:
    """Writes a bot session in the RuneLite recorder's JSONL format (gzip)."""

    SCHEMA = 5

    def __init__(self, path: str | Path, t0_ms: int | None = None, tick_ms: int = 600):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = gzip.open(self.path, "wt", encoding="utf-8")
        self.t0 = int(time.time() * 1000) if t0_ms is None else t0_ms
        self.tick_ms = tick_ms
        self.n_ticks = 0
        self.t_ms = 0.0  # set by the caller before each click (env.t_ms), else the tick time
        self._xp: dict[str, int] = {}
        self.focused = True
        self._write({"type": "session_start", "schema": self.SCHEMA, "world": 0, "source": "rsmod-agent-bridge"})

    def _write(self, rec: dict) -> None:
        rec.setdefault("t", self.t0 + int(self.t_ms))
        self.f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def tick(self, s: dict, camera_moved: bool = False) -> None:
        self.n_ticks += 1
        self.t_ms = max(self.t_ms, self.n_ticks * self.tick_ms)
        npcs = []
        if s.get("crab") and s["crab"]["visible"]:
            c = s["crab"]
            npcs.append({"index": c["slot"], "name": "Gemstone Crab", "dx": c["x"] - s["x"], "dy": c["z"] - s["z"]})
        for sh in s["shells"]:
            if sh["visible"]:
                npcs.append({"index": sh["slot"], "name": "Gemstone Crab shell",
                             "dx": sh["x"] - s["x"], "dy": sh["z"] - s["z"]})
        inter = s.get("interaction")
        self._write({
            "type": "tick", "t": self.t0 + self.n_ticks * self.tick_ms, "tick": s["cycle"],
            "x": s["x"], "y": s["z"], "plane": s["level"],
            "interacting": "Gemstone Crab" if s["attacking"] else (inter["kind"] if inter else None),
            "run": s["run_energy"], "camera_moved": camera_moved, "bank_open": False, "focused": self.focused,
            "inv": s["inv"], "npcs": npcs,
            "players": [{"pid": p["slot"], "dx": p["x"] - s["x"], "dy": p["z"] - s["z"]} for p in s["players"]],
        })
        for msg in s.get("messages", []):
            self._write({"type": "game_message", "t": self.t0 + self.n_ticks * self.tick_ms, "msg": msg})
        for skill, xp in s["xp"].items():
            if self._xp.get(skill) != xp:
                if skill in self._xp:
                    self._write({"type": "xp", "t": self.t0 + self.n_ticks * self.tick_ms,
                                 "skill": skill.upper(), "xp": xp, "level": s["levels"][skill]})
                self._xp[skill] = xp

    def click(self, kind: str, s: dict) -> None:
        option, target = CLICK_TEXT.get(kind, ("Walk here", ""))
        self._write({"type": "menu_click", "option": option, "target": target, "tick": s["cycle"]})

    def input(self, intent: int, option: str = "Skills") -> None:
        """Camera moves are written on the next tick record; tab checks as target-less clicks."""
        if intent == CHECK_TAB:
            self._write({"type": "menu_click", "option": option or "Skills", "target": ""})

    def focus(self, focused: bool) -> None:
        """A focus change, as the RuneLite recorder writes it; ticks carry the current state."""
        self.focused = bool(focused)
        self._write({"type": "focus", "focused": self.focused})

    def close(self) -> None:
        self._write({"type": "session_end", "reason": "episode end"})
        self.f.close()


__all__ = ["BridgeClient", "BridgeError", "RemoteGemCrab", "RecordingWriter", "INTERACT", "WALK"]
