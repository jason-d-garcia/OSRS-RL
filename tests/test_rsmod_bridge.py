"""RemoteGemCrab against a tiny fake of the RSMod agent bridge (no server needed).

The fake speaks the bridge's protocol with a toy world: one crab at the north mine that the
player hits for 10 xp a tick once attacked, and a shell that appears when the crab burrows.
It checks the plumbing: lockstep stepping, the local frame, attacking, burrow events, gem
progress and the recorder-format file reading back through recordings.py.
"""

import json
import socket
import threading

import numpy as np

from osrs_rl.agents.controllers import ScriptedBot, run_episode
from osrs_rl.env import OBS_DIM, OsrsEnv
from osrs_rl.recordings import to_session
from osrs_rl.rsmod_bridge import RecordingWriter, RemoteGemCrab

MINES = [
    {"name": "north", "crab_x": 1273, "crab_z": 3173, "tunnel_x": 1278, "tunnel_z": 3168, "exit_x": 1277, "exit_z": 3169},
    {"name": "east", "crab_x": 1353, "crab_z": 3112, "tunnel_x": 1351, "tunnel_z": 3124, "exit_x": 1352, "exit_z": 3122},
    {"name": "south", "crab_x": 1239, "crab_z": 3043, "tunnel_x": 1246, "tunnel_z": 3036, "exit_x": 1246, "exit_z": 3038},
]
BURROW_AT = 60
GEM = 1625  # uncut opal


class FakeBridge(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.cycle = 0
        self.x, self.z = 1268, 3170
        self.attacking = False
        self.xp = 1_000_000
        self.gems = 0
        self.age = 0
        self.mined = False
        self.pending_mine = -1

    def tick(self, results):
        self.cycle += 1
        self.age += 1
        if self.attacking and self.age <= BURROW_AT:
            self.xp += 10
        if self.age == BURROW_AT + 1:
            self.attacking = False
        if self.pending_mine >= 0 and self.cycle >= self.pending_mine:
            self.gems += 3
            self.pending_mine = -1
            self.mined = True
        crab = None if self.age > BURROW_AT else {
            "slot": 7, "mine": 0, "x": 1273, "z": 3173, "age": self.age, "visible": True,
            "my_damage": 0, "rank": 1, "attackers": 1, "my_last_hit_cycle": -1}
        shells = [] if self.age <= BURROW_AT or self.age > BURROW_AT + 150 else [{
            "slot": 8, "mine": 0, "x": 1273, "z": 3173, "ticks_left": 150, "visible": True,
            "eligible": True, "mined": self.mined}]
        return {
            "type": "tick", "cycle": self.cycle, "x": self.x, "z": self.z, "level": 0,
            "run_energy": 100.0, "running": True, "moving": False, "delayed": False,
            "interaction": {"kind": "npc", "slot": 7, "op": 2} if self.attacking else None,
            "face_npc": 7 if self.attacking else -1, "attacking": self.attacking,
            "levels": {k: 80 for k in ("attack", "strength", "defence", "hitpoints", "ranged", "magic", "mining")},
            "xp": {"attack": self.xp, "strength": 0, "defence": 0, "hitpoints": 0, "ranged": 0, "magic": 0, "mining": 0},
            "inv": [[GEM, 1]] * self.gems, "gems": self.gems, "crab": crab, "shells": shells,
            "players": [{"slot": 3, "x": 1271, "z": 3171, "target_npc": 7}],
            "messages": [], "results": results,
        }

    def run(self):
        conn, _ = self.srv.accept()
        f = conn.makefile("rw", encoding="utf-8", newline="\n")

        def send(o):
            f.write(json.dumps(o) + "\n")
            f.flush()

        for line in f:
            cmd = json.loads(line)
            op = cmd["op"]
            if op == "hello":
                send({"type": "hello", "mines": MINES, "alive_ticks": 1000, "shell_ticks": 150,
                      "tunnel_ticks": 3, "mine_ticks": 4})
            elif op == "spawn":
                send({"type": "spawned", "name": cmd["name"], "slot": 1, "headless": True, "lockstep": True})
                send(self.tick([]))
            elif op == "reset":
                self.attacking, self.age, self.mined = False, 0, False
                send({"type": "reset_done", "cycle": self.cycle})
            elif op == "step":
                results = []
                for a in cmd["actions"]:
                    if a["type"] == "attack_crab" and self.age <= BURROW_AT:
                        self.attacking = True
                    if a["type"] == "mine_shell":
                        self.pending_mine = self.cycle + 4
                    results.append({"action": a, "ok": True})
                send(self.tick(results))
            elif op == "logout":
                break
        conn.close()


def test_remote_gemcrab_runs_an_episode(tmp_path):
    fake = FakeBridge()
    fake.start()
    rec = RecordingWriter(tmp_path / "bot.jsonl.gz")
    sim = RemoteGemCrab(name="t", port=fake.port, recorder=rec)
    env = OsrsEnv("gemcrab", sim=sim, episode_ticks=120)
    sess, tr = run_episode(env, ScriptedBot(0), seed=0, record=True)
    rec.close()
    sim.close()

    assert tr[0][0].shape == (OBS_DIM,)
    assert sim.tick_count == 120
    kinds = [r["kind"] for r in sess["log"]]
    assert kinds[0] == "crab" and "shell" in kinds  # attacked, then mined the shell after the burrow
    assert sim.gems == 3
    events = {e for _, e in env.events_log}
    assert {"target_gone", "shell_ready", "idle"} <= events
    # 10 xp a tick for the ticks attacked, plus three gems
    assert sess["progress"] > 300 and sess["progress"] < 600 + 3 * 150 + 1
    # local frame: the crab sits at the origin, the player a few tiles south-west
    assert (sim.px, sim.py) == (1268 - 1273, 3170 - 3173)
    # the recorder-format file reads back to the same inputs
    back = to_session(rec.path)
    assert [r["kind"] for r in back["log"]] == kinds
    assert np.isclose(back["progress"], sess["progress"], rtol=0.05)
