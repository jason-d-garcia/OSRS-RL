import numpy as np

from osrs_rl.agents.controllers import ScriptedBot, SyntheticHuman, run_episode
from osrs_rl.env import OBS_DIM, OsrsEnv
from osrs_rl.metrics import FEATURE_NAMES, session_features
from osrs_rl.sim.base import level_for_xp, stat_random, xp_for_level


def test_xp_table():
    assert xp_for_level(2) == 83
    assert xp_for_level(99) == 13_034_431
    assert level_for_xp(13_034_431) == 99


def test_stat_random_rate():
    rng = np.random.default_rng(0)
    p = np.mean([stat_random(rng, 40, 56, 175) for _ in range(20000)])
    expected = (1 + (56 * 59 + 175 * 39) // 98) / 256
    assert abs(p - expected) < 0.01


def test_episodes_run_and_progress():
    for act in ("willows", "gemcrab"):
        for ctrl in (ScriptedBot(0), SyntheticHuman(0, 0)):
            env = OsrsEnv(act, seed=1)
            sess, tr = run_episode(env, ctrl, seed=1, record=True)
            assert sess["progress"] > 0
            assert env.sim.tick_count >= env.episode_ticks
            assert tr[0][0].shape == (OBS_DIM,)
            assert len(session_features(sess)) == len(FEATURE_NAMES)


def test_willows_inventory_and_bank():
    env = OsrsEnv("willows", seed=3)
    sess, _ = run_episode(env, ScriptedBot(3), seed=3)
    deposits = [r for r in sess["log"] if r["intent"] == 2]
    assert deposits, "scripted bot should bank at least once in 30 minutes"
    assert env.sim.logs <= 28


def test_gemcrab_cycle():
    env = OsrsEnv("gemcrab", seed=4)
    sess, _ = run_episode(env, ScriptedBot(4), seed=4)
    kinds = {r["kind"] for r in sess["log"]}
    assert {"crab", "cave"} <= kinds


def test_dialogue_reads():
    from osrs_rl.recordings import dialogue_reads

    recs = [
        {"type": "dialogue", "t": 1000, "kind": "npc", "chars": 40, "words": 8},
        {"type": "mouse_move", "t": 1500},
        {"type": "menu_click", "t": 3000, "option": "Continue"},
        {"type": "dialogue", "t": 3100, "kind": "options", "chars": 16, "words": 3, "options": ["a", "b"]},
        {"type": "menu_click", "t": 4100, "option": "Continue"},
        {"type": "dialogue", "t": 4200, "kind": "npc", "chars": 10, "words": 2},
        {"type": "dialogue", "t": 4300, "kind": None},
        {"type": "menu_click", "t": 5000, "option": "Walk here"},
        {"type": "dialogue", "t": 6000, "kind": "npc", "chars": 10, "words": 2},
        {"type": "key_press", "t": 6400, "key": "\u2423"},
    ]
    reads = dialogue_reads(recs)
    assert [d["ms"] for d in reads] == [2000, 1000, 400]
    assert reads[1]["options"] == 2


def test_delay_interrupted_by_event():
    """A long delay is cut short when the crab burrows, and the pending intent is dropped."""
    from osrs_rl.env import INTERACT, WAIT
    env = OsrsEnv("gemcrab", seed=5, randomize=False)
    env.reset(seed=5)
    env.step({"intent": INTERACT, "target": env.slot_of(env.sim.crab), "delay_ms": 600.0})
    env.sim.crab_timer = 50  # burrows in 30 s
    n_log = len(env.log)
    env.step({"intent": INTERACT, "target": 0, "delay_ms": 299_000.0})  # would click long after the burrow
    assert env.trigger in ("target_gone", "shell_ready")
    assert len(env.log) == n_log, "the interrupted click must not be logged"
    assert env.t_ms < 299_000.0 + 1200.0
    env.step({"intent": WAIT, "target": 0, "delay_ms": 5_000.0})
    assert env.trigger == "wait_done"


def test_walk_intent():
    from osrs_rl.env import WALK
    env = OsrsEnv("gemcrab", seed=6)
    env.reset(seed=6)
    cave = env.slot_of(env.sim.cave)
    start = (env.sim.px, env.sim.py)
    env.step({"intent": WALK, "target": cave, "delay_ms": 600.0})
    rec = env.log[-1]
    assert rec["intent"] == WALK and rec["kind"] == "" and rec["rank"] == -1
    assert env.trigger == "busy_check"
    env.step({"intent": 0, "target": 0, "delay_ms": 60_000.0})  # wait; the walk ends with "idle"
    assert env.trigger == "idle"
    assert (env.sim.px, env.sim.py) != start
    assert not env.sim.attacking


def test_tutorial_next_action_follows_the_stage():
    from osrs_rl import tutorial as T

    base = {"x": 3094, "z": 3107, "inv": [], "dialogue": None,
            "npcs": [{"slot": 5, "id": T.NPC_GUIDE, "x": 3094, "z": 3108}],
            "locs": [{"id": T.LOC_GUIDE_DOOR, "kind": "guide_door", "x": 3098, "z": 3107}]}
    s = dict(base, tutorial={"stage": T.TALK_TO_GUIDE})
    assert T.next_action(s) == ("act", {"type": "npc", "slot": 5, "op": 1})
    s = dict(base, tutorial={"stage": T.OPEN_SETTINGS})
    assert T.next_action(s)[1] == {"type": "tab", "name": "settings"}
    s = dict(base, tutorial={"stage": T.OPEN_DOOR})
    assert T.next_action(s)[1] == {"type": "loc", "id": T.LOC_GUIDE_DOOR, "op": 1}
    s = dict(base, tutorial={"stage": T.TALK_TO_GUIDE}, dialogue={"kind": "npc"})
    assert T.next_action(s) == ("continue", {"type": "continue"})
    s = dict(base, tutorial={"stage": T.TALK_TO_GUIDE}, dialogue={"kind": "options", "options": ["a", "b", "c"]})
    assert T.next_action(s) == ("option", {"type": "option", "n": 3})
    s = dict(base, tutorial={"stage": T.MEET_CHEF})
    assert T.next_action(s) is None
    # a fire already on the tile: step aside before trying again
    s = dict(base, tutorial={"stage": T.LIGHT_FIRE})
    assert T.next_action(s)[1]["type"] == "use_on_item"
    assert T.next_action(s, fire_blocked=1)[1] == {"type": "walk", "x": 3095, "z": 3107}
    assert T.next_action(s, fire_blocked=2)[1] == {"type": "walk", "x": 3094, "z": 3106}


def test_tutorial_timing_matches_the_live_run():
    import numpy as np
    from osrs_rl.tutorial import HumanTiming

    rng = np.random.default_rng(0)
    t = HumanTiming()
    cont = [t.draw(rng, "continue") for _ in range(4000)]
    react = [t.draw(rng, "act") for _ in range(4000)]
    assert 280 < np.median(cont) < 370  # Jason: 317 ms
    assert 1000 < np.median(react) < 1350  # Jason: 1.16 s
