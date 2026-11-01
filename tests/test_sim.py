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
