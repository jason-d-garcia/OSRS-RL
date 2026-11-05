import numpy as np

from osrs_rl.afk import AttentionState, fit_heavy, sample_heavy
from osrs_rl.agents.controllers import ScriptedBot, run_episode
from osrs_rl.env import CHECK_TAB, OsrsEnv, set_afk_model


def _prior(away_s=60.0, fidget_per_min=2.0, switch_per_hour=0.0):
    a = {"n": 50, "mu": float(np.log(away_s)), "sigma": 0.1, "u": away_s * 2, "w_tail": 0.0, "alpha": 2.0}
    reg = {"share": 1.0,
           "away": {"shell_ready": a, "target_gone": a, "arrived": a, "start": a},
           "fight_gap": {"per_min": 0.0, "length": a},
           "fidget": {"per_min": fidget_per_min, "options": {"Friends List": 1}},
           "focus_hold": {"median_ms": 5000.0, "sigma": 0.5}}
    return {"regimes": {"unfocused": reg}, "switch_per_hour": switch_per_hour, "mood_sigma": 0.0,
            "tab_options": {"Friends List": 1}}


def test_fit_and_sample_heavy():
    rng = np.random.default_rng(0)
    x = np.concatenate([np.exp(rng.normal(3.0, 0.5, 900)), 60 * (1 - rng.random(100)) ** (-1 / 1.5)])
    p = fit_heavy(x)
    assert 0.1 < p["w_tail"] < 0.3 and p["alpha"] > 0.5
    s = np.array([sample_heavy(p, rng) for _ in range(5000)])
    assert abs(np.median(s) - np.median(x)) / np.median(x) < 0.25
    assert s.max() <= 900.0


def test_attention_state_switches_regime():
    p = _prior(switch_per_hour=1e6)
    p["regimes"]["focused"] = dict(p["regimes"]["unfocused"])
    st = AttentionState(p, np.random.default_rng(0))
    first = st.regime
    assert st.update(1.0) and st.regime != first


def test_env_away_after_events_and_fidgets():
    set_afk_model(_prior(away_s=60.0, fidget_per_min=2.0))
    try:
        env = OsrsEnv("gemcrab", seed=2)
        sess, _ = run_episode(env, ScriptedBot(2), seed=2)
    finally:
        set_afk_model(None)
    assert env.away_log, "a prompting event should send the player away"
    # Nothing but tab fidgets happens inside an away period.
    for start, ms, _ in env.away_log:
        inside = [r for r in sess["log"] if start < r["t_ms"] < start + ms]
        assert all(r["intent"] == CHECK_TAB for r in inside)
    tabs = [r for r in sess["log"] if r["intent"] == CHECK_TAB]
    assert tabs and all(r["option"] == "Friends List" for r in tabs)


def test_env_without_afk_is_unchanged():
    env = OsrsEnv("gemcrab", seed=2)
    sess, _ = run_episode(env, ScriptedBot(2), seed=2)
    assert env.attn is None and not env.away_log and sess["progress"] > 0


def test_away_slows_progress():
    rates = []
    for prior in (None, _prior(away_s=120.0, fidget_per_min=0.0)):
        set_afk_model(prior)
        try:
            env = OsrsEnv("gemcrab", seed=4)
            sess, _ = run_episode(env, ScriptedBot(4), seed=4)
        finally:
            set_afk_model(None)
        rates.append(sess["progress"])
    assert rates[1] < rates[0]


def test_camera_cap_turns_camera_into_wait():
    from osrs_rl.env import CAMERA, set_camera_cap

    set_camera_cap(0.0)
    try:
        env = OsrsEnv("gemcrab", seed=5)
        env.reset(seed=5)
        for _ in range(20):
            env.step({"intent": CAMERA, "target": 0, "delay_ms": 1000.0})
    finally:
        set_camera_cap(None)
    assert not any(r["intent"] == CAMERA for r in env.log)
