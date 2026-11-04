import numpy as np
import torch

from osrs_rl.agents.controllers import SyntheticHuman, run_episode
from osrs_rl.env import OBS_DIM, OsrsEnv
from osrs_rl.policy import Policy, PolicyConfig, policy_from_checkpoint, timing_match_loss, timing_targets
from osrs_rl.train import PolicyController, episode_starts, train_bc, train_ppo


def _transitions(n_sessions=2, ticks=600):
    out = []
    for k in range(n_sessions):
        env = OsrsEnv("gemcrab", seed=k, episode_ticks=ticks)
        out += _roll(env, SyntheticHuman(k, k), k)[1]
    return out


def _roll(env, ctrl, seed):
    return run_episode(env, ctrl, seed=seed, record=True)


def test_default_config_matches_old_state_dict():
    # A checkpoint without "config" (every one saved before this change) must load unchanged.
    old = Policy()
    ckpt = {"bc": old.state_dict()}
    new = policy_from_checkpoint(ckpt, "bc")
    obs = torch.randn(5, OBS_DIM)
    a, b = old(obs), new(obs)
    for k in ("intent", "mu", "log_std", "value"):
        assert torch.allclose(a[k], b[k])
    assert not new.recurrent


def test_options_build_and_sample():
    for cfg in (PolicyConfig(n_mix=5), PolicyConfig(wide_tail=True), PolicyConfig(situated_delay=True),
                PolicyConfig(recurrent=True), PolicyConfig(5, True, True, True)):
        pol = Policy(config=cfg)
        obs = torch.randn(4, OBS_DIM)
        if cfg.recurrent:
            i, t, d, lp, v, hx = pol.sample(obs, hx=pol.initial_state(4))
            assert hx.shape == (4, pol.hidden)
        else:
            i, t, d, lp, v = pol.sample(obs)
        assert torch.isfinite(lp).all() and d.shape == (4,)


def test_forward_seq_matches_stepping():
    torch.manual_seed(0)
    pol = Policy(config=PolicyConfig(recurrent=True))
    T, B = 6, 3
    obs = torch.randn(T, B, OBS_DIM)
    starts = torch.zeros(T, B, dtype=torch.bool)
    starts[0] = True
    starts[3, 1] = True  # env 1 starts a new episode at step 3
    out = pol.forward_seq(obs, None, starts)
    hx = torch.zeros(B, pol.hidden)
    for t in range(T):
        hx = hx * (1.0 - starts[t].float())[:, None]
        o = pol(obs[t], hx)
        hx = o["hx"]
        assert torch.allclose(out["intent"][t * B : (t + 1) * B], o["intent"], atol=1e-6)


def test_timing_match_loss_is_differentiable_and_zero_at_target():
    pol = Policy()
    obs = torch.randn(64, OBS_DIM)
    intent = torch.zeros(64, dtype=torch.long)
    out = pol(obs)
    m, v = Policy.delay_moments(out, intent)
    target = {0: (float(m.mean()), float((v.mean() + m.var(unbiased=False)).sqrt()))}
    assert float(timing_match_loss(out, intent, target)) < 1e-8
    loss = timing_match_loss(out, intent, {0: (9.0, 1.7)})
    loss.backward()
    assert pol.delay.weight.grad is not None and pol.delay.weight.grad.abs().sum() > 0


def test_episode_starts_from_clock():
    O = torch.zeros(6, OBS_DIM)
    O[:, 2] = torch.tensor([0.0, 0.2, 0.5, 0.1, 0.4, 0.0])
    assert episode_starts(O).tolist() == [True, False, False, True, False, True]


def test_recurrent_bc_ppo_and_controller_smoke():
    tr = _transitions()
    targets = timing_targets(tr)
    cfg = PolicyConfig(n_mix=4, wide_tail=True, situated_delay=True, recurrent=True)
    bc, hist = train_bc(tr, tr[:200], epochs=2, config=cfg, seq_len=32, burn_in=8, log=lambda m: None)
    assert bc.recurrent and np.isfinite(hist[-1]["val_nll"])
    rl, _ = train_ppo("gemcrab", bc, bc, {"reward_scale": 10.0, "human_median_rate_per_hour": 70000.0},
                      total_ticks=3000, n_envs=2, horizon=64, epochs=1, log=lambda m: None,
                      timing_targets=targets, timing_coef=1.0)
    assert rl.recurrent
    env = OsrsEnv("gemcrab", seed=7, episode_ticks=300)
    sess, _ = _roll(env, PolicyController(rl), 7)
    assert sess["ms"] > 0
