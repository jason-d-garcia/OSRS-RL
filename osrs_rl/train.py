"""Behaviour cloning and PPO fine-tuning with a KL penalty to the BC prior."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from .agents.controllers import Controller
from .env import OsrsEnv
from .policy import Policy, to_actions

torch.set_num_threads(4)


class PolicyController(Controller):
    def __init__(self, policy: Policy):
        self.policy = policy

    def __call__(self, env):
        obs = torch.from_numpy(env._obs()[None])
        i, t, d, _, _ = self.policy.sample(obs)
        return to_actions(i, t, d)[0]


# ----------------------------------------------------------------------
def transitions_to_tensors(transitions):
    obs = torch.from_numpy(np.stack([o for o, _ in transitions]).astype(np.float32))
    intent = torch.tensor([a["intent"] for _, a in transitions])
    target = torch.tensor([a["target"] for _, a in transitions])
    log_delay = torch.tensor([np.log(np.clip(a["delay_ms"], 120.0, 300_000.0)) for _, a in transitions], dtype=torch.float32)
    return obs, intent, target, log_delay


def train_bc(train_tr, val_tr, epochs: int = 40, lr: float = 1e-3, seed: int = 0, log=print, select_best: bool = False):
    """Behaviour cloning. With ``select_best`` the epoch with the lowest validation NLL is kept."""
    torch.manual_seed(seed)
    pol = Policy()
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    O, I, T, D = transitions_to_tensors(train_tr)
    vO, vI, vT, vD = transitions_to_tensors(val_tr)
    n = len(O)
    hist = []
    best, best_nll = None, float("inf")
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for k in range(0, n, 512):
            idx = perm[k : k + 512]
            out = pol(O[idx])
            lp, _ = Policy.log_prob(out, I[idx], T[idx], D[idx])
            loss = -lp.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        with torch.no_grad():
            vlp, parts = Policy.log_prob(pol(vO), vI, vT, vD)
            acc = (pol(vO)["intent"].argmax(-1) == vI).float().mean().item()
        v_nll = -vlp.mean().item()
        hist.append({"epoch": ep, "train_nll": tot / n, "val_nll": v_nll, "val_intent_acc": acc})
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  bc epoch {ep:3d} train_nll {tot / n:.3f} val_nll {v_nll:.3f} val_intent_acc {acc:.3f}")
        if select_best and v_nll < best_nll:
            best_nll, best = v_nll, (ep, copy.deepcopy(pol.state_dict()))
    if select_best and best is not None:
        pol.load_state_dict(best[1])
        log(f"  bc keeps epoch {best[0]} (best val_nll {best_nll:.3f})")
    return pol, hist


# ----------------------------------------------------------------------
def train_ppo(
    activity: str,
    init: Policy | None,
    ref: Policy | None,
    reward_cfg: dict,
    total_ticks: int = 5_000_000,
    n_envs: int = 16,
    horizon: int = 128,
    kl_coef: float = 0.1,
    lr: float = 1e-4,
    gamma: float = 0.995,
    lam: float = 0.95,
    clip: float = 0.2,
    epochs: int = 4,
    minibatch: int = 512,
    ent_coef: float = 0.0,
    seed: int = 0,
    log=print,
    detector_reward=None,
):
    """PPO with a KL leash to ``ref``. ``detector_reward`` (osrs_rl.adversarial.DetectorReward) adds a
    bonus at the end of each episode for looking human to the current detectors, which are retrained
    every few updates on the agent's latest play."""
    torch.manual_seed(seed)
    pol = copy.deepcopy(init) if init is not None else Policy()
    ref = ref.eval() if ref is not None else None
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    envs = [OsrsEnv(activity, seed=seed * 1000 + i, reward_cfg=reward_cfg) for i in range(n_envs)]
    obs = np.stack([e.reset(seed=seed * 1000 + i)[0] for i, e in enumerate(envs)])
    ep_ret = np.zeros(n_envs)
    ep_kl = np.zeros(n_envs)
    finished = []
    curve = []
    ticks_done = 0
    t0 = time.time()
    upd = 0
    # warm up the value head when starting from BC so early advantages are not noise
    while ticks_done < total_ticks:
        buf = {k: [] for k in ("obs", "intent", "target", "logd", "lp", "val", "rew", "done")}
        for _ in range(horizon):
            ot = torch.from_numpy(obs.astype(np.float32))
            i, t, d, lp, v = pol.sample(ot)
            kl = np.zeros(n_envs)
            if ref is not None and kl_coef > 0:
                with torch.no_grad():
                    lp_ref, _ = Policy.log_prob(ref(ot), i, t, d)
                kl = (lp - lp_ref).numpy()
            acts = to_actions(i, t, d)
            rews, dones, nobs = np.zeros(n_envs), np.zeros(n_envs), []
            for k, (e, a) in enumerate(zip(envs, acts)):
                before = e.sim.tick_count
                o, r, done, _, info = e.step(a)
                ticks_done += e.sim.tick_count - before
                rews[k] = r - kl_coef * kl[k]
                ep_ret[k] += r
                ep_kl[k] += kl[k]
                if done:
                    bonus = 0.0
                    if detector_reward is not None:
                        bonus = detector_reward.bonus({"log": e.log, "progress": info["episode_progress"],
                                                       "ms": info["episode_ms"], "activity": e.activity_name})
                        rews[k] += bonus
                    finished.append({"progress": info["episode_progress"], "ms": info["episode_ms"], "ret": ep_ret[k],
                                     "kl": ep_kl[k], "decisions": e.n_decisions, "actions": len(e.log),
                                     "bonus": bonus})
                    ep_ret[k] = ep_kl[k] = 0
                    o, _ = e.reset(seed=int(np.random.randint(1 << 30)))
                dones[k] = done
                nobs.append(o)
            for key, val in (("obs", ot), ("intent", i), ("target", t), ("logd", d), ("lp", lp), ("val", v)):
                buf[key].append(val)
            buf["rew"].append(torch.tensor(rews, dtype=torch.float32))
            buf["done"].append(torch.tensor(dones, dtype=torch.float32))
            obs = np.stack(nobs)
        with torch.no_grad():
            last_v = pol(torch.from_numpy(obs.astype(np.float32)))["value"]
        R, D, V = torch.stack(buf["rew"]), torch.stack(buf["done"]), torch.stack(buf["val"])
        adv = torch.zeros_like(R)
        gae = torch.zeros(n_envs)
        for s in reversed(range(horizon)):
            nv = last_v if s == horizon - 1 else V[s + 1]
            delta = R[s] + gamma * nv * (1 - D[s]) - V[s]
            gae = delta + gamma * lam * (1 - D[s]) * gae
            adv[s] = gae
        ret = adv + V
        flat = lambda x: torch.cat(x) if isinstance(x, list) else x.reshape(-1)
        O = torch.cat(buf["obs"])
        I, T, Dl, LP = flat(buf["intent"]), flat(buf["target"]), flat(buf["logd"]), flat(buf["lp"])
        A, RET = flat(adv), flat(ret)
        A = (A - A.mean()) / (A.std() + 1e-8)
        n = len(O)
        value_only = upd < 3 and init is not None
        for _ in range(epochs):
            perm = torch.randperm(n)
            for k in range(0, n, minibatch):
                idx = perm[k : k + minibatch]
                out = pol(O[idx])
                v_loss = F.mse_loss(out["value"], RET[idx])
                if value_only:
                    loss = v_loss
                else:
                    lp_new, _ = Policy.log_prob(out, I[idx], T[idx], Dl[idx])
                    ratio = (lp_new - LP[idx]).clamp(-20.0, 20.0).exp()
                    pg = -torch.min(ratio * A[idx], ratio.clamp(1 - clip, 1 + clip) * A[idx]).mean()
                    loss = pg + 0.5 * v_loss - ent_coef * Policy.entropy_proxy(out).mean()
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
                opt.step()
        upd += 1
        if detector_reward is not None:
            detector_reward.maybe_retrain(upd)
        if finished:
            recent = finished[-32:]
            row = {
                "update": upd,
                "ticks": ticks_done,
                "progress_per_h": float(np.mean([f["progress"] / (f["ms"] / 3.6e6) for f in recent])),
                "return": float(np.mean([f["ret"] for f in recent])),
                "kl_per_episode": float(np.mean([f["kl"] for f in recent])),
                "actions_per_episode": float(np.mean([f["actions"] for f in recent])),
                "decisions_per_episode": float(np.mean([f["decisions"] for f in recent])),
                "detector_bonus": float(np.mean([f["bonus"] for f in recent])),
                "wall_s": time.time() - t0,
            }
            curve.append(row)
            if upd % 10 == 0:
                log(
                    f"  ppo upd {upd:4d} ticks {ticks_done / 1e6:5.2f}M prog/h {row['progress_per_h']:8.0f} "
                    f"ret {row['return']:6.2f} kl/ep {row['kl_per_episode']:6.2f} actions/ep {row['actions_per_episode']:5.0f} "
                    f"det {row['detector_bonus']:+.2f} ({row['wall_s']:.0f}s)"
                )
    return pol, curve
