"""First experiment, end to end, from one command.

    python run_experiment.py                      # both activities, full budget
    python run_experiment.py --activity gemcrab --ticks 500000   # quick run

For each activity:
  1. synthetic human sessions (12 made-up players x 10 sessions; 8 train, 4 held out)
  2. scripted bot sessions (uniform 600-1800 ms sleeps)
  3. behaviour cloning on the 8 training players
  4. PPO from the BC prior with a KL penalty, efficiency cap and humanness shaping
  5. a pure-RL ablation (no BC prior, no KL, no shaping)
  6. detectors trained on human-train vs each bot, tested on held-out players and
     held-out bot sessions; per-timescale metrics against held-out humans
Results go to results/<activity>/.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from osrs_rl.agents.controllers import ScriptedBot, SyntheticHuman, run_episode
from osrs_rl.detector import evaluate_detectors
from osrs_rl.env import OsrsEnv
from osrs_rl.metrics import reaction_delays, timescale_metrics
from osrs_rl.train import PolicyController, train_bc, train_ppo

N_PLAYERS, TRAIN_PLAYERS, SESSIONS_PER_PLAYER = 12, 8, 10
N_BOT_SESSIONS = 100  # 50 to train each detector, 50 to test it


def rate(s):
    return s["progress"] / (s["ms"] / 3.6e6)


def gen_sessions(activity, make_controller, seeds, record=False):
    sessions, transitions = [], []
    for sd in seeds:
        env = OsrsEnv(activity, seed=sd)
        sess, tr = run_episode(env, make_controller(sd), seed=sd, record=record)
        sessions.append(sess)
        transitions += tr
    return sessions, transitions


def strip(sess_list):
    return [{k: v for k, v in s.items()} for s in sess_list]


def run(activity, ticks, out_dir, kl_coef, seed, log):
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"== {activity} ==")
    # 1. synthetic humans
    humans = {}
    trans_by_player = {}
    for p in range(N_PLAYERS):
        seeds = [100_000 * (p + 1) + k for k in range(SESSIONS_PER_PLAYER)]
        ss, tr = gen_sessions(activity, lambda sd, p=p: SyntheticHuman(p, sd), seeds, record=True)
        humans[p], trans_by_player[p] = ss, tr
    h_train = [s for p in range(TRAIN_PLAYERS) for s in humans[p]]
    h_test = [s for p in range(TRAIN_PLAYERS, N_PLAYERS) for s in humans[p]]
    hr = np.array([rate(s) for s in h_train])
    rd = np.concatenate([reaction_delays(s["log"]) for s in h_train])
    reward_cfg = {
        "human_median_rate_per_hour": float(np.median(hr)),
        "cap_rate_per_hour": float(np.percentile(hr, 75)),
        "reaction_floor_ms": float(np.percentile(rd, 2)),
        "shaping_weight": 0.2,
        "camera_weight": 0.05,
        "reward_scale": 10.0,
    }
    log(f"  human median {reward_cfg['human_median_rate_per_hour']:.0f}/h, p75 cap {reward_cfg['cap_rate_per_hour']:.0f}/h, "
        f"reaction floor {reward_cfg['reaction_floor_ms']:.0f} ms")

    # 2. BC (val = one of the training players, so held-out players stay untouched)
    tr_train = [t for p in range(TRAIN_PLAYERS - 1) for t in trans_by_player[p]]
    tr_val = trans_by_player[TRAIN_PLAYERS - 1]
    log(f"  bc on {len(tr_train)} decisions from {TRAIN_PLAYERS - 1} players")
    bc, bc_hist = train_bc(tr_train, tr_val, seed=seed, log=log)

    # 3. PPO from BC with KL, and the pure-RL ablation
    log(f"  ppo from bc, kl {kl_coef}, {ticks / 1e6:.1f}M ticks")
    rl, rl_curve = train_ppo(activity, bc, bc, reward_cfg, total_ticks=ticks, kl_coef=kl_coef, seed=seed, log=log)
    log("  pure rl ablation: no bc, no kl, no shaping or cap")
    raw_cfg = {"human_median_rate_per_hour": reward_cfg["human_median_rate_per_hour"], "reward_scale": 10.0}
    pure, pure_curve = train_ppo(activity, None, None, raw_cfg, total_ticks=ticks, kl_coef=0.0, lr=3e-4,
                                 ent_coef=0.01, seed=seed + 1, log=log)

    # 4. evaluation sessions on fresh seeds
    def pol_sessions(pol, base):
        torch.manual_seed(base)
        return gen_sessions(activity, lambda sd: PolicyController(pol), range(base, base + N_BOT_SESSIONS))[0]

    bots = {
        "scripted": gen_sessions(activity, lambda sd: ScriptedBot(sd), range(9_000_000, 9_000_000 + N_BOT_SESSIONS))[0],
        "bc": pol_sessions(bc, 9_100_000),
        "rl_kl": pol_sessions(rl, 9_200_000),
        "pure_rl": pol_sessions(pure, 9_300_000),
    }
    results = {"activity": activity, "reward_cfg": reward_cfg, "ticks": ticks, "kl_coef": kl_coef, "bots": {}}
    human_rates = np.array([rate(s) for s in h_test])
    results["human_heldout"] = {
        "progress_per_h_median": float(np.median(human_rates)),
        "progress_per_h_iqr": [float(x) for x in np.percentile(human_rates, [25, 75])],
        "actions_per_session": float(np.mean([len(s["log"]) for s in h_test])),
    }
    for name, ss in bots.items():
        half = N_BOT_SESSIONS // 2
        det = evaluate_detectors(h_train, h_test, ss[:half], ss[half:])
        det.pop("_feature_detector", None)
        ts = timescale_metrics(ss[half:], h_test)
        results["bots"][name] = {
            "detectors": det,
            "timescales": ts,
            "progress_per_h_median": float(np.median([rate(s) for s in ss])),
            "actions_per_session": float(np.mean([len(s["log"]) for s in ss])),
        }
        log(f"  {name:9s} prog/h {results['bots'][name]['progress_per_h_median']:7.0f}  "
            f"rules {det['rules_floor']['bot_flag_rate']:.2f}  gbt acc {det['feature_gbt']['balanced_acc']:.2f} "
            f"auc {det['feature_gbt']['auc']:.2f}  gru acc {det['sequence_gru']['balanced_acc']:.2f} "
            f"auc {det['sequence_gru']['auc']:.2f}  reaction KS {ts['reaction_ks']:.2f}")
    results["rules_floor_human_false_flag"] = float(np.mean(
        [__import__("osrs_rl.detector", fromlist=["rules_floor"]).rules_floor(s) for s in h_test]))
    results["bc_history"] = bc_hist
    results["rl_curve"] = rl_curve
    results["pure_rl_curve"] = pure_curve
    results["wall_s"] = time.time() - t0
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    torch.save({"bc": bc.state_dict(), "rl_kl": rl.state_dict(), "pure_rl": pure.state_dict(), "reward_cfg": reward_cfg},
               out_dir / "policies.pt")
    # a few sample sessions for inspection
    samples = {"human": h_test[:3], **{k: v[:3] for k, v in bots.items()}}
    (out_dir / "sample_sessions.json").write_text(json.dumps(samples))
    log(f"  done in {time.time() - t0:.0f}s")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activity", default="both", choices=["both", "gemcrab", "willows"])
    ap.add_argument("--ticks", type=int, default=5_000_000)
    ap.add_argument("--kl", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    acts = ["gemcrab", "willows"] if args.activity == "both" else [args.activity]
    out = pathlib.Path(args.out)
    logf = open(out.mkdir(parents=True, exist_ok=True) or out / "log.txt", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    for a in acts:
        run(a, args.ticks, out / a, args.kl, args.seed, log)


if __name__ == "__main__":
    main()
