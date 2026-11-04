"""First experiment against real play: Jason's recorded gem crab sessions as the human side.

    python run_real.py --recordings ~/.runelite/osrs-rl-recordings --out results/real

What changes from run_experiment.py:

* The humans the detectors see are 15-minute windows of real gem crab play, cut from the
  RuneLite recordings (``recordings.crab_windows``). Every simulated session is cut into the
  same 15-minute windows so the two sides cover the same length of play.
* There is one real player and only a few hours, so the real windows are split by recording
  file (leave-files-out cross-validation): a detector is never tested on a window from a file
  it trained on. Predictions from all folds are pooled into one AUC with a bootstrap 90%
  interval.
* Behaviour cloning still learns from the synthetic players. The recordings do not yet carry
  the simulator's observation, so they cannot train a policy directly; they only judge it.
* One extra comparison: synthetic humans vs real play. If a detector separates those easily,
  the simulator's model of a person is the gap, not the agent.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from osrs_rl.agents.controllers import ScriptedBot, SyntheticHuman
from osrs_rl.detector import FeatureDetector, SequenceDetector, rules_floor
from osrs_rl.env import OsrsEnv
from osrs_rl.metrics import FEATURE_NAMES, gaps, reaction_delays, session_features, timescale_metrics
from osrs_rl.recordings import crab_windows
from osrs_rl.train import PolicyController, train_bc, train_ppo

WINDOW_TICKS = 1500  # 15 minutes
TICK_MS = 600
N_PLAYERS, TRAIN_PLAYERS, SESSIONS_PER_PLAYER = 12, 8, 10
N_BOT_EPISODES = 40  # each 30-minute episode gives two 15-minute windows
N_FOLDS = 4


def rate(s):
    return s["progress"] / (s["ms"] / 3.6e6)


def rollout_windows(env, controller, seed, record=False):
    """Run one 30-minute episode and cut it into 15-minute windows with their own progress."""
    obs, _ = env.reset(seed=seed)
    controller.reset(env)
    transitions, marks, done = [], [0.0], False
    next_cut = WINDOW_TICKS
    while not done:
        a = controller(env)
        if record:
            transitions.append((obs, a))
        obs, _, done, _, _ = env.step(a)
        while env.sim.tick_count >= next_cut:
            marks.append(env.sim.progress)
            next_cut += WINDOW_TICKS
    wins = []
    win_ms = WINDOW_TICKS * TICK_MS
    for k in range(len(marks) - 1):
        lo, hi = k * win_ms, (k + 1) * win_ms
        rows = [dict(r) for r in env.log if lo <= r["t_ms"] < hi]
        for r in rows:
            r["t_ms"] -= lo
        wins.append({"log": rows, "progress": marks[k + 1] - marks[k], "ms": float(win_ms),
                     "activity": env.activity_name})
    return wins, transitions


def gen(make_controller, seeds, record=False):
    wins, trans = [], []
    for sd in seeds:
        w, tr = rollout_windows(OsrsEnv("gemcrab", seed=sd), make_controller(sd), sd, record)
        wins += w
        trans += tr
    return wins, trans


def bootstrap_auc(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]


def cv_detect(real, real_groups, other, seed=0):
    """Real play (label 0) vs ``other`` (label 1), folds grouped by recording file."""
    rng = np.random.default_rng(seed)
    files = sorted(set(real_groups))
    rng.shuffle(files)
    fold_of_file = {f: i % N_FOLDS for i, f in enumerate(files)}
    real_fold = np.array([fold_of_file[g] for g in real_groups])
    other_fold = rng.permutation(np.arange(len(other)) % N_FOLDS)
    preds = {"feature_gbt": [], "sequence_gru": []}
    ys = []
    for k in range(N_FOLDS):
        tr = [s for s, f in zip(real, real_fold) if f != k] + [s for s, f in zip(other, other_fold) if f != k]
        ytr = [0] * int(np.sum(real_fold != k)) + [1] * int(np.sum(other_fold != k))
        te = [s for s, f in zip(real, real_fold) if f == k] + [s for s, f in zip(other, other_fold) if f == k]
        yte = [0] * int(np.sum(real_fold == k)) + [1] * int(np.sum(other_fold == k))
        if not any(y == 0 for y in yte):
            continue
        ys += yte
        preds["feature_gbt"] += list(FeatureDetector().fit(tr, ytr).predict(te))
        preds["sequence_gru"] += list(SequenceDetector(seed=seed).fit(tr, ytr).predict(te))
    out = {}
    for name, p in preds.items():
        out[name] = {"auc": float(roc_auc_score(ys, p)), "auc_90ci": bootstrap_auc(ys, p)}
    best = max(out, key=lambda k: out[k]["auc"])
    out["worst_case"] = {"name": best, **out[best]}
    return out


def describe(sessions):
    g = np.concatenate([gaps(s["log"]) for s in sessions]) if sessions else np.zeros(0)
    rd = np.concatenate([reaction_delays(s["log"]) for s in sessions]) if sessions else np.zeros(0)
    feats = np.stack([session_features(s) for s in sessions])
    f = lambda name: float(np.mean(feats[:, FEATURE_NAMES.index(name)]))  # mean: most windows have none
    return {
        "windows": len(sessions),
        "progress_per_h_median": float(np.median([rate(s) for s in sessions])),
        "actions_per_15min_median": float(np.median([len(s["log"]) for s in sessions])),
        "reaction_ms_median": float(np.median(rd)) if len(rd) else None,
        "gap_s_median": float(np.median(g) / 1000) if len(g) else None,
        "frac_gaps_over_60s": float(np.mean(g > 60_000)) if len(g) else None,
        "camera_per_h": f("camera_per_h"),
        "check_tab_per_h": f("check_tab_per_h"),
        "rules_floor_flag_rate": float(np.mean([rules_floor(s) for s in sessions])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--ticks", type=int, default=5_000_000)
    ap.add_argument("--kl", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--bc", default="synthetic", choices=["synthetic", "real", "mixed"],
                    help="what behaviour cloning copies: the synthetic players, Jason's recordings, or both")
    ap.add_argument("--adv-weight", type=float, default=0.0,
                    help="also train PPO with the detectors in the reward (GAIL-style) at this weight; 0 = off")
    ap.add_argument("--adv-retrain", type=int, default=10, help="retrain the in-loop detectors every N PPO updates")
    ap.add_argument("--bc-val-files", type=int, default=3,
                    help="hold out this many of the most recent recordings to validate BC and pick its best epoch")
    ap.add_argument("--skip-pure", action="store_true", help="skip the pure-RL baseline (for leash sweeps)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    real, groups = [], []
    for p in sorted(Path(args.recordings).expanduser().glob("session-*.jsonl.gz")):
        w = crab_windows(p, WINDOW_TICKS)
        real += w
        groups += [p.name] * len(w)
        log(f"{p.name}: {len(w)} crab windows")
    log(f"real: {len(real)} windows of 15 min from {len(set(groups))} files")
    if len(set(groups)) < N_FOLDS:
        raise SystemExit("need crab windows from at least 4 recording files")

    humans, trans = {}, {}
    for p in range(N_PLAYERS):
        seeds = [100_000 * (p + 1) + k for k in range(SESSIONS_PER_PLAYER)]
        humans[p], trans[p] = gen(lambda sd, p=p: SyntheticHuman(p, sd), seeds, record=True)
    synth = [s for p in range(N_PLAYERS) for s in humans[p]]

    # Reward targets come from real play now: its progress rate and reaction floor.
    rr = np.array([rate(s) for s in real])
    rd = np.concatenate([reaction_delays(s["log"]) for s in real])
    reward_cfg = {
        "human_median_rate_per_hour": float(np.median(rr)),
        "cap_rate_per_hour": float(np.percentile(rr, 75)),
        # Clicks queued before an event register as 0 ms; never let the floor drop below 200 ms.
        "reaction_floor_ms": max(float(np.percentile(rd, 2)), 200.0) if len(rd) else 250.0,
        "shaping_weight": 0.2,
        "camera_weight": 0.05,
        "reward_scale": 10.0,
    }
    log(f"real median {reward_cfg['human_median_rate_per_hour']:.0f}/h, p75 cap {reward_cfg['cap_rate_per_hour']:.0f}/h, "
        f"reaction floor {reward_cfg['reaction_floor_ms']:.0f} ms")

    tr_train = [t for p in range(TRAIN_PLAYERS - 1) for t in trans[p]]
    tr_val = trans[TRAIN_PLAYERS - 1]
    real_tr = {}
    if args.bc != "synthetic":
        from osrs_rl.real_bc import recording_transitions
        for p in sorted(Path(args.recordings).expanduser().glob("session-*.jsonl.gz")):
            tr = recording_transitions(p)
            if tr:
                real_tr[p.name] = tr
        # Hold out the most recent recordings (file names sort by date) for validation: they are the
        # current recorder schema and Jason's current habits. Keep at least 3 files for training.
        files = sorted(real_tr)
        n_val = min(args.bc_val_files, max(len(files) - 3, 1))
        val_files = files[-n_val:]
        r_train = [t for f in files if f not in val_files for t in real_tr[f]]
        r_val = [t for f in val_files for t in real_tr[f]]
        n_int = sum(a["intent"] != 0 for _, a in r_train)
        log(f"real bc data: {len(r_train)} decisions ({n_int} inputs) from {len(files) - n_val} files, "
            f"val {len(r_val)} decisions from {', '.join(val_files)}")
        if args.bc == "real":
            tr_train, tr_val = r_train, r_val
        else:
            # Upweight the real decisions so they are about half of the data.
            k = max(1, len(tr_train) // max(len(r_train), 1))
            tr_train, tr_val = tr_train + r_train * k, r_val
    bc, _ = train_bc(tr_train, tr_val, seed=args.seed, log=log, select_best=args.bc != "synthetic")
    log(f"ppo from bc, kl {args.kl}, {args.ticks / 1e6:.1f}M ticks")
    rl, rl_curve = train_ppo("gemcrab", bc, bc, reward_cfg, total_ticks=args.ticks, kl_coef=args.kl,
                             seed=args.seed, log=log)

    def pol(p, base, n=N_BOT_EPISODES):
        torch.manual_seed(base)
        return gen(lambda sd: PolicyController(p), range(base, base + n))[0]

    adv = adv_curve = None
    if args.adv_weight > 0:
        from osrs_rl.adversarial import DetectorReward
        dr = DetectorReward(real, weight=args.adv_weight, retrain_every=args.adv_retrain, seed=args.seed, log=log)
        dr.fit(pol(bc, 8_000_000, n=24))  # start from detectors that already know the BC agent
        log(f"ppo from bc with detector reward, weight {args.adv_weight}, kl {args.kl}")
        adv, adv_curve = train_ppo("gemcrab", bc, bc, reward_cfg, total_ticks=args.ticks, kl_coef=args.kl,
                                   seed=args.seed, log=log, detector_reward=dr)
        adv_history = dr.history

    pure = None
    if not args.skip_pure:
        raw_cfg = {"human_median_rate_per_hour": reward_cfg["human_median_rate_per_hour"], "reward_scale": 10.0}
        pure, _ = train_ppo("gemcrab", None, None, raw_cfg, total_ticks=args.ticks, kl_coef=0.0, lr=3e-4,
                            ent_coef=0.01, seed=args.seed + 1, log=log)

    others = {
        "synthetic_humans": synth,
        "scripted": gen(lambda sd: ScriptedBot(sd), range(9_000_000, 9_000_000 + N_BOT_EPISODES))[0],
        "bc": pol(bc, 9_100_000),
        "rl_kl": pol(rl, 9_200_000),
    }
    if adv is not None:
        others["rl_adv"] = pol(adv, 9_400_000)
    if pure is not None:
        others["pure_rl"] = pol(pure, 9_300_000)
    results = {"window_minutes": WINDOW_TICKS * TICK_MS / 60000, "real_files": sorted(set(groups)),
               "reward_cfg": reward_cfg, "ticks": args.ticks, "kl_coef": args.kl, "bc_source": args.bc,
               "real": describe(real), "vs_real": {}}
    for name, ss in others.items():
        det = cv_detect(real, groups, ss, seed=args.seed)
        ts = timescale_metrics(ss, real)
        results["vs_real"][name] = {"describe": describe(ss), "detectors": det, "reaction_ks": ts["reaction_ks"],
                                    "gap_ks": ts["gap_ks"]}
        w = det["worst_case"]
        log(f"{name:16s} prog/h {results['vs_real'][name]['describe']['progress_per_h_median']:7.0f}  "
            f"gbt auc {det['feature_gbt']['auc']:.2f}  gru auc {det['sequence_gru']['auc']:.2f}  "
            f"worst {w['auc']:.2f} [{w['auc_90ci'][0]:.2f}, {w['auc_90ci'][1]:.2f}]  reaction KS {ts['reaction_ks']:.2f}")
    results["rl_curve"] = rl_curve
    if adv is not None:
        results["adv"] = {"weight": args.adv_weight, "retrain_every": args.adv_retrain, "curve": adv_curve,
                          "detector_history": adv_history}
    results["wall_s"] = time.time() - t0
    (out / "results.json").write_text(json.dumps(results, indent=1))
    log(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
