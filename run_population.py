"""Population test: does a bot have to look like *Jason*, or only like *a plausible player*?

The usual score (``cv_detect``) asks one window at a time: Jason's play (0) or this bot (1)? With a
single human in the data, a detector can win that by learning Jason's habits, so a high AUC may
mean "not Jason" rather than "not human". This script adds three views:

1. Progress per hour, per 15-minute window, for every source. Is the bot inside Jason's range?
2. Jason vs Jason: the same detectors trained to tell one of Jason's days from his other days
   (folds grouped by file). That AUC is the floor of "different from this exact recording set"
   that a real person already hits. A bot near it is as different as Jason is from himself.
3. Accounts: groups of N windows, drawn at random as all-Jason, all one bot, or a mix (an account
   that bots part of the time). Each window gets an out-of-fold detector score; an account's score
   is the mean. We report account-level AUC as N grows (what an anti-cheat sees over hours), and
   for mixed accounts, whether the windows that are bots can be picked out. Both with all features
   and with the skill features (progress/h, actions/h) removed, since a slower or faster player
   is still a plausible player.

    python run_population.py --recordings ~/.runelite/osrs-rl-recordings \\
        --bots scripted=results/server-score/scripted/recordings \\
               bc=results/server-score-gail/bc/recordings rl_kl=results/server-score-gail/rl_kl/recordings \\
               rl_adv=results/server-score-gail/rl_adv/recordings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from osrs_rl.detector import SequenceDetector
from osrs_rl.metrics import FEATURE_NAMES, session_features
from osrs_rl.recordings import crab_windows
from run_real import N_FOLDS, WINDOW_TICKS, assign_folds, bootstrap_auc, cv_detect, rate

SKILL = ("progress_per_h_1e4", "actions_per_h")


def gbt(X, y):
    w = np.where(y == 1, 0.5 / max(y.mean(), 1e-6), 0.5 / max(1 - y.mean(), 1e-6))
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          random_state=0).fit(X, y, sample_weight=w)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def oof_scores(real, rgroups, bots, bgroups, seed):
    """Out-of-fold P(bot) for every window, from detectors trained on Jason vs all bots pooled."""
    rf, bf = assign_folds(rgroups, len(bots), bgroups, seed)
    win = real + bots
    fold = np.concatenate([rf, bf])
    y = np.array([0] * len(real) + [1] * len(bots))
    X = np.stack([session_features(s) for s in win])
    keep = [i for i, n in enumerate(FEATURE_NAMES) if n not in SKILL]
    out = {k: np.full(len(win), np.nan) for k in ("gbt", "gbt_no_skill", "gru")}
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        if not te.any():
            continue
        out["gbt"][te] = gbt(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        out["gbt_no_skill"][te] = gbt(X[tr][:, keep], y[tr]).predict_proba(X[te][:, keep])[:, 1]
        sd = SequenceDetector(seed=seed).fit([win[i] for i in np.flatnonzero(tr)], list(y[tr]))
        out["gru"][te] = sd.predict([win[i] for i in np.flatnonzero(te)])
    return out


def accounts(real_idx, bot_idx, sizes, n_acc, rng):
    """Random accounts: (kind, window indices, per-window is_bot)."""
    names = list(bot_idx)
    acc = []
    for n in sizes:
        for _ in range(n_acc):
            kind = rng.choice(["human", "bot", "mixed"], p=[0.4, 0.3, 0.3])
            b = names[rng.integers(len(names))]
            if kind == "human":
                is_bot = np.zeros(n, bool)
            elif kind == "bot":
                is_bot = np.ones(n, bool)
            else:  # each window botted with a per-account probability; at least one of each
                is_bot = rng.random(n) < rng.uniform(0.2, 0.8)
                if n > 1 and (is_bot.all() or not is_bot.any()):
                    is_bot[rng.integers(n)] ^= True
                if n == 1:
                    is_bot[:] = rng.random() < 0.5
            idx = np.array([rng.choice(bot_idx[b]) if ib else rng.choice(real_idx) for ib in is_bot])
            acc.append({"n": n, "kind": kind, "bot": b if is_bot.any() else None, "idx": idx, "is_bot": is_bot})
    return acc


def account_table(acc, score):
    """Account AUC (any bot vs all Jason) per size and per bot, and within-mixed window picking."""
    L = logit(score)
    res = {}
    for n in sorted({a["n"] for a in acc}):
        A = [a for a in acc if a["n"] == n]
        y = np.array([a["is_bot"].any() for a in A])
        s = np.array([L[a["idx"]].mean() for a in A])
        row = {"any_bot_auc": round(float(roc_auc_score(y, s)), 3),
               "any_bot_auc_90ci": [round(v, 3) for v in bootstrap_auc(list(y), list(s))]}
        for b in sorted({a["bot"] for a in A if a["bot"]}):
            sel = [i for i, a in enumerate(A) if not a["is_bot"].any() or (a["bot"] == b)]
            pure = [i for i in sel if not A[i]["is_bot"].any() or A[i]["kind"] == "bot"]
            mix = [i for i in sel if not A[i]["is_bot"].any() or A[i]["kind"] == "mixed"]
            row[b] = {"all_bot_auc": round(float(roc_auc_score(y[pure], s[pure])), 3) if len(set(y[pure])) == 2 else None,
                      "part_bot_auc": round(float(roc_auc_score(y[mix], s[mix])), 3) if len(set(y[mix])) == 2 else None}
        mixed = [a for a in A if a["kind"] == "mixed" and a["is_bot"].any() and not a["is_bot"].all()]
        if mixed:
            yb = np.concatenate([a["is_bot"] for a in mixed])
            raw = np.concatenate([L[a["idx"]] for a in mixed])
            rel = np.concatenate([L[a["idx"]] - L[a["idx"]].mean() for a in mixed])
            row["which_windows_in_mixed"] = {"raw_auc": round(float(roc_auc_score(yb, raw)), 3),
                                             "relative_to_account_auc": round(float(roc_auc_score(yb, rel)), 3)}
        res[n] = row
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--bots", nargs="+", required=True, help="name=dir of bot-*.jsonl.gz")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--accounts", type=int, default=600, help="accounts per size")
    ap.add_argument("--out", default="results/population")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)

    real, rgroups = [], []
    for p in sorted(Path(args.recordings).expanduser().glob("session-*.jsonl.gz")):
        got = crab_windows(p, WINDOW_TICKS)
        real += got
        rgroups += [p.name] * len(got)
    bots, bgroups, bname = [], [], []
    for spec in args.bots:
        name, d = spec.split("=", 1)
        for p in sorted(Path(d).glob("bot-*.jsonl.gz")):
            got = crab_windows(p, WINDOW_TICKS)
            bots += got
            bgroups += [f"{name}:{p.name}"] * len(got)
            bname += [name] * len(got)
    print(f"{len(real)} Jason windows from {len(set(rgroups))} files; bots "
          + ", ".join(f"{n} {bname.count(n)}" for n in dict.fromkeys(bname)), flush=True)
    results = {"n_real": len(real), "n_bots": {n: bname.count(n) for n in dict.fromkeys(bname)}}

    # 1. progress per hour per window
    pct = lambda v: {q: round(float(np.percentile(v, q))) for q in (10, 25, 50, 75, 90)}
    prog = {"jason": pct([rate(s) for s in real])}
    for n in dict.fromkeys(bname):
        prog[n] = pct([rate(s) for s, b in zip(bots, bname) if b == n])
    jr = np.array([rate(s) for s in real])
    for n in dict.fromkeys(bname):
        br = np.array([rate(s) for s, b in zip(bots, bname) if b == n])
        prog[n]["share_inside_jason_p10_p90"] = round(float(np.mean((br >= np.percentile(jr, 10)) & (br <= np.percentile(jr, 90)))), 2)
    results["progress_per_h"] = prog
    print("progress/h percentiles:", json.dumps(prog), flush=True)

    # 2. Jason vs Jason: each day against his other days
    day = [g.split("-")[1] for g in rgroups]
    jj = {}
    for d in sorted(set(day)):
        a = [i for i, x in enumerate(day) if x == d]
        b = [i for i, x in enumerate(day) if x != d]
        if len(a) < 8 or len(b) < 8:
            continue
        r = cv_detect([real[i] for i in b], [rgroups[i] for i in b], [real[i] for i in a],
                      seed=args.seed, other_groups=[rgroups[i] for i in a])
        jj[d] = {k: round(v["auc"], 3) for k, v in r.items() if "auc" in v} | {"n_windows": len(a)}
        print(f"Jason {d} vs his other days: {jj[d]}", flush=True)
    results["jason_vs_jason"] = jj

    # 3. accounts
    sc = oof_scores(real, rgroups, bots, bgroups, args.seed)
    real_idx = np.arange(len(real))
    bot_idx = {n: len(real) + np.flatnonzero(np.array(bname) == n) for n in dict.fromkeys(bname)}
    win = {}
    for k, s in sc.items():
        win[k] = {n: round(float(roc_auc_score([0] * len(real) + [1] * len(i),
                                               np.concatenate([s[real_idx], s[i]]))), 3) for n, i in bot_idx.items()}
    results["window_auc_oof"] = win
    print("per-window AUC (one detector for all bots):", json.dumps(win), flush=True)
    acc = accounts(real_idx, bot_idx, args.sizes, args.accounts, rng)
    results["accounts"] = {k: account_table(acc, s) for k, s in sc.items()}
    for k, t in results["accounts"].items():
        print(f"[{k}] accounts: " + "  ".join(f"N={n}: {r['any_bot_auc']}" for n, r in t.items()), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
