"""Data-efficiency curve: how much of Jason's labelled play does behaviour cloning need?

    python run_efficiency.py --recordings ~/.runelite/osrs-rl-recordings --out results/efficiency

Trains BC on a fraction of the recording *files* (100 / 50 / 25 / 10 / 5 %), rolls the clone out
in the simulator, and scores it against all of Jason's 15-minute crab windows with the same
by-file 4-fold detectors as ``run_real.py``. Each fraction below 100 % is repeated on a few
random file subsets so the spread is visible. Two curves:

* ``real``: BC learns from Jason's decisions only.
* ``mixed``: the synthetic players plus Jason's decisions upweighted to about half the data
  (the setup that scored best in docs/real-data.md); fraction 0 is the synthetic-only floor.

No PPO: the question is what the labelled data buys the prior, and BC is the cheap, repeatable
part. Besides the usual AUC on every real window (which includes the files BC trained on), each
run also reports the AUC on windows from files BC never saw, so the curve is not flattered by
memorised play. One mid-sized file is always held out of training for the validation NLL, as in
``run_real.py``, so that held-out AUC exists even at 100 %.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from osrs_rl.agents.controllers import SyntheticHuman
from osrs_rl.detector import FeatureDetector, SequenceDetector
from osrs_rl.metrics import timescale_metrics
from osrs_rl.real_bc import recording_transitions
from osrs_rl.recordings import crab_windows
from osrs_rl.train import PolicyController, train_bc
from run_real import (N_BOT_EPISODES, N_FOLDS, SESSIONS_PER_PLAYER, TRAIN_PLAYERS, WINDOW_TICKS, bootstrap_auc,
                      describe, gen)


def cv_detect_detail(real, real_groups, other, seed=0):
    """``run_real.cv_detect`` but it also returns the pooled predictions and which file each real
    window came from, so AUC can be recomputed on a subset of files."""
    rng = np.random.default_rng(seed)
    files = sorted(set(real_groups))
    rng.shuffle(files)
    fold_of_file = {f: i % N_FOLDS for i, f in enumerate(files)}
    real_fold = np.array([fold_of_file[g] for g in real_groups])
    other_fold = rng.permutation(np.arange(len(other)) % N_FOLDS)
    preds = {"feature_gbt": [], "sequence_gru": []}
    ys, src = [], []
    for k in range(N_FOLDS):
        tr = [s for s, f in zip(real, real_fold) if f != k] + [s for s, f in zip(other, other_fold) if f != k]
        ytr = [0] * int(np.sum(real_fold != k)) + [1] * int(np.sum(other_fold != k))
        te = [s for s, f in zip(real, real_fold) if f == k] + [s for s, f in zip(other, other_fold) if f == k]
        yte = [0] * int(np.sum(real_fold == k)) + [1] * int(np.sum(other_fold == k))
        if not any(y == 0 for y in yte):
            continue
        ys += yte
        src += [g for g, f in zip(real_groups, real_fold) if f == k] + [None] * int(np.sum(other_fold == k))
        preds["feature_gbt"] += list(FeatureDetector().fit(tr, ytr).predict(te))
        preds["sequence_gru"] += list(SequenceDetector(seed=seed).fit(tr, ytr).predict(te))
    return np.array(ys), {k: np.array(v) for k, v in preds.items()}, src


def score(ys, preds, keep):
    """Worst-case (best detector) AUC with a bootstrap 90% interval over the windows in ``keep``."""
    out = {}
    for name, p in preds.items():
        y, q = ys[keep], p[keep]
        if y.min() == y.max():
            return None
        out[name] = {"auc": float(roc_auc_score(y, q)), "auc_90ci": bootstrap_auc(y, q)}
    best = max(out, key=lambda k: out[k]["auc"])
    out["worst_case"] = {"name": best, **out[best]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--out", default="results/efficiency")
    ap.add_argument("--fractions", default="1,0.5,0.25,0.1,0.05")
    ap.add_argument("--subsets", type=int, default=3, help="random file subsets per fraction below 100%")
    ap.add_argument("--modes", default="real,mixed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
    fractions = [float(x) for x in args.fractions.split(",")]
    modes = args.modes.split(",")

    real, groups, real_tr = [], [], {}
    for p in sorted(Path(args.recordings).expanduser().glob("session-*.jsonl.gz")):
        w = crab_windows(p, WINDOW_TICKS)
        real += w
        groups += [p.name] * len(w)
        tr = recording_transitions(p)
        if tr:
            real_tr[p.name] = tr
        log(f"{p.name}: {len(w)} crab windows, {len(tr)} decisions ({sum(a['intent'] != 0 for _, a in tr)} inputs)")
    log(f"real: {len(real)} windows of 15 min from {len(set(groups))} files; decisions from {len(real_tr)} files")
    if len(set(groups)) < N_FOLDS:
        raise SystemExit("need crab windows from at least 4 recording files")
    files = sorted(real_tr, key=lambda f: len(real_tr[f]))
    val_file = files[len(files) // 2]  # same choice as run_real.py
    train_files = [f for f in files if f != val_file]
    r_val = real_tr[val_file]
    log(f"validation file {val_file} ({len(r_val)} decisions); {len(train_files)} files available for training")

    synth_tr = []
    if "mixed" in modes:
        for p in range(TRAIN_PLAYERS - 1):
            seeds = [100_000 * (p + 1) + k for k in range(SESSIONS_PER_PLAYER)]
            synth_tr += gen(lambda sd, p=p: SyntheticHuman(p, sd), seeds, record=True)[1]
        log(f"synthetic bc data: {len(synth_tr)} decisions from {TRAIN_PLAYERS - 1} players")

    runs = []
    if (out / "results.json").exists():  # resume: keep finished runs on the same recordings
        prev = json.loads((out / "results.json").read_text())
        if prev.get("recordings") == sorted(set(groups)) and prev.get("val_file") == val_file:
            runs = prev["runs"]
            log(f"resuming: {len(runs)} runs already in {out / 'results.json'}")
    done = {(r["mode"], r["fraction"], r["subset"]) for r in runs}
    for mode in modes:
        fr = list(fractions) + ([0.0] if mode == "mixed" else [])
        for frac in fr:
            n_sub = 1 if frac in (1.0, 0.0) else args.subsets
            k = 0 if frac == 0.0 else max(1, int(round(frac * len(train_files))))
            for sub in range(n_sub):
                if (mode, frac, sub) in done:
                    continue
                rng = np.random.default_rng(1000 * args.seed + 100 * sub + int(frac * 1000))
                chosen = sorted(rng.choice(train_files, k, replace=False).tolist()) if 0 < k < len(train_files) else (train_files if k else [])
                r_train = [t for f in chosen for t in real_tr[f]]
                if mode == "real":
                    tr_train = r_train
                else:
                    w = max(1, len(synth_tr) // max(len(r_train), 1)) if r_train else 0
                    tr_train = synth_tr + r_train * w
                n_int = sum(a["intent"] != 0 for _, a in r_train)
                tag = f"{mode} {frac:.2f} #{sub}"
                log(f"{tag}: {len(chosen)} files, {len(r_train)} real decisions ({n_int} inputs), {len(tr_train)} total")
                bc, hist = train_bc(tr_train, r_val, seed=args.seed + sub, log=lambda m: None)
                torch.manual_seed(9_100_000 + sub)
                wins = gen(lambda sd: PolicyController(bc), range(9_100_000 + 1000 * sub, 9_100_000 + 1000 * sub + N_BOT_EPISODES))[0]
                ys, preds, src = cv_detect_detail(real, groups, wins, seed=args.seed)
                keep_all = np.ones(len(ys), bool)
                keep_held = np.array([s is None or s not in chosen for s in src])
                det_all, det_held = score(ys, preds, keep_all), score(ys, preds, keep_held)
                ts = timescale_metrics(wins, real)
                row = {"mode": mode, "fraction": frac, "subset": sub, "files": chosen, "n_real_decisions": len(r_train),
                       "n_real_inputs": n_int, "n_train_decisions": len(tr_train),
                       "val_nll": hist[-1]["val_nll"], "val_intent_acc": hist[-1]["val_intent_acc"],
                       "describe": describe(wins), "detectors_all": det_all, "detectors_heldout_files": det_held,
                       "n_heldout_real_windows": int(sum(1 for s in src if s is not None and s not in chosen)),
                       "reaction_ks": ts["reaction_ks"], "gap_ks": ts["gap_ks"]}
                runs.append(row)
                wa, wh = det_all["worst_case"], (det_held or {}).get("worst_case")
                log(f"{tag}: prog/h {row['describe']['progress_per_h_median']:7.0f}  val_nll {row['val_nll']:.3f}  "
                    f"worst AUC all {wa['auc']:.2f} [{wa['auc_90ci'][0]:.2f}, {wa['auc_90ci'][1]:.2f}]"
                    + (f"  held-out files {wh['auc']:.2f} [{wh['auc_90ci'][0]:.2f}, {wh['auc_90ci'][1]:.2f}] "
                       f"({row['n_heldout_real_windows']} windows)" if wh else "")
                    + f"  reaction KS {ts['reaction_ks']:.2f}")
                (out / "results.json").write_text(json.dumps({
                    "recordings": sorted(set(groups)), "val_file": val_file, "fractions": fractions, "subsets": args.subsets,
                    "real": describe(real), "runs": runs, "wall_s": time.time() - t0}, indent=1))

    # Summary table: mean over subsets.
    log("summary (mean over subsets):")
    log(f"{'mode':6s} {'frac':>5s} {'files':>5s} {'inputs':>6s} {'prog/h':>7s} {'val_nll':>7s} {'AUC all':>8s} {'AUC held':>8s} {'KS':>5s}")
    for mode in modes:
        for frac in sorted({r["fraction"] for r in runs if r["mode"] == mode}, reverse=True):
            rs = [r for r in runs if r["mode"] == mode and r["fraction"] == frac]
            m = lambda f: float(np.mean([f(r) for r in rs]))
            held = [r["detectors_heldout_files"]["worst_case"]["auc"] for r in rs if r["detectors_heldout_files"]]
            log(f"{mode:6s} {frac:5.2f} {m(lambda r: len(r['files'])):5.1f} {m(lambda r: r['n_real_inputs']):6.0f} "
                f"{m(lambda r: r['describe']['progress_per_h_median']):7.0f} {m(lambda r: r['val_nll']):7.3f} "
                f"{m(lambda r: r['detectors_all']['worst_case']['auc']):8.2f} "
                f"{(np.mean(held) if held else float('nan')):8.2f} {m(lambda r: r['reaction_ks']):5.2f}")
    log(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
