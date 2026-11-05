"""Format-leak check: can the detectors tell real play from bots by *how the file was written*?

Real sessions come from the RuneLite recorder; server bots are written by
``rsmod_bridge.RecordingWriter``. They differ in things that have nothing to do with behaviour
(tick timestamps on an exact 600 ms grid, click target text, which NPCs are listed, XP record
cadence, ...). If a detector reads those, a low or high AUC says nothing about humanness.

This script re-saves each real recording through ``RecordingWriter``: the same ticks, clicks,
camera moves, XP and messages, rebuilt as the server-side state the bridge would have sent, with
each click placed at its real offset from the tick before it. Then it scores:

1. original real vs re-saved real. About 0.5 means no format leak.
2. re-saved real vs each set of server-bot recordings, next to original real vs the same bots.
3. each detector's top features. For the GBT detector that's permutation importance on held-out
   folds; for the GRU it's the AUC drop when one token channel is shuffled across the test fold.

    python run_leak_check.py --recordings ~/.runelite/osrs-rl-recordings \\
        --bots results/server-score-gail/rl_adv/recordings results/server-score/scripted/recordings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from osrs_rl.detector import FeatureDetector, SequenceDetector, _pad
from osrs_rl.env import INTENTS
from osrs_rl.metrics import FEATURE_NAMES, session_features, session_tokens
from osrs_rl.recordings import IGNORED_OPTIONS, _kind, crab_windows, load
from osrs_rl.rsmod_bridge import RecordingWriter
from osrs_rl.sim.base import KINDS
from run_real import N_FOLDS, WINDOW_TICKS, assign_folds, cv_detect

CHECK_TAB = INTENTS.index("check_tab")


def resave(src: Path, dst: Path) -> None:
    """Rewrite one RuneLite recording as RecordingWriter would have written the same play."""
    recs = list(load(src))
    ticks = [r for r in recs if r["type"] == "tick"]
    if not ticks:
        return
    w = RecordingWriter(dst, t0_ms=int(ticks[0]["t"]))
    xp, levels, msgs = {}, {}, []
    last_tick_t, s = None, None
    for r in recs:
        typ = r["type"]
        if typ == "xp":
            xp[r["skill"].lower()] = r["xp"]
            levels[r["skill"].lower()] = r.get("level", 1)
        elif typ == "game_message":
            msgs.append(r.get("msg", ""))
        elif typ == "tick":
            x, y = r.get("x", 0), r.get("y", 0)
            crab, shells = None, []
            for n in r.get("npcs", []) or []:
                name = (n.get("name") or "").lower()
                if name == "gemstone crab":
                    crab = {"visible": True, "slot": n.get("index", 0), "x": x + n["dx"], "z": y + n["dy"]}
                elif "gemstone crab" in name and "shell" in name:
                    shells.append({"visible": True, "slot": n.get("index", 0), "x": x + n["dx"], "z": y + n["dy"]})
            inter = r.get("interacting")
            iname = (inter.get("name") if isinstance(inter, dict) else inter) or ""
            attacking = iname.lower() == "gemstone crab"
            s = {"cycle": r.get("tick", 0), "x": x, "z": y, "level": r.get("plane", 0),
                 "crab": crab, "shells": shells, "attacking": attacking,
                 "interaction": ({"kind": iname} if inter and not attacking else None),
                 "run_energy": r.get("run", 100), "inv": r.get("inv", []),
                 "players": [{"slot": p.get("pid", 0), "x": x + p.get("dx", 0), "z": y + p.get("dy", 0)}
                             for p in r.get("players", []) or []],
                 "messages": msgs, "xp": dict(xp), "levels": dict(levels)}
            w.tick(s, camera_moved=bool(r.get("camera_moved")))
            msgs, last_tick_t = [], r["t"]
        elif typ == "menu_click" and s is not None:
            opt = (r.get("option") or "").lower()
            tgt = r.get("target") or ""
            if opt in IGNORED_OPTIONS:
                continue
            # Same offset from the last tick as in the real file, on the writer's own tick clock.
            w.t_ms = w.n_ticks * w.tick_ms + max(0, r["t"] - (last_tick_t or r["t"]))
            if opt == "walk here":
                w.click("walk", s)
            elif not tgt:
                w.input(CHECK_TAB)
            else:
                kind = _kind(tgt, opt)
                if kind in ("crab", "shell", "cave"):
                    w.click(kind, s)
                # deposit / close / other targets: no server-side equivalent at the crab
    w.close()


def windows_from(paths):
    ws, groups = [], []
    for p in paths:
        got = crab_windows(p, WINDOW_TICKS)
        ws += got
        groups += [p.name] * len(got)
    return ws, groups


# ----------------------------------------------------------------------
TOKEN_CHANNELS = {"gap": [0], "since_event": [1], "change_event": [2], "position_in_window": [3]}
_b = 4
TOKEN_CHANNELS["intent"] = list(range(_b, _b + len(INTENTS)))
_b += len(INTENTS)
TOKEN_CHANNELS["kind"] = list(range(_b, _b + len(KINDS)))
_b += len(KINDS)
TOKEN_CHANNELS.update({"redundant": [_b], "valid": [_b + 1], "rank": [_b + 2]})


def top_features(real, groups, other, seed=0, top=8, other_groups=None):
    """Fold-averaged importances for both detectors (real = 0, other = 1)."""
    rf, of = assign_folds(groups, len(other), other_groups, seed)
    gbt = np.zeros(len(FEATURE_NAMES))
    gru = {k: [] for k in TOKEN_CHANNELS}
    n_f = 0
    for k in range(N_FOLDS):
        tr = [s for s, f in zip(real, rf) if f != k] + [s for s, f in zip(other, of) if f != k]
        ytr = [0] * int((rf != k).sum()) + [1] * int((of != k).sum())
        te = [s for s, f in zip(real, rf) if f == k] + [s for s, f in zip(other, of) if f == k]
        yte = np.array([0] * int((rf == k).sum()) + [1] * int((of == k).sum()))
        if len(set(yte)) < 2:
            continue
        n_f += 1
        fd = FeatureDetector().fit(tr, ytr)
        Xte = np.stack([session_features(s) for s in te])
        pi = permutation_importance(fd.m, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=seed)
        gbt += pi.importances_mean
        sd = SequenceDetector(seed=seed).fit(tr, ytr)
        x, L = _pad([session_tokens(s) for s in te])
        with torch.no_grad():
            base = roc_auc_score(yte, torch.sigmoid(sd.m(x, L)).numpy())
            g = torch.Generator().manual_seed(seed)
            for name, ch in TOKEN_CHANNELS.items():
                drops = []
                for _ in range(5):
                    xp = x.clone()
                    # shuffle this channel's values across all (session, step) cells of the fold
                    flat = xp[:, :, ch].reshape(-1, len(ch))
                    xp[:, :, ch] = flat[torch.randperm(len(flat), generator=g)].reshape(xp.shape[0], xp.shape[1], len(ch))
                    drops.append(base - roc_auc_score(yte, torch.sigmoid(sd.m(xp, L)).numpy()))
                gru[name].append(float(np.mean(drops)))
    gbt /= max(n_f, 1)
    order = np.argsort(-gbt)[:top]
    return {
        "feature_gbt": [{"feature": FEATURE_NAMES[i], "auc_drop": round(float(gbt[i]), 4)} for i in order],
        "sequence_gru": sorted(({"channel": k, "auc_drop": round(float(np.mean(v)), 4)} for k, v in gru.items() if v),
                               key=lambda d: -d["auc_drop"])[:top],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--bots", nargs="+", required=True, help="directories of server-bot recordings (bot-*.jsonl.gz)")
    ap.add_argument("--out", default="results/leak-check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    resaved_dir = out / "resaved"
    srcs = sorted(Path(args.recordings).expanduser().glob("session-*.jsonl.gz"))
    for p in srcs:
        resave(p, resaved_dir / p.name)
    real, groups = windows_from(srcs)
    res, rgroups = windows_from(sorted(resaved_dir.glob("session-*.jsonl.gz")))
    print(f"{len(real)} real windows, {len(res)} re-saved windows from {len(srcs)} files", flush=True)
    results = {"n_real": len(real), "n_resaved": len(res)}

    # Each re-saved file is tested in the same fold as its original, so a test window's near twin is
    # never in training with the other label (that pushed the AUC well below 0.5).
    results["real_vs_resaved"] = cv_detect(real, groups, res, seed=args.seed, other_groups=rgroups)
    results["real_vs_resaved_top"] = top_features(real, groups, res, seed=args.seed, other_groups=rgroups)
    print("original vs re-saved real:", json.dumps({k: round(v["auc"], 3) for k, v in results["real_vs_resaved"].items()
                                                     if "auc" in v}), flush=True)
    results["bots"] = {}
    for d in args.bots:
        d = Path(d)
        bots, bgroups = [], []
        for p in sorted(d.glob("bot-*.jsonl.gz")):
            got = crab_windows(p, WINDOW_TICKS)
            bots += got
            bgroups += [f"bot:{p.name}"] * len(got)  # both windows of an episode go to one fold
        name = f"{d.parent.name}/{d.parent.parent.name}" if d.name == "recordings" else d.name
        a = cv_detect(real, groups, bots, seed=args.seed, other_groups=bgroups)
        b = cv_detect(res, rgroups, bots, seed=args.seed, other_groups=bgroups)
        results["bots"][name] = {
            "n_windows": len(bots),
            "original_real_vs_bot": a,
            "resaved_real_vs_bot": b,
            "top_features_original": top_features(real, groups, bots, seed=args.seed, other_groups=bgroups),
            "top_features_resaved": top_features(res, rgroups, bots, seed=args.seed, other_groups=bgroups),
        }
        print(f"{name}: original {a['feature_gbt']['auc']:.3f}/{a['sequence_gru']['auc']:.3f}  "
              f"re-saved {b['feature_gbt']['auc']:.3f}/{b['sequence_gru']['auc']:.3f} (gbt/gru)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
