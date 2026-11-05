"""Are window focus and tab clicks a tell? GBT AUC with and without them.

The real client tells the server when its window gains or loses focus, and RuneLite records which
side-panel tab each click opened (Skills, Friends List, ...). Server bots used to send neither: no
focus changes at all and every tab click written as "Skills". This scores Jason vs each set of bot
recordings with three feature sets:

* ``base``: the usual session features without the tab-click rate;
* ``+tabs``: plus tab clicks per hour, split by tab, and the spread over tabs (entropy);
* ``+tabs+focus``: plus the share of time focused and focus changes per hour.

    python run_focus_tells.py --recordings ~/.runelite/osrs-rl-recordings \\
        --bots gear=results/server-score-jasongear/rl_adv/recordings afk=results/server-afk/afk-gail-rl_adv/recordings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from osrs_rl.env import CHECK_TAB
from osrs_rl.metrics import FEATURE_NAMES, session_features
from osrs_rl.recordings import crab_windows
from run_real import N_FOLDS, WINDOW_TICKS, assign_folds, bootstrap_auc

TABS = ("Skills", "Friends List", "Inventory")
BASE_KEEP = [i for i, n in enumerate(FEATURE_NAMES) if n != "check_tab_per_h"]


def tab_features(s):
    h = s["ms"] / 3.6e6
    opts = [r.get("option", "") for r in s["log"] if r["intent"] == CHECK_TAB]
    counts = [sum(1 for o in opts if o == t) for t in TABS] + [sum(1 for o in opts if o not in TABS)]
    p = np.array(counts, float) / max(len(opts), 1)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return [c / h for c in counts] + [ent]


def focus_features(s):
    f = s.get("focus") or {"share": 1.0, "changes": 0}
    return [f["share"], f["changes"] / (s["ms"] / 3.6e6)]


def feats(s, kind):
    x = list(session_features(s)[BASE_KEEP])
    if kind in ("+tabs", "+tabs+focus"):
        x += tab_features(s)
    if kind == "+tabs+focus":
        x += focus_features(s)
    return x


def cv_auc(real, rgroups, bots, bgroups, kind, seed=0):
    rf, bf = assign_folds(rgroups, len(bots), bgroups, seed)
    X = np.array([feats(s, kind) for s in real + bots])
    y = np.array([0] * len(real) + [1] * len(bots))
    fold = np.concatenate([rf, bf])
    p = np.full(len(y), np.nan)
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        if not te.any() or len(set(y[tr])) < 2:
            continue
        w = np.where(y[tr] == 1, 0.5 / y[tr].mean(), 0.5 / (1 - y[tr].mean()))
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
        p[te] = m.fit(X[tr], y[tr], sample_weight=w).predict_proba(X[te])[:, 1]
    ok = ~np.isnan(p)
    return {"auc": round(float(roc_auc_score(y[ok], p[ok])), 3),
            "auc_90ci": [round(v, 3) for v in bootstrap_auc(list(y[ok]), list(p[ok]))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--bots", nargs="+", required=True, help="name=dir of bot-*.jsonl.gz")
    ap.add_argument("--out", default="results/focus-tells")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    real, rg = [], []
    for p in sorted(Path(args.recordings).expanduser().glob("session-*.jsonl.gz")):
        got = crab_windows(p, WINDOW_TICKS)
        real += got
        rg += [p.name] * len(got)
    describe = lambda ws: {"focus_share_median": round(float(np.median([focus_features(s)[0] for s in ws])), 3),
                           "focus_changes_per_h_median": round(float(np.median([focus_features(s)[1] for s in ws])), 1),
                           "tabs_per_h_by_tab": dict(zip(list(TABS) + ["other"], np.round(np.mean([tab_features(s)[:4] for s in ws], 0), 1).tolist()))}
    results = {"jason": describe(real), "bots": {}}
    print("jason:", results["jason"], flush=True)
    for spec in args.bots:
        name, d = spec.split("=", 1)
        bots, bg = [], []
        for p in sorted(Path(d).glob("bot-*.jsonl.gz")):
            got = crab_windows(p, WINDOW_TICKS)
            bots += got
            bg += [f"{name}:{p.name}"] * len(got)
        row = {"n": len(bots), "describe": describe(bots)}
        for kind in ("base", "+tabs", "+tabs+focus"):
            row[kind] = cv_auc(real, rg, bots, bg, kind, args.seed)
        results["bots"][name] = row
        print(f"{name}: n={len(bots)} " + "  ".join(f"{k} {row[k]['auc']}" for k in ("base", "+tabs", "+tabs+focus"))
              + f"  {row['describe']}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
