"""Learn from the other players at the crab: an inverse-dynamics model over their visible state.

    python -m osrs_rl.bystanders ~/.runelite/osrs-rl-recordings/session-*.jsonl.gz

prints the IDM's accuracy on Jason (where the true clicks are known), how much bystander play it
recovers, and how many bystanders the bot filter drops. Nothing identifying is used: bystanders
are the recorder's per-session anonymous ``pid`` (schema 4+), and no names are ever read.

Why
---
About 23 other players stand at the crab with Jason at any time, roughly 20x his own hours.
Their inputs are not logged, but their consequences are: a player who clicks Attack starts
interacting with the crab a tick or two later, one who clicks Walk here starts moving, one who
clicks the cave walks to it and vanishes. Reading inputs back from those state changes is an
inverse-dynamics model (IDM). Only some intents are visible this way (attack, shell, cave, walk;
never tab checks or camera moves, and never a re-click on a crab you're already attacking), so
everything compared against bystanders goes through ``observable``, which keeps exactly those.

The IDM
-------
Rules, with one fitted parameter per intent: the lag from the click to the visible change. Each
rule is checked on Jason's own state, where the true click is known (``evaluate_idm``), and the
lag is the median over his files. A learned classifier would have almost nothing to learn: with
a tick snapshot the visible change is near-deterministic, and the accuracy table says how close
the rules get.

* ``INTERACT crab|shell``: the player's target becomes the crab (shell) NPC.
* ``INTERACT cave``: the player walks to a cave and leaves (a bystander vanishes within a few
  tiles of it; Jason's tile jumps to another mine).
* ``WALK``: the player starts moving without a new target and doesn't end at a cave.

Bot filter
----------
Bystanders include bots. Two filters, both reported:

* Gear farms: the exact same visible gear (all nine slots) and combat level on several
  different pids, across more than one recording, is a farm signature people rarely share.
* Detector: a feature detector trained on Jason vs the scripted bot (both through
  ``observable``) scores each bystander window; windows it's sure are bots are dropped.

Damage per player isn't visible (hitsplats show who took a hit, not who dealt it), so attack
cadence from their animation stands in for DPS in the gear summary.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict

import numpy as np

from .env import CAMERA, CHECK_TAB, INTERACT, WALK
from .recordings import IGNORED_OPTIONS, in_crab_area, load

TICK_MS = 600
WINDOW_MS = 15 * 60 * 1000
CAVE_NEAR = 4  # tiles: a player who vanishes this close to a cave went through it
JUMP = 10  # tiles in one tick: came out of a cave at another mine
OBSERVABLE = (INTERACT, WALK)
# Click -> visible change, ms. Overwritten by fit_lags() on Jason's files; these are the defaults
# for a 600 ms tick (the server applies a click on the next tick, the client sees it a tick later).
LAGS = {"interact": 900.0, "move": 900.0}


def _name(n):
    import re
    return re.sub(r"<[^>]+>", "", n or "").lower()


def _npc_kind(name):
    nm = _name(name)
    if "gemstone crab" in nm:
        return "shell" if "shell" in nm else "crab"
    return None


# ----------------------------------------------------------------------
# Tracks: one row per tick per player, in world tiles
def _scene(tick, cave_ids):
    """Per-tick world facts: npc index -> kind, cave tiles, whether a crab/shell is up."""
    kinds = {n.get("index"): _npc_kind(n.get("name")) for n in tick.get("npcs", [])}
    caves = [(tick["x"] + o["dx"] + (o.get("w", 1) - 1) / 2, tick["y"] + o["dy"] + (o.get("h", 1) - 1) / 2)
             for o in tick.get("objects", []) if o.get("id") in cave_ids]
    return kinds, caves


def read_tracks(path):
    """Jason's track, bystander tracks keyed by pid, per-tick scene info, Jason's true clicks.

    A track row is ``(t_ms, tick, x, y, anim, target_kind)``; ``target_kind`` is "crab", "shell",
    another string for other targets, or None.
    """
    recs = list(load(path))
    cave_ids = {r.get("id") for r in recs if r["type"] == "menu_click"
                and (r.get("option") or "").lower() == "crawl-through"}
    me, others, scene, clicks = [], defaultdict(list), [], []
    equip, combat = {}, {}
    in_area = False
    for r in recs:
        typ = r["type"]
        if typ == "tick":
            in_area = in_crab_area(r)
            if not in_area:
                continue
            kinds, caves = _scene(r, cave_ids)
            scene.append((r["t"], r.get("tick"), kinds, caves, r["x"], r["y"]))
            inter = r.get("interacting")
            tk = _npc_kind(inter.get("name")) if isinstance(inter, dict) and inter.get("name") else None
            if isinstance(inter, dict) and tk is None:
                tk = "other"
            me.append((r["t"], r.get("tick"), r["x"], r["y"], r.get("anim", -1), tk))
            for p in r.get("players", []):
                pid = p.get("pid")
                if pid is None:
                    continue  # schema < 4: no stable id
                tn = p.get("target_npc")
                tk = kinds.get(tn) if tn is not None else ("other" if p.get("target_player") is not None else None)
                if tn is not None and tk is None:
                    tk = "other"
                others[pid].append((r["t"], r.get("tick"), r["x"] + p["dx"], r["y"] + p["dy"], p.get("anim", -1), tk))
                if "equip" in p:
                    equip[pid] = tuple(p["equip"])
                combat[pid] = p.get("combat")
        elif typ == "menu_click" and in_area:
            opt = (r.get("option") or "").strip().lower()
            tgt = _name(r.get("target"))
            if opt in IGNORED_OPTIONS:
                continue
            if opt == "walk here":
                clicks.append((r["t"], WALK, ""))
            elif "gemstone crab" in tgt:
                clicks.append((r["t"], INTERACT, "shell" if "shell" in tgt else "crab"))
            elif opt == "crawl-through" or tgt == "cave":
                clicks.append((r["t"], INTERACT, "cave"))
            elif not tgt:
                clicks.append((r["t"], CHECK_TAB, ""))
    return {"me": me, "others": dict(others), "scene": scene, "clicks": clicks, "equip": equip,
            "combat": combat, "file": getattr(path, "name", str(path))}


def _segments(track, max_gap_ticks=3):
    """Split a track where the player was out of view for more than a few ticks."""
    seg = []
    for row in track:
        if seg and row[1] is not None and seg[-1][1] is not None and row[1] - seg[-1][1] > max_gap_ticks:
            yield seg
            seg = []
        seg.append(row)
    if seg:
        yield seg


def _near_cave(x, y, caves):
    return any(max(abs(x - cx), abs(y - cy)) <= CAVE_NEAR for cx, cy in caves)


# ----------------------------------------------------------------------
# The IDM
def infer_inputs(seg, caves_at, vanished_at_end: bool, lags=None):
    """Infer (t_ms, intent, kind) inputs from one track segment.

    ``caves_at(t)`` gives the cave tiles in view at time t. ``vanished_at_end``: the segment ends
    because the player left the scene (not because the recording or the crab area ended).
    Times are the visible change minus the fitted lag, i.e. the estimated click time.
    """
    lags = lags or LAGS
    out = []
    n = len(seg)
    i = 1
    while i < n:
        t, _, x, y, _, tk = seg[i]
        _, _, px, py, _, ptk = seg[i - 1]
        step = max(abs(x - px), abs(y - py))
        if tk in ("crab", "shell") and ptk != tk:
            out.append((t - lags["interact"], INTERACT, tk))
            # the approach walk that follows belongs to this click
            while i + 1 < n and seg[i + 1][2:4] != seg[i][2:4] and step <= JUMP:
                i += 1
            i += 1
            continue
        if 0 < step <= JUMP:
            j = i
            while j + 1 < n and 0 < max(abs(seg[j + 1][2] - seg[j][2]), abs(seg[j + 1][3] - seg[j][3])) <= JUMP \
                    and seg[j + 1][5] == tk:
                j += 1
            end_x, end_y = seg[j][2], seg[j][3]
            # Crawling takes a few ticks at the cave before the player reappears at another mine
            # (Jason) or leaves the scene (a bystander).
            k = j + 1
            while k < n and k <= j + 6 and seg[k][2:4] == (end_x, end_y):
                k += 1
            jumped = k < n and k <= j + 6 and max(abs(seg[k][2] - end_x), abs(seg[k][3] - end_y)) > JUMP
            gone = k >= n and vanished_at_end
            new_target = j + 1 < n and seg[j + 1][5] in ("crab", "shell") and seg[j + 1][5] != tk
            if (jumped or gone) and _near_cave(end_x, end_y, caves_at(seg[j][0])):
                out.append((t - lags["move"], INTERACT, "cave"))
                i = k + 1 if jumped else n
                continue
            elif not new_target:
                out.append((t - lags["move"], WALK, ""))
            i = j + 1
            continue
        if step > JUMP and _near_cave(px, py, caves_at(seg[i - 1][0])):
            # a cave hop with no visible walk (already standing at the cave)
            out.append((seg[i - 1][0] - lags["move"], INTERACT, "cave"))
        i += 1
    return out


def _caves_lookup(scene):
    ts = np.array([s[0] for s in scene]) if scene else np.zeros(0)

    def caves_at(t):
        if not len(ts):
            return []
        k = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
        return scene[k][3]
    return caves_at


def evaluate_idm(tracks, lags=None, tol_ms=1800.0):
    """Run the IDM on Jason's own track and match it against his real clicks.

    Returns per-intent counts and the raw lags (visible change - click) of matched pairs, which
    ``fit_lags`` turns into the lag table. Clicks the IDM can't see by construction (a re-click on
    the crab he is already attacking) are counted separately.
    """
    zero = {"interact": 0.0, "move": 0.0}
    caves_at = _caves_lookup(tracks["scene"])
    inferred = []
    for seg in _segments(tracks["me"]):
        inferred += infer_inputs(seg, caves_at, vanished_at_end=False, lags=zero)
    truth = [c for c in tracks["clicks"] if c[1] in OBSERVABLE]
    # A crab click while already attacking leaves no trace.
    me_t = np.array([r[0] for r in tracks["me"]]) if tracks["me"] else np.zeros(0)

    def attacking_at(t):
        if not len(me_t):
            return False
        k = int(np.searchsorted(me_t, t)) - 1
        return k >= 0 and tracks["me"][k][5] == "crab"
    res = defaultdict(lambda: {"truth": 0, "invisible": 0, "inferred": 0, "matched": 0, "lags": []})
    used = set()
    for t, it, kind in truth:
        key = "walk" if it == WALK else kind
        if it == INTERACT and kind == "crab" and attacking_at(t):
            res[key]["invisible"] += 1
            continue
        res[key]["truth"] += 1
        best = None
        for j, (ti, ii, ki) in enumerate(inferred):
            if j in used or ii != it or ki != kind:
                continue
            lag = ti - t
            if -300 <= lag <= tol_ms + 1200 and (best is None or abs(lag) < abs(best[1])):
                best = (j, lag)
        if best is not None:
            used.add(best[0])
            res[key]["matched"] += 1
            res[key]["lags"].append(best[1])
    for ti, ii, ki in inferred:
        res["walk" if ii == WALK else ki]["inferred"] += 1
    return dict(res)


def fit_lags(evals) -> dict:
    inter = [l for e in evals for k in ("crab", "shell") for l in e.get(k, {}).get("lags", [])]
    move = [l for e in evals for k in ("walk", "cave") for l in e.get(k, {}).get("lags", [])]
    return {"interact": float(np.median(inter)) if inter else LAGS["interact"],
            "move": float(np.median(move)) if move else LAGS["move"]}


# ----------------------------------------------------------------------
# Bystander sessions
def _events(scene):
    """Shared world events at Jason's mine: (t_ms, event)."""
    out, prev = [], None
    for t, _, kinds, _, x, y in scene:
        crab = "crab" in kinds.values()
        shell = "shell" in kinds.values()
        if prev is not None:
            if prev[0] and not crab:
                out.append((t, "target_gone"))
            if shell and not prev[1]:
                out.append((t, "shell_ready"))
        prev = (crab, shell)
    return out


def _rows(inputs, events, t0, start_event="start"):
    """Turn inferred inputs into env-format log rows, measured from t0."""
    rows, ev_i = [], 0
    last_event, last_event_t, last_t = start_event, t0, t0
    for t, intent, kind in sorted(inputs):
        while ev_i < len(events) and events[ev_i][0] <= t:
            if events[ev_i][0] >= t0:
                last_event, last_event_t = events[ev_i][1], events[ev_i][0]
            ev_i += 1
        rows.append({"t_ms": t - t0, "intent": intent, "kind": kind, "uid": -1, "valid": True,
                     "trigger": last_event, "since_event_ms": t - last_event_t, "last_event": last_event,
                     "gap_ms": t - last_t, "redundant": False, "rank": 0 if kind in ("crab", "shell", "cave") else -1})
        last_t = t
    return rows


def bystander_windows(tracks, lags=None, min_ms=WINDOW_MS, attack_rate_per_ms=None):
    """15-minute windows of bystander play in the env's session format (observable intents only).

    ``progress`` is not visible for bystanders; it is set from time spent attacking times
    ``attack_rate_per_ms`` (Jason's rate) when given, else 0. ``observable`` zeroes it for
    everyone anyway.
    """
    lags = lags or LAGS
    caves_at = _caves_lookup(tracks["scene"])
    events = _events(tracks["scene"])
    end_t = tracks["scene"][-1][0] if tracks["scene"] else 0
    out = []
    for pid, track in tracks["others"].items():
        for seg in _segments(track):
            t_first, t_last = seg[0][0], seg[-1][0]
            if t_last - t_first < min_ms:
                continue
            vanished = t_last < end_t - 3 * TICK_MS
            arrived = _near_cave(seg[0][2], seg[0][3], caves_at(t_first))
            inputs = infer_inputs(seg, caves_at, vanished, lags)
            n_win = int((t_last - t_first) // min_ms)
            for k in range(n_win):
                lo, hi = t_first + k * min_ms, t_first + (k + 1) * min_ms
                win_in = [x for x in inputs if lo <= x[0] < hi]
                rows = _rows(win_in, [e for e in events if e[0] < hi], lo,
                             start_event="arrived" if (k == 0 and arrived) else "start")
                att = sum(1 for r in seg if lo <= r[0] < hi and r[5] == "crab")
                prog = att * TICK_MS * attack_rate_per_ms if attack_rate_per_ms else 0.0
                anims = [r[4] for r in seg if lo <= r[0] < hi and r[4] not in (-1, None)]
                out.append({"log": rows, "progress": float(prog), "ms": float(min_ms), "activity": "bystander",
                            "source": f"{tracks['file']}#pid{pid}#{k}", "pid": pid, "file": tracks["file"],
                            "equip": tracks["equip"].get(pid), "combat": tracks["combat"].get(pid),
                            "attack_anim": Counter(anims).most_common(1)[0][0] if anims else -1})
    return out


def observable(sess, tick_ms: int = TICK_MS) -> dict:
    """What a bystander's IDM could see of a session: attack/shell/cave/walk inputs only, no
    re-clicks on a target already being attacked, times on the tick grid, no progress.

    Everything compared with bystanders (Jason, agents) goes through this, so a detector can't
    separate them by things the IDM can't see.
    """
    rows, last_t = [], 0.0
    for r in sess["log"]:
        if r["intent"] not in OBSERVABLE or r.get("redundant") or not r.get("valid", True):
            continue
        t = float(np.floor(r["t_ms"] / tick_ms) * tick_ms)
        since = float(np.floor(r["since_event_ms"] / tick_ms) * tick_ms)
        rows.append({**r, "t_ms": t, "since_event_ms": since, "gap_ms": t - last_t,
                     "rank": 0 if r.get("kind") in ("crab", "shell", "cave") else r.get("rank", -1)})
        last_t = t
    return {**sess, "log": rows, "progress": 0.0}


# ----------------------------------------------------------------------
# Bot filter
def gear_farms(windows, min_pids: int = 4) -> set:
    """Gear signatures (all nine visible slots + combat level) worn by many different pids in more
    than one recording: a farm signature. Returns the set of (equip, combat) keys flagged."""
    pids, files = defaultdict(set), defaultdict(set)
    for w in windows:
        if not w.get("equip") or not any(w["equip"]):
            continue
        key = (tuple(w["equip"]), w.get("combat"))
        pids[key].add((w["file"], w["pid"]))
        files[key].add(w["file"])
    return {k for k in pids if len(pids[k]) >= min_pids and len(files[k]) >= 2}


def filter_bots(bwin, jason_windows, bot_windows, p_max: float = 0.8, seed: int = 0, log=print):
    """Drop gear-farm bystanders, then windows a Jason-vs-scripted-bot detector is sure are bots."""
    from .detector import FeatureDetector
    farms = gear_farms(bwin)
    keep = [w for w in bwin if not (w.get("equip") and (tuple(w["equip"]), w.get("combat")) in farms)]
    n_gear = len(bwin) - len(keep)
    h = [observable(s) for s in jason_windows]
    b = [observable(s) for s in bot_windows]
    fd = FeatureDetector().fit(h + b, [0] * len(h) + [1] * len(b))
    p = fd.predict([observable(w) for w in keep]) if keep else np.zeros(0)
    out = [w for w, q in zip(keep, p) if q < p_max]
    log(f"bystanders: {len(bwin)} windows from {len({(w['file'], w['pid']) for w in bwin})} players; "
        f"gear farms drop {n_gear} ({len(farms)} signatures), detector drops {len(keep) - len(out)}, keep {len(out)}")
    return out, {"windows": len(bwin), "gear_farm_windows": n_gear, "gear_farm_signatures": len(farms),
                 "detector_dropped": len(keep) - len(out), "kept": len(out)}


# ----------------------------------------------------------------------
def main(paths):
    import json
    all_tracks = [read_tracks(p) for p in paths]
    evals = [evaluate_idm(t) for t in all_tracks]
    # Leave-one-file-out lags, so each file's accuracy uses lags fitted without it.
    agg = defaultdict(lambda: Counter())
    for i, tr in enumerate(all_tracks):
        lags = fit_lags([e for j, e in enumerate(evals) if j != i])
        ev = evaluate_idm(tr, lags=lags)
        for k, v in ev.items():
            for f in ("truth", "invisible", "inferred", "matched"):
                agg[k][f] += v[f]
    lags = fit_lags(evals)
    table = {k: {**dict(v), "precision": round(v["matched"] / max(v["inferred"], 1), 3),
                 "recall": round(v["matched"] / max(v["truth"], 1), 3)} for k, v in agg.items()}
    bw = [w for t in all_tracks for w in bystander_windows(t, lags)]
    farms = gear_farms(bw)
    acts = [len(w["log"]) for w in bw]
    print(json.dumps({
        "files": len(paths), "files_with_pids": sum(1 for t in all_tracks if t["others"]),
        "lags_ms": lags, "idm_on_jason": table,
        "bystander_windows": len(bw), "bystander_players": len({(w["file"], w["pid"]) for w in bw}),
        "bystander_hours": round(len(bw) * WINDOW_MS / 3.6e6, 1),
        "inputs_per_window_p50": float(np.median(acts)) if acts else 0,
        "gear_farm_signatures": len(farms),
        "gear_farm_windows": sum(1 for w in bw if w.get("equip") and (tuple(w["equip"]), w.get("combat")) in farms),
        "attack_anims": Counter(w["attack_anim"] for w in bw).most_common(8),
    }, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
