"""Read sessions written by the RuneLite recorder (recorder/).

    python -m osrs_rl.recordings ~/.runelite/osrs-rl-recordings/session-*.jsonl.gz

prints a summary per file. ``to_session`` turns a recording into the same
observable-action log the simulator writes, so the detector and the humanness
metrics run on real play unchanged. That mapping is a first pass: it infers the
game events (target gone, inventory full, shell ready) from tick snapshots and
menu clicks, and should be checked against a few annotated sessions.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter

from .env import CAMERA, CHECK_TAB, DEPOSIT, INTERACT, CLOSE, WALK

KIND_WORDS = [
    ("shell", "shell"),
    ("gemstone crab", "crab"),
    ("tunnel", "cave"),
    ("cave", "cave"),
    ("willow", "tree"),
    ("tree", "tree"),
    ("bank", "bank"),
]


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        try:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        break  # a file cut off mid-line by a crash; keep what we have
        except (EOFError, gzip.BadGzipFile):
            return  # a session still being written, or a client that was killed; keep what we have


def summary(path) -> dict:
    recs = list(load(path))
    if not recs:
        return {"file": str(path), "records": 0}
    types = Counter(r["type"] for r in recs)
    t0, t1 = recs[0]["t"], recs[-1]["t"]
    hours = max(t1 - t0, 1) / 3.6e6
    xp = [r for r in recs if r["type"] == "xp"]
    gained = Counter()
    first = {}
    for r in xp:
        first.setdefault(r["skill"], r["xp"])
        gained[r["skill"]] = r["xp"] - first[r["skill"]]
    unfocused, last_off = 0.0, None
    for r in recs:
        if r["type"] == "focus":
            if not r["focused"]:
                last_off = r["t"]
            elif last_off is not None:
                unfocused += r["t"] - last_off
                last_off = None
    moves = [r["t"] for r in recs if r["type"] in ("mouse_move", "mouse_drag")]
    return {
        "file": str(path),
        "minutes": round((t1 - t0) / 60000, 1),
        "records": len(recs),
        "ticks": types.get("tick", 0),
        "menu_clicks_per_h": round(types.get("menu_click", 0) / hours, 1),
        "mouse_samples": len(moves),
        "unfocused_minutes": round(unfocused / 60000, 1),
        "xp_gained": dict(gained),
        "types": dict(types),
    }


def _kind(target: str, option: str) -> str:
    s = f"{option} {target}".lower()
    for word, kind in KIND_WORDS:
        if word in s:
            return kind
    return ""


IGNORED_OPTIONS = {"play", "logout", "cancel", "examine", "click here to play"}

UNCUT_GEMS = {1617, 1619, 1621, 1623, 1625, 1627, 1629, 1631}  # diamond..dragonstone, opal, jade, red topaz
GEM_VALUE = 150.0  # same progress units as osrs_rl/sim/params.py

# The three gem crab mines and the area between them (world tiles, with a margin).
CRAB_AREA = (1225, 3025, 1370, 3190)


def in_crab_area(r) -> bool:
    x0, y0, x1, y1 = CRAB_AREA
    return x0 <= r.get("x", -1) <= x1 and y0 <= r.get("y", -1) <= y1


def crab_windows(path, window_ticks: int = 1500, max_gap_ticks: int = 100) -> list[dict]:
    """Cut a recording into sessions of ``window_ticks`` spent at the gem crab.

    A stretch counts as crab time while the player's tile is in ``CRAB_AREA``; leaving it for up
    to ``max_gap_ticks`` (a cave hop, a quick bank trip) does not break the stretch. Each full
    window becomes one session in the simulator's format; leftovers shorter than a window are
    dropped so every session covers the same length of play.
    """
    recs = list(load(path))
    ticks = [i for i, r in enumerate(recs) if r["type"] == "tick"]
    runs, start, last_in = [], None, None
    for n, i in enumerate(ticks):
        if in_crab_area(recs[i]):
            if start is None:
                start = n
            elif n - last_in > max_gap_ticks:
                runs.append((start, last_in))
                start = n
            last_in = n
    if start is not None:
        runs.append((start, last_in))
    out = []
    for a, b in runs:
        for w in range(a, b + 1 - window_ticks + 1, window_ticks):
            lo, hi = ticks[w], ticks[w + window_ticks - 1]
            sess = records_to_session(recs[lo : hi + 1])
            sess["source"] = f"{getattr(path, 'name', str(path))}#{(w - a) // window_ticks}"
            out.append(sess)
    return out


def to_session(path) -> dict:
    """Map a recording to the simulator's session format (observable actions only)."""
    return records_to_session(list(load(path)))


def records_to_session(recs: list) -> dict:
    log = []
    t0 = recs[0]["t"] if recs else 0
    last_event, last_event_t = "start", t0
    last_action_t = t0
    prev_interacting = None
    prev_inv_full = False
    shell_seen = False
    prev_bank = False
    xp_first, last = {}, {}
    gems, prev_gems = 0, None

    def add(t, intent, kind="", redundant=False, rank=-1):
        nonlocal last_action_t
        log.append({
            "t_ms": t - t0, "intent": intent, "kind": kind, "uid": -1, "valid": True,
            "trigger": last_event, "since_event_ms": t - last_event_t, "last_event": last_event,
            "gap_ms": t - last_action_t, "redundant": redundant, "rank": rank,
        })
        last_action_t = t

    prev_xy = None
    for r in recs:
        typ, t = r["type"], r["t"]
        if typ == "tick":
            xy = (r.get("x"), r.get("y"))
            if prev_xy is not None and None not in xy and None not in prev_xy \
                    and max(abs(xy[0] - prev_xy[0]), abs(xy[1] - prev_xy[1])) > 10:
                last_event, last_event_t = "arrived", t  # came out of the cave at another mine
            prev_xy = xy
            inter = r.get("interacting")
            if prev_interacting and not inter:
                last_event, last_event_t = "target_gone", t
            prev_interacting = inter
            n_items = sum(1 for it in r.get("inv", []) if it and it[0] > 0)
            if n_items >= 28 and not prev_inv_full:
                last_event, last_event_t = "inv_full", t
            prev_inv_full = n_items >= 28
            has_shell = any("shell" in (n.get("name") or "").lower() for n in r.get("npcs", []))
            if has_shell and not shell_seen:
                last_event, last_event_t = "shell_ready", t
            shell_seen = has_shell
            if r.get("camera_moved"):
                add(t, CAMERA)
            if r.get("bank_open") and not prev_bank:
                last_event, last_event_t = "bank_open", t
            prev_bank = bool(r.get("bank_open"))
            n_gems = sum(it[1] for it in r.get("inv", []) if it and it[0] in UNCUT_GEMS)
            if prev_gems is not None and n_gems > prev_gems:
                gems += n_gems - prev_gems
            prev_gems = n_gems
        elif typ == "menu_click":
            opt = (r.get("option") or "").lower()
            tgt = r.get("target") or ""
            if opt in IGNORED_OPTIONS:
                continue  # login screen, logout, cancelling a menu, examine text
            if "deposit" in opt:
                add(t, DEPOSIT)
            elif opt in ("close",):
                add(t, CLOSE)
            elif opt == "walk here":
                add(t, WALK, "")
            elif not tgt:
                add(t, CHECK_TAB)  # side-panel tabs: Skills, Inventory, Friends List, ...
            else:
                kind = _kind(tgt, opt)
                # Each mine has one crab, one shell and one cave, so a click on any of them is a
                # click on the nearest of its kind, as the simulator records it (rank 0).
                add(t, INTERACT, kind, redundant=bool(prev_interacting) and kind == "crab",
                    rank=0 if kind in ("crab", "shell", "cave") else -1)
        elif typ == "xp":
            xp_first.setdefault(r["skill"], r["xp"])
            last[r["skill"]] = r["xp"]
    progress = float(sum(last[s] - xp_first[s] for s in last)) + gems * GEM_VALUE
    ms = (recs[-1]["t"] - t0) if recs else 0
    return {"log": log, "progress": progress, "ms": float(ms), "activity": "recorded"}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(summary(p), indent=1))


def focus_after_input_ms(path) -> list[float]:
    """For each input in the crab area, ms until the game window lost focus (when it did so before
    the next input). Sets the env's attention model (``env.set_focus_model``)."""
    out, t_input, prev_in_area = [], None, False
    for r in load(path):
        typ = r["type"]
        if typ == "tick":
            prev_in_area = in_crab_area(r)
        elif typ == "menu_click" and prev_in_area:
            if (r.get("option") or "").strip().lower() not in IGNORED_OPTIONS:
                t_input = r["t"]
        elif typ == "focus" and not r.get("focused", True) and t_input is not None:
            out.append(float(r["t"] - t_input))
            t_input = None
    return out
