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

from .env import CAMERA, CHECK_TAB, DEPOSIT, INTERACT, CLOSE

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
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    break  # a file cut off mid-line by a crash; keep what we have


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


def to_session(path) -> dict:
    """Map a recording to the simulator's session format (observable actions only)."""
    recs = list(load(path))
    log = []
    t0 = recs[0]["t"] if recs else 0
    last_event, last_event_t = "start", t0
    last_action_t = t0
    prev_interacting = None
    prev_inv_full = False
    shell_seen = False
    prev_bank = False
    xp_first, last = {}, {}

    def add(t, intent, kind="", redundant=False):
        nonlocal last_action_t
        log.append({
            "t_ms": t - t0, "intent": intent, "kind": kind, "uid": -1, "valid": True,
            "trigger": last_event, "since_event_ms": t - last_event_t, "last_event": last_event,
            "gap_ms": t - last_action_t, "redundant": redundant, "rank": -1,
        })
        last_action_t = t

    for r in recs:
        typ, t = r["type"], r["t"]
        if typ == "tick":
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
        elif typ == "menu_click":
            opt = (r.get("option") or "").lower()
            tgt = r.get("target") or ""
            if "deposit" in opt:
                add(t, DEPOSIT)
            elif opt in ("close",):
                add(t, CLOSE)
            elif opt in ("walk here",) or not tgt:
                add(t, CHECK_TAB if not tgt else INTERACT, "")
            else:
                kind = _kind(tgt, opt)
                add(t, INTERACT, kind, redundant=bool(prev_interacting) and kind == "crab")
        elif typ == "xp":
            xp_first.setdefault(r["skill"], r["xp"])
            last[r["skill"]] = r["xp"]
    progress = float(sum(last[s] - xp_first[s] for s in last))
    ms = (recs[-1]["t"] - t0) if recs else 0
    return {"log": log, "progress": progress, "ms": float(ms), "activity": "recorded"}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(summary(p), indent=1))
