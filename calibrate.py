"""Measure the real gem crab numbers the simulator should match.

    python calibrate.py ~/.runelite/osrs-rl-recordings/session-*.jsonl.gz

Prints one JSON object of aggregate statistics (no per-click data), for setting
osrs_rl/sim/params.py and checking the recording converter.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

from osrs_rl.recordings import UNCUT_GEMS, in_crab_area, load

TICK_S = 0.6


def pct(x, qs=(10, 25, 50, 75, 90)):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    return {f"p{q}": round(float(np.percentile(x, q)), 2) for q in qs} | {"n": int(len(x))}


def is_crab(name):
    return "gemstone crab" in (name or "").lower() and "shell" not in (name or "").lower()


def is_shell(name):
    return "shell" in (name or "").lower()


def main(paths):
    xp_drops, attack_gaps, crab_ticks_attacking = [], [], 0
    combat_xp_attacking = 0.0
    shell_durations, crab_lifetimes, gems_per_mine = [], [], []
    react = {"shell_ready": [], "target_gone": [], "crab_surfaced": []}
    yaw_changes, camera_flag_ticks, yaw_moved_ticks, ticks_total = [], 0, 0, 0
    no_target_opts, all_opts = Counter(), Counter()
    others_near, unfocused_ticks, focus_off = [], 0, []
    hours_crab = 0.0

    for path in paths:
        recs = list(load(path))
        prev = None
        last_xp_tick = None
        shell_since = None
        crab_since = None
        pending = {}  # event -> time it happened, waiting for the next click
        prev_gems = None
        mined_gems = None
        off_t = None
        for r in recs:
            typ = r["type"]
            if typ == "focus":
                if not r["focused"]:
                    off_t = r["t"]
                elif off_t is not None:
                    focus_off.append((r["t"] - off_t) / 1000)
                    off_t = None
            if typ == "menu_click":
                opt = (r.get("option") or "").strip()
                all_opts[opt] += 1
                if not (r.get("target") or "").strip():
                    no_target_opts[opt] += 1
                for ev, t in list(pending.items()):
                    react[ev].append((r["t"] - t) / 1000)
                    del pending[ev]
                continue
            if typ != "tick":
                continue
            if not in_crab_area(r):
                prev = None
                continue
            ticks_total += 1
            hours_crab += TICK_S / 3600
            if not r.get("focused", True):
                unfocused_ticks += 1
            npcs = r.get("npcs", [])
            crab_here = any(is_crab(n.get("name")) for n in npcs)
            shell_here = any(is_shell(n.get("name")) for n in npcs)
            others_near.append(len(r.get("players", [])))
            inter = r.get("interacting")
            attacking = isinstance(inter, dict) and is_crab(inter.get("name"))
            r["_attacking"] = attacking
            if attacking:
                crab_ticks_attacking += 1
            # camera
            if r.get("camera_moved"):
                camera_flag_ticks += 1
            if prev is not None and r.get("yaw") is not None and prev.get("yaw") is not None:
                d = abs(r["yaw"] - prev["yaw"]) % 2048
                d = min(d, 2048 - d)
                if d > 0:
                    yaw_changes.append(d)
                    yaw_moved_ticks += 1
            # crab / shell lifecycle as seen from this spot
            if shell_here and shell_since is None:
                shell_since = r["tick"]
                pending["shell_ready"] = r["t"]
                if crab_since is not None:
                    crab_lifetimes.append(r["tick"] - crab_since)
                    crab_since = None
            if not shell_here and shell_since is not None:
                shell_durations.append(r["tick"] - shell_since)
                shell_since = None
            if crab_here and prev is not None and not prev.get("_crab_here"):
                crab_since = r["tick"]
                pending["crab_surfaced"] = r["t"]
            r["_crab_here"] = crab_here
            if prev is not None and prev.get("_attacking") and not attacking:
                pending["target_gone"] = r["t"]
            # gems
            n_gems = sum(it[1] for it in r.get("inv", []) if it and it[0] in UNCUT_GEMS)
            if prev_gems is not None and n_gems > prev_gems:
                gems_per_mine.append(n_gems - prev_gems)
            prev_gems = n_gems
            prev = r
        # xp drops while at the crab
        last = {}
        prev_tick_attacking = False
        cur_tick = None
        for r in recs:
            if r["type"] == "tick":
                prev_tick_attacking = bool(r.get("_attacking"))
                cur_tick = r.get("tick")
            elif r["type"] == "xp":
                s = r["skill"]
                if s in last and prev_tick_attacking and cur_tick is not None:
                    gain = r["xp"] - last[s]
                    if gain > 0:
                        xp_drops.append((s, gain))
                        combat_xp_attacking += gain
                        if s.lower() == "hitpoints":
                            if last_xp_tick is not None and 0 < cur_tick - last_xp_tick < 20:
                                attack_gaps.append(cur_tick - last_xp_tick)
                            last_xp_tick = cur_tick
                last[s] = r["xp"]

    hp = [g for s, g in xp_drops if s.lower() == "hitpoints"]
    by_skill = Counter()
    for s, g in xp_drops:
        by_skill[s] += g
    hours_att = crab_ticks_attacking * TICK_S / 3600
    out = {
        "hours_in_crab_area": round(hours_crab, 2),
        "frac_ticks_attacking_crab": round(crab_ticks_attacking / max(ticks_total, 1), 3),
        "combat_xp_per_hour_while_attacking": round(combat_xp_attacking / max(hours_att, 1e-6)),
        "combat_xp_per_hour_in_area": round(combat_xp_attacking / max(hours_crab, 1e-6)),
        "xp_by_skill_while_attacking": dict(by_skill),
        "hp_xp_per_drop": pct(hp),
        "implied_damage_per_hit": pct(np.array(hp) * 3 / 4),
        "ticks_between_hp_drops": pct(attack_gaps),
        "crab_lifetime_ticks_seen": pct(crab_lifetimes),
        "shell_visible_ticks": pct(shell_durations),
        "gems_per_mine": dict(Counter(gems_per_mine)),
        "reaction_s": {k: pct(v) for k, v in react.items()},
        "camera_flag_ticks_per_h": round(camera_flag_ticks / max(hours_crab, 1e-6), 1),
        "yaw_changed_ticks_per_h": round(yaw_moved_ticks / max(hours_crab, 1e-6), 1),
        "yaw_change_size": pct(yaw_changes),
        "other_players_in_view": pct(others_near),
        "frac_ticks_unfocused": round(unfocused_ticks / max(ticks_total, 1), 3),
        "focus_off_seconds": pct(focus_off),
        "top_menu_options": all_opts.most_common(20),
        "no_target_options": no_target_opts.most_common(15),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
