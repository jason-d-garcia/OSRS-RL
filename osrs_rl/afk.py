"""Attention model: when a player is at the keyboard, and what they fidget with while idle.

Gem crab is AFK content. Between fights Jason is often doing something else, and how long he takes
to come back after the crab dies depends on what that is. He has two regimes, visible in the
recorder's per-tick ``focused`` flag:

* focused: RuneLite has focus most of the time (another game on the other monitor, glancing over);
* unfocused: RuneLite is in the background (working on another machine), so he comes back late.

``fit_prior`` reads his recordings and writes a JSON prior with, per regime:

* ``away``: per prompting event (crab dies -> ``shell_ready``, crab burrows -> ``target_gone``,
  arrival at a mine -> ``arrived``, session start -> ``start``), the time from the event to his next
  game input (attack/walk/cave), as a heavy-tailed mixture: log-normal body + Pareto tail;
* ``fight_gap``: the rate and length of away bouts with no event (a long pause mid-fight);
* ``fidget``: tab clicks (Skills, Friends List, ...) per minute while away, and which tabs;
* ``focus_hold``: how long the window stays focused after an input (log-normal);

and across regimes the share of time in each, the switch rate within a session, and a per-session
"mood" spread (how much a session's away times scale up or down as a whole).

``AttentionState`` is what the env uses (``env.set_afk_model``): one per episode, it picks a regime
and a mood, switches regime at the fitted rate, and samples away times and fidgets.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .env import CAMERA, CHECK_TAB, INTERACT, WALK
from .recordings import in_crab_area, load, records_to_session

PROMPTS = ("shell_ready", "target_gone", "arrived", "start")
GAME = (INTERACT, WALK)
BLOCK_MS = 5 * 60 * 1000  # regime is labelled per 5-minute block of crab time
FOCUSED_SHARE = 0.25  # a block with RuneLite focused at least this much is the focused regime
MAX_AWAY_S = 900.0
FIGHT_GAP_S = 60.0  # a gap this long with no event in between counts as an away bout mid-fight
FIGHT_HAZARD_PER_MIN = 0.003  # see fit_prior


# ----------------------------------------------------------------------
# heavy-tailed fit
def fit_heavy(x, q_tail: float = 0.8) -> dict:
    """Log-normal body below the ``q_tail`` quantile, Pareto tail above it (Hill estimate)."""
    x = np.asarray([v for v in x if v > 0], dtype=float)
    if len(x) < 5:
        return {"n": int(len(x)), "mu": math.log(max(np.median(x), 1.0)) if len(x) else 0.0, "sigma": 1.0,
                "u": float(np.max(x)) if len(x) else 1.0, "w_tail": 0.0, "alpha": 2.0}
    u = float(np.quantile(x, q_tail))
    body = np.log(x[x <= u])
    tail = x[x > u]
    alpha = float(len(tail) / max(np.sum(np.log(tail / u)), 1e-6)) if len(tail) else 2.0
    return {"n": int(len(x)), "mu": float(body.mean()), "sigma": float(max(body.std(), 0.05)), "u": u,
            "w_tail": float(len(tail) / len(x)), "alpha": float(min(max(alpha, 0.3), 10.0))}


def sample_heavy(p: dict, rng, scale: float = 1.0) -> float:
    if rng.random() < p["w_tail"]:
        v = p["u"] * (1.0 - rng.random()) ** (-1.0 / p["alpha"])
    else:
        for _ in range(20):
            v = math.exp(p["mu"] + p["sigma"] * rng.standard_normal())
            if v <= p["u"]:
                break
        else:
            v = p["u"]
    return float(min(v * scale, MAX_AWAY_S))


def heavy_quantiles(p: dict, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99), n: int = 20000, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    s = np.array([sample_heavy(p, rng) for _ in range(n)])
    return [round(float(np.quantile(s, q)), 1) for q in qs]


# ----------------------------------------------------------------------
# reading the recordings
def _focus_timeline(recs):
    """(t_ms, focused) per crab-area tick."""
    return [(r["t"], bool(r.get("focused", True))) for r in recs if r["type"] == "tick" and in_crab_area(r)]


def _block_regimes(timeline):
    """Block start -> 'focused' / 'unfocused' (5-minute blocks of crab time)."""
    out = {}
    if not timeline:
        return out
    t0 = timeline[0][0]
    by = defaultdict(list)
    for t, f in timeline:
        by[(t - t0) // BLOCK_MS].append(f)
    for b, fs in by.items():
        if len(fs) >= 100:  # at least a minute of ticks
            out[b] = "focused" if np.mean(fs) >= FOCUSED_SHARE else "unfocused"
    return out


def session_events(path):
    """Everything fit_prior needs from one recording."""
    recs = list(load(path))
    if not recs:
        return None
    sess = records_to_session(recs)
    t0 = recs[0]["t"]
    timeline = _focus_timeline(recs)
    if len(timeline) < 500:
        return None
    blocks = _block_regimes(timeline)
    tl0 = timeline[0][0]
    regime_at = lambda t: blocks.get((t - tl0) // BLOCK_MS)
    tabs = [(r["t"], r.get("option") or "") for r in recs
            if r["type"] == "menu_click" and not (r.get("target") or "") and (r.get("option") or "").lower()
            not in ("walk here", "cancel", "play", "logout", "close")]
    # focus holds: time from an input to the next focus loss
    focus_ev = [(r["t"], bool(r.get("focused"))) for r in recs if r["type"] == "focus" and "focused" in r]
    inputs = [t0 + row["t_ms"] for row in sess["log"] if row["intent"] in GAME + (CHECK_TAB,)]
    holds = []
    lost = [t for t, f in focus_ev if not f]
    j = 0
    for t in inputs:
        while j < len(lost) and lost[j] < t:
            j += 1
        if j < len(lost):
            holds.append(((lost[j] - t) / 1000.0, regime_at(t)))
    return {"file": Path(path).name, "t0": t0, "log": sess["log"], "regime_at": regime_at, "blocks": blocks,
            "tabs": tabs, "holds": holds, "focused_share": float(np.mean([f for _, f in timeline]))}


def fit_prior(paths, out: str | Path | None = None) -> dict:
    """Fit the attention prior from Jason's recordings (see the module docstring)."""
    away = defaultdict(lambda: defaultdict(list))  # regime -> event -> [s]
    fight_gaps = defaultdict(list)
    fight_time = Counter()
    fidget_n, fidget_time = Counter(), Counter()
    tab_opts = defaultdict(Counter)
    holds = defaultdict(list)
    block_counts, switches, crab_hours = Counter(), 0, 0.0
    mood = []
    for p in paths:
        ev = session_events(p)
        if ev is None:
            continue
        t0, log, regime_at = ev["t0"], ev["log"], ev["regime_at"]
        b = [ev["blocks"][k] for k in sorted(ev["blocks"])]
        block_counts.update(b)
        switches += sum(1 for x, y in zip(b, b[1:]) if x != y)
        crab_hours += len(b) * BLOCK_MS / 3.6e6
        for s, r in ev["holds"]:
            if r:
                holds[r].append(s)
        tabs = ev["tabs"]
        seen = set()
        game = [row for row in log if row["intent"] in GAME]
        sess_lat = []  # (regime, latency) of this session's shell_ready responses
        for row in game:
            e_t = t0 + row["t_ms"] - row["since_event_ms"]
            key = (row["last_event"], e_t)
            reg = regime_at(e_t)
            if reg is None:
                continue
            if row["last_event"] in PROMPTS and key not in seen:
                seen.add(key)
                lat = row["since_event_ms"] / 1000.0
                away[reg][row["last_event"]].append(lat)
                if row["last_event"] == "shell_ready":
                    sess_lat.append((reg, lat))
                # fidgets while away: tab clicks between the event and this input
                t_in = t0 + row["t_ms"]
                n = sum(1 for t, _ in tabs if e_t <= t < t_in)
                fidget_n[reg] += n
                fidget_time[reg] += lat / 60.0
                for t, o in tabs:
                    if e_t <= t < t_in:
                        tab_opts[reg][o] += 1
        # mid-fight pauses: gaps between game inputs with no event in between
        for a, c in zip(game, game[1:]):
            same_event = (c["t_ms"] - c["since_event_ms"]) == (a["t_ms"] - a["since_event_ms"])
            reg = regime_at(t0 + a["t_ms"])
            if not same_event or reg is None:
                continue
            g = (c["t_ms"] - a["t_ms"]) / 1000.0
            fight_time[reg] += g / 60.0
            if g >= FIGHT_GAP_S:
                fight_gaps[reg].append(g)
        if len(sess_lat) >= 3:
            mood.append((ev["file"], sess_lat))
    # Session mood: how far a session's shell_ready responses sit from their regime's median, in
    # log space, after the regime is accounted for.
    reg_med = {r: float(np.median(np.log(v["shell_ready"]))) for r, v in away.items() if v["shell_ready"]}
    resid = [float(np.median([np.log(l) - reg_med[r] for r, l in lat])) for _, lat in mood if len(lat) >= 3]
    regimes = {}
    total_blocks = sum(block_counts.values())
    for reg in ("focused", "unfocused"):
        if not block_counts[reg]:
            continue
        hl = np.log(np.clip(holds[reg], 0.6, None)) if holds[reg] else np.log([20.0])
        regimes[reg] = {
            "share": block_counts[reg] / total_blocks,
            "away": {e: fit_heavy(away[reg][e]) for e in PROMPTS},
            "away_quantiles_s": {e: ([round(float(np.quantile(away[reg][e], q)), 1)
                                      for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)] if away[reg][e] else None)
                                 for e in PROMPTS},
            # Pauses mid-fight are invisible while the character keeps attacking, and a pause that
            # spans the crab dying is already in the shell_ready response. What's left is idle
            # time with the crab up and no attack: 0.4% of Jason's idle bouts there pass 60 s
            # (about 10 bouts per 15 minutes), so the hazard is ~0.003 per minute.
            "fight_gap": {"per_min": FIGHT_HAZARD_PER_MIN,
                          "length": fit_heavy([g - FIGHT_GAP_S + 1 for g in fight_gaps[reg]] or [30.0])},
            "fidget": {"per_min": fidget_n[reg] / max(fidget_time[reg], 1e-6),
                       "options": {o: n for o, n in tab_opts[reg].most_common() if n >= 2}},
            "focus_hold": {"median_ms": float(np.exp(np.median(hl)) * 1000), "sigma": float(max(np.std(hl), 0.1))},
        }
    all_opts = Counter()
    for c in tab_opts.values():
        all_opts.update(c)
    prior = {
        "regimes": regimes,
        "switch_per_hour": switches / max(crab_hours, 1e-6),
        "mood_sigma": float(np.std(resid)) if len(resid) >= 3 else 0.5,
        "tab_options": dict(all_opts.most_common()) or {"Skills": 1},
        "source": "Jason's RuneLite recordings",
    }
    if out is not None:
        Path(out).write_text(json.dumps(prior, indent=1))
    return prior


# ----------------------------------------------------------------------
class AttentionState:
    """One episode's attention: regime, mood, regime switches, away times and fidgets."""

    def __init__(self, prior: dict, rng):
        self.p = prior
        self.rng = rng
        regs = list(prior["regimes"])
        shares = np.array([prior["regimes"][r]["share"] for r in regs])
        self.regime = regs[int(rng.choice(len(regs), p=shares / shares.sum()))]
        self.mood = float(math.exp(prior.get("mood_sigma", 0.5) * rng.standard_normal()))
        self._next_switch_ms = self._switch_gap_ms(0.0)

    def _switch_gap_ms(self, t_ms):
        rate = self.p.get("switch_per_hour", 0.0)
        return t_ms + (self.rng.exponential(3.6e6 / rate) if rate > 0 else math.inf)

    def update(self, t_ms: float) -> bool:
        """Switch regime when its time is up. Returns True on a switch."""
        if t_ms < self._next_switch_ms or len(self.p["regimes"]) < 2:
            return False
        self.regime = next(r for r in self.p["regimes"] if r != self.regime)
        self._next_switch_ms = self._switch_gap_ms(t_ms)
        return True

    @property
    def r(self) -> dict:
        return self.p["regimes"][self.regime]

    def away_ms(self, event: str) -> float:
        a = self.r["away"].get(event)
        if not a or a["n"] < 5:
            return 0.0
        return 1000.0 * sample_heavy(a, self.rng, self.mood)

    def fight_away_ms(self, span_ms: float) -> float:
        """An away bout starting within the next ``span_ms`` of steady fighting, or 0."""
        fg = self.r["fight_gap"]
        if self.rng.random() >= 1.0 - math.exp(-fg["per_min"] * span_ms / 60_000.0):
            return 0.0
        return 1000.0 * (FIGHT_GAP_S + sample_heavy(fg["length"], self.rng, self.mood))

    def fidget_times(self, start_ms: float, end_ms: float) -> list:
        rate = self.r["fidget"]["per_min"] / 60_000.0
        out, t = [], start_ms
        while rate > 0:
            t += self.rng.exponential(1.0 / rate)
            if t >= end_ms:
                break
            out.append(t)
        return out

    def tab_option(self) -> str:
        opts = self.r["fidget"]["options"] or self.p["tab_options"]
        names = list(opts)
        w = np.array([opts[n] for n in names], dtype=float)
        return names[int(self.rng.choice(len(names), p=w / w.sum()))]

    def focus_model(self) -> dict:
        return self.r["focus_hold"]


def load_prior(path) -> dict:
    return json.loads(Path(path).read_text())


__all__ = ["AttentionState", "fit_prior", "fit_heavy", "sample_heavy", "heavy_quantiles", "load_prior", "PROMPTS"]


if __name__ == "__main__":
    import sys

    prior = fit_prior(sorted(Path(sys.argv[1]).expanduser().glob("session-*.jsonl.gz")), sys.argv[2])
    print(json.dumps({r: {k: v for k, v in d.items() if k in ("share", "away_quantiles_s", "fidget")}
                      for r, d in prior["regimes"].items()}, indent=1))
    print("switch/h", round(prior["switch_per_hour"], 2), "mood sigma", round(prior["mood_sigma"], 2))
