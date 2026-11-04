"""Session-level humanness features and per-timescale metrics.

Everything here is computed only from what a server or client could observe:
timestamped input events (clicks, camera moves, tab checks) and the game events
they followed. Waits are invisible, so they never appear.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from .env import CAMERA, CHECK_TAB, CLOSE, DEPOSIT, INTENTS, INTERACT, WALK
from .sim.base import KINDS

CHANGE = {"start", "inv_full", "target_gone", "bank_open", "shell_ready", "arrived", "invalid"}
GAP_BINS = np.linspace(np.log(120), np.log(300_000), 11)


def reaction_delays(log):
    """Delay from a game event that needs a response to the first input after it."""
    out, seen = [], set()
    for r in log:
        key = (r["last_event"], round(r["t_ms"] - r["since_event_ms"]))
        if r["last_event"] in CHANGE and key not in seen:
            seen.add(key)
            out.append(r["since_event_ms"])
    return np.array(out, dtype=float)


def gaps(log):
    return np.array([r["gap_ms"] for r in log[1:]], dtype=float)


def _stats(x):
    if len(x) == 0:
        return [0.0] * 6
    lx = np.log(np.maximum(x, 1.0))
    return [lx.mean(), lx.std(), *np.percentile(lx, [10, 50, 90]), float(len(x))]


def session_features(sess) -> np.ndarray:
    log = sess["log"]
    hours = max(sess["ms"], 1.0) / 3.6e6
    g = gaps(log)
    rd = reaction_delays(log)
    lg = np.log(np.maximum(g, 1.0)) if len(g) else np.zeros(1)
    ac = float(np.corrcoef(lg[:-1], lg[1:])[0, 1]) if len(lg) > 3 and lg.std() > 1e-6 else 0.0
    cv = float(g.std() / g.mean()) if len(g) > 1 and g.mean() > 0 else 0.0
    intents = np.array([r["intent"] for r in log]) if log else np.zeros(0)
    rate = lambda k: float(np.sum(intents == k)) / hours
    interacts = [r for r in log if r["intent"] == INTERACT]
    red = np.mean([r["redundant"] for r in interacts]) if interacts else 0.0
    inval = np.mean([not r["valid"] for r in log]) if log else 0.0
    nearest = np.mean([r["rank"] == 0 for r in interacts]) if interacts else 0.0
    hist = np.histogram(lg, bins=GAP_BINS)[0] / max(len(lg), 1)
    f = [
        len(log) / hours,
        sess["progress"] / hours / 1e4,
        *_stats(g),
        *_stats(rd),
        cv,
        np.nan_to_num(ac),
        float(np.mean(g > 60_000)) if len(g) else 0.0,
        float(np.log(g.max())) if len(g) else 0.0,
        rate(CAMERA),
        rate(CHECK_TAB),
        rate(CLOSE),
        rate(DEPOSIT),
        rate(INTERACT),
        rate(WALK),
        red,
        inval,
        nearest,
        *hist,
    ]
    return np.nan_to_num(np.array(f, dtype=np.float32))


FEATURE_NAMES = (
    ["actions_per_h", "progress_per_h_1e4"]
    + [f"gap_{s}" for s in ("logmean", "logstd", "p10", "p50", "p90", "n")]
    + [f"react_{s}" for s in ("logmean", "logstd", "p10", "p50", "p90", "n")]
    + ["gap_cv", "gap_autocorr", "frac_gap_over_60s", "log_max_gap"]
    + [f"{INTENTS[k]}_per_h" for k in (CAMERA, CHECK_TAB, CLOSE, DEPOSIT, INTERACT, WALK)]
    + ["redundant_click_rate", "invalid_rate", "nearest_target_rate"]
    + [f"gap_hist_{i}" for i in range(10)]
)


def session_tokens(sess, max_len: int = 256) -> np.ndarray:
    """Per-action tokens for the sequence detector."""
    rows = []
    for r in sess["log"][:max_len]:
        v = np.zeros(4 + len(INTENTS) + len(KINDS) + 3, dtype=np.float32)
        v[0] = (np.log(max(r["gap_ms"], 1.0)) - 8.0) / 2.0
        v[1] = (np.log(max(r["since_event_ms"], 1.0)) - 8.0) / 2.0
        v[2] = float(r["last_event"] in CHANGE)
        v[3] = (r["t_ms"] / 1.8e6)
        v[4 + r["intent"]] = 1.0
        if r["kind"]:
            v[4 + len(INTENTS) + KINDS.index(r["kind"])] = 1.0
        b = 4 + len(INTENTS) + len(KINDS)
        v[b] = float(r["redundant"])
        v[b + 1] = float(r["valid"])
        v[b + 2] = min(max(r["rank"], 0), 5) / 5.0
        rows.append(v)
    if not rows:
        rows.append(np.zeros(4 + len(INTENTS) + len(KINDS) + 3, dtype=np.float32))
    return np.stack(rows)


def timescale_metrics(bot_sessions, human_sessions) -> dict:
    """The plan's per-timescale table, for the timescales the simulator has."""
    def pool(sessions, fn):
        xs = [fn(s["log"]) for s in sessions]
        return np.concatenate(xs) if xs else np.zeros(0)

    out = {}
    b_rd, h_rd = pool(bot_sessions, reaction_delays), pool(human_sessions, reaction_delays)
    b_g, h_g = pool(bot_sessions, gaps), pool(human_sessions, gaps)
    out["reaction_ks"] = float(ks_2samp(b_rd, h_rd).statistic) if len(b_rd) and len(h_rd) else float("nan")
    out["gap_ks"] = float(ks_2samp(b_g, h_g).statistic) if len(b_g) and len(h_g) else float("nan")

    def acf(sessions):
        vals = [session_features(s)[FEATURE_NAMES.index("gap_autocorr")] for s in sessions]
        return float(np.mean(vals))

    out["gap_autocorr_bot"], out["gap_autocorr_human"] = acf(bot_sessions), acf(human_sessions)
    rate = lambda ss: np.array([s["progress"] / (s["ms"] / 3.6e6) for s in ss])
    br, hr = rate(bot_sessions), rate(human_sessions)
    q25, q75 = np.percentile(hr, [25, 75])
    out["progress_per_h_bot_median"] = float(np.median(br))
    out["progress_per_h_human_iqr"] = [float(q25), float(q75)]
    out["progress_in_human_iqr_frac"] = float(np.mean((br >= q25) & (br <= q75)))
    cam = lambda ss: float(np.mean([sum(r["intent"] == CAMERA for r in s["log"]) / (s["ms"] / 3.6e6) for s in ss]))
    out["camera_per_h_bot"], out["camera_per_h_human"] = cam(bot_sessions), cam(human_sessions)
    lapse = lambda ss: float(np.mean([np.mean(gaps(s["log"]) > 60_000) if len(s["log"]) > 1 else 0 for s in ss]))
    out["frac_gaps_over_60s_bot"], out["frac_gaps_over_60s_human"] = lapse(bot_sessions), lapse(human_sessions)
    return out
