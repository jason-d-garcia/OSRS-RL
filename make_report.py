"""Turn results/<activity>/results.json into docs/first-experiment.md tables and a
compact JSON the results page reads."""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).parent
NAMES = {"scripted": "Scripted bot", "bc": "Behaviour cloning", "rl_kl": "PPO from BC + KL", "pure_rl": "Pure RL (no prior)"}


def load(act):
    p = ROOT / "results" / act / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def table(res):
    h = res["human_heldout"]
    rows = [
        "| Agent | Progress/h | Actions per 30 min | Rules floor flags | GBT detector acc / AUC | GRU detector acc / AUC | Reaction-time KS |",
        "|---|---|---|---|---|---|---|",
        f"| Held-out synthetic humans | {h['progress_per_h_median']:,.0f} (IQR {h['progress_per_h_iqr'][0]:,.0f}–{h['progress_per_h_iqr'][1]:,.0f}) | {h['actions_per_session']:.0f} | {res['rules_floor_human_false_flag']:.0%} (false flags) | – | – | – |",
    ]
    for k, b in res["bots"].items():
        d, ts = b["detectors"], b["timescales"]
        rows.append(
            f"| {NAMES[k]} | {b['progress_per_h_median']:,.0f} | {b['actions_per_session']:.0f} | {d['rules_floor']['bot_flag_rate']:.0%} | "
            f"{d['feature_gbt']['balanced_acc']:.2f} / {d['feature_gbt']['auc']:.2f} | {d['sequence_gru']['balanced_acc']:.2f} / {d['sequence_gru']['auc']:.2f} | {ts['reaction_ks']:.2f} |"
        )
    return "\n".join(rows)


def compact(res):
    out = {k: res[k] for k in ("activity", "ticks", "kl_coef", "reward_cfg", "human_heldout", "rules_floor_human_false_flag")}
    out["bots"] = res["bots"]
    out["rl_curve"] = [{k: r[k] for k in ("ticks", "progress_per_h", "kl_per_episode", "actions_per_episode")} for r in res["rl_curve"]]
    out["pure_rl_curve"] = [{k: r[k] for k in ("ticks", "progress_per_h", "actions_per_episode")} for r in res["pure_rl_curve"]]
    out["bc_history"] = res["bc_history"]
    return out


def dashboard_rows(summary):
    """Flat rows for the results page datasets."""
    agents, curves, bc = [], [], []
    for act, res in summary.items():
        h = res["human_heldout"]
        agents.append({"activity": act, "agent": "Held-out humans (synthetic)", "order": 0,
                       "progress_per_h": round(h["progress_per_h_median"]), "iqr_lo": round(h["progress_per_h_iqr"][0]),
                       "iqr_hi": round(h["progress_per_h_iqr"][1]), "actions_per_30min": round(h["actions_per_session"], 1),
                       "rules_flag_rate": round(res["rules_floor_human_false_flag"], 3), "gbt_acc": None, "gbt_auc": None,
                       "gru_acc": None, "gru_auc": None, "best_auc": None, "reaction_ks": None})
        for i, (k, b) in enumerate(res["bots"].items(), start=1):
            d, ts = b["detectors"], b["timescales"]
            agents.append({"activity": act, "agent": NAMES[k], "order": i,
                           "progress_per_h": round(b["progress_per_h_median"]), "iqr_lo": None, "iqr_hi": None,
                           "actions_per_30min": round(b["actions_per_session"], 1),
                           "rules_flag_rate": round(d["rules_floor"]["bot_flag_rate"], 3),
                           "gbt_acc": round(d["feature_gbt"]["balanced_acc"], 3), "gbt_auc": round(d["feature_gbt"]["auc"], 3),
                           "gru_acc": round(d["sequence_gru"]["balanced_acc"], 3), "gru_auc": round(d["sequence_gru"]["auc"], 3),
                           "best_auc": round(max(d["feature_gbt"]["auc"], d["sequence_gru"]["auc"]), 3),
                           "reaction_ks": round(ts["reaction_ks"], 3)})
        for run, key in (("PPO from BC + KL", "rl_curve"), ("Pure RL (no prior)", "pure_rl_curve")):
            for r in res[key]:
                curves.append({"activity": act, "run": run, "ticks_m": round(r["ticks"] / 1e6, 3),
                               "progress_per_h": round(r["progress_per_h"]), "actions_per_episode": round(r["actions_per_episode"], 1)})
        for r in res["bc_history"]:
            bc.append({"activity": act, "epoch": r["epoch"], "val_nll": round(r["val_nll"], 4), "val_intent_acc": round(r["val_intent_acc"], 4)})
    return agents, curves, bc


if __name__ == "__main__":
    summary = {}
    for act in ("gemcrab", "willows"):
        res = load(act)
        if res:
            print(f"### {act}\n\n{table(res)}\n")
            summary[act] = compact(res)
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1))
    agents, curves, bc = dashboard_rows(summary)
    for name, rows in (("agents", agents), ("curves", curves), ("bc", bc)):
        (ROOT / "results" / f"dash_{name}.json").write_text(json.dumps(rows))
