"""Play Tutorial Island (Gielinor Guide and Survival Expert) on the RSMod server with the scripted bot.

    # server: RSMOD_TICK_MS=0 gradlew run   (with rsmod-content/tutorial-island and agent-bridge)
    python run_tutorial.py --episodes 4

Each episode spawns a headless bot, puts it at the start of the island through the agent bridge's
tutorial scene and plays to the gate out of the Survival Expert's area. The bot knows the order
of the steps; its delays come from ``HumanTiming``, fitted to Jason's live run. Prints the time per
section next to Jason's and writes ``results/tutorial/runs.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from osrs_rl.tutorial import STAGES, TutorialBot, jason_sections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--max-ticks", type=int, default=3000)
    ap.add_argument("--stage", type=int, default=STAGES[0], help="start at this tutorial stage")
    ap.add_argument("--name", default="tut1")
    ap.add_argument("--port", type=int, default=45600)
    ap.add_argument("--realtime", action="store_true", help="don't lockstep; watch at 600 ms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/tutorial")
    args = ap.parse_args()

    bot = TutorialBot(args.name, lockstep=not args.realtime, port=args.port, seed=args.seed)
    runs = []
    try:
        for ep in range(args.episodes):
            t0 = time.time()
            log = bot.play(max_ticks=args.max_ticks, stage=args.stage)
            failed = [a for a in log.actions if len(a) > 3 and not a[3]]
            run = {
                "episode": ep,
                "finished": log.finished,
                "ticks": log.ticks,
                "game_seconds": log.ticks * 0.6,
                "wall_seconds": round(time.time() - t0, 2),
                "sections": log.section_seconds(),
                "stage_ticks": log.stage_ticks,
                "actions": len(log.actions),
                "failed_actions": len(failed),
                "median_dialogue_ms": float(np.median(log.dialogue_delays)) if log.dialogue_delays else None,
            }
            runs.append(run)
            print(json.dumps({k: v for k, v in run.items() if k != "stage_ticks"}))
    finally:
        bot.close()

    print("\nSection times (s): bot median vs Jason's live run")
    for name, jason in jason_sections().items():
        vals = [r["sections"][name] for r in runs if name in r["sections"]]
        bot_s = f"{np.median(vals):.0f}" if vals else "-"
        print(f"  {name:16s} bot {bot_s:>4s}   Jason {jason:.0f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs.json").write_text(json.dumps(runs, indent=1))


if __name__ == "__main__":
    main()
