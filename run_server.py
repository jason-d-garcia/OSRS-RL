"""Play gem crab on the real RSMod server and score the sessions like real play.

    # server: RSMOD_TICK_MS=0 gradlew run   (headless, as fast as the agent; see rsmod-content/README.md)
    python run_server.py --controller policy --policy results/gemcrab/policies.pt --which rl_kl \
        --episodes 8 --out results/server --recordings ~/.runelite/osrs-rl-recordings

Each episode is 30 minutes of game time on the server, driven through the agent bridge
(``osrs_rl/rsmod_bridge.py``). For every episode this writes

* ``recordings/bot-<controller>-<n>.jsonl.gz``: the session in the RuneLite recorder's format,
  so it reads like one of your own recordings (``recordings.crab_windows``);
* the env's own action log, cut into the same 15-minute windows ``run_real.py`` uses.

With ``--recordings`` (your real sessions) it then trains the detectors on real windows vs. these
server windows, with folds grouped by recording file as in ``run_real.py``, and prints the AUCs
next to a summary of both sides. An AUC near 0.5 means the detectors can't tell the bot from you.

``--realtime`` leaves the server at its own pace (use it with ``RSMOD_TICK_MS`` unset, at 600 ms)
so you can log in with RuneLite and watch; ``--attach NAME`` drives your own logged-in character
instead of a headless bot.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from osrs_rl.agents.controllers import ScriptedBot, SyntheticHuman
from osrs_rl.env import OsrsEnv
from osrs_rl.policy import policy_from_checkpoint
from osrs_rl.recordings import crab_windows
from osrs_rl.rsmod_bridge import RecordingWriter, RemoteGemCrab
from osrs_rl.train import PolicyController
from run_real import N_FOLDS, WINDOW_TICKS, cv_detect, describe, rollout_windows


def make_controller(args, seed):
    if args.controller == "scripted":
        return ScriptedBot(seed=seed)
    if args.controller == "human":
        return SyntheticHuman(player_id=seed % 12, session_seed=seed)
    ckpt = torch.load(args.policy, map_location="cpu")
    pol = policy_from_checkpoint(ckpt, args.which)
    pol.eval()
    return PolicyController(pol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="scripted", choices=["scripted", "human", "policy"])
    ap.add_argument("--policy", help="policies.pt from run_experiment.py or run_real.py")
    ap.add_argument("--which", default="rl_kl", help="which policy in the checkpoint: bc, rl_kl, pure_rl (rl_adv from run_real.py --adv-weight)")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--ticks", type=int, default=3000, help="game ticks per episode (3000 = 30 min)")
    ap.add_argument("--name", default="agent1", help="bot account name on the server")
    ap.add_argument("--attach", help="drive this logged-in player instead of spawning a headless bot")
    ap.add_argument("--realtime", action="store_true", help="don't hold the server in lockstep")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=45600)
    ap.add_argument("--recordings", help="your RuneLite recordings, to score the bot against")
    ap.add_argument("--observable", action="store_true",
                    help="also score on server-visible inputs only (attack/shell/cave/walk, tick-quantised, "
                         "no progress), as a server-side anti-cheat would see them")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/server")
    args = ap.parse_args()
    if args.controller == "policy" and not args.policy:
        raise SystemExit("--controller policy needs --policy")
    out = Path(args.out)
    (out / "recordings").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    sim = RemoteGemCrab(name=args.name, attach=args.attach, lockstep=not args.realtime,
                        host=args.host, port=args.port)
    windows, files = [], []
    try:
        for ep in range(args.episodes):
            seed = args.seed * 1000 + ep
            path = out / "recordings" / f"bot-{args.controller}-{ep:03d}.jsonl.gz"
            sim.recorder = RecordingWriter(path)
            env = OsrsEnv("gemcrab", sim=sim, episode_ticks=args.ticks)
            w, _ = rollout_windows(env, make_controller(args, seed), seed)
            sim.recorder.close()
            windows += w
            files.append(path)
            rates = [s["progress"] / (s["ms"] / 3.6e6) for s in w]
            log(f"episode {ep}: {sim.tick_count} ticks, {len(env.log)} inputs, "
                f"{sim.gems} gems, {np.mean(rates) if rates else 0:.0f} progress/h -> {path.name}")
    finally:
        sim.close()

    results = {"controller": args.controller, "which": args.which if args.controller == "policy" else None,
               "episodes": args.episodes, "bot": describe(windows) if windows else None}
    # The recorder-format files read back through the same path as your own recordings.
    reread = [s for p in files for s in crab_windows(p, WINDOW_TICKS)]
    results["bot_from_recordings"] = describe(reread) if reread else None
    if args.recordings:
        real, groups = [], []
        for p in sorted(Path(args.recordings).expanduser().glob("session-*.jsonl.gz")):
            ws = crab_windows(p, WINDOW_TICKS)
            real += ws
            groups += [p.name] * len(ws)
        if len(set(groups)) < N_FOLDS:
            raise SystemExit(f"need crab windows from at least {N_FOLDS} recording files")
        results["real"] = describe(real)
        results["detect_real_vs_server_bot"] = cv_detect(real, groups, windows, seed=args.seed)
        log(f"detectors, real vs server bot: {json.dumps(results['detect_real_vs_server_bot']['worst_case'])}")
        if args.observable:
            from osrs_rl.bystanders import observable

            obs = cv_detect([observable(w) for w in real], groups, [observable(w) for w in windows], seed=args.seed)
            results["detect_real_vs_server_bot_observable"] = obs
            log(f"detectors, observable inputs only: {json.dumps(obs['worst_case'])}")
    (out / "results.json").write_text(json.dumps(results, indent=1))
    log(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
