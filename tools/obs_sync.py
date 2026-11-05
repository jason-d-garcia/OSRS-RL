#!/usr/bin/env python3
"""Start/stop OBS recordings and write a sync file that ties video time to the RuneLite recorder.

The recorder stamps every event with wall-clock milliseconds (``t``). This script asks OBS (over
its built-in WebSocket server, OBS 28+) to start recording, measures the wall-clock time of the
video's first frame, and on stop writes ``<video>.sync.json`` next to the video:

    {"video": ".../2026-11-05 01-23-45.mkv", "start_epoch_ms": 1793...,
     "fps": 30.0, "width": 1280, "height": 720, "host": "macbook", ...}

so an event at recorder time ``t`` is at ``(t - start_epoch_ms) / 1000`` seconds into the video.
OBS and RuneLite must run on the same machine (same clock).

Setup (once):
    pip install obsws-python
    OBS: Tools -> WebSocket Server Settings -> Enable, set a password.
    OBS: Settings -> Output -> Recording format mkv (survives crashes); Video -> 30 fps.
    OBS: add a window-capture source for the RuneLite window only.

Use:
    export OBS_PASSWORD=...           # or pass --password
    python obs_sync.py run            # starts recording; Ctrl+C stops it and writes the sync file
    python obs_sync.py start / stop   # the same in two steps
    python obs_sync.py check VIDEO.sync.json ~/.runelite/osrs-rl-recordings/session-....jsonl.gz
                                      # prints the first clicks with their video times to spot-check
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

STATE = Path.home() / ".obs_sync_state.json"


def client(args):
    try:
        import obsws_python as obs
    except ImportError:
        sys.exit("needs: pip install obsws-python")
    return obs.ReqClient(host=args.host, port=args.port, password=args.password or os.environ.get("OBS_PASSWORD", ""),
                         timeout=5)


def start(args):
    cl = client(args)
    if cl.get_record_status().output_active:
        sys.exit("OBS is already recording; stop it first so the start time is known")
    cl.start_record()
    # The first frame's wall-clock time: now minus how long OBS says it has been recording,
    # sampled a few times (the median smooths out request latency).
    samples = []
    deadline = time.time() + 10
    while len(samples) < 7 and time.time() < deadline:
        t_before = time.time() * 1000
        st = cl.get_record_status()
        t_after = time.time() * 1000
        if st.output_active and st.output_duration > 0:
            samples.append((t_before + t_after) / 2 - st.output_duration)
        time.sleep(0.3)
    if not samples:
        sys.exit("OBS did not report an active recording within 10 s")
    vs = cl.get_video_settings()
    state = {"start_epoch_ms": round(statistics.median(samples), 1),
             "start_spread_ms": round(max(samples) - min(samples), 1),
             "fps": vs.fps_numerator / vs.fps_denominator,
             "width": vs.output_width, "height": vs.output_height,
             "host": socket.gethostname(), "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    STATE.write_text(json.dumps(state))
    print(f"recording; first frame at epoch {state['start_epoch_ms']:.0f} ms "
          f"(+/- {state['start_spread_ms'] / 2:.0f} ms), {state['fps']:g} fps {state['width']}x{state['height']}")


def stop(args):
    if not STATE.exists():
        sys.exit("no recording started by this script (missing ~/.obs_sync_state.json)")
    cl = client(args)
    res = cl.stop_record()
    video = getattr(res, "output_path", None)
    state = json.loads(STATE.read_text())
    state["stopped_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["video"] = video
    out = Path(f"{video}.sync.json") if video else Path.home() / f"obs-sync-{int(state['start_epoch_ms'])}.json"
    out.write_text(json.dumps(state, indent=1))
    STATE.unlink()
    print(f"stopped; wrote {out}")


def run(args):
    start(args)
    print("Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    stop(args)


def check(args):
    sync = json.loads(Path(args.sync).read_text())
    t0 = sync["start_epoch_ms"]
    n = 0
    with gzip.open(Path(args.recording).expanduser(), "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") != "menu_click" or r["t"] < t0:
                continue
            s = (r["t"] - t0) / 1000
            print(f"{int(s // 60):3d}:{s % 60:06.3f}  frame {int(s * sync['fps']):7d}  {r.get('option')} {r.get('target') or ''}")
            n += 1
            if n >= args.n:
                break
    if not n:
        print("no clicks after the video start; is this the right recording?")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["start", "stop", "run", "check"])
    ap.add_argument("sync", nargs="?", help="check: the .sync.json")
    ap.add_argument("recording", nargs="?", help="check: a recorder .jsonl.gz")
    ap.add_argument("-n", type=int, default=15, help="check: how many clicks to list")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=4455)
    ap.add_argument("--password")
    args = ap.parse_args()
    {"start": start, "stop": stop, "run": run, "check": check}[args.cmd](args)


if __name__ == "__main__":
    main()
