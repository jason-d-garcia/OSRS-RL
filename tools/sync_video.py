"""Line up an OBS recording of the RuneLite window with a recorder session.

    python tools/sync_video.py "2026-11-05 19-30-12.mkv" session-20261105-193015.jsonl.gz
    python tools/sync_video.py VIDEO RECORDING --sync "VIDEO.sync.json"   # check obs_sync.py's answer

Use it when you hit Record in OBS by hand (no obs_sync.py), or to double-check
obs_sync.py from the picture itself.

The recorder stamps every record with epoch milliseconds. OBS only gives the
video's start time to the second (its file name), plus a little start-up lag, so
that alone can be off by a second or more. This script:

1. takes a coarse start time from the OBS file name (local time, or --tz),
   falling back to the file's modification time minus its duration;
2. measures how much the picture changes from frame to frame (ffmpeg, grayscale,
   downscaled), and lines that up with the recorder's clicks and key presses. A
   click changes the screen within a tick, so the lag where the two match best is
   the true offset;
3. repeats that per 10-minute block to catch drift, and fits a line;
4. writes <video>.sync.json (offset, drift, confidence) and <video>.events.csv
   (every input record with its video time and frame number).

Only numpy and ffmpeg/ffprobe are needed. On a Mac: brew install ffmpeg.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import os
import re
import subprocess
import sys

import numpy as np

INPUT_TYPES = ("menu_click", "key_press", "mouse_press")
EXPORT_TYPES = (
    "menu_click", "key_press", "key_release", "mouse_press", "mouse_release",
    "wheel", "focus", "widget_open", "widget_close", "dialogue",
)
OBS_NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})[ _T](\d{2})-(\d{2})-(\d{2})")


def load_records(path):
    opener = gzip.open if path.endswith(".gz") else open
    out = []
    with opener(path, "rt", encoding="utf-8") as f:
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        except (EOFError, gzip.BadGzipFile):
            pass
    return out


def probe(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,r_frame_rate:format=duration",
         "-of", "json", video],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    st = info["streams"][0]
    num, den = (st.get("avg_frame_rate") or st["r_frame_rate"]).split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    return {
        "width": int(st["width"]),
        "height": int(st["height"]),
        "fps": fps,
        "duration_s": float(info["format"]["duration"]),
    }


def coarse_start_ms(video, duration_s, tz):
    m = OBS_NAME.search(os.path.basename(video))
    if m:
        naive = dt.datetime(*map(int, m.groups()))
        if tz:
            from zoneinfo import ZoneInfo
            aware = naive.replace(tzinfo=ZoneInfo(tz))
        else:
            aware = naive.astimezone()  # this machine's local time zone
        return int(aware.timestamp() * 1000), "file name"
    return int((os.path.getmtime(video) - duration_s) * 1000), "file mtime - duration"


def motion_signal(video, rate, width=160):
    """Mean absolute frame-to-frame difference, sampled at `rate` frames per second."""
    p = probe(video)
    h = max(2, int(round(p["height"] * width / p["width"] / 2)) * 2)
    cmd = ["ffmpeg", "-v", "error", "-i", video, "-vf",
           f"fps={rate},scale={width}:{h},format=gray", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = width * h
    prev, sig = None, []
    while True:
        buf = proc.stdout.read(size * 256)
        if not buf:
            break
        n = len(buf) // size
        frames = np.frombuffer(buf[: n * size], dtype=np.uint8).reshape(n, size).astype(np.int16)
        if prev is not None:
            frames = np.vstack([prev[None], frames])
            sig.extend(np.abs(np.diff(frames, axis=0)).mean(axis=1))
        else:
            sig.append(0.0)
            sig.extend(np.abs(np.diff(frames, axis=0)).mean(axis=1))
        prev = frames[-1]
    proc.wait()
    s = np.asarray(sig, dtype=np.float64)
    # keep the spikes, drop slow changes (camera pans, lighting): subtract a 2 s running median-ish baseline
    k = max(3, int(rate * 2) | 1)
    pad = np.pad(s, k // 2, mode="edge")
    base = np.lib.stride_tricks.sliding_window_view(pad, k)
    s = s - np.percentile(base, 50, axis=1)
    s = np.clip(s, 0, np.percentile(s, 99.5) or None)
    return s


def best_lag(signal, event_bins, rate, search_s, response_s=(0.0, 0.1)):
    """Lag (in bins) to add to event bins so they line up with the motion spikes."""
    n = len(signal)
    ev = np.zeros(n)
    for b in event_bins:
        if 0 <= b < n:
            ev[b] += 1
    # the click marker, cursor and menu react on the next client frame, so a short window is enough;
    # anything that waits for the game tick only adds a constant the IDM can learn
    lo, hi = int(response_s[0] * rate), max(1, int(response_s[1] * rate))
    kern = np.zeros(hi + 1)
    kern[lo:] = 1
    ev = np.convolve(ev, kern)[:n]
    sig = (signal - signal.mean()) / (signal.std() or 1)
    lags = np.arange(-int(search_s * rate), int(search_s * rate) + 1)
    scores = np.empty(len(lags))
    for i, L in enumerate(lags):
        if L >= 0:
            a, b = ev[: n - L], sig[L:]
        else:
            a, b = ev[-L:], sig[: n + L]
        scores[i] = float(np.dot(a, b)) / max(1, a.sum())
    i = int(np.argmax(scores))
    spread = np.median(np.abs(scores - np.median(scores))) or 1e-9
    confidence = (scores[i] - np.median(scores)) / spread
    return int(lags[i]), float(confidence)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("recording", help="session-*.jsonl.gz from ~/.runelite/osrs-rl-recordings")
    ap.add_argument("--sync", help="<video>.sync.json from obs_sync.py: use its start time as the coarse start and just verify it")
    ap.add_argument("--tz", help="time zone the video was recorded in, e.g. America/Los_Angeles (default: this machine's)")
    ap.add_argument("--rate", type=float, default=20.0, help="motion samples per second (default 20 = 50 ms)")
    ap.add_argument("--search", type=float, default=10.0, help="how far either side of the coarse start to search, seconds")
    ap.add_argument("--block-min", type=float, default=10.0, help="block length for the drift check, minutes")
    ap.add_argument("--out", help="output prefix (default: next to the video)")
    a = ap.parse_args(argv)

    info = probe(a.video)
    if a.sync:
        with open(a.sync) as f:
            start_ms, start_src = int(json.load(f)["start_epoch_ms"]), "obs_sync.py"
        a.search = min(a.search, 3.0)
    else:
        start_ms, start_src = coarse_start_ms(a.video, info["duration_s"], a.tz)
    recs = load_records(a.recording)
    if not recs:
        sys.exit("recording is empty")
    sess = next((r for r in recs if r["type"] == "session_start"), {})

    inputs = np.array([r["t"] for r in recs if r["type"] in INPUT_TYPES], dtype=np.int64)
    rate = a.rate
    print(f"video {info['width']}x{info['height']} @ {info['fps']:.2f} fps, {info['duration_s']/60:.1f} min; "
          f"coarse start from {start_src}: {dt.datetime.fromtimestamp(start_ms/1000)}")
    in_video = inputs[(inputs >= start_ms - a.search * 1000) & (inputs <= start_ms + info["duration_s"] * 1000 + a.search * 1000)]
    print(f"{len(in_video)} clicks/keys fall inside the video; measuring motion ...")
    if len(in_video) < 20:
        sys.exit("fewer than 20 inputs overlap the video: wrong recording, or the time zone is off (try --tz)")

    sig = motion_signal(a.video, rate)
    bins = np.round((in_video - start_ms) / 1000 * rate).astype(int)
    lag, conf = best_lag(sig, bins, rate, a.search)
    offset_ms = lag * 1000 / rate

    # drift: same fit per block, searching only near the global answer
    blocks = []
    block_bins = int(a.block_min * 60 * rate)
    for b0 in range(0, len(sig), block_bins):
        sel = bins[(bins >= b0) & (bins < b0 + block_bins)]
        if len(sel) < 20:
            continue
        sub = sig[b0 : b0 + block_bins]
        l, c = best_lag(sub, sel - b0 + lag, rate, 1.5)
        blocks.append({"t_min": round(b0 / rate / 60, 1), "offset_ms": offset_ms + l * 1000 / rate, "confidence": round(c, 1)})
    drift_ppm = 0.0
    if len(blocks) >= 3:
        good = [b for b in blocks if b["confidence"] >= 4]
        if len(good) >= 3:
            x = np.array([b["t_min"] * 60000 for b in good])
            y = np.array([b["offset_ms"] for b in good])
            drift_ppm = float(np.polyfit(x, y, 1)[0] * 1e6)

    # video time of an event = (t - start_ms) + offset_ms + drift
    def vtime_ms(t):
        rel = t - start_ms
        return rel + offset_ms + rel * drift_ppm / 1e6

    prefix = a.out or os.path.splitext(a.video)[0]
    report = {
        "video": os.path.abspath(a.video),
        "recording": os.path.abspath(a.recording),
        "video_size": [info["width"], info["height"]],
        "canvas_size": [sess.get("canvas_w"), sess.get("canvas_h")],
        "stretched": sess.get("stretched"),
        "fps": info["fps"],
        "coarse_start_epoch_ms": start_ms,
        "coarse_start_source": start_src,
        "offset_ms": offset_ms,
        "drift_ppm": round(drift_ppm, 2),
        "confidence": round(conf, 1),
        "blocks": blocks,
        "formula": "video_ms = (t - coarse_start_epoch_ms) * (1 + drift_ppm/1e6) + offset_ms",
    }
    with open(prefix + ".sync.json", "w") as f:
        json.dump(report, f, indent=2)

    n = 0
    with open(prefix + ".events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "video_ms", "frame", "type", "x", "y", "button", "key", "option", "target", "extra"])
        for r in recs:
            if r["type"] not in EXPORT_TYPES:
                continue
            v = vtime_ms(r["t"])
            if v < 0 or v > info["duration_s"] * 1000:
                continue
            extra = {k: r[k] for k in ("focused", "group", "kind", "id", "item", "rot") if k in r}
            w.writerow([r["t"], round(v), int(v / 1000 * info["fps"]), r["type"], r.get("x", ""), r.get("y", ""),
                        r.get("button", ""), r.get("key", ""), r.get("option", ""), r.get("target", ""),
                        json.dumps(extra) if extra else ""])
            n += 1

    verdict = "good" if conf >= 8 else "check it" if conf >= 4 else "unreliable"
    print(f"offset {offset_ms:+.0f} ms (confidence {conf:.1f}: {verdict}), drift {drift_ppm:+.1f} ppm")
    if report["canvas_size"][0] and report["canvas_size"] != report["video_size"]:
        print(f"note: video is {info['width']}x{info['height']} but the game canvas was "
              f"{report['canvas_size'][0]}x{report['canvas_size'][1]}; click x/y are canvas pixels "
              "(the capture includes RuneLite's title bar/side panel, or is scaled)")
    print(f"wrote {prefix}.sync.json and {prefix}.events.csv ({n} events)")


if __name__ == "__main__":
    main()
