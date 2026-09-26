#!/usr/bin/env python3
"""stream.py - Stream the headless Exult LLM playthrough to Twitch (or a file).

The engine renders real frames even when running headless (SDL dummy video +
software renderer): the LLM agent bridge exposes a `screenshot` command that
writes a PNG of the current game screen. This script:

  1. Frame pump: on its own TCP connection to the Exult bridge (so it never
     disturbs driver.py's connection), calls {"cmd":"screenshot"} at a fixed
     rate and copies the resulting PNG to a stable path (stream_frame.png).
  2. Encoder: runs ffmpeg, which loop-reads that PNG as a video input, encodes
     H.264, adds a silent audio track (Twitch requires an audio stream), and
     pushes RTMPS to Twitch - or writes an MP4 file for a local dry run.

Usage:
  # Local test (no Twitch): write a 30s mp4 you can inspect
  python stream.py --dry-run --seconds 30 --out /tmp/exult_stream.mp4

  # Go live: needs a Twitch stream key (keep it OUT of source/history)
  export TWITCH_STREAM_KEY=live_xxxxxxxxxxxxxxxxx
  python stream.py --fps 4

Env / files:
  Reads bridge host/port from agent.env (AGENT_BRIDGE_HOST/PORT) like driver.py.
  TWITCH_STREAM_KEY may be set in the environment or in a gitignored twitch.env.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME_PATH = os.path.join(HERE, "stream_frame.png")
# Where the engine writes agent_shot.png (LLM_agent::screenshot -> <SAVEGAME>).
DEFAULT_SHOT = os.path.expanduser("~/.exult/blackgate/agent_shot.png")
TWITCH_RTMPS = "rtmps://live.twitch.tv/app/"

_stop = False


def _load_env_file(name: str) -> None:
    path = os.path.join(HERE, name)
    if not os.path.isfile(path):
        return
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    except OSError:
        pass


def _send(host: str, port: int, obj: dict, timeout: float = 15.0) -> dict:
    s = socket.create_connection((host, port), timeout=timeout)
    try:
        s.sendall((json.dumps(obj) + "\n").encode("utf-8"))
        buf = b""
        s.settimeout(timeout)
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
    finally:
        s.close()


def _handle_sigint(signum, frame):
    global _stop
    _stop = True


def build_ffmpeg_cmd(args) -> list:
    """ffmpeg reads the continuously-updated PNG on a loop and encodes a CBR
    H.264 stream sized/timed for Twitch, plus a silent AAC track."""
    common_in = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        # Loop the single image input; -re paces it at the output framerate.
        "-re", "-framerate", str(args.fps), "-loop", "1", "-i", FRAME_PATH,
        # Silent audio source (Twitch requires an audio stream).
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
    ]
    # Scale to an even, 16:9-friendly height Twitch likes; keep aspect via pad.
    vf = (f"scale={args.width}:{args.height}:force_original_aspect_ratio=decrease,"
          f"pad={args.width}:{args.height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p")
    enc = [
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
        "-pix_fmt", "yuv420p", "-g", str(args.fps * 2), "-r", str(args.fps),
        "-b:v", args.vbitrate, "-maxrate", args.vbitrate, "-bufsize",
        args.vbitrate,
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
    ]
    if args.dry_run:
        return common_in + enc + ["-t", str(args.seconds), "-y", args.out]
    if args.serve:
        # Serve MPEG-TS over HTTP: ffmpeg listens; a viewer (ffplay/mpv/VLC)
        # connects to http://<this-host>:<port>/ and watches live. No VNC or X
        # server needed. -tune stillimage + low latency friendly.
        url = f"http://{args.serve_host}:{args.serve_port}/"
        return common_in + enc + [
            "-f", "mpegts", "-listen", "1", url,
        ]
    key = os.environ.get("TWITCH_STREAM_KEY", "")
    if not key:
        print("[stream] TWITCH_STREAM_KEY not set (env or twitch.env). Refusing "
              "to go live. Use --dry-run or --serve to test locally.",
              file=sys.stderr)
        sys.exit(2)
    return common_in + enc + ["-f", "flv", TWITCH_RTMPS + key]


def main():
    _load_env_file("agent.env")
    _load_env_file("twitch.env")
    ap = argparse.ArgumentParser(description="Stream headless Exult to Twitch.")
    ap.add_argument("--host", default=os.environ.get("AGENT_BRIDGE_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("AGENT_BRIDGE_PORT", "45999")))
    ap.add_argument("--fps", type=int, default=4,
                    help="frames pumped/encoded per second (default 4)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--vbitrate", default="2500k")
    ap.add_argument("--dry-run", action="store_true",
                    help="write to --out mp4 instead of Twitch")
    ap.add_argument("--serve", action="store_true",
                    help="serve live MPEG-TS over HTTP for local viewing "
                         "(no Twitch, no VNC); watch with ffplay/mpv/VLC")
    ap.add_argument("--serve-host", default="0.0.0.0",
                    help="interface to bind the HTTP stream (default all)")
    ap.add_argument("--serve-port", type=int, default=8090)
    ap.add_argument("--from-file", nargs="?", const=DEFAULT_SHOT,
                    default=None,
                    help="read the game frame from a PNG the driver writes "
                         "(no bridge connection; avoids single-client contention). "
                         f"Default path: {DEFAULT_SHOT}")
    ap.add_argument("--seconds", type=int, default=30,
                    help="dry-run duration")
    ap.add_argument("--out", default="/tmp/exult_stream.mp4")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("[stream] ffmpeg not found. Install it: sudo emerge -av "
              "media-video/ffmpeg", file=sys.stderr)
        sys.exit(1)

    # Establish the first frame before ffmpeg opens the input (the file must
    # exist). In --from-file mode we never touch the bridge; otherwise we ping
    # it and pull the first screenshot.
    if args.from_file:
        print(f"[stream] from-file mode: serving frames from {args.from_file}")
        for _ in range(30):
            if _pump_once(args):
                break
            time.sleep(0.5)
        if not os.path.isfile(FRAME_PATH):
            print(f"[stream] no frame at {args.from_file} yet. Is the driver "
                  f"running with --screenshot?", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            pong = _send(args.host, args.port, {"cmd": "ping"})
        except OSError as e:
            print(f"[stream] cannot reach Exult bridge {args.host}:{args.port}: {e}",
                  file=sys.stderr)
            sys.exit(1)
        if not pong.get("ok"):
            print(f"[stream] unexpected ping reply: {pong}", file=sys.stderr)
            sys.exit(1)

        if not _pump_once(args):
            print("[stream] initial screenshot failed; is a game loaded?",
                  file=sys.stderr)
            sys.exit(1)

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    cmd = build_ffmpeg_cmd(args)
    print("[stream] starting ffmpeg:", " ".join(cmd))
    if args.serve:
        print(f"[stream] serving live video. Open in VLC/ffplay/mpv:\n"
              f"           http://<this-host>:{args.serve_port}/\n"
              f"         (auto-relistens after a viewer disconnects; Ctrl-C to stop)")
    proc = subprocess.Popen(cmd)

    # Frame pump loop: keep FRAME_PATH fresh while ffmpeg loop-reads it.
    interval = 1.0 / max(1, args.fps)
    frames = 0
    start = time.time()
    try:
        while not _stop:
            if proc.poll() is not None:
                # ffmpeg exited.
                if args.serve and not _stop:
                    # In serve mode a client disconnecting ends ffmpeg; relaunch
                    # so the next viewer can connect again.
                    print("[stream] viewer disconnected; re-listening...")
                    proc = subprocess.Popen(cmd)
                else:
                    break
            t0 = time.time()
            if _pump_once(args):
                frames += 1
            dt = time.time() - t0
            time.sleep(max(0.0, interval - dt))
            if args.dry_run and (time.time() - start) >= args.seconds + 2:
                break
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        print(f"[stream] stopped after {frames} frames pumped.")


def _pump_once(args) -> bool:
    """Refresh FRAME_PATH from the current game frame.

    In --from-file mode, copy the PNG the driver writes (no bridge). Otherwise
    ask the engine for a screenshot over the bridge.
    """
    if getattr(args, "from_file", None):
        src = args.from_file
        if not os.path.isfile(src):
            return False
    else:
        try:
            r = _send(args.host, args.port, {"cmd": "screenshot"})
        except OSError:
            return False
        if not r.get("ok"):
            return False
        src = r.get("path")
        if not src or not os.path.isfile(src):
            return False
    # Copy to a stable path so ffmpeg's -loop input always sees a complete file.
    tmp = FRAME_PATH + ".tmp"
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, FRAME_PATH)  # atomic swap
    except OSError:
        return False
    return True


if __name__ == "__main__":
    main()
