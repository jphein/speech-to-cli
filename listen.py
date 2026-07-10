#!/usr/bin/env python3
"""
Listen-CLI: one-shot scriptable listen — record until silence, print the transcript.

Built for AI CLI agents driving voice conversations from Bash when the MCP
server isn't connected in the session: pair with tts.py for a speak → listen
loop (speak a turn, listen for the reply, respond, repeat).

  python3 tts.py "What do you think?" && python3 listen.py

Uses the same STT pipeline as the MCP server (streaming WebSocket with VAD +
energy gating, REST fallback) via stt.stt(). Progress/VU noise goes to stderr;
only the transcript (plus optional markers) goes to stdout, so it's safe to
capture:  reply=$(python3 listen.py --markers 2>/dev/null)

Usage:
  python3 listen.py                       # listen until silence, print transcript
  python3 listen.py --seconds 20          # cap recording at 20 s (default 30)
  python3 listen.py --silence-timeout 2   # stop after 2 s of silence (default 3)
  python3 listen.py --mode vad            # force a backend: streaming|vad|whisper|fixed
  python3 listen.py --markers             # wrap stdout in ===TRANSCRIPT_START/END=== markers
  python3 listen.py --copy                # also copy the transcript to the clipboard

Exit codes: 0 = transcript printed · 1 = no speech detected · 2 = config/STT error
"""

import argparse
import subprocess
import sys

import state
from stt import stt


def copy_to_clipboard(text):
    for cmd in [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]:
        try:
            subprocess.run(cmd, input=text.encode(), check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return False


def main():
    parser = argparse.ArgumentParser(
        description="One-shot listen: record until silence, print the transcript to stdout.")
    parser.add_argument("--seconds", type=int, default=30,
                        help="max recording length in seconds (1-30, default 30)")
    parser.add_argument("--silence-timeout", type=float, default=3.0,
                        help="stop after this many seconds of silence (default 3)")
    parser.add_argument("--mode", choices=["streaming", "vad", "whisper", "fixed"],
                        default=None, help="force an STT backend (default: auto)")
    parser.add_argument("--markers", action="store_true",
                        help="wrap the transcript in ===TRANSCRIPT_START/END=== markers")
    parser.add_argument("--copy", action="store_true",
                        help="also copy the transcript to the clipboard")
    args = parser.parse_args()

    if not state.CONFIG.get("key"):
        print("Error: No Azure Speech API key found.", file=sys.stderr)
        print("Set AZURE_SPEECH_KEY or create ~/.config/speech-to-cli/config.json",
              file=sys.stderr)
        sys.exit(2)

    try:
        result = stt(seconds=args.seconds, mode=args.mode,
                     silence_timeout=args.silence_timeout)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(2)

    text = (result or {}).get("text", "").strip()

    if args.markers:
        print("===TRANSCRIPT_START===")
        print(text if text else "(no speech detected)")
        print("===TRANSCRIPT_END===")
    elif text:
        print(text)

    if not text:
        if not args.markers:
            print("(no speech detected)", file=sys.stderr)
        sys.exit(1)

    if args.copy:
        copy_to_clipboard(text)


if __name__ == "__main__":
    main()
