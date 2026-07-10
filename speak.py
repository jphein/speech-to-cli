#!/usr/bin/env python3
"""
Speak-CLI: agent-friendly text → speech with a per-call voice override.

Like tts.py, but adds --voice (no env-var juggling — pick a character per
call) and prefers the `tts_region` config key when synthesizing: HD voices
(e.g. en-US-Ava:DragonHDLatestNeural) are only deployed in some regions, so
STT can stay in your closer region while TTS uses `tts_region`.

Built for AI CLI agents driving voice conversations from Bash when the MCP
server isn't connected in the session — pair with listen.py for a
speak → listen loop:

  python3 speak.py --voice en-US-AndrewNeural "What do you think?" \
    && python3 listen.py --markers

Usage:
  python3 speak.py "Hello there"
  python3 speak.py --voice en-US-AndrewNeural "A different character"
  echo "hello" | python3 speak.py
  python3 speak.py -o out.mp3 "save me"       # plays AND saves (tts.py flags)
"""

import argparse
import os
import sys

from state import load_config
from tts import get_text, play_audio, save_audio, synthesize


def main():
    parser = argparse.ArgumentParser(
        description="Text-to-speech via Azure Speech, with per-call voice override")
    parser.add_argument("text", nargs="*", help="Text to speak (or reads from stdin/clipboard)")
    parser.add_argument("-v", "--voice", default=None,
                        help="Azure voice name (default: config voice, e.g. en-US-AndrewNeural)")
    parser.add_argument("-o", "--output", help="Save audio to file (MP3 or WAV)")
    parser.add_argument("-s", "--silent", action="store_true",
                        help="Save only, don't play (requires -o)")
    args = parser.parse_args()

    cfg = load_config()
    if not cfg.get("key"):
        print("Error: No Azure Speech API key found.", file=sys.stderr)
        print("Set AZURE_SPEECH_KEY or create ~/.config/speech-to-cli/config.json",
              file=sys.stderr)
        sys.exit(2)

    region = cfg.get("tts_region") or cfg.get("region")
    voice = args.voice or cfg.get("voice")

    text = get_text(args.text)
    if not text:
        print('No text to speak. Usage: speak.py "text" or copy text to clipboard first.',
              file=sys.stderr)
        sys.exit(1)

    if args.silent and not args.output:
        print("Error: --silent requires --output", file=sys.stderr)
        sys.exit(1)

    preview = text[:80] + "..." if len(text) > 80 else text
    print(f"\033[93m🔊 [{voice}] {preview}\033[0m", file=sys.stderr)

    audio = synthesize(text, cfg["key"], region, voice)
    if not audio:
        sys.exit(2)

    if args.output:
        if save_audio(audio, args.output):
            size = os.path.getsize(os.path.expanduser(args.output))
            print(f"\033[92m💾 Saved: {args.output} ({size} bytes)\033[0m", file=sys.stderr)
        else:
            print(f"\033[91m❌ Failed to save: {args.output}\033[0m", file=sys.stderr)
    if not args.silent:
        play_audio(audio)


if __name__ == "__main__":
    main()
