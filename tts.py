#!/usr/bin/env python3
"""
TTS-CLI: Read clipboard or stdin text aloud via Azure Speech.

Usage:
  python3 tts.py                  # speak clipboard contents
  python3 tts.py "hello world"    # speak argument
  echo "hello" | python3 tts.py   # speak stdin
  python3 tts.py -o out.mp3 "hi" # save to file (plays AND saves)
  python3 tts.py -o out.mp3 -s "hi"  # save only (silent)
"""

import argparse
import os
import subprocess
import sys
import requests

from state import load_config_standalone


def get_text(args_text):
    """Get text from args, stdin, or clipboard."""
    if args_text:
        return " ".join(args_text)
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    # Try clipboard
    for cmd in [["xclip", "-selection", "clipboard", "-o"], ["xsel", "--clipboard", "--output"]]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except FileNotFoundError:
            continue
    return None


def synthesize(text, key, region, voice):
    """Send text to Azure TTS, return audio bytes."""
    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    ssml = (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        f'<voice name="{voice}">{safe}</voice></speak>'
    )
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
    }
    resp = requests.post(url, headers=headers, data=ssml.encode("utf-8"), timeout=30)
    if resp.status_code != 200:
        print(f"Azure TTS error {resp.status_code}: {resp.text}", file=sys.stderr)
        return None
    return resp.content


def play_audio(audio_data):
    """Play WAV audio via aplay."""
    subprocess.run(
        ["aplay", "-D", "default", "-"],
        input=audio_data,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def save_audio(audio_data, output_file):
    """Save WAV audio to file. Converts to MP3 if extension is .mp3."""
    output_file = os.path.expanduser(output_file)
    parent = os.path.dirname(output_file)
    if parent:
        os.makedirs(parent, exist_ok=True)

    ext = os.path.splitext(output_file)[1].lower()
    if ext == ".mp3":
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-i", "pipe:0", "-codec:a", "libmp3lame", "-q:a", "2", output_file],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc.communicate(audio_data, timeout=30)
        return proc.returncode == 0
    else:
        if not ext:
            output_file += ".wav"
        with open(output_file, "wb") as f:
            f.write(audio_data)
        return True


def main():
    parser = argparse.ArgumentParser(description="Text-to-speech via Azure Speech")
    parser.add_argument("text", nargs="*", help="Text to speak (or reads from stdin/clipboard)")
    parser.add_argument("-o", "--output", help="Save audio to file (MP3 or WAV)")
    parser.add_argument("-s", "--silent", action="store_true", help="Save only, don't play (requires -o)")
    args = parser.parse_args()

    key, region, voice = load_config_standalone(need_voice=True)
    text = get_text(args.text)
    if not text:
        print("No text to speak. Usage: tts.py \"text\" or copy text to clipboard first.", file=sys.stderr)
        sys.exit(1)

    if args.silent and not args.output:
        print("Error: --silent requires --output", file=sys.stderr)
        sys.exit(1)

    # Truncate display for long text
    preview = text[:80] + "..." if len(text) > 80 else text
    print(f"\033[93m🔊 {preview}\033[0m", file=sys.stderr)

    audio = synthesize(text, key, region, voice)
    if audio:
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
