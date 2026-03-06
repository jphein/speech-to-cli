#!/usr/bin/env python3
"""
TTS-CLI: Read clipboard or stdin text aloud via Azure Speech.

Usage:
  python3 tts.py                  # speak clipboard contents
  python3 tts.py "hello world"    # speak argument
  echo "hello" | python3 tts.py   # speak stdin
"""

import json
import os
import subprocess
import sys
import requests

DEFAULTS_PATH = os.path.expanduser("~/.config/speech-to-cli/config.json")


def load_config():
    cfg = {}
    if os.path.exists(DEFAULTS_PATH):
        with open(DEFAULTS_PATH) as f:
            cfg = json.load(f)
    key = os.environ.get("AZURE_SPEECH_KEY") or cfg.get("key")
    region = os.environ.get("AZURE_SPEECH_REGION") or cfg.get("region", "westus2")
    voice = os.environ.get("AZURE_SPEECH_VOICE") or cfg.get("voice", "en-US-Ava:DragonHDLatestNeural")
    if not key:
        print("Error: No Azure Speech API key found.", file=sys.stderr)
        print("Set AZURE_SPEECH_KEY or create ~/.config/speech-to-cli/config.json", file=sys.stderr)
        sys.exit(1)
    return key, region, voice


def get_text():
    """Get text from args, stdin, or clipboard."""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
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
    ssml = (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        f'<voice name="{voice}">{text}</voice></speak>'
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


def main():
    key, region, voice = load_config()
    text = get_text()
    if not text:
        print("No text to speak. Usage: tts.py \"text\" or copy text to clipboard first.", file=sys.stderr)
        sys.exit(1)

    # Truncate display for long text
    preview = text[:80] + "..." if len(text) > 80 else text
    print(f"\033[93m🔊 {preview}\033[0m", file=sys.stderr)

    audio = synthesize(text, key, region, voice)
    if audio:
        play_audio(audio)


if __name__ == "__main__":
    main()
