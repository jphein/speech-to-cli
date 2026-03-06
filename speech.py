#!/usr/bin/env python3
"""
Speech-to-CLI: Record from mic, transcribe via Azure Speech, copy to clipboard.

Uses Azure Speech Services REST API (no SDK needed, no AVX2 required).
Press Enter to start recording, Enter again to stop and transcribe.
Press Ctrl+C to quit.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import requests

DEFAULTS_PATH = os.path.expanduser("~/.config/speech-to-cli/config.json")


def load_config():
    cfg = {}
    if os.path.exists(DEFAULTS_PATH):
        with open(DEFAULTS_PATH) as f:
            cfg = json.load(f)
    key = os.environ.get("AZURE_SPEECH_KEY") or cfg.get("key")
    region = os.environ.get("AZURE_SPEECH_REGION") or cfg.get("region", "westus2")
    if not key:
        print("Error: No Azure Speech API key found.", file=sys.stderr)
        print("Set AZURE_SPEECH_KEY or create ~/.config/speech-to-cli/config.json", file=sys.stderr)
        sys.exit(1)
    return key, region


def record_audio(path):
    proc = subprocess.Popen(
        ["arecord", "-D", "default", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav", path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        input()
    except EOFError:
        pass
    proc.send_signal(signal.SIGINT)
    proc.wait()
    return os.path.exists(path) and os.path.getsize(path) > 44


def transcribe(audio_path, key, region):
    url = f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
    }
    with open(audio_path, "rb") as f:
        resp = requests.post(
            url, params={"language": "en-US", "format": "detailed"},
            headers=headers, data=f, timeout=30,
        )
    if resp.status_code != 200:
        print(f"Azure error {resp.status_code}: {resp.text}", file=sys.stderr)
        return None
    result = resp.json()
    if result.get("RecognitionStatus") == "Success":
        nbest = result.get("NBest", [])
        return nbest[0]["Display"] if nbest else result.get("DisplayText", "")
    return None


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
    key, region = load_config()
    print(f"🎤 Speech-to-CLI (Azure {region})")
    print(f"   Enter = start recording, Enter = stop & transcribe, Ctrl+C = quit\n")

    while True:
        try:
            input("\033[90m[Enter to record]\033[0m ")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            print("\033[91m● REC\033[0m  (Enter to stop)", end="", flush=True)
            if not record_audio(tmp_path):
                print("\r\033[K  No audio captured.")
                continue

            print("\r\033[K\033[93m⏳ Transcribing...\033[0m", end="", flush=True)
            text = transcribe(tmp_path, key, region)

            if text:
                print(f"\r\033[K\033[92m✓\033[0m {text}")
                if copy_to_clipboard(text):
                    print(f"  \033[90m(copied to clipboard)\033[0m")
            else:
                print("\r\033[K  (no speech detected)")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
