#!/usr/bin/env python3
"""
Voice companion for Copilot CLI.

Run this in a separate terminal alongside your Copilot CLI session.
Flow:
  1. Press Enter → record your voice → transcribed & copied to clipboard
  2. Ctrl+Shift+V to paste into Copilot CLI
  3. Select Copilot's response, Ctrl+Shift+C to copy it
  4. Press Enter here → it speaks the response aloud
  5. Then immediately starts recording your next message
  ... and loops forever. Ctrl+C to quit.
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
    voice = os.environ.get("AZURE_SPEECH_VOICE") or cfg.get("voice", "en-US-Ava:DragonHDLatestNeural")
    if not key:
        print("Error: No Azure Speech API key found.", file=sys.stderr)
        print("Set AZURE_SPEECH_KEY or create ~/.config/speech-to-cli/config.json", file=sys.stderr)
        sys.exit(1)
    return key, region, voice


def get_clipboard():
    try:
        r = subprocess.run(["xclip", "-selection", "clipboard", "-o"],
                           capture_output=True, text=True, timeout=2)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def set_clipboard(text):
    try:
        subprocess.run(["xclip", "-selection", "clipboard"],
                       input=text.encode(), check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


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


def stt(audio_path, key, region):
    url = f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
    }
    with open(audio_path, "rb") as f:
        resp = requests.post(url, params={"language": "en-US", "format": "detailed"},
                             headers=headers, data=f, timeout=30)
    if resp.status_code != 200:
        return None
    result = resp.json()
    if result.get("RecognitionStatus") == "Success":
        nbest = result.get("NBest", [])
        return nbest[0]["Display"] if nbest else result.get("DisplayText", "")
    return None


def tts(text, key, region, voice):
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
    resp = requests.post(url, headers=headers, data=ssml.encode("utf-8"), timeout=60)
    if resp.status_code == 200:
        subprocess.run(["aplay", "-D", "default", "-q", "-"],
                       input=resp.content, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    key, region, voice = load_config()

    print("🎙️  Voice Chat for Copilot CLI")
    print("─" * 40)
    print("  1. Enter → record your voice")
    print("  2. Enter → stop & transcribe (auto-copied)")
    print("  3. Paste into Copilot CLI (Ctrl+Shift+V)")
    print("  4. Copy Copilot's response (Ctrl+Shift+C)")
    print("  5. Enter → speaks response, then records again")
    print("  Ctrl+C to quit\n")

    last_clipboard = get_clipboard()
    first_round = True

    while True:
        try:
            if first_round:
                input("\033[90m[Enter to start recording]\033[0m")
                first_round = False
            else:
                input("\033[90m[Enter when you've copied the response]\033[0m")
                # Check if clipboard has new content (Copilot's response)
                clip = get_clipboard()
                if clip and clip != last_clipboard:
                    preview = clip[:100] + "..." if len(clip) > 100 else clip
                    print(f"\033[94m🔊 Speaking response...\033[0m")
                    tts(clip, key, region, voice)
                    last_clipboard = clip
                else:
                    print("\033[90m  (no new clipboard content to speak)\033[0m")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        # Record
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            print("\033[91m● REC\033[0m  (Enter to stop)", end="", flush=True)
            if not record_audio(tmp_path):
                print("\r\033[K  No audio captured.")
                continue

            print("\r\033[K\033[93m⏳ Transcribing...\033[0m", end="", flush=True)
            text = stt(tmp_path, key, region)

            if text:
                print(f"\r\033[K\033[92m✓ You:\033[0m {text}")
                set_clipboard(text)
                last_clipboard = text
                print(f"  \033[90m→ Paste into Copilot CLI (Ctrl+Shift+V)\033[0m")
            else:
                print("\r\033[K  (no speech detected)")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
