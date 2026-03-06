import time
import requests
import os
import json

DEFAULTS_PATH = os.path.expanduser("~/.config/speech-to-cli/config.json")

def load_config():
    cfg = {}
    if os.path.exists(DEFAULTS_PATH):
        with open(DEFAULTS_PATH) as f:
            cfg = json.load(f)
    return {
        "key": os.environ.get("AZURE_SPEECH_KEY") or cfg.get("key"),
        "region": os.environ.get("AZURE_SPEECH_REGION") or cfg.get("region", "westus2"),
    }

CONFIG = load_config()

VOICES = [
    "en-US-AvaNeural",
    "en-US-AndrewNeural",
    "en-US-EmmaNeural",
    "en-US-BrianNeural",
    "en-US-Ava:DragonHDLatestNeural",
]

def test_voice(voice):
    url = f"https://{CONFIG['region']}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": CONFIG["key"],
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3",
    }
    ssml = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US"><voice name="{voice}">Testing latency for this voice.</voice></speak>'
    
    start = time.time()
    resp = requests.post(url, headers=headers, data=ssml.encode("utf-8"), stream=True)
    
    first_byte = None
    for chunk in resp.iter_content(chunk_size=1):
        if first_byte is None:
            first_byte = time.time()
        break
    
    end = time.time()
    
    if resp.status_code == 200:
        return {
            "voice": voice,
            "ttfb": (first_byte - start) * 1000,
            "total": (end - start) * 1000
        }
    else:
        return {"voice": voice, "error": resp.status_code}

print(f"{'Voice':<35} | {'TTFB (ms)':<10} | {'Total (ms)':<10}")
print("-" * 60)

for voice in VOICES:
    res = test_voice(voice)
    if "error" in res:
        print(f"{res['voice']:<35} | Error: {res['error']}")
    else:
        print(f"{res['voice']:<35} | {res['ttfb']:>9.1f} | {res['total']:>10.1f}")
