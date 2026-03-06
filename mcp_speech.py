#!/usr/bin/env python3
"""
Azure Speech MCP Server for Copilot CLI.

Provides 'listen' (STT) and 'speak' (TTS) tools so Copilot can
have voice conversations.
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
    return {
        "key": os.environ.get("AZURE_SPEECH_KEY") or cfg.get("key"),
        "region": os.environ.get("AZURE_SPEECH_REGION") or cfg.get("region", "westus2"),
        "voice": os.environ.get("AZURE_SPEECH_VOICE") or cfg.get("voice", "en-US-Ava:DragonHDLatestNeural"),
    }


CONFIG = load_config()


def stt(seconds=None):
    """Record from mic and transcribe via Azure STT."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = ["arecord", "-D", "default", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav"]
        if seconds:
            cmd += ["-d", str(seconds)]
        cmd.append(tmp_path)
        proc = subprocess.run(cmd, capture_output=True, timeout=max(60, (seconds or 10) + 5))
        if proc.returncode != 0 and not os.path.exists(tmp_path):
            return {"error": "Recording failed"}

        url = f"https://{CONFIG['region']}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": CONFIG["key"],
            "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
        }
        with open(tmp_path, "rb") as f:
            resp = requests.post(url, params={"language": "en-US", "format": "detailed"},
                                 headers=headers, data=f, timeout=30)
        if resp.status_code != 200:
            return {"error": f"Azure STT error {resp.status_code}"}
        result = resp.json()
        if result.get("RecognitionStatus") == "Success":
            nbest = result.get("NBest", [])
            text = nbest[0]["Display"] if nbest else result.get("DisplayText", "")
            return {"text": text}
        return {"text": "", "status": result.get("RecognitionStatus", "Unknown")}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def tts(text):
    """Speak text aloud via Azure TTS."""
    # Escape XML special chars
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    url = f"https://{CONFIG['region']}.tts.speech.microsoft.com/cognitiveservices/v1"
    ssml = (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        f'<voice name="{CONFIG["voice"]}">{text}</voice></speak>'
    )
    headers = {
        "Ocp-Apim-Subscription-Key": CONFIG["key"],
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
    }
    resp = requests.post(url, headers=headers, data=ssml.encode("utf-8"), timeout=60)
    if resp.status_code != 200:
        return {"error": f"Azure TTS error {resp.status_code}"}
    subprocess.run(["aplay", "-D", "default", "-q", "-"],
                   input=resp.content, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"spoken": True, "chars": len(text)}


# --- MCP Protocol (stdio JSON-RPC) ---

TOOLS = [
    {
        "name": "listen",
        "description": "Record audio from the user's microphone and transcribe it to text using Azure Speech-to-Text. Use this to hear what the user is saying via voice. The recording duration is in seconds (default 5, max 30).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Recording duration in seconds (default 5, max 30)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 30,
                }
            },
        },
    },
    {
        "name": "speak",
        "description": "Speak text aloud to the user using Azure Text-to-Speech. Use this to read your responses out loud so the user can hear them.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to speak aloud",
                }
            },
            "required": ["text"],
        },
    },
]


def handle_request(req):
    method = req.get("method")
    params = req.get("params", {})
    req_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "azure-speech", "version": "1.0.0"},
            },
        }
    elif method == "notifications/initialized":
        return None
    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})
        if tool_name == "listen":
            result = stt(seconds=args.get("seconds", 5))
            text = result.get("text", result.get("error", ""))
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": text or "(no speech detected)"}]},
            }
        elif tool_name == "speak":
            result = tts(args.get("text", ""))
            msg = "Spoke the text aloud." if result.get("spoken") else result.get("error", "Failed")
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": msg}]},
            }
        else:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }
    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    else:
        if req_id is not None:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }
        return None


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
