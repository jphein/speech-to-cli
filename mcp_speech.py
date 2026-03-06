#!/usr/bin/env python3
"""
Azure Speech MCP Server for Copilot CLI.

Provides 'listen' (STT) and 'speak' (TTS) tools so Copilot can
have voice conversations.

STT modes (selected automatically):
  - streaming: Real-time WebSocket recognition (fastest, ~0.5s after speech ends)
  - vad:       Record with voice activity detection, upload when silence detected (~2-4s)
  - fixed:     Record for a fixed duration, then upload (fallback)
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import requests

try:
    import webrtcvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False

try:
    import websocket
    HAS_WS = True
except ImportError:
    HAS_WS = False

DEFAULTS_PATH = os.path.expanduser("~/.config/speech-to-cli/config.json")

# VAD settings
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = SAMPLE_RATE * 2 * FRAME_MS // 1000  # 960 bytes per 30ms frame
SILENCE_TIMEOUT = 1.5  # seconds of silence before stopping
MIN_SPEECH_DURATION = 0.3  # minimum speech before accepting silence
VAD_AGGRESSIVENESS = 3  # 0-3, higher = more aggressive filtering of non-speech


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


def stt_streaming(max_seconds=30):
    """Real-time STT via Azure WebSocket — transcribes as you speak."""
    region = CONFIG["region"]
    key = CONFIG["key"]
    conn_id = uuid.uuid4().hex

    ws_url = (
        f"wss://{region}.stt.speech.microsoft.com/speech/recognition"
        f"/conversation/cognitiveservices/v1?language=en-US&format=detailed"
    )
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "X-ConnectionId": conn_id,
    }

    result = {"text": "", "done": False, "error": None}
    lock = threading.Lock()

    def on_message(ws, message):
        try:
            if isinstance(message, str):
                # Parse the Azure Speech protocol message
                parts = message.split("\r\n\r\n", 1)
                if len(parts) < 2:
                    return
                headers_part, body = parts
                if "speech.phrase" in headers_part.lower():
                    data = json.loads(body)
                    if data.get("RecognitionStatus") == "Success":
                        nbest = data.get("NBest", [])
                        text = nbest[0]["Display"] if nbest else data.get("DisplayText", "")
                        with lock:
                            result["text"] = (result["text"] + " " + text).strip()
                elif "turn.end" in headers_part.lower():
                    with lock:
                        result["done"] = True
                    ws.close()
        except Exception:
            pass

    def on_error(ws, error):
        with lock:
            result["error"] = str(error)
            result["done"] = True

    def on_close(ws, status, msg):
        with lock:
            result["done"] = True

    def on_open(ws):
        def send_audio():
            # Send speech config
            speech_config = {
                "context": {
                    "system": {"version": "1.0.00000"},
                    "os": {"platform": "Linux", "name": "speech-to-cli"},
                    "audio": {"source": {"connectivity": "Unknown", "manufacturer": "Unknown",
                                         "model": "Unknown", "type": "Unknown"}},
                }
            }
            config_msg = (
                f"Path: speech.config\r\n"
                f"X-RequestId: {uuid.uuid4().hex}\r\n"
                f"X-Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}\r\n"
                f"Content-Type: application/json\r\n\r\n"
                + json.dumps(speech_config)
            )
            ws.send(config_msg)

            # Start arecord (raw PCM)
            proc = subprocess.Popen(
                ["arecord", "-D", "default", "-f", "S16_LE", "-r", "16000", "-c", "1",
                 "-t", "raw", "-d", str(max_seconds)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )

            request_id = uuid.uuid4().hex

            def make_audio_binary(audio_data):
                """Build Azure Speech binary frame: 2-byte header len + ASCII header + audio."""
                header_str = (
                    f"Path: audio\r\n"
                    f"X-RequestId: {request_id}\r\n"
                    f"X-Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}\r\n"
                    f"Content-Type: audio/x-wav\r\n"
                )
                header_bytes = header_str.encode('ascii')
                return struct.pack('>H', len(header_bytes)) + header_bytes + audio_data

            # First chunk includes RIFF/WAV header
            wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
                b'RIFF', 0, b'WAVE', b'fmt ', 16, 1, 1, 16000, 32000, 2, 16, b'data', 0)
            ws.send(make_audio_binary(wav_header), opcode=websocket.ABNF.OPCODE_BINARY)

            vad = webrtcvad.Vad(VAD_AGGRESSIVENESS) if HAS_VAD else None
            silence_frames = 0
            speech_frames = 0
            max_silence = int(SILENCE_TIMEOUT * 1000 / FRAME_MS)
            min_speech = int(MIN_SPEECH_DURATION * 1000 / FRAME_MS)

            try:
                while not result["done"]:
                    chunk = proc.stdout.read(FRAME_BYTES)
                    if not chunk:
                        break
                    ws.send(make_audio_binary(chunk), opcode=websocket.ABNF.OPCODE_BINARY)

                    # VAD: stop on silence after speech detected
                    if vad and len(chunk) == FRAME_BYTES:
                        is_speech = vad.is_speech(chunk, SAMPLE_RATE)
                        if is_speech:
                            speech_frames += 1
                            silence_frames = 0
                        else:
                            silence_frames += 1
                        if speech_frames >= min_speech and silence_frames >= max_silence:
                            break
            except Exception:
                pass
            finally:
                proc.terminate()
                proc.wait()

            # Send empty audio to signal end of stream
            ws.send(make_audio_binary(b""), opcode=websocket.ABNF.OPCODE_BINARY)

        threading.Thread(target=send_audio, daemon=True).start()

    ws_app = websocket.WebSocketApp(
        ws_url, header=headers,
        on_open=on_open, on_message=on_message,
        on_error=on_error, on_close=on_close,
    )
    ws_thread = threading.Thread(target=ws_app.run_forever, daemon=True)
    ws_thread.start()

    # Wait for result
    deadline = time.time() + max_seconds + 5
    while time.time() < deadline:
        with lock:
            if result["done"]:
                break
        time.sleep(0.1)

    if result["error"]:
        return {"error": result["error"]}
    return {"text": result["text"]}


def stt_vad(max_seconds=30):
    """Record with VAD — stops when silence is detected after speech."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        proc = subprocess.Popen(
            ["arecord", "-D", "default", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        frames = []
        silence_frames = 0
        speech_frames = 0
        max_silence = int(SILENCE_TIMEOUT * 1000 / FRAME_MS)
        min_speech = int(MIN_SPEECH_DURATION * 1000 / FRAME_MS)
        max_frames = int(max_seconds * 1000 / FRAME_MS)

        for _ in range(max_frames):
            chunk = proc.stdout.read(FRAME_BYTES)
            if not chunk or len(chunk) < FRAME_BYTES:
                break
            frames.append(chunk)
            is_speech = vad.is_speech(chunk, SAMPLE_RATE)
            if is_speech:
                speech_frames += 1
                silence_frames = 0
            else:
                silence_frames += 1
            if speech_frames >= min_speech and silence_frames >= max_silence:
                break

        proc.terminate()
        proc.wait()

        if not frames:
            return {"text": "", "status": "NoAudio"}

        # Write WAV file
        raw_data = b"".join(frames)
        with open(tmp_path, "wb") as f:
            f.write(struct.pack('<4sI4s4sIHHIIHH4sI',
                b'RIFF', 36 + len(raw_data), b'WAVE', b'fmt ', 16, 1, 1,
                SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16, b'data', len(raw_data)))
            f.write(raw_data)

        # Upload to Azure STT
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


def stt_fixed(seconds=5):
    """Record for a fixed duration, then upload (original fallback)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = ["arecord", "-D", "default", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav",
               "-d", str(seconds), tmp_path]
        subprocess.run(cmd, capture_output=True, timeout=seconds + 5)
        if not os.path.exists(tmp_path):
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


def stt(seconds=None, mode=None):
    """Speech-to-text with automatic mode selection.

    Modes: 'streaming' (fastest), 'vad' (stop on silence), 'fixed' (original).
    If mode is None, picks the best available.
    """
    max_seconds = min(seconds or 30, 30)

    if mode is None:
        if HAS_WS and HAS_VAD:
            mode = "streaming"
        elif HAS_VAD:
            mode = "vad"
        else:
            mode = "fixed"

    if mode == "streaming" and HAS_WS:
        return stt_streaming(max_seconds)
    elif mode == "vad" and HAS_VAD:
        return stt_vad(max_seconds)
    else:
        return stt_fixed(seconds or 5)


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
        "description": (
            "Record audio from the user's microphone and transcribe it to text using Azure Speech-to-Text. "
            "Use this to hear what the user is saying via voice. "
            "By default uses the fastest available mode: streaming (real-time WebSocket), "
            "vad (stops on silence), or fixed duration."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Max recording duration in seconds (default 30, max 30). With streaming/vad modes, recording stops early when silence is detected.",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 30,
                },
                "mode": {
                    "type": "string",
                    "description": "STT mode: 'streaming' (fastest, real-time), 'vad' (stop on silence), 'fixed' (record full duration). Default: best available.",
                    "enum": ["streaming", "vad", "fixed"],
                },
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
            result = stt(seconds=args.get("seconds"), mode=args.get("mode"))
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
