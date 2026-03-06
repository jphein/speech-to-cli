#!/usr/bin/env python3
"""
Azure Speech MCP Server for Copilot CLI.

Provides 'listen' (STT), 'speak' (TTS), and 'converse' tools so Copilot
can have voice conversations.

STT modes (selected automatically):
  - streaming: Real-time WebSocket recognition + energy-gated VAD (fastest)
  - vad:       Record with energy-gated VAD, upload on silence
  - whisper:   Local transcription via faster-whisper (no network, CPU-only)
  - fixed:     Record for a fixed duration, then upload (fallback)

TTS: Streams audio playback as chunks arrive from Azure for lower latency.
"""

import json
import math
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

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
    _whisper_model = None
except ImportError:
    HAS_WHISPER = False

DEFAULTS_PATH = os.path.expanduser("~/.config/speech-to-cli/config.json")
CHIME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chime.wav")

# Audio settings
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = SAMPLE_RATE * 2 * FRAME_MS // 1000  # 960 bytes per 30ms frame

# VAD + energy gate settings
SILENCE_TIMEOUT = 0.4  # seconds of silence before stopping
NO_SPEECH_TIMEOUT = 3.0  # bail out if no speech detected at all for this long
MIN_SPEECH_DURATION = 0.15  # minimum speech before accepting silence
VAD_AGGRESSIVENESS = 3  # 0-3, higher filters more non-speech
ENERGY_CALIBRATION_FRAMES = 5  # ~0.15s of ambient noise sampling
ENERGY_THRESHOLD_MULTIPLIER = 2.5  # speech must be Nx louder than ambient
NOISE_CACHE_TTL = 30.0  # seconds before re-calibrating noise floor

# Cached state for persistent connections and noise floor
_cached_noise_threshold = None
_cached_noise_time = 0.0
_http_session = None
_persistent_ws = None
_persistent_ws_time = 0.0
WS_IDLE_TIMEOUT = 540  # Azure closes at ~600s


def load_config():
    cfg = {}
    if os.path.exists(DEFAULTS_PATH):
        with open(DEFAULTS_PATH) as f:
            cfg = json.load(f)
    return {
        "key": os.environ.get("AZURE_SPEECH_KEY") or cfg.get("key"),
        "region": os.environ.get("AZURE_SPEECH_REGION") or cfg.get("region", "westus2"),
        "voice": os.environ.get("AZURE_SPEECH_VOICE") or cfg.get("voice", "en-US-Ava:DragonHDLatestNeural"),
        "fast_voice": cfg.get("fast_voice", "en-US-AvaNeural"),
    }


CONFIG = load_config()


def get_http_session():
    """Reuse HTTP session for connection pooling (saves ~150ms per TTS call)."""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
    return _http_session


def _prewarm():
    """Pre-establish connections in background so first call is fast."""
    try:
        get_http_session()
    except Exception:
        pass
    try:
        if HAS_WS:
            _get_stt_ws()  # noqa: defined below
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def play_chime():
    """Play a short ready chime (non-blocking, ~80ms audio)."""
    if os.path.exists(CHIME_PATH):
        subprocess.Popen(
            ["aplay", "-D", "default", "-q", CHIME_PATH],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def rms_energy(frame_bytes):
    """Calculate RMS energy of a 16-bit PCM frame."""
    n_samples = len(frame_bytes) // 2
    if n_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{n_samples}h", frame_bytes[:n_samples * 2])
    return math.sqrt(sum(s * s for s in samples) / n_samples)


def calibrate_noise(proc, n_frames=ENERGY_CALIBRATION_FRAMES):
    """Estimate ambient noise energy. Uses cached threshold if fresh (saves ~150ms)."""
    global _cached_noise_threshold, _cached_noise_time
    if _cached_noise_threshold is not None and (time.time() - _cached_noise_time) < NOISE_CACHE_TTL:
        chunk = proc.stdout.read(FRAME_BYTES)
        frames = [chunk] if chunk and len(chunk) == FRAME_BYTES else []
        return _cached_noise_threshold, frames
    energies = []
    frames = []
    for _ in range(n_frames):
        chunk = proc.stdout.read(FRAME_BYTES)
        if not chunk or len(chunk) < FRAME_BYTES:
            break
        frames.append(chunk)
        energies.append(rms_energy(chunk))
    if not energies:
        return 500.0, frames
    ambient = sum(energies) / len(energies)
    threshold = max(ambient * ENERGY_THRESHOLD_MULTIPLIER, 300.0)
    _cached_noise_threshold = threshold
    _cached_noise_time = time.time()
    return threshold, frames


def is_speech_energy(chunk, vad, energy_threshold):
    """Combined VAD + energy gate: both must agree it's speech."""
    energy = rms_energy(chunk)
    if energy < energy_threshold:
        return False
    if vad:
        return vad.is_speech(chunk, SAMPLE_RATE)
    return True


def get_whisper_model():
    """Lazy-load whisper model (first call downloads ~150MB)."""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def write_wav(path, raw_data):
    """Write raw PCM data as a WAV file."""
    with open(path, "wb") as f:
        f.write(struct.pack('<4sI4s4sIHHIIHH4sI',
            b'RIFF', 36 + len(raw_data), b'WAVE', b'fmt ', 16, 1, 1,
            SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16, b'data', len(raw_data)))
        f.write(raw_data)


def record_with_vad(proc, max_seconds):
    """Record with energy-gated VAD, returning (raw_frames, energy_threshold)."""
    energy_threshold, calibration_frames = calibrate_noise(proc)
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS) if HAS_VAD else None
    frames = list(calibration_frames)
    silence_frames = 0
    speech_frames = 0
    max_silence = int(SILENCE_TIMEOUT * 1000 / FRAME_MS)
    max_no_speech = int(NO_SPEECH_TIMEOUT * 1000 / FRAME_MS)
    min_speech = int(MIN_SPEECH_DURATION * 1000 / FRAME_MS)
    max_frames = int(max_seconds * 1000 / FRAME_MS)
    total_frames = 0

    for _ in range(max_frames):
        chunk = proc.stdout.read(FRAME_BYTES)
        if not chunk or len(chunk) < FRAME_BYTES:
            break
        frames.append(chunk)
        total_frames += 1
        if is_speech_energy(chunk, vad, energy_threshold):
            speech_frames += 1
            silence_frames = 0
        else:
            silence_frames += 1
        # Stop after silence following speech
        if speech_frames >= min_speech and silence_frames >= max_silence:
            break
        # Bail out if no speech detected at all for too long
        if speech_frames == 0 and total_frames >= max_no_speech:
            break

    return frames, energy_threshold


# ---------------------------------------------------------------------------
# STT backends
# ---------------------------------------------------------------------------

def _get_stt_ws():
    """Get or create persistent WebSocket to Azure STT (saves ~230ms on reuse)."""
    global _persistent_ws, _persistent_ws_time
    now = time.time()
    if _persistent_ws is not None and (now - _persistent_ws_time) < WS_IDLE_TIMEOUT:
        _persistent_ws_time = now
        return _persistent_ws
    if _persistent_ws is not None:
        try:
            _persistent_ws.close()
        except Exception:
            pass
        _persistent_ws = None
    region = CONFIG["region"]
    key = CONFIG["key"]
    _persistent_ws = websocket.create_connection(
        f"wss://{region}.stt.speech.microsoft.com/speech/recognition"
        f"/conversation/cognitiveservices/v1?language=en-US&format=detailed",
        header=[
            f"Ocp-Apim-Subscription-Key: {key}",
            f"X-ConnectionId: {uuid.uuid4().hex}",
        ],
        timeout=10,
    )
    _persistent_ws_time = time.time()
    return _persistent_ws


def _invalidate_stt_ws():
    """Close and discard the persistent WebSocket."""
    global _persistent_ws
    if _persistent_ws is not None:
        try:
            _persistent_ws.close()
        except Exception:
            pass
        _persistent_ws = None


# Pre-warm connections in background on module load
threading.Thread(target=_prewarm, daemon=True).start()


def stt_streaming(max_seconds=30):
    """Real-time STT via persistent Azure WebSocket + energy-gated VAD."""

    def _do_streaming(ws, max_seconds):
        request_id = uuid.uuid4().hex

        # Drain any stale messages from previous turn
        try:
            ws.settimeout(0.05)
            while True:
                ws.recv()
        except Exception:
            pass

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
            f"X-RequestId: {request_id}\r\n"
            f"X-Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}\r\n"
            f"Content-Type: application/json\r\n\r\n"
            + json.dumps(speech_config)
        )
        ws.send(config_msg)

        def make_audio_binary(audio_data):
            header_str = (
                f"Path: audio\r\n"
                f"X-RequestId: {request_id}\r\n"
                f"X-Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}\r\n"
                f"Content-Type: audio/x-wav\r\n"
            )
            header_bytes = header_str.encode('ascii')
            return struct.pack('>H', len(header_bytes)) + header_bytes + audio_data

        play_chime()
        proc = subprocess.Popen(
            ["arecord", "-D", "default", "-f", "S16_LE", "-r", "16000", "-c", "1",
             "-t", "raw", "-d", str(max_seconds)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        result_text = []
        sender_done = threading.Event()

        def send_audio():
            try:
                energy_threshold, calibration_frames = calibrate_noise(proc)
                wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
                    b'RIFF', 0, b'WAVE', b'fmt ', 16, 1, 1, 16000, 32000, 2, 16, b'data', 0)
                ws.send(make_audio_binary(wav_header), opcode=websocket.ABNF.OPCODE_BINARY)

                for frame in calibration_frames:
                    ws.send(make_audio_binary(frame), opcode=websocket.ABNF.OPCODE_BINARY)

                vad = webrtcvad.Vad(VAD_AGGRESSIVENESS) if HAS_VAD else None
                silence_frames = 0
                speech_frames = 0
                total_frames = 0
                max_silence = int(SILENCE_TIMEOUT * 1000 / FRAME_MS)
                max_no_speech = int(NO_SPEECH_TIMEOUT * 1000 / FRAME_MS)
                min_speech = int(MIN_SPEECH_DURATION * 1000 / FRAME_MS)

                while True:
                    chunk = proc.stdout.read(FRAME_BYTES)
                    if not chunk:
                        break
                    ws.send(make_audio_binary(chunk), opcode=websocket.ABNF.OPCODE_BINARY)
                    if len(chunk) == FRAME_BYTES:
                        total_frames += 1
                        if is_speech_energy(chunk, vad, energy_threshold):
                            speech_frames += 1
                            silence_frames = 0
                        else:
                            silence_frames += 1
                        if speech_frames >= min_speech and silence_frames >= max_silence:
                            break
                        if speech_frames == 0 and total_frames >= max_no_speech:
                            break
            except Exception:
                pass
            finally:
                proc.terminate()
                proc.wait()
                try:
                    ws.send(make_audio_binary(b""), opcode=websocket.ABNF.OPCODE_BINARY)
                except Exception:
                    pass
                sender_done.set()

        sender = threading.Thread(target=send_audio, daemon=True)
        sender.start()

        # Read responses until turn.end
        deadline = time.time() + max_seconds + 5
        got_phrase = False
        while time.time() < deadline:
            try:
                ws.settimeout(1.0)
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                if sender_done.is_set():
                    if got_phrase:
                        break  # Already have result, don't wait longer
                    try:
                        ws.settimeout(2.0)
                        msg = ws.recv()
                    except Exception:
                        break
                else:
                    continue
            except Exception:
                break

            if isinstance(msg, str):
                parts = msg.split("\r\n\r\n", 1)
                if len(parts) < 2:
                    continue
                hdr, body = parts
                if "speech.phrase" in hdr.lower():
                    data = json.loads(body)
                    if data.get("RecognitionStatus") == "Success":
                        nbest = data.get("NBest", [])
                        text = nbest[0]["Display"] if nbest else data.get("DisplayText", "")
                        result_text.append(text)
                    got_phrase = True
                    # Early exit: if sender done, quick drain for turn.end
                    if sender_done.is_set():
                        try:
                            ws.settimeout(0.5)
                            end_msg = ws.recv()
                        except Exception:
                            pass
                        break
                elif "turn.end" in hdr.lower():
                    break

        sender.join(timeout=2)
        return " ".join(result_text).strip()

    # Try persistent connection, retry once on failure
    for attempt in range(2):
        try:
            ws = _get_stt_ws()
            text = _do_streaming(ws, max_seconds)
            return {"text": text}
        except Exception as e:
            _invalidate_stt_ws()
            if attempt == 0:
                continue
            return {"error": str(e)}


def stt_vad(max_seconds=30):
    """Record with energy-gated VAD, stop on silence, upload via REST."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        play_chime()
        proc = subprocess.Popen(
            ["arecord", "-D", "default", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        frames, _ = record_with_vad(proc, max_seconds)
        proc.terminate()
        proc.wait()

        if not frames:
            return {"text": "", "status": "NoAudio"}

        raw_data = b"".join(frames)
        write_wav(tmp_path, raw_data)

        url = f"https://{CONFIG['region']}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": CONFIG["key"],
            "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
        }
        with open(tmp_path, "rb") as f:
            resp = get_http_session().post(url, params={"language": "en-US", "format": "detailed"},
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


def stt_whisper(max_seconds=30):
    """Local STT via faster-whisper - no network needed."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        play_chime()
        proc = subprocess.Popen(
            ["arecord", "-D", "default", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        if HAS_VAD:
            frames, _ = record_with_vad(proc, max_seconds)
            proc.terminate()
            proc.wait()
            raw_data = b"".join(frames)
        else:
            raw_data = proc.stdout.read(SAMPLE_RATE * 2 * max_seconds)
            proc.terminate()
            proc.wait()

        if not raw_data:
            return {"text": "", "status": "NoAudio"}

        write_wav(tmp_path, raw_data)

        model = get_whisper_model()
        segments, _ = model.transcribe(tmp_path, beam_size=5, language="en")
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return {"text": text}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def stt_fixed(seconds=5):
    """Record for a fixed duration, then upload (fallback)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        play_chime()
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
            resp = get_http_session().post(url, params={"language": "en-US", "format": "detailed"},
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

    Modes: 'streaming', 'vad', 'whisper', 'fixed'.
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
    elif mode == "whisper" and HAS_WHISPER:
        return stt_whisper(max_seconds)
    elif mode == "vad" and HAS_VAD:
        return stt_vad(max_seconds)
    else:
        return stt_fixed(seconds or 5)


# ---------------------------------------------------------------------------
# TTS (streaming playback)
# ---------------------------------------------------------------------------

def tts(text, quality="fast", speed=1.0):
    """Speak text aloud via Azure TTS with streaming playback.
    
    quality: 'fast' uses standard Neural voice (~120ms), 'hd' uses DragonHD (~1200ms).
    speed: playback speed multiplier (1.0 = normal, 1.2 = 20% faster).
    """
    voice = CONFIG["fast_voice"] if quality == "fast" else CONFIG["voice"]
    safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    url = f"https://{CONFIG['region']}.tts.speech.microsoft.com/cognitiveservices/v1"
    # Use SSML prosody rate to generate faster speech in fast mode (smaller audio, faster playback)
    if quality == "fast":
        body_ssml = f'<prosody rate="+15%">{safe_text}</prosody>'
    else:
        body_ssml = safe_text
    ssml = (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        f'<voice name="{voice}">{body_ssml}</voice></speak>'
    )
    headers = {
        "Ocp-Apim-Subscription-Key": CONFIG["key"],
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3",
    }
    resp = get_http_session().post(url, headers=headers, data=ssml.encode("utf-8"), timeout=60, stream=True)
    if resp.status_code != 200:
        return {"error": f"Azure TTS error {resp.status_code}"}

    # Stream MP3 audio via mpv/ffplay for immediate playback
    speed_args = ["--speed=" + str(speed)] if speed != 1.0 else []
    for player_cmd in [
        ["mpv", "--no-terminal", "--no-video"] + speed_args + ["-"],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"],
    ]:
        try:
            proc = subprocess.Popen(
                player_cmd,
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            break
        except FileNotFoundError:
            continue
    else:
        # Fallback: aplay can't play MP3, so buffer and convert
        import tempfile
        audio_data = resp.content
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        subprocess.run(["ffmpeg", "-y", "-i", tmp_path, "-f", "wav", "-acodec", "pcm_s16le", "-"],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return {"spoken": True, "chars": len(text)}

    try:
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                proc.stdin.write(chunk)
        proc.stdin.close()
        proc.wait()
    except BrokenPipeError:
        pass
    return {"spoken": True, "chars": len(text)}


# ---------------------------------------------------------------------------
# MCP Protocol (stdio JSON-RPC)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "listen",
        "description": (
            "Record audio from the user's microphone and transcribe it to text. "
            "Uses energy-gated voice activity detection to stop automatically when the user "
            "finishes speaking (ignores background noise). "
            "Modes: 'streaming' (real-time Azure WebSocket, fastest), "
            "'vad' (record then upload), 'whisper' (local, no network), 'fixed' (timed). "
            "Default: best available."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Max recording duration in seconds (default 30, max 30). With streaming/vad/whisper modes, recording stops early on silence.",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 30,
                },
                "mode": {
                    "type": "string",
                    "description": "STT mode: 'streaming' (fastest, real-time Azure), 'vad' (stop on silence, upload), 'whisper' (local, offline), 'fixed' (full duration). Default: best available.",
                    "enum": ["streaming", "vad", "whisper", "fixed"],
                },
            },
        },
    },
    {
        "name": "speak",
        "description": "Speak text aloud to the user using Azure Text-to-Speech with streaming playback. Use quality='fast' for conversation (10x faster) or 'hd' for DragonHD quality.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to speak aloud",
                },
                "quality": {
                    "type": "string",
                    "description": "Voice quality: 'fast' (standard Neural, ~120ms) or 'hd' (DragonHD, ~1200ms). Default: fast.",
                    "enum": ["fast", "hd"],
                    "default": "fast",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "converse",
        "description": (
            "Have a voice conversation turn: listen to the user via microphone, return their speech as text, "
            "AND after you process it and call 'speak' with your reply, call 'converse' again to keep the "
            "conversation going. This creates a natural voice chat loop. "
            "Equivalent to calling 'listen' but signals conversational intent."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Max recording duration in seconds (default 30).",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 30,
                },
                "mode": {
                    "type": "string",
                    "description": "STT mode (same as listen).",
                    "enum": ["streaming", "vad", "whisper", "fixed"],
                },
            },
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
                "serverInfo": {"name": "azure-speech", "version": "3.0.0"},
            },
        }
    elif method == "notifications/initialized":
        return None
    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})
        if tool_name in ("listen", "converse"):
            result = stt(seconds=args.get("seconds"), mode=args.get("mode"))
            text = result.get("text", result.get("error", ""))
            content_text = text or "(no speech detected)"
            if tool_name == "converse":
                content_text += "\n\n[Voice conversation active - speak your response using the 'speak' tool, then call 'converse' again to keep listening.]"
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": content_text}]},
            }
        elif tool_name == "speak":
            result = tts(args.get("text", ""), quality=args.get("quality", "fast"))
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
