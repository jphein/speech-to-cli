#!/usr/bin/env python3
"""
Azure Speech MCP Server — compatible with Copilot CLI, Claude Code, and Gemini CLI.

Provides 'listen' (STT), 'speak' (TTS), and 'converse' tools so AI assistants
can have voice conversations.

STT modes (selected automatically):
  - streaming: Real-time WebSocket recognition + energy-gated VAD (fastest)
  - vad:       Record with energy-gated VAD, upload on silence
  - whisper:   Local transcription via faster-whisper (no network, CPU-only)
  - fixed:     Record for a fixed duration, then upload (fallback)

TTS: Streams audio playback as chunks arrive from Azure for lower latency.
"""

import array
import json
import math
import os
import random
import re
import shutil
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
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHIME_PATH = os.path.join(_SCRIPT_DIR, "chime_ready.wav")
CHIME_PROCESSING = os.path.join(_SCRIPT_DIR, "chime_processing.wav")
CHIME_SPEAK = os.path.join(_SCRIPT_DIR, "chime_speak.wav")
CHIME_DONE = os.path.join(_SCRIPT_DIR, "chime_done.wav")
CHIME_HUM = os.path.join(_SCRIPT_DIR, "chime_hum.wav")

# Audio settings
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = SAMPLE_RATE * 2 * FRAME_MS // 1000  # 960 bytes per 30ms frame

# VAD + energy gate settings
SILENCE_TIMEOUT = 3.0  # seconds of silence before stopping (increased for longer pauses)
NO_SPEECH_TIMEOUT = 7.0  # bail out if no speech detected at all for this long
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
_hum_proc = None
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
        "chime_ready": cfg.get("chime_ready", True),
        "chime_processing": cfg.get("chime_processing", False),
        "chime_speak": cfg.get("chime_speak", False),
        "chime_done": cfg.get("chime_done", False),
        "chime_hum": cfg.get("chime_hum", False),
        "visual_indicator": cfg.get("visual_indicator", True),
        "live_subtitles": cfg.get("live_subtitles", True),
        "vu_meter": cfg.get("vu_meter", True),
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

def _generate_chimes():
    """Generate all status chime WAV files if they don't exist."""
    import wave
    rate = 16000

    def _make(path, tones):
        if os.path.exists(path):
            return
        samples = []
        for freq, dur, vol in tones:
            if freq == 0:  # silence gap
                samples.extend([0] * int(rate * dur))
            else:
                for i in range(int(rate * dur)):
                    t = i / rate
                    env = min(1.0, min(t * 50, (dur - t) * 50))
                    samples.append(int(vol * env * math.sin(2 * math.pi * freq * t) * 32767))
        raw = struct.pack(f"<{len(samples)}h", *samples)
        try:
            with open(path, "wb") as f:
                w = wave.open(f, "wb")
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
                w.writeframes(raw); w.close()
        except OSError:
            pass

    _make(CHIME_PATH, [(880, 0.03, 0.3), (1175, 0.05, 0.4)])          # ready: ascending
    _make(CHIME_PROCESSING, [(1320, 0.025, 0.2)])                       # processing: soft blip
    _make(CHIME_SPEAK, [(1175, 0.03, 0.25), (880, 0.04, 0.3)])         # speak: descending
    _make(CHIME_DONE, [(1047, 0.02, 0.15), (0, 0.02, 0), (1047, 0.02, 0.15)])  # done: double tap
    _make(CHIME_HUM, [(150, 1.0, 0.1)])                                # hum: 150Hz steady

_generate_chimes()


def _play_sound(path):
    """Play a WAV file non-blocking via aplay."""
    if os.path.exists(path):
        try:
            subprocess.Popen(
                ["aplay", "-D", "default", "-q", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass


_tty_fd = None

def _get_tty():
    global _tty_fd
    if _tty_fd is None:
        # 1. Try common TTY devices
        # 2. Try the current TTY of the parent process
        # 3. Try to find the active TTY on Linux
        paths = ["/dev/tty", "/dev/pts/0", "/dev/pts/1", "/dev/pts/2"]
        try:
            ppid = os.getppid()
            ptty = subprocess.check_output(["ps", "-o", "tty=", "-p", str(ppid)], text=True).strip()
            if ptty and not ptty.startswith("?"):
                paths.insert(0, f"/dev/{ptty}")
        except Exception:
            pass
            
        for path in paths:
            try:
                f = open(path, "w")
                if f.isatty():
                    _tty_fd = f
                    break
                f.close()
            except Exception:
                continue
                
    return _tty_fd


_cached_tty_width = None
_cached_tty_width_time = 0.0

def _get_tty_width():
    """Get terminal width, cached for 2s to avoid repeated /dev/tty opens."""
    global _cached_tty_width, _cached_tty_width_time
    now = time.monotonic()
    if _cached_tty_width is not None and (now - _cached_tty_width_time) < 2.0:
        return _cached_tty_width
    width = 120
    if "COLUMNS" in os.environ:
        try:
            width = int(os.environ["COLUMNS"])
            _cached_tty_width = width
            _cached_tty_width_time = now
            return width
        except ValueError:
            pass
    try:
        with open("/dev/tty", "r") as tty:
            width = os.get_terminal_size(tty.fileno()).columns
    except Exception:
        width = shutil.get_terminal_size(fallback=(120, 24)).columns
    _cached_tty_width = width
    _cached_tty_width_time = now
    return width


def _print_status(text, color_code="90"):
    """Print a status indicator to stderr."""
    if CONFIG.get("visual_indicator", True):
        # Gemini CLI shows stderr output during tool execution
        sys.stderr.write(f"{text}\n")
        sys.stderr.flush()


def play_chime():
    """Ready chime — ascending tone, signals 'speak now'."""
    if CONFIG.get("chime_ready", True):
        _play_sound(CHIME_PATH)
    _print_status("🎤 Listening...", "92")  # Green


def play_processing():
    """Processing blip — signals STT is done, thinking."""
    global _hum_proc
    if CONFIG.get("chime_processing", False):
        _play_sound(CHIME_PROCESSING)
        
    if CONFIG.get("chime_hum", False):
        _print_status("🧠 Thinking...", "94")  # Blue
        try:
            _hum_proc = subprocess.Popen(
                ["bash", "-c", "while true; do aplay -D default -q -- \"$1\"; done", "_", CHIME_HUM],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            pass


def stop_hum():
    """Stop the looping hum if active."""
    global _hum_proc
    _print_status("") # Clear indicator
    if _hum_proc:
        try:
            import signal
            os.killpg(os.getpgid(_hum_proc.pid), signal.SIGTERM)
            _hum_proc.wait(timeout=0.1)
        except Exception:
            pass
        _hum_proc = None


def play_speak():
    """Speak chime — descending tone, signals TTS starting."""
    if CONFIG.get("chime_speak", False):
        _play_sound(CHIME_SPEAK)


def play_done():
    """Done tap — signals TTS finished."""
    if CONFIG.get("chime_done", False):
        _play_sound(CHIME_DONE)


def rms_energy(frame_bytes):
    """Calculate RMS energy of a 16-bit PCM frame (array module for speed)."""
    n_samples = len(frame_bytes) // 2
    if n_samples == 0:
        return 0.0
    a = array.array("h")
    a.frombytes(frame_bytes[:n_samples * 2])
    return math.sqrt(sum(s * s for s in a) / n_samples)


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


def stt_streaming(max_seconds=30, progress_token=None):
    """Real-time STT via persistent Azure WebSocket + energy-gated VAD."""

    def _do_streaming(ws, max_seconds):
        request_id = uuid.uuid4().hex
        send_progress(progress_token, 0, 100, "🎤 Ready...")

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
        send_progress(progress_token, 5, 100, "🎤 Calibrating...")

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
        send_progress(progress_token, 0, 100, "🎤 Listening...")
        proc = subprocess.Popen(
            ["arecord", "-D", "default", "-f", "S16_LE", "-r", "16000", "-c", "1",
             "-t", "raw", "-d", str(max_seconds)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        result_text = []
        sender_done = threading.Event()
        shared_state = {"partial": "Listening..."}

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

                max_frames = int(max_seconds * 1000 / FRAME_MS)
                last_progress_pct = 0
                bars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
                
                while True:
                    chunk = proc.stdout.read(FRAME_BYTES)
                    if not chunk:
                        break
                    ws.send(make_audio_binary(chunk), opcode=websocket.ABNF.OPCODE_BINARY)
                    if len(chunk) == FRAME_BYTES:
                        total_frames += 1
                        
                        # Calculate progress incrementally (0% to 70% during listening)
                        current_pct = int((total_frames / max_frames) * 70)
                        
                        # Calculate VU meter
                        energy = rms_energy(chunk)
                        # typical energy ranges 0 to ~1500 for normal speech
                        idx = max(0, min(7, int((energy / 1200.0) * 8)))
                        vu = bars[idx]
                        
                        # Update UI with VU meter and partial text ~every 90ms (3 frames)
                        if total_frames % 3 == 0 or current_pct > last_progress_pct:
                            parts = ["🎤"]
                            if CONFIG.get("vu_meter", True):
                                parts.append(vu)
                            if CONFIG.get("live_subtitles", True):
                                parts.append(shared_state["partial"])
                            else:
                                parts.append("Listening...")
                                
                            msg = " ".join(parts)
                            send_progress(progress_token, current_pct, 100, msg)
                            last_progress_pct = current_pct
                            
                        if is_speech_energy(chunk, vad, energy_threshold):
                            speech_frames += 1
                            silence_frames = 0
                        else:
                            silence_frames += 1
                        if speech_frames >= min_speech and silence_frames >= max_silence:
                            send_progress(progress_token, 80, 100, "🧠 Transcribing...")
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
                    send_progress(progress_token, 100, 100, "✅ Done")
                    # Early exit: if sender done, quick drain for turn.end
                    if sender_done.is_set():
                        try:
                            ws.settimeout(0.5)
                            end_msg = ws.recv()
                        except Exception:
                            pass
                        break
                elif "speech.hypothesis" in hdr.lower():
                    data = json.loads(body)
                    partial_text = data.get("Text", "")
                    if partial_text:
                        term_width = _get_tty_width()
                        window_size = max(40, term_width - 25)
                        
                        # Truncate left side if too long so Gemini CLI doesn't right-truncate with "..."
                        display_text = partial_text if len(partial_text) < window_size else "..." + partial_text[-(window_size-3):]
                        shared_state["partial"] = display_text
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


def stt_vad(max_seconds=30, progress_token=None):
    """Record with energy-gated VAD, stop on silence, upload via REST."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        play_chime()
        send_progress(progress_token, 10, 100, "🎤 Listening...")
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

        send_progress(progress_token, 50, 100, "🧠 Transcribing...")
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
            send_progress(progress_token, 100, 100, "✅ Done")
            return {"text": text}
        return {"text": "", "status": result.get("RecognitionStatus", "Unknown")}
    finally:
        try:
            os.unlink(tmp_path)
            send_progress(progress_token, 100, 100, "✅ Done")
        except OSError:
            pass


def stt_whisper(max_seconds=30, progress_token=None):
    """Local STT via faster-whisper - no network needed."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        play_chime()
        send_progress(progress_token, 0, 100, "🎤 Listening...")
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

        send_progress(progress_token, 50, 100, "🧠 Transcribing...")
        model = get_whisper_model()
        segments, _ = model.transcribe(tmp_path, beam_size=5, language="en")
        text = " ".join(seg.text.strip() for seg in segments).strip()
        send_progress(progress_token, 100, 100, "✅ Done")
        return {"text": text}
    finally:
        try:
            os.unlink(tmp_path)
            send_progress(progress_token, 100, 100, "✅ Done")
        except OSError:
            pass


def stt_fixed(seconds=5, progress_token=None):
    """Record for a fixed duration, then upload (fallback)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        play_chime()
        send_progress(progress_token, 0, 100, "🎤 Listening...")
        cmd = ["arecord", "-D", "default", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav",
               "-d", str(seconds), tmp_path]
        subprocess.run(cmd, capture_output=True, timeout=seconds + 5)
        if not os.path.exists(tmp_path):
            return {"error": "Recording failed"}

        send_progress(progress_token, 50, 100, "🧠 Transcribing...")
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
            send_progress(progress_token, 100, 100, "✅ Done")
            return {"text": text}
        return {"text": "", "status": result.get("RecognitionStatus", "Unknown")}
    finally:
        try:
            os.unlink(tmp_path)
            send_progress(progress_token, 100, 100, "✅ Done")
        except OSError:
            pass


_stt_lock = threading.Lock()

def stt(seconds=None, mode=None, silence_timeout=None, vad_aggressiveness=None, energy_multiplier=None, progress_token=None):
    """Speech-to-text with automatic mode selection."""
    max_seconds = max(1, min(int(seconds or 30), 30))

    with _stt_lock:
        global SILENCE_TIMEOUT, VAD_AGGRESSIVENESS, ENERGY_THRESHOLD_MULTIPLIER
        old_silence = SILENCE_TIMEOUT
        old_vad = VAD_AGGRESSIVENESS
        old_energy = ENERGY_THRESHOLD_MULTIPLIER

        if silence_timeout is not None:
            SILENCE_TIMEOUT = max(0.1, min(float(silence_timeout), 10.0))
        if vad_aggressiveness is not None:
            VAD_AGGRESSIVENESS = max(0, min(int(vad_aggressiveness), 3))
        if energy_multiplier is not None:
            ENERGY_THRESHOLD_MULTIPLIER = max(0.5, min(float(energy_multiplier), 20.0))

        try:
            if mode is None:
                if HAS_WS and HAS_VAD:
                    mode = "streaming"
                elif HAS_VAD:
                    mode = "vad"
                else:
                    mode = "fixed"

            if mode == "streaming" and HAS_WS:
                result = stt_streaming(max_seconds, progress_token=progress_token)
            elif mode == "whisper" and HAS_WHISPER:
                result = stt_whisper(max_seconds, progress_token=progress_token)
            elif mode == "vad" and HAS_VAD:
                result = stt_vad(max_seconds, progress_token=progress_token)
            else:
                result = stt_fixed(max_seconds, progress_token=progress_token)

            if result.get("text"):
                play_processing()
            return result
        finally:
            SILENCE_TIMEOUT = old_silence
            VAD_AGGRESSIVENESS = old_vad
            ENERGY_THRESHOLD_MULTIPLIER = old_energy


# ---------------------------------------------------------------------------
# TTS (streaming playback)
# ---------------------------------------------------------------------------

_SSML_SAFE_RE = re.compile(r'^[a-zA-Z0-9\-_.:+%() ]+$')

def _sanitize_ssml_attr(value, default="default"):
    """Reject values that could inject SSML markup."""
    if not value or not isinstance(value, str):
        return default
    value = value.strip()[:64]
    if not _SSML_SAFE_RE.match(value):
        return default
    return value

_MAX_TTS_CHARS = 5000

def tts(text, quality="fast", speed=1.0, voice=None, pitch="default", volume="default", progress_token=None):
    """Speak text aloud via Azure TTS with streaming playback."""
    stop_hum()
    send_progress(progress_token, 0, 100, "🔊 Synthesizing...")

    if not text or not isinstance(text, str):
        return {"error": "No text provided"}
    text = text[:_MAX_TTS_CHARS]

    if not voice:
        voice = CONFIG["fast_voice"] if quality == "fast" else CONFIG["voice"]
    voice = _sanitize_ssml_attr(voice, CONFIG["fast_voice"])
    pitch = _sanitize_ssml_attr(pitch, "default")
    volume = _sanitize_ssml_attr(volume, "default")

    safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    url = f"https://{CONFIG['region']}.tts.speech.microsoft.com/cognitiveservices/v1"
    
    # Construct prosody tag with all modifiers
    prosody_attrs = []
    if quality == "fast":
        prosody_attrs.append('rate="+15%"')
    elif speed != 1.0:
        rate_pct = int((speed - 1.0) * 100)
        sign = "+" if rate_pct >= 0 else ""
        prosody_attrs.append(f'rate="{sign}{rate_pct}%"')
    
    if pitch != "default":
        prosody_attrs.append(f'pitch="{pitch}"')
    if volume != "default":
        prosody_attrs.append(f'volume="{volume}"')
    
    prosody_str = " ".join(prosody_attrs)
    if prosody_str:
        body_ssml = f'<prosody {prosody_str}>{safe_text}</prosody>'
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
        return {"error": f"Azure TTS error {resp.status_code}: {resp.text}"}

    send_progress(progress_token, 5, 100, "🔊 Speaking...")
    play_speak()

    # Stream MP3 audio via mpv/ffplay for immediate playback
    for player_cmd in [
        ["mpv", "--no-terminal", "--no-video", "-"],
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
        # We download chunks much faster than they play.
        # To make a smooth progress bar, we calculate estimated duration:
        # fast voice is ~20 chars/sec, hd is slower. Let's use 22 for fast to ensure it finishes early.
        speed_factor = 22.0 if quality == "fast" else 15.0
        estimated_duration = max(1.0, len(text) / speed_factor)
        start_time = time.time()
        
        def download_audio():
            try:
                for chunk in resp.iter_content(chunk_size=16384):
                    if chunk:
                        proc.stdin.write(chunk)
                proc.stdin.close()
            except Exception:
                pass

        dl_thread = threading.Thread(target=download_audio, daemon=True)
        dl_thread.start()

        bars = [" ", "▂", "▃", "▄", "▅"]
        total_len = len(text)
        last_msg = ""
        show_vu = CONFIG.get("vu_meter", True)
        show_subs = CONFIG.get("live_subtitles", True)

        while proc.poll() is None:
            elapsed = time.time() - start_time
            current_pct = int(min(1.0, elapsed / estimated_duration) * 100)

            vu_prefix = f"{random.choice(bars)} " if show_vu else ""

            if show_subs and total_len > 0:
                char_idx = int((current_pct / 100.0) * total_len)
                window_size = max(40, _get_tty_width() - 25)
                start_idx = max(0, char_idx - window_size)
                snippet = text[start_idx:char_idx]
                base_msg = f"Speaking: {snippet}"
            else:
                base_msg = "Speaking..."

            msg = f"🔊 {vu_prefix}{base_msg}"
            if msg != last_msg or show_vu:
                send_progress(progress_token, current_pct, 100, msg)
                last_msg = msg

            time.sleep(0.1)
            
        dl_thread.join()
        
    except BrokenPipeError:
        pass
    finally:
        stop_hum()
        send_progress(progress_token, 100, 100, "✅ Done")
    play_done()
    return {"spoken": True, "chars": len(text)}


def get_voices():
    """Fetch available voices from Azure."""
    url = f"https://{CONFIG['region']}.tts.speech.microsoft.com/cognitiveservices/voices/list"
    headers = {"Ocp-Apim-Subscription-Key": CONFIG["key"]}
    try:
        resp = get_http_session().get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"Azure error {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# MCP Protocol (stdio JSON-RPC)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "listen",
        "description": (
            "Listen through the microphone and return what the user said as text. "
            "Stops automatically when the user finishes speaking. "
            "This is listen-only — if you also need to speak, use 'talk' instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Max recording duration (default 30).",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 30,
                },
                "mode": {
                    "type": "string",
                    "description": "STT mode: 'streaming' (fastest), 'vad', 'whisper', 'fixed'.",
                    "enum": ["streaming", "vad", "whisper", "fixed"],
                },
                "silence_timeout": {
                    "type": "number",
                    "description": "Seconds of silence before stopping (default 0.8).",
                },
                "vad_aggressiveness": {
                    "type": "integer",
                    "description": "VAD level 0-3 (3 is most aggressive, default 3).",
                    "minimum": 0,
                    "maximum": 3,
                },
                "energy_multiplier": {
                    "type": "number",
                    "description": "Energy threshold multiplier for noise gating (default 2.5).",
                },
            },
        },
    },
    {
        "name": "speak",
        "description": (
            "Say something out loud (text-to-speech) WITHOUT listening for a reply. "
            "Use this for one-way announcements or final messages. "
            "If you want to speak AND hear the user's response, use 'talk' instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to speak aloud",
                },
                "quality": {
                    "type": "string",
                    "description": "Voice quality: 'fast' or 'hd'.",
                    "enum": ["fast", "hd"],
                    "default": "fast",
                },
                "voice": {
                    "type": "string",
                    "description": "Azure voice name (e.g. 'en-US-AvaNeural').",
                },
                "speed": {
                    "type": "number",
                    "description": "Playback speed multiplier (default 1.0).",
                    "default": 1.0,
                },
                "pitch": {
                    "type": "string",
                    "description": "Pitch: 'high', 'low', '+20%', or 'default'.",
                },
                "volume": {
                    "type": "string",
                    "description": "Volume: 'loud', 'soft', '+10%', or 'default'.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "get_voices",
        "description": "List available Azure Speech voices for the current region.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "converse",
        "description": (
            "Listen to the user through their microphone and return what they said as text. "
            "Use this to START a voice conversation (listen first, then respond with 'talk'). "
            "For ongoing back-and-forth, prefer 'talk' which speaks AND listens in one step."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Max recording duration (default 30).",
                    "default": 30,
                },
                "mode": {
                    "type": "string",
                    "description": "STT mode (streaming, vad, whisper, fixed).",
                },
                "silence_timeout": {
                    "type": "number",
                    "description": "Seconds of silence before stopping.",
                },
            },
        },
    },
    {
        "name": "talk",
        "description": (
            "Say something out loud, then immediately listen for the user's reply. "
            "This is the PRIMARY tool for voice conversations — it speaks your response "
            "and waits for the user to talk back, all in one fast step. "
            "Just pass your message as 'text' and you'll get back what the user said. "
            "Keep calling 'talk' each turn to continue the conversation. "
            "Use 'speak' instead only when you want to say something final with no reply expected."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to speak aloud before listening.",
                },
                "quality": {
                    "type": "string",
                    "description": "Voice quality: 'fast' or 'hd'.",
                    "enum": ["fast", "hd"],
                    "default": "fast",
                },
                "speed": {
                    "type": "number",
                    "description": "Playback speed multiplier (default 1.0).",
                    "default": 1.0,
                },
                "voice": {
                    "type": "string",
                    "description": "Azure voice name.",
                },
                "pitch": {
                    "type": "string",
                    "description": "Pitch: 'high', 'low', '+20%', or 'default'.",
                },
                "volume": {
                    "type": "string",
                    "description": "Volume: 'loud', 'soft', '+10%', or 'default'.",
                },
                "seconds": {
                    "type": "integer",
                    "description": "Max recording duration (default 30).",
                    "default": 30,
                },
                "mode": {
                    "type": "string",
                    "description": "STT mode (streaming, vad, whisper, fixed).",
                },
                "silence_timeout": {
                    "type": "number",
                    "description": "Seconds of silence before stopping.",
                },
            },
            "required": ["text"],
        },
    },
]


def send_progress(token, progress, total=None, description=None):
    """Send an MCP progress notification if a token is provided."""
    if token is None:
        return
    msg = {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {
            "progressToken": token,
            "progress": progress,
        }
    }
    if total is not None:
        msg["params"]["total"] = total
    if description:
        # Gemini CLI uses the 'message' field to show status text
        msg["params"]["message"] = description
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def handle_request(req):
    method = req.get("method")
    params = req.get("params", {})
    req_id = req.get("id")
    # MCP progress tokens can be in the params
    progress_token = params.get("_meta", {}).get("progressToken")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": True}},
                "serverInfo": {"name": "azure-speech", "version": "3.5.0"},
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
            result = stt(
                seconds=args.get("seconds"),
                mode=args.get("mode"),
                silence_timeout=args.get("silence_timeout"),
                vad_aggressiveness=args.get("vad_aggressiveness"),
                energy_multiplier=args.get("energy_multiplier"),
                progress_token=progress_token
            )
            text = result.get("text", result.get("error", ""))
            content_text = text or "(no speech detected)"
            if tool_name == "converse":
                content_text += "\n\n[Voice conversation active — call 'talk' with your response to speak and listen in one step, or call 'speak' for a final message with no reply needed.]"
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": content_text}]},
            }
        elif tool_name == "speak":
            speak_text = args.get("text", "")
            if not speak_text or not isinstance(speak_text, str):
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": "Error: 'text' is required."}]},
                }
            quality = args.get("quality", "fast")
            if quality not in ("fast", "hd"):
                quality = "fast"
            speed_val = args.get("speed", 1.0)
            if not isinstance(speed_val, (int, float)):
                speed_val = 1.0
            speed_val = max(0.5, min(float(speed_val), 3.0))
            result = tts(
                speak_text,
                quality=quality,
                voice=args.get("voice"),
                speed=speed_val,
                pitch=args.get("pitch", "default"),
                volume=args.get("volume", "default"),
                progress_token=progress_token
            )
            msg = "Spoke the text aloud." if result.get("spoken") else result.get("error", "Failed")
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": msg}]},
            }
        elif tool_name == "talk":
            # Combo: speak then immediately listen — eliminates round-trip latency
            speak_text = args.get("text", "")
            if not speak_text or not isinstance(speak_text, str):
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": "Error: 'text' is required."}]},
                }
            quality = args.get("quality", "fast")
            if quality not in ("fast", "hd"):
                quality = "fast"
            speed_val = args.get("speed", 1.0)
            if not isinstance(speed_val, (int, float)):
                speed_val = 1.0
            speed_val = max(0.5, min(float(speed_val), 3.0))
            tts_result = tts(
                speak_text,
                quality=quality,
                voice=args.get("voice"),
                speed=speed_val,
                pitch=args.get("pitch", "default"),
                volume=args.get("volume", "default"),
                progress_token=progress_token,
            )
            if not tts_result.get("spoken"):
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": tts_result.get("error", "TTS failed")}]},
                }
            # Now immediately listen
            stt_result = stt(
                seconds=args.get("seconds"),
                mode=args.get("mode"),
                silence_timeout=args.get("silence_timeout"),
                progress_token=progress_token,
            )
            user_said = stt_result.get("text", stt_result.get("error", "")) if isinstance(stt_result, dict) else str(stt_result)
            content_text = user_said or "(no speech detected)"
            content_text += "\n\n[Voice conversation active — call 'talk' again with your response to continue, or 'speak' for a final message with no reply needed.]"
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": content_text}]},
            }
        elif tool_name == "get_voices":
            voices = get_voices()
            if isinstance(voices, dict) and "error" in voices:
                text = voices["error"]
            else:
                # Format a subset of info for brevity
                lines = [f"{v['ShortName']} ({v['Gender']}, {v['LocaleName']})" for v in voices[:50]]
                text = "Available voices (first 50):\n" + "\n".join(lines)
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": text}]},
            }
        else:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }
    # Methods that Claude Code / Gemini CLI may probe
    elif method == "resources/list":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "resources": [
                    {
                        "uri": "speech://readme",
                        "name": "Speech-to-CLI Documentation",
                        "description": "The complete README and documentation for the speech-to-cli MCP server, including features, usage, and configuration.",
                        "mimeType": "text/markdown"
                    },
                    {
                        "uri": "speech://config-schema",
                        "name": "Configuration Schema",
                        "description": "The current configuration settings loaded by the MCP server.",
                        "mimeType": "application/json"
                    }
                ]
            }
        }
    elif method == "resources/read":
        uri = params.get("uri")
        if uri == "speech://readme":
            try:
                readme_path = os.path.join(_SCRIPT_DIR, "README.md")
                with open(readme_path, "r") as f:
                    content = f.read()
            except Exception as e:
                content = f"Error reading README: {e}"
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "text/markdown",
                            "text": content
                        }
                    ]
                }
            }
        elif uri == "speech://config-schema":
            safe_config = {k: ("***" if k == "key" else v) for k, v in CONFIG.items()}
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(safe_config, indent=2)
                        }
                    ]
                }
            }
        else:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32602, "message": f"Resource not found: {uri}"}
            }
    elif method in ("resources/templates/list", "prompts/list", "completion/complete"):
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    elif method == "notifications/cancelled":
        return None
    else:
        # Ignore unknown notifications (no id); error on unknown requests
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
