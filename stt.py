"""Speech-to-text backends: streaming WebSocket, VAD+REST, Whisper, fixed-duration.

Import pattern: `import state` at top. Reassignable globals accessed as `state.X`.
"""

import json
import os
import struct
import subprocess
import tempfile
import threading
import time
import uuid

import state
from state import (CONFIG, HAS_VAD, HAS_WS, HAS_WHISPER,
                   is_cancelled, register_proc, unregister_proc,
                   get_http_session, send_progress, _get_iso_timestamp)
from audio import (play_chime, play_processing, calibrate_noise, is_speech_energy,
                   record_with_vad, write_wav, rms_energy, get_whisper_model,
                   _build_rec_cmd, _take_prewarmed_rec, _get_tty_width, _colorize)

if HAS_WS:
    import websocket
if HAS_VAD:
    import webrtcvad


# ---------------------------------------------------------------------------
# WebSocket helpers
# ---------------------------------------------------------------------------

def _make_ws_audio_msg(request_id, audio_data):
    """Build a binary WebSocket message for Azure STT audio streaming."""
    hdr = (
        f"Path: audio\r\n"
        f"X-RequestId: {request_id}\r\n"
        f"X-Timestamp: {_get_iso_timestamp()}\r\n"
        f"Content-Type: audio/x-wav\r\n"
    ).encode('ascii')
    return struct.pack('>H', len(hdr)) + hdr + audio_data


def _get_stt_ws():
    """Get or create persistent WebSocket to Azure STT (saves ~230ms on reuse)."""
    now = time.time()
    if state._persistent_ws is not None and (now - state._persistent_ws_time) < state.WS_IDLE_TIMEOUT:
        state._persistent_ws_time = now
        return state._persistent_ws
    if state._persistent_ws is not None:
        try:
            state._persistent_ws.close()
        except Exception:
            pass
        state._persistent_ws = None
    region = CONFIG["region"]
    key = CONFIG["key"]
    state._persistent_ws = websocket.create_connection(
        f"wss://{region}.stt.speech.microsoft.com/speech/recognition"
        f"/conversation/cognitiveservices/v1?language=en-US&format=detailed",
        header=[
            f"Ocp-Apim-Subscription-Key: {key}",
            f"X-ConnectionId: {uuid.uuid4().hex}",
        ],
        timeout=10,
    )
    state._persistent_ws_time = time.time()
    return state._persistent_ws


def _invalidate_stt_ws():
    """Close and discard the persistent WebSocket."""
    if state._persistent_ws is not None:
        try:
            state._persistent_ws.close()
        except Exception:
            pass
        state._persistent_ws = None


# ---------------------------------------------------------------------------
# STT backends
# ---------------------------------------------------------------------------

def stt_streaming(max_seconds=30, progress_token=None):
    """Real-time STT via persistent Azure WebSocket + energy-gated VAD."""

    def _do_streaming(ws, max_seconds):
        request_id = uuid.uuid4().hex

        # Take pre-warmed recorder FIRST (gives it max time while we set up WS)
        proc = _take_prewarmed_rec()
        if proc is None:
            proc = subprocess.Popen(
                _build_rec_cmd(max_seconds=max_seconds),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )

        # Drain stale WS messages + send config while recorder initializes
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
            f"X-Timestamp: {_get_iso_timestamp()}\r\n"
            f"Content-Type: application/json\r\n\r\n"
            + json.dumps(speech_config)
        )
        ws.send(config_msg)

        play_chime()
        send_progress(progress_token, 0, 100, "🎤 Listening...")
        register_proc(proc)

        result_text = []
        sender_done = threading.Event()
        shared_state = {"partial": "Listening..."}

        def send_audio():
            try:
                energy_threshold, calibration_frames = calibrate_noise(proc)
                wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
                    b'RIFF', 0, b'WAVE', b'fmt ', 16, 1, 1, 16000, 32000, 2, 16, b'data', 0)
                ws.send(_make_ws_audio_msg(request_id, wav_header), opcode=websocket.ABNF.OPCODE_BINARY)

                for frame in calibration_frames:
                    ws.send(_make_ws_audio_msg(request_id, frame), opcode=websocket.ABNF.OPCODE_BINARY)

                vad = webrtcvad.Vad(state.VAD_AGGRESSIVENESS) if HAS_VAD else None
                silence_frames = 0
                speech_frames = 0
                total_frames = 0
                max_silence = int(state.SILENCE_TIMEOUT * 1000 / state.FRAME_MS)
                max_no_speech = int(state.NO_SPEECH_TIMEOUT * 1000 / state.FRAME_MS)
                min_speech = int(state.MIN_SPEECH_DURATION * 1000 / state.FRAME_MS)

                max_frames = int(max_seconds * 1000 / state.FRAME_MS)
                last_progress_pct = 0
                bars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
                _show_vu = CONFIG.get("vu_meter", True)
                _show_subs = CONFIG.get("live_subtitles", True)
                _user_color = CONFIG.get("subtitle_color_user")

                while True:
                    if is_cancelled():
                        break
                    chunk = proc.stdout.read(state.FRAME_BYTES)
                    if not chunk:
                        break
                    ws.send(_make_ws_audio_msg(request_id, chunk), opcode=websocket.ABNF.OPCODE_BINARY)
                    if len(chunk) == state.FRAME_BYTES:
                        total_frames += 1

                        # Calculate progress incrementally (0% to 70% during listening)
                        current_pct = int((total_frames / max_frames) * 70)

                        # Calculate VU meter
                        energy = rms_energy(chunk)
                        idx = max(0, min(7, int((energy / 1200.0) * 8)))
                        vu = bars[idx]

                        # Update UI with VU meter and partial text ~every 90ms (3 frames)
                        if total_frames % 3 == 0 or current_pct > last_progress_pct:
                            parts = ["🎤"]
                            if _show_vu:
                                parts.append(vu)
                            if _show_subs:
                                parts.append(_colorize(shared_state["partial"], _user_color))
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
                unregister_proc(proc)
                try:
                    ws.send(_make_ws_audio_msg(request_id, b""), opcode=websocket.ABNF.OPCODE_BINARY)
                except Exception:
                    pass
                sender_done.set()

        sender = threading.Thread(target=send_audio, daemon=True)
        sender.start()

        # Read responses until turn.end
        deadline = time.time() + max_seconds + 5
        got_phrase = False
        while time.time() < deadline and not is_cancelled():
            try:
                ws.settimeout(1.0)
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                if sender_done.is_set():
                    if got_phrase:
                        break
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
            if is_cancelled():
                return {"text": "", "cancelled": True}
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
        proc = _take_prewarmed_rec()
        if proc is None:
            proc = subprocess.Popen(
                _build_rec_cmd(),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
        register_proc(proc)

        frames, _ = record_with_vad(proc, max_seconds)
        proc.terminate()
        proc.wait()
        unregister_proc(proc)

        if is_cancelled():
            return {"text": "", "cancelled": True}

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
        proc = _take_prewarmed_rec()
        if proc is None:
            proc = subprocess.Popen(
                _build_rec_cmd(),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
        register_proc(proc)

        if HAS_VAD:
            frames, _ = record_with_vad(proc, max_seconds)
            proc.terminate()
            proc.wait()
            raw_data = b"".join(frames)
        else:
            raw_data = proc.stdout.read(state.SAMPLE_RATE * 2 * max_seconds)
            proc.terminate()
            proc.wait()
        unregister_proc(proc)

        if is_cancelled():
            return {"text": "", "cancelled": True}

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
        rec = CONFIG.get("recorder", "auto")
        mic = CONFIG.get("mic_source") or "default"
        if rec == "arecord" or rec == "auto":
            cmd = ["arecord", "-D", mic, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav",
                   "-d", str(seconds), tmp_path]
        else:
            cmd = ["pw-record", "--format", "s16", "--rate", "16000", "--channels", "1"]
            if CONFIG.get("mic_source"):
                cmd += ["--target", CONFIG["mic_source"]]
            cmd.append(tmp_path)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        register_proc(proc)
        try:
            proc.wait(timeout=seconds + 5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait()
        unregister_proc(proc)

        if is_cancelled():
            return {"text": "", "cancelled": True}
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


def stt(seconds=None, mode=None, silence_timeout=None, vad_aggressiveness=None,
        energy_multiplier=None, progress_token=None):
    """Speech-to-text with automatic mode selection."""
    max_seconds = max(1, min(int(seconds or 30), 30))

    with state._stt_lock:
        old_silence = state.SILENCE_TIMEOUT
        old_vad = state.VAD_AGGRESSIVENESS
        old_energy = state.ENERGY_THRESHOLD_MULTIPLIER

        if silence_timeout is not None:
            state.SILENCE_TIMEOUT = max(0.1, min(float(silence_timeout), 10.0))
        if vad_aggressiveness is not None:
            state.VAD_AGGRESSIVENESS = max(0, min(int(vad_aggressiveness), 3))
        if energy_multiplier is not None:
            state.ENERGY_THRESHOLD_MULTIPLIER = max(0.5, min(float(energy_multiplier), 20.0))

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
            state.SILENCE_TIMEOUT = old_silence
            state.VAD_AGGRESSIVENESS = old_vad
            state.ENERGY_THRESHOLD_MULTIPLIER = old_energy
