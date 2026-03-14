"""Text-to-speech: Azure TTS with streaming playback, multi_speak, talk (full-duplex).

Import pattern: `import state` at top. Reassignable globals accessed as `state.X`.
"""

import json
import os
import random
import re
import struct
import subprocess
import tempfile
import threading
import time
import uuid

import state
from state import (CONFIG, HAS_WS, HAS_VAD,
                   is_cancelled, register_proc, unregister_proc,
                   get_http_session, send_progress, _pause_event)
from audio import (has_echo_cancel, play_chime, play_speak, play_done, stop_hum,
                   play_processing, calibrate_noise, is_speech_energy, write_wav,
                   rms_energy, _build_rec_cmd, _build_player_cmd,
                   _start_player, _take_prewarmed_player, _get_tty_width,
                   _schedule_warmup, _classify_voice_cmd, _colorize)
from stt import (_make_ws_audio_msg, _get_stt_ws, _invalidate_stt_ws,
                 _make_logger, _check_end_word, _strip_end_word,
                 _silence_icon, _init_stt_ws_session, _parse_ws_msg,
                 _rest_stt_fallback)

if HAS_WS:
    import websocket
if HAS_VAD:
    import webrtcvad


# ---------------------------------------------------------------------------
# TTS helpers
# ---------------------------------------------------------------------------

_SSML_SAFE_RE = re.compile(r'^[a-zA-Z0-9\-_.:+%() ]+$')


def _tts_lead_in_ms():
    """Lead-in when device is cold (first call or idle >10s)."""
    if state._last_tts_end == 0.0 or (time.monotonic() - state._last_tts_end) > 10.0:
        return 200
    return 0


def _mark_tts_end():
    """Record when TTS playback finished (for adaptive lead-in)."""
    state._last_tts_end = time.monotonic()


def _sanitize_ssml_attr(value, default="default"):
    """Reject values that could inject SSML markup."""
    if not value or not isinstance(value, str):
        return default
    value = value.strip()[:64]
    if not _SSML_SAFE_RE.match(value):
        return default
    return value


def _build_ssml(text, voice, quality, speed, pitch, volume):
    """Build SSML payload for Azure TTS."""
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    attrs = []
    if quality == "fast":
        attrs.append('rate="+15%"')
    elif speed != 1.0:
        pct = int((speed - 1.0) * 100)
        attrs.append(f'rate="{("+" if pct >= 0 else "")}{pct}%"')
    if pitch != "default":
        attrs.append(f'pitch="{pitch}"')
    if volume != "default":
        attrs.append(f'volume="{volume}"')
    body = f'<prosody {" ".join(attrs)}>{safe}</prosody>' if attrs else safe
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        f'<voice name="{voice}">{body}</voice></speak>'
    )


def _prepare_tts(text, quality, speed, voice, pitch, volume):
    """Common TTS preparation: sanitize inputs, build SSML, return (ssml, tts_rate, headers, url)."""
    if not voice:
        voice = CONFIG["fast_voice"] if quality == "fast" else CONFIG["voice"]
    voice = _sanitize_ssml_attr(voice, CONFIG["fast_voice"])
    pitch = _sanitize_ssml_attr(pitch, "default")
    volume = _sanitize_ssml_attr(volume, "default")
    ssml = _build_ssml(text, voice, quality, speed, pitch, volume)
    tts_rate = 48000 if quality == "hd" else 24000
    url = f"https://{CONFIG['region']}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": CONFIG["key"],
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": f"raw-{tts_rate // 1000}khz-16bit-mono-pcm",
    }
    return ssml, tts_rate, headers, url


def _build_multi_voice_ssml(segments, quality="fast"):
    """Build SSML with multiple <voice> tags for single-request multi-voice TTS."""
    voices_xml = []
    for seg in segments:
        text = seg.get("text", "")[:state._MAX_TTS_CHARS]
        voice = seg.get("voice") or (CONFIG["fast_voice"] if quality == "fast" else CONFIG["voice"])
        voice = _sanitize_ssml_attr(voice, CONFIG["fast_voice"])
        if not text:
            continue
        # Escape XML special chars
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        # Add prosody for fast mode
        if quality == "fast":
            body = f'<prosody rate="+15%">{safe}</prosody>'
        else:
            body = safe
        voices_xml.append(f'<voice name="{voice}">{body}</voice>')

    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        + "".join(voices_xml)
        + '</speak>'
    )


def multi_speak_stream(segments, quality="fast", progress_token=None):
    """Stream multi-voice TTS in a SINGLE request using Azure's multi-voice SSML.

    Much faster than multi_speak - one API call instead of N calls.
    Voices switch automatically within the audio stream.
    """
    stop_hum()
    if not segments:
        return {"error": "No segments"}

    send_progress(progress_token, 1, 100, "🔊 Building multi-voice SSML...")

    # Build single SSML with all voices
    ssml = _build_multi_voice_ssml(segments, quality)
    tts_rate = 48000 if quality == "hd" else 24000
    url = f"https://{CONFIG['region']}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": CONFIG["key"],
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": f"raw-{tts_rate // 1000}khz-16bit-mono-pcm",
    }

    send_progress(progress_token, 5, 100, f"🔊 Streaming {len(segments)} voices...")

    # Stream the response
    session = get_http_session()
    try:
        resp = session.post(url, headers=headers, data=ssml.encode("utf-8"), timeout=120, stream=True)
        if resp.status_code != 200:
            return {"error": f"TTS error {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"error": f"TTS request failed: {e}"}

    # Start player
    proc = _start_player(tts_rate)
    if proc is None:
        return {"error": "No audio player found"}
    register_proc(proc)

    # Calculate total expected duration for progress
    total_chars = sum(len(seg.get("text", "")) for seg in segments)
    estimated_duration = max(2.0, total_chars / 15.0)  # ~15 chars/sec speaking rate

    # Stream audio chunks to player
    lead_ms = _tts_lead_in_ms()
    if lead_ms:
        try:
            silence = b"\x00" * (tts_rate * 2 * lead_ms // 1000)
            proc.stdin.write(silence)
        except Exception:
            pass

    bytes_written = 0
    start_time = time.time()
    _ms_show_subs = CONFIG.get("live_subtitles", True)
    _ms_window = max(40, _get_tty_width() - 25)

    # Combine all text for subtitle display
    all_text = " | ".join(seg.get("text", "")[:100] for seg in segments if seg.get("text"))

    try:
        for chunk in resp.iter_content(chunk_size=4096):
            if is_cancelled():
                proc.kill()
                return {"cancelled": True}
            if chunk:
                try:
                    proc.stdin.write(chunk)
                    bytes_written += len(chunk)
                except (BrokenPipeError, OSError):
                    break

            # Update progress based on time elapsed
            elapsed = time.time() - start_time
            pct = min(95, 5 + int((elapsed / estimated_duration) * 90))

            if _ms_show_subs and all_text:
                char_idx = int(min(1.0, elapsed / estimated_duration) * len(all_text))
                start_idx = max(0, char_idx - _ms_window)
                snippet = all_text[start_idx:char_idx]
                send_progress(progress_token, pct, 100, f"🔊 {snippet}")
            else:
                send_progress(progress_token, pct, 100, f"🔊 Streaming...")

        proc.stdin.close()
    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        return {"error": f"Streaming failed: {e}"}

    # Wait for playback to finish
    try:
        proc.wait(timeout=estimated_duration + 30)
    except Exception:
        pass
    finally:
        unregister_proc(proc)

    _mark_tts_end()
    send_progress(progress_token, 100, 100, f"🔊 Done — {len(segments)} voices")
    return {"spoken": len(segments), "bytes": bytes_written, "streamed": True}


# ---------------------------------------------------------------------------
# multi_speak
# ---------------------------------------------------------------------------

def multi_speak(segments, quality="fast", progress_token=None):
    """Speak multiple text+voice segments. TTS requests fire in parallel, playback is sequential."""
    stop_hum()
    if not segments:
        return {"error": "No segments"}

    send_progress(progress_token, 1, 100, "🔊 Starting...")

    tts_rate = 48000 if quality == "hd" else 24000
    session = get_http_session()
    n_seg = len(segments)
    audio_buffers = [None] * n_seg
    errors = [None] * n_seg

    def fetch_one(idx, seg):
        text = seg.get("text", "")[:state._MAX_TTS_CHARS]
        voice = seg.get("voice")
        if not text:
            errors[idx] = "empty text"
            return
        ssml, _, headers, url = _prepare_tts(text, quality, 1.0, voice, "default", "default")
        try:
            resp = session.post(url, headers=headers, data=ssml.encode("utf-8"), timeout=30)
            if resp.status_code != 200:
                errors[idx] = f"TTS error {resp.status_code}"
                return
            audio_buffers[idx] = resp.content
        except Exception as e:
            errors[idx] = str(e)

    # Fire all TTS requests in parallel
    threads = []
    for i, seg in enumerate(segments):
        t = threading.Thread(target=fetch_one, args=(i, seg), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=35)

    # Unified smooth progress 0→100% during playback
    spoken = 0
    lead_ms = _tts_lead_in_ms()
    silence_bytes = b"\x00" * (tts_rate * 2 * lead_ms // 1000)
    pct = 1
    _ms_show_subs = CONFIG.get("live_subtitles", True)
    _ms_default_color = CONFIG.get("subtitle_color_tts")
    _ms_window = max(40, _get_tty_width() - 25)

    for i, audio in enumerate(audio_buffers):
        if is_cancelled():
            return {"cancelled": True}
        if audio is None:
            continue

        proc = _start_player(tts_rate)
        if proc is None:
            return {"error": "No audio player found"}

        register_proc(proc)

        # Write audio in background thread — large buffers block on pipe,
        # delaying seg_start and causing subtitle lag
        def _write_audio(p, data, lead):
            try:
                if lead:
                    p.stdin.write(lead)
                p.stdin.write(data)
                p.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        threading.Thread(target=_write_audio,
                         args=(proc, audio, silence_bytes if i == 0 else b""),
                         daemon=True).start()
        seg_start = time.time()

        try:
            seg_text = segments[i].get("text", "")
            seg_color = segments[i].get("subtitle_color") or _ms_default_color
            seg_target = int(((i + 1) / n_seg) * 99)
            seg_dur = max(1.0, len(seg_text) / 18.0)
            last_msg = ""
            while proc.poll() is None:
                pct = min(pct + 2, seg_target)
                if _ms_show_subs and seg_text:
                    elapsed = time.time() - seg_start
                    char_idx = int(min(1.0, elapsed / seg_dur) * len(seg_text))
                    start_idx = max(0, char_idx - _ms_window)
                    snippet = seg_text[start_idx:char_idx]
                    msg = f"🔊 {_colorize(snippet, seg_color)}"
                else:
                    msg = f"🔊 Playing {i + 1}/{n_seg}..."
                if msg != last_msg:
                    send_progress(progress_token, pct, 100, msg)
                    last_msg = msg
                time.sleep(0.2)
            spoken += 1
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        finally:
            unregister_proc(proc)

        pct = seg_target
        send_progress(progress_token, pct, 100, f"🔊 Played {i + 1}/{n_seg}")

    # Show last segment text so subtitles persist
    last_text = ""
    for seg in reversed(segments):
        if seg.get("text"):
            last_color = seg.get("subtitle_color") or _ms_default_color
            last_text = _colorize(seg["text"], last_color)
            break
    send_progress(progress_token, 100, 100, f"🔊 {last_text}" if last_text else f"🔊 Done — {spoken} segments")
    return {"spoken": spoken}


# ---------------------------------------------------------------------------
# tts (single segment, streaming playback)
# ---------------------------------------------------------------------------

def tts(text, quality="fast", speed=1.0, voice=None, pitch="default", volume="default", progress_token=None, subtitle_color=None):
    """Speak text aloud via Azure TTS with streaming playback."""
    stop_hum()
    send_progress(progress_token, 0, 100, "🔊 Synthesizing...")

    if not text or not isinstance(text, str):
        return {"error": "No text provided"}
    text = text[:state._MAX_TTS_CHARS]

    ssml, tts_rate, headers, url = _prepare_tts(text, quality, speed, voice, pitch, volume)

    # Take pre-warmed player or start fresh (overlaps with TTS API latency)
    proc = _take_prewarmed_player(tts_rate) or _start_player(tts_rate)
    if proc is None:
        return {"error": "No audio player found — set 'player' in config"}

    # Fire TTS request (player is already waiting for stdin data)
    resp = get_http_session().post(url, headers=headers, data=ssml.encode("utf-8"), timeout=60, stream=True)
    if resp.status_code != 200:
        proc.stdin.close()
        proc.wait()
        return {"error": f"Azure TTS error {resp.status_code}: {resp.text}"}

    send_progress(progress_token, 5, 100, "🔊 Speaking...")
    play_speak()

    register_proc(proc)
    try:
        speed_factor = 22.0 if quality == "fast" else 15.0
        estimated_duration = max(1.0, len(text) / speed_factor)
        start_time = time.time()

        def download_audio():
            try:
                lead_ms = _tts_lead_in_ms()
                silence_bytes = b"\x00" * (tts_rate * 2 * lead_ms // 1000)
                proc.stdin.write(silence_bytes)
                proc.stdin.flush()
                for chunk in resp.iter_content(chunk_size=16384):
                    if is_cancelled():
                        break
                    while _pause_event.is_set() and not is_cancelled():
                        time.sleep(0.05)
                    if chunk:
                        proc.stdin.write(chunk)
                        proc.stdin.flush()
                        last_write_time[0] = time.time()
                proc.stdin.close()
            except Exception:
                pass

        last_write_time = [time.time()]
        dl_thread = threading.Thread(target=download_audio, daemon=True)
        dl_thread.start()

        bars = [" ", "▂", "▃", "▄", "▅"]
        total_len = len(text)
        last_msg = ""
        current_pct = 0
        show_vu = CONFIG.get("vu_meter", True)
        show_subs = CONFIG.get("live_subtitles", True)
        tts_color = subtitle_color or CONFIG.get("subtitle_color_tts")
        window_size = max(40, _get_tty_width() - 25)

        player_timeout = estimated_duration + 5.0
        pause_start = None
        while proc.poll() is None:
            if is_cancelled():
                proc.terminate()
                break

            time_since_write = time.time() - last_write_time[0]
            if not _pause_event.is_set() and (time.time() - start_time) > player_timeout and time_since_write > 10.0:
                proc.terminate()
                break

            if _pause_event.is_set():
                if pause_start is None:
                    pause_start = time.time()
                    send_progress(progress_token, current_pct, 100, "⏸ Paused")
                time.sleep(0.2)
                continue
            elif pause_start is not None:
                start_time += time.time() - pause_start
                pause_start = None

            elapsed = time.time() - start_time
            current_pct = int(min(1.0, elapsed / estimated_duration) * 100)

            vu_prefix = f"{random.choice(bars)} " if show_vu else ""

            if show_subs and total_len > 0:
                char_idx = int((current_pct / 100.0) * total_len)
                start_idx = max(0, char_idx - window_size)
                snippet = text[start_idx:char_idx]
                base_msg = f"Speaking: {_colorize(snippet, tts_color)}"
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
        unregister_proc(proc)
        stop_hum()
        if is_cancelled():
            send_progress(progress_token, 100, 100, "⏹ Cancelled")
            return {"spoken": False, "cancelled": True}
        # Show full text in final message so subtitles persist
        tts_color = subtitle_color or CONFIG.get("subtitle_color_tts")
        send_progress(progress_token, 100, 100, f"🔊 {_colorize(text, tts_color)}")
    play_done()
    _mark_tts_end()
    return {"spoken": True, "chars": len(text)}


# ---------------------------------------------------------------------------
# talk_fullduplex (speak + listen simultaneously)
# ---------------------------------------------------------------------------

def talk_fullduplex(text, quality="fast", speed=1.0, voice=None, pitch="default",
                    volume="default", seconds=30, mode=None, silence_timeout=None,
                    progress_token=None, subtitle_color=None):
    """Speak and listen simultaneously.

    TTS plays to the default output while STT records from the default mic.
    If echo cancellation nodes are available, routes through them instead.
    """
    stop_hum()

    if not text or not isinstance(text, str):
        return {"spoken": False, "error": "No text provided", "text": ""}
    text = text[:state._MAX_TTS_CHARS]

    ssml, tts_rate, headers, url = _prepare_tts(text, quality, speed, voice, pitch, volume)

    # --- Start STT recording ---
    cap = int(CONFIG.get("max_record_seconds", 120))
    max_seconds = max(1, min(int(seconds or cap), cap))
    stt_silence = float(silence_timeout) if silence_timeout else CONFIG.get("talk_silence_timeout", 1.5)

    rec_cmd = _build_rec_cmd()
    if has_echo_cancel() and "--target" not in rec_cmd:
        if "pw-record" in rec_cmd[0]:
            idx = rec_cmd.index("-")
            rec_cmd[idx:idx] = ["--target", state.EC_SOURCE]

    rec_proc = subprocess.Popen(
        rec_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    register_proc(rec_proc)

    # Start player FIRST so it's ready when TTS audio arrives
    ec_target = state.EC_SINK if has_echo_cancel() else None
    player_proc = _start_player(tts_rate, target=ec_target)
    if player_proc is None:
        rec_proc.terminate()
        rec_proc.wait()
        unregister_proc(rec_proc)
        return {"spoken": False, "error": "No audio player found", "text": ""}

    # Fire TTS request
    send_progress(progress_token, 0, 100, "🔊 Synthesizing...")
    resp = get_http_session().post(url, headers=headers, data=ssml.encode("utf-8"), timeout=60, stream=True)
    if resp.status_code != 200:
        player_proc.stdin.close()
        player_proc.wait()
        rec_proc.terminate()
        rec_proc.wait()
        unregister_proc(rec_proc)
        return {"spoken": False, "error": f"Azure TTS error {resp.status_code}: {resp.text}", "text": ""}

    # Shared state for the recording thread
    rec_frames = []
    rec_done = threading.Event()
    rec_speech_detected = threading.Event()
    tts_done = threading.Event()
    tts_paused = threading.Event()

    # Streaming STT state
    stt_ws = [None]
    stt_request_id = [None]
    stt_phrases = []  # Accumulate all recognized phrases
    stt_ws_ready = threading.Event()

    # Audio buffer for repeat support
    tts_audio_buf = []

    register_proc(player_proc)
    send_progress(progress_token, 5, 100, "🔊🎤 Speaking + listening...")

    # --- Set up streaming STT WebSocket early ---
    use_streaming_stt = HAS_WS
    if use_streaming_stt:
        try:
            ws = _get_stt_ws()
            rid = uuid.uuid4().hex
            stt_request_id[0] = rid
            _init_stt_ws_session(ws, rid)
            stt_ws[0] = ws
        except Exception:
            use_streaming_stt = False
            _invalidate_stt_ws()

    # --- Live subtitle state ---
    stt_partial = [""]
    stt_phrase_done = threading.Event()
    stt_turn_ended = threading.Event()  # Azure closed STT session
    silence_ratio = [0.0]  # 0.0 → 1.0 as silence approaches cutoff
    end_word_detected = threading.Event()
    _end_word = CONFIG.get("end_word", "over")

    def _recv_ws():
        """Non-blocking read of WS messages, using shared parser."""
        ws = stt_ws[0]
        if not ws:
            return
        try:
            ws.settimeout(0.001)
            msg = ws.recv()
        except Exception:
            return
        mtype = _parse_ws_msg(msg, stt_phrases, stt_partial, end_word_detected, _end_word, _log)
        if mtype == "phrase":
            stt_phrase_done.set()
        elif mtype == "turn_end":
            _log(f"turn.end during recording (phrases={len(stt_phrases)}) — Azure ended STT session")
            stt_turn_ended.set()
            stt_phrase_done.set()
            stt_ws[0] = None
            _invalidate_stt_ws()

    # --- Barge-in config (experimental, disabled by default) ---
    barge_trigger_frames = max(1, int(CONFIG.get("barge_in_frames", 3)))
    barge_silence_sec = max(0.3, float(CONFIG.get("barge_in_silence", 1.0)))

    # --- Background: read mic frames with VAD ---
    _log = _make_logger("talk-vad")

    def record_with_vad_bg():
        try:
            vad = webrtcvad.Vad(state.VAD_AGGRESSIVENESS) if HAS_VAD else None
            energy_threshold, cal_frames = calibrate_noise(rec_proc)
            rec_frames.extend(cal_frames)
            _log(f"calibrated: threshold={energy_threshold:.0f}, cal_frames={len(cal_frames)}")

            # Ding as soon as mic is ready — even over TTS playback
            play_chime()

            silence_frames = 0
            speech_frames = 0
            max_silence = int(stt_silence * 1000 / state.FRAME_MS)
            max_no_speech = int(state.NO_SPEECH_TIMEOUT * 1000 / state.FRAME_MS)
            min_speech = int(state.MIN_SPEECH_DURATION * 1000 / state.FRAME_MS)
            max_total = int(max_seconds * 1000 / state.FRAME_MS)
            total_frames = 0
            barge_speech = 0
            barge_silence = 0
            barge_silence_frames = int(barge_silence_sec * 1000 / state.FRAME_MS)

            post_tts_frames = 0
            ws_reconnected = False
            tts_dead_frames = []  # frames buffered after WS died during TTS
            _log(f"limits: max_silence={max_silence} max_no_speech={max_no_speech} min_speech={min_speech} stt_silence={stt_silence}")

            hard_cap = max_total * 3  # safety cap: 3x max to allow TTS + full recording
            while (post_tts_frames < max_total or not tts_done.is_set()) and total_frames < hard_cap:
                if is_cancelled():
                    _log("cancelled")
                    break
                chunk = rec_proc.stdout.read(state.FRAME_BYTES)
                if not chunk or len(chunk) < state.FRAME_BYTES:
                    _log(f"no data at frame {total_frames}")
                    break
                total_frames += 1
                is_speech = is_speech_energy(chunk, vad, energy_threshold)

                # --- During TTS playback ---
                if not tts_done.is_set():
                    ws = stt_ws[0]
                    rid = stt_request_id[0]
                    if ws and rid:
                        try:
                            ws.send(_make_ws_audio_msg(rid, chunk),
                                    opcode=websocket.ABNF.OPCODE_BINARY)
                            _recv_ws()
                        except Exception as _e:
                            _log(f"WS send error during TTS (frame {total_frames}): {_e}")
                            stt_ws[0] = None  # mark dead so we don't keep trying
                    elif stt_ws[0] is None:
                        # WS is dead — buffer frames so we can replay on reconnect
                        tts_dead_frames.append(chunk)

                    # Experimental barge-in (disabled by default)
                    if CONFIG.get("enable_barge_in", False) and is_speech:
                        barge_speech += 1
                        barge_silence = 0
                        rec_frames.append(chunk)
                        if barge_speech >= barge_trigger_frames and not tts_paused.is_set():
                            tts_paused.set()
                    elif tts_paused.is_set():
                        barge_silence += 1
                        rec_frames.append(chunk)
                        if barge_silence >= barge_silence_frames:
                            tts_paused.clear()
                            barge_speech = 0
                            barge_silence = 0
                    else:
                        barge_speech = 0
                        barge_silence = 0
                    continue

                # --- After TTS: normal recording ---
                if post_tts_frames == 0:
                    _log(f"TTS done at frame {total_frames}, old threshold={energy_threshold:.0f}")
                    # Reset to minimum threshold — TTS inflates the calibrated value,
                    # and re-calibration is unreliable (catches residual noise).
                    # Ambient silence is ~40-100, speech is 300+, so 300 works reliably.
                    # Let webrtcvad do the heavy lifting for speech detection.
                    energy_threshold = 300.0
                    _log(f"reset threshold to {energy_threshold:.0f}")

                post_tts_frames += 1
                rec_frames.append(chunk)

                ws = stt_ws[0]
                rid = stt_request_id[0]
                if ws and rid:
                    try:
                        ws.send(_make_ws_audio_msg(rid, chunk),
                                opcode=websocket.ABNF.OPCODE_BINARY)
                        _recv_ws()
                    except Exception as _e:
                        _log(f"WS send error post-TTS (frame {total_frames}): {_e}")
                        stt_ws[0] = None  # mark dead
                elif stt_ws[0] is None and use_streaming_stt and not ws_reconnected:
                    # Azure killed WS during TTS — start fresh session
                    ws_reconnected = True
                    _log(f"WS dead at post-TTS frame {post_tts_frames}, starting fresh STT session")
                    try:
                        ws_new = _get_stt_ws()
                        rid_new = uuid.uuid4().hex
                        stt_request_id[0] = rid_new
                        _init_stt_ws_session(ws_new, rid_new)
                        # Replay buffered frames: TTS-phase dead frames + post-TTS frames
                        replay = tts_dead_frames + rec_frames
                        for prev_frame in replay:
                            ws_new.send(_make_ws_audio_msg(rid_new, prev_frame),
                                        opcode=websocket.ABNF.OPCODE_BINARY)
                        stt_ws[0] = ws_new
                        stt_turn_ended.clear()
                        stt_phrase_done.clear()
                        stt_phrases.clear()
                        _log(f"fresh WS started: rid={rid_new[:8]}, replayed {len(tts_dead_frames)}+{len(rec_frames)} frames")
                    except Exception as _e:
                        _log(f"fresh WS FAILED: {_e}")

                if is_speech:
                    speech_frames += 1
                    silence_frames = 0
                    if not rec_speech_detected.is_set():
                        rec_speech_detected.set()
                else:
                    silence_frames += 1

                # Track silence ratio for countdown display
                if max_silence > 0:
                    silence_ratio[0] = silence_frames / max_silence

                # Log every ~1 second of post-TTS recording
                if post_tts_frames % 33 == 0:
                    _log(f"post-TTS f={post_tts_frames} speech={speech_frames} silence={silence_frames} energy={rms_energy(chunk):.0f} thresh={energy_threshold:.0f}")

                if end_word_detected.is_set():
                    _log(f"STOP: end word '{_end_word}' detected. speech={speech_frames}")
                    break
                if speech_frames >= min_speech and silence_frames >= max_silence:
                    _log(f"STOP: silence timeout. speech={speech_frames} silence={silence_frames}/{max_silence}")
                    break
                if speech_frames == 0 and post_tts_frames >= max_no_speech:
                    _log(f"STOP: no speech timeout. post_tts={post_tts_frames}/{max_no_speech}")
                    break
            _log(f"REC END: speech={speech_frames} post_tts={post_tts_frames} total={total_frames} ws_alive={stt_ws[0] is not None} phrases={len(stt_phrases)} text={repr(' '.join(stt_phrases)[:100])}")
        except Exception as e:
            _log(f"exception: {e}")
        finally:
            rec_done.set()

    rec_thread = threading.Thread(target=record_with_vad_bg, daemon=True)
    rec_thread.start()

    # --- Download TTS audio and pipe to player ---
    dl_done = threading.Event()

    def download_audio():
        try:
            lead_ms = _tts_lead_in_ms()
            silence_bytes = b"\x00" * (tts_rate * 2 * lead_ms // 1000)
            try:
                player_proc.stdin.write(silence_bytes)
                player_proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            for chunk in resp.iter_content(chunk_size=16384):
                if is_cancelled():
                    break
                if chunk:
                    tts_audio_buf.append(chunk)
                    while tts_paused.is_set() and not is_cancelled():
                        try:
                            player_proc.stdin.write(b'\x00' * 4800)
                            player_proc.stdin.flush()
                        except (BrokenPipeError, OSError):
                            break
                        time.sleep(0.1)
                    try:
                        player_proc.stdin.write(chunk)
                        player_proc.stdin.flush()
                        last_write_time[0] = time.time()
                    except (BrokenPipeError, OSError):
                        break
            try:
                player_proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        except Exception:
            pass
        finally:
            dl_done.set()

    last_write_time = [time.time()]
    dl_thread = threading.Thread(target=download_audio, daemon=True)
    dl_thread.start()

    # --- Helper: replay buffered TTS audio ---
    def _replay_audio():
        nonlocal player_proc
        unregister_proc(player_proc)
        new_player = _start_player(tts_rate, target=ec_target)
        if new_player is None:
            return False
        player_proc = new_player
        register_proc(player_proc)
        def _write_buf():
            try:
                for chunk in tts_audio_buf:
                    player_proc.stdin.write(chunk)
                player_proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        threading.Thread(target=_write_buf, daemon=True).start()
        return True

    # --- Wait for TTS to finish playing, showing progress ---
    speed_factor = 22.0 if quality == "fast" else 15.0
    estimated_duration = max(1.0, len(text) / speed_factor)
    player_timeout = estimated_duration + 5.0
    start_time = time.time()
    bars = [" ", "▂", "▃", "▄", "▅"]
    show_vu = CONFIG.get("vu_meter", True)
    show_subs = CONFIG.get("live_subtitles", True)
    tts_color = subtitle_color or CONFIG.get("subtitle_color_tts")
    user_color = CONFIG.get("subtitle_color_user")
    text_len = len(text)
    window_size = max(40, _get_tty_width() - 25)
    current_pct = 0
    last_msg = ""

    while player_proc.poll() is None:
        if is_cancelled():
            player_proc.terminate()
            break

        time_since_write = time.time() - last_write_time[0]
        if (time.time() - start_time) > player_timeout and time_since_write > 10.0:
            player_proc.terminate()
            break

        elapsed = time.time() - start_time
        current_pct = int(min(1.0, elapsed / estimated_duration) * 100)
        vu_prefix = f"{random.choice(bars)} " if show_vu else ""

        if show_subs and text_len > 0:
            char_idx = int((current_pct / 100.0) * text_len)
            start_idx = max(0, char_idx - window_size)
            snippet = text[start_idx:char_idx]
            user_partial = stt_partial[0]
            if user_partial:
                base_msg = f"🗣 {_colorize(user_partial, user_color)}\n🔊 {_colorize(snippet, tts_color)}"
            else:
                base_msg = f"Speaking: {_colorize(snippet, tts_color)}"
        else:
            base_msg = "Speaking + Listening..."

        msg = f"🔊🎤 {vu_prefix}{base_msg}"
        if msg != last_msg or show_vu:
            send_progress(progress_token, current_pct, 100, msg)
            last_msg = msg
        time.sleep(0.1)

    tts_done.set()
    dl_thread.join(timeout=max(5, estimated_duration * 0.5))
    unregister_proc(player_proc)
    _mark_tts_end()

    if is_cancelled():
        rec_proc.terminate()
        rec_proc.wait()
        unregister_proc(rec_proc)
        stop_hum()
        send_progress(progress_token, 100, 100, "⏹ Cancelled")
        return {"spoken": False, "cancelled": True, "text": ""}

    # --- Wait for recording thread to finish ---
    listen_start = time.time()
    listen_timeout = state.NO_SPEECH_TIMEOUT + stt_silence + 2
    listen_bars = [" ", "▂", "▃", "▄", "▅"]
    while rec_thread.is_alive():
        elapsed = time.time() - listen_start
        if elapsed > listen_timeout:
            break
        pct = int(70 + min(1.0, elapsed / listen_timeout) * 15)
        vu = random.choice(listen_bars) if show_vu else ""
        timer_icon = _silence_icon(silence_ratio[0])
        # Keep TTS text visible + show user's speech below it
        tts_line = f"🔊 {_colorize(text, tts_color)}" if show_subs else ""
        if rec_speech_detected.is_set():
            partial = stt_partial[0]
            if partial and show_subs:
                send_progress(progress_token, pct, 100, f"{tts_line}\n🎤 {vu} {_colorize(partial, user_color)}{timer_icon}")
            else:
                send_progress(progress_token, pct, 100, f"{tts_line}\n🎤 {vu} Hearing you...{timer_icon}")
        else:
            send_progress(progress_token, pct, 100, f"{tts_line}\n🎤 {vu} Listening for your reply...")
        rec_thread.join(timeout=0.15)

    # Stop recording if still running
    try:
        rec_proc.terminate()
        rec_proc.wait(timeout=1)
    except Exception:
        pass
    unregister_proc(rec_proc)

    if is_cancelled():
        stop_hum()
        send_progress(progress_token, 100, 100, "⏹ Cancelled")
        return {"spoken": True, "cancelled": True, "text": ""}

    # --- Get transcription result ---
    raw_data = b"".join(rec_frames)
    if not raw_data or len(raw_data) < state.FRAME_BYTES:
        if use_streaming_stt and stt_ws[0]:
            try:
                stt_ws[0].send(_make_ws_audio_msg(stt_request_id[0], b""),
                               opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception:
                pass
        stop_hum()
        play_done()
        _mark_tts_end()
        tts_color = subtitle_color or CONFIG.get("subtitle_color_tts")
        send_progress(progress_token, 100, 100, f"🔊 {_colorize(text, tts_color)}")
        return {"spoken": True, "text": ""}

    user_text = ""
    if use_streaming_stt and stt_ws[0]:
        send_progress(progress_token, 88, 100, "🧠 Finishing transcription...")
        _log(f"DRAIN START: phrases_so_far={len(stt_phrases)} phrase_done={stt_phrase_done.is_set()}")
        try:
            stt_ws[0].send(_make_ws_audio_msg(stt_request_id[0], b""),
                           opcode=websocket.ABNF.OPCODE_BINARY)
            _log("sent end-of-stream marker")
        except Exception as _e:
            _log(f"end-of-stream send FAILED: {_e}")
        if not stt_phrase_done.is_set():
            deadline = time.time() + 5
            ws = stt_ws[0]
            while time.time() < deadline:
                try:
                    ws.settimeout(1.0)
                    msg = ws.recv()
                except Exception as _e:
                    _log(f"drain recv error: {_e}")
                    break
                if isinstance(msg, str):
                    parts = msg.split("\r\n\r\n", 1)
                    if len(parts) < 2:
                        continue
                    hdr, body = parts
                    _log(f"drain msg: {hdr.split(chr(13))[0][:60]}")
                    if "speech.phrase" in hdr.lower():
                        try:
                            data = json.loads(body)
                            status = data.get("RecognitionStatus", "?")
                            _log(f"drain phrase status={status}")
                            if status == "Success":
                                nbest = data.get("NBest", [])
                                phrase = nbest[0]["Display"] if nbest else data.get("DisplayText", "")
                                if phrase:
                                    stt_phrases.append(phrase)
                                    _log(f"drain phrase: {phrase[:80]}")
                            else:
                                _log(f"drain phrase NON-SUCCESS: {body[:200]}")
                        except Exception as _e:
                            _log(f"drain phrase parse error: {_e}")
                        break
                    elif "turn.end" in hdr.lower():
                        _log("drain: turn.end received")
                        break
        else:
            _log("phrase_done already set, skipping drain")
        user_text = " ".join(stt_phrases)
        _log(f"FINAL RESULT: phrases={len(stt_phrases)} text={repr(user_text[:150])}")
    if not user_text:
        # Fallback: REST STT (WS failed, died, or returned nothing)
        _log(f"WS returned nothing (ws_alive={stt_ws[0] is not None} turn_ended={stt_turn_ended.is_set()} raw_bytes={len(raw_data)}), falling back to REST STT")
        send_progress(progress_token, 85, 100, "🧠 Transcribing...")
        user_text = _rest_stt_fallback(rec_frames, _log)

    stop_hum()
    play_done()
    _mark_tts_end()

    user_text = _strip_end_word(user_text, _end_word)

    # Show final state: TTS text + user reply so subtitles persist
    tts_color = subtitle_color or CONFIG.get("subtitle_color_tts")
    user_color = CONFIG.get("subtitle_color_user")
    if user_text:
        send_progress(progress_token, 100, 100, f"🎤 {_colorize(user_text, user_color)}")
    else:
        send_progress(progress_token, 100, 100, f"🔊 {_colorize(text, tts_color)}")

    # Pre-warm for next call
    _schedule_warmup()

    return {"spoken": True, "text": user_text}


# ---------------------------------------------------------------------------
# get_voices
# ---------------------------------------------------------------------------

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
