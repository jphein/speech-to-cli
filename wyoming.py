#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal Wyoming protocol client + Azure circuit breaker.

Wyoming (rhasspy) wire format: one JSON header line per event, optionally
followed by `data_length` bytes of extended-data JSON and `payload_length`
bytes of binary payload. Used as the offline fallback for STT/TTS when Azure
is unreachable — the server (Piper / ONNX-ASR) lives on the LAN and is
configured via wyoming_* keys in config.json; the feature is off when
wyoming_host is empty.
"""

import json
import os
import socket
import threading
import time

import state


class WyomingError(Exception):
    pass


# -- Azure circuit breaker ----------------------------------------------------

_azure_down_until = 0.0


def force_offline():
    return os.environ.get("SPEECH_FORCE_OFFLINE", "") not in ("", "0")


def enabled():
    return bool(state.CONFIG.get("wyoming_host", ""))


def prefer_local():
    """True when config says the LAN Wyoming server is the PRIMARY backend.

    `speech_backend`: "azure" (default) | "local". Set from gnome-speaks prefs
    (Voice & Sound → Speech Backend). Requires a configured wyoming_host;
    without one there is nothing local to prefer. Replaces the old
    SPEECH_FORCE_OFFLINE drop-in for the common case, with one difference:
    this one FALLS BACK to Azure when the local server fails; forced never does.
    """
    return enabled() and str(state.CONFIG.get("speech_backend", "azure")).strip().lower() == "local"


_local_down_until = 0.0


def local_down():
    """True while the LAN server is on cooldown after a failure."""
    return time.time() < _local_down_until


def mark_local_down(cooldown=60):
    global _local_down_until
    _local_down_until = time.time() + cooldown


def mark_local_up():
    global _local_down_until
    _local_down_until = 0.0


def skip_azure():
    """True while Azure should not even be attempted.

    Three reasons, in priority order (see skip_reason): forced offline via
    SPEECH_FORCE_OFFLINE; the local server is preferred and not on cooldown;
    Azure is on cooldown after a failure.
    """
    if force_offline():
        return True
    if not enabled():
        return False
    if time.time() < _azure_down_until:
        return True
    return prefer_local() and not local_down()


def skip_reason():
    """Why Azure is being skipped, for logs and /status. None = Azure is live.

    'forced'        SPEECH_FORCE_OFFLINE is set (no Azure fallback at all)
    'azure_down'    Azure failed within the cooldown window
    'prefer_local'  speech_backend=local and the LAN server is healthy
    """
    if force_offline():
        return "forced"
    if not enabled():
        return None
    if time.time() < _azure_down_until:
        return "azure_down"
    if prefer_local() and not local_down():
        return "prefer_local"
    return None


def azure_fallback_allowed():
    """After a LOCAL failure, may the caller try Azure? Never when forced."""
    return not force_offline()


def mark_azure_down(cooldown=60):
    global _azure_down_until
    _azure_down_until = time.time() + cooldown


def mark_azure_up():
    global _azure_down_until
    _azure_down_until = 0.0


# -- Wire helpers ---------------------------------------------------------------

def _send_event(sock, etype, data=None, payload=b""):
    header = {"type": etype, "data": data or {}}
    if payload:
        header["payload_length"] = len(payload)
    sock.sendall(json.dumps(header).encode() + b"\n")
    if payload:
        sock.sendall(payload)


def _read_exact(f, n):
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise WyomingError("connection closed mid-event")
        buf += chunk
    return buf


def _read_event(f):
    line = f.readline()
    if not line:
        raise WyomingError("connection closed")
    try:
        header = json.loads(line)
    except ValueError as e:
        raise WyomingError(f"bad header: {e}")
    data = header.get("data") or {}
    dlen = header.get("data_length")
    if dlen:
        try:
            data.update(json.loads(_read_exact(f, dlen)))
        except ValueError as e:
            raise WyomingError(f"bad data block: {e}")
    payload = b""
    plen = header.get("payload_length")
    if plen:
        payload = _read_exact(f, plen)
    return header.get("type", ""), data, payload


def _connect(host, port, timeout):
    try:
        sock = socket.create_connection((host, port), timeout=2.0)
        sock.settimeout(timeout)
        return sock
    except OSError as e:
        raise WyomingError(f"connect {host}:{port}: {e}")


# -- Public API -------------------------------------------------------------------

def synthesize_stream(host, port, text, voice=None, timeout=10.0):
    """TTS via Wyoming, streamed. Generator: the FIRST item is the format
    tuple (rate, width, channels), every later item is a bytes PCM chunk as
    the server sends it.

    Piper streams progressively (#19: first audio-chunk at ~0.2-0.5 s, audio-stop
    at 1.4 s / 4.5 s for a line / a paragraph), so a caller that plays chunks as
    they arrive starts speaking seconds before the collect-all `synthesize()`
    would even return. The format is yielded at `audio-start`, before any PCM,
    so the caller can spawn its player while synthesis is still running.
    Raises WyomingError on connect failure, ANY socket error (a mid-stream
    RST included -- #20 review), timeout, or an empty stream; the socket is
    closed when the generator finishes or is closed early. Nothing but
    WyomingError escapes this generator's wire handling, so a caller that
    also owns a player pipe can tell "the server went away" from "the player
    went away" by exception class.

    `timeout` bounds the time spent WAITING ON THE SERVER, summed over the
    stream -- not wall-clock. A consumer that plays as it reads is throttled
    by the player, so an utterance longer than `timeout` seconds must not
    "time out" mid-playback while the server is keeping up.
    """
    data = {"text": text}
    if voice:
        data["voice"] = {"name": voice}
    sock = _connect(host, port, timeout)
    try:
        f = sock.makefile("rb")
        try:
            _send_event(sock, "synthesize", data)
        except OSError as e:
            raise WyomingError(f"synthesize send: {e}")
        rate, width, channels = 22050, 2, 1
        started = False
        got_audio = False
        waited = 0.0
        while True:
            t0 = time.monotonic()
            try:
                etype, edata, payload = _read_event(f)
            except socket.timeout:
                # Subclass of OSError -- must stay ABOVE the generic clause.
                raise WyomingError("synthesize stalled")
            except OSError as e:
                # Mirror _connect(): a ConnectionResetError from a mid-stream
                # RST used to escape as a bare OSError, land in _tts_wyoming's
                # player-pipe handler, skip the player's stdin.close() and
                # report {'ok': True} with the player orphaned on stdin.
                raise WyomingError(f"synthesize read: {e}")
            waited += time.monotonic() - t0
            if waited > timeout:
                raise WyomingError("synthesize timed out")
            if etype == "audio-start":
                rate = int(edata.get("rate", rate))
                width = int(edata.get("width", width))
                channels = int(edata.get("channels", channels))
                if not started:
                    started = True
                    yield rate, width, channels
            elif etype == "audio-chunk":
                if not started:
                    # No audio-start seen: fall back to the wire defaults.
                    started = True
                    yield rate, width, channels
                if payload:
                    got_audio = True
                    yield payload
            elif etype == "audio-stop":
                break
        if not got_audio:
            raise WyomingError("no audio returned")
    finally:
        sock.close()


def synthesize(host, port, text, voice=None, timeout=10.0):
    """TTS via Wyoming. Returns (rate, width, channels, pcm_bytes).

    Collect-all wrapper over synthesize_stream() for callers that want a WAV
    (tts.py, speak.py). Live playback should consume the stream instead.
    """
    gen = synthesize_stream(host, port, text, voice=voice, timeout=timeout)
    rate, width, channels = next(gen)
    pcm = bytearray()
    for chunk in gen:
        pcm.extend(chunk)
    return rate, width, channels, bytes(pcm)


def transcribe(host, port, pcm, rate=16000, width=2, channels=1, timeout=20.0):
    """STT via Wyoming. pcm is raw s16le audio. Returns transcript text."""
    if not pcm:
        return ""
    sock = _connect(host, port, timeout)
    try:
        f = sock.makefile("rb")
        fmt = {"rate": rate, "width": width, "channels": channels}
        _send_event(sock, "transcribe", {})
        _send_event(sock, "audio-start", dict(fmt))
        chunk_sz = 4096
        for i in range(0, len(pcm), chunk_sz):
            _send_event(sock, "audio-chunk", dict(fmt), pcm[i:i + chunk_sz])
        _send_event(sock, "audio-stop", dict(fmt))
        deadline = time.time() + timeout
        while True:
            if time.time() > deadline:
                raise WyomingError("transcribe timed out")
            etype, edata, _ = _read_event(f)
            if etype == "transcript":
                return (edata.get("text") or "").strip()
    finally:
        sock.close()


def detect_stream(host, port, model, chunk_iter, timeout=5.0):
    """Stream mic audio to a Wyoming wake-word server until detection.

    chunk_iter yields raw s16le 16 kHz mono chunks and stops when the caller
    wants to disarm. A reader thread blocks on server events (select() over a
    BufferedReader misses buffered events — thread avoids the trap). Returns
    the detection name, or None if chunk_iter ended first. Raises
    WyomingError on connect failure.
    """
    sock = _connect(host, port, timeout)
    sock.settimeout(None)  # reader thread blocks indefinitely
    f = sock.makefile("rb")
    detected = []
    done = threading.Event()

    def _reader():
        try:
            while not done.is_set():
                etype, edata, _payload = _read_event(f)
                if etype == "detection":
                    detected.append(edata.get("name") or model)
                    done.set()
                    return
        except Exception:
            done.set()

    reader = threading.Thread(target=_reader, daemon=True,
                              name="wyoming-wake-reader")
    try:
        _send_event(sock, "detect", {"names": [model]})
        fmt = {"rate": 16000, "width": 2, "channels": 1}
        _send_event(sock, "audio-start", dict(fmt))
        reader.start()
        for chunk in chunk_iter:
            if done.is_set():
                break
            try:
                _send_event(sock, "audio-chunk", dict(fmt), chunk)
            except OSError as e:
                raise WyomingError(f"wake stream write: {e}")
        # Give an in-flight detection a moment to land after the last chunk
        done.wait(timeout=0.5)
        return detected[0] if detected else None
    finally:
        done.set()
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()
