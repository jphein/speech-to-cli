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


def skip_azure():
    """True while Azure should not even be attempted."""
    return force_offline() or (enabled() and time.time() < _azure_down_until)


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

def synthesize(host, port, text, voice=None, timeout=10.0):
    """TTS via Wyoming. Returns (rate, width, channels, pcm_bytes)."""
    data = {"text": text}
    if voice:
        data["voice"] = {"name": voice}
    sock = _connect(host, port, timeout)
    try:
        f = sock.makefile("rb")
        _send_event(sock, "synthesize", data)
        rate, width, channels = 22050, 2, 1
        pcm = bytearray()
        deadline = time.time() + timeout
        while True:
            if time.time() > deadline:
                raise WyomingError("synthesize timed out")
            etype, edata, payload = _read_event(f)
            if etype == "audio-start":
                rate = int(edata.get("rate", rate))
                width = int(edata.get("width", width))
                channels = int(edata.get("channels", channels))
            elif etype == "audio-chunk":
                pcm.extend(payload)
            elif etype == "audio-stop":
                break
        if not pcm:
            raise WyomingError("no audio returned")
        return rate, width, channels, bytes(pcm)
    finally:
        sock.close()


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
