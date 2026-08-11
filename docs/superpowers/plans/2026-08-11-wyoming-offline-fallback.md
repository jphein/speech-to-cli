# Wyoming Offline Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** STT/TTS automatically fall back to the LAN Wyoming stack (Piper TTS, ONNX-ASR STT) when Azure is unreachable.

**Architecture:** New stdlib-only `wyoming.py` (protocol client + Azure circuit breaker reading `state.CONFIG`). `speech_tts.tts()` gains a Wyoming path (fresh player at the Wyoming sample rate, same cancel/pause/VU discipline). `stt.py`'s `_rest_stt_fallback` (and the batch REST site) fall through to Wyoming only on *network-class* Azure failures. `SPEECH_FORCE_OFFLINE=1` forces the breaker open (manual lever + test seam).

**Tech Stack:** Python 3 stdlib (`socket`, `json`, `struct`, `time`, `os`). Config keys in `~/.config/speech-to-cli/config.json` only (public repo — no LAN hosts in code). No test framework: `python3 -c` round-trips + forced-offline live checks.

**Spec:** `docs/superpowers/specs/2026-08-11-wyoming-offline-fallback-design.md`

**File map:**
- Create `wyoming.py` — protocol client (`synthesize`, `transcribe`), `WyomingError`, breaker (`skip_azure`, `mark_azure_down`, `mark_azure_up`, `enabled`)
- Modify `speech_tts.py` — `_tts_wyoming()` helper + hooks in `tts()`
- Modify `stt.py` — `_wyoming_stt()` helper + restructured `_rest_stt_fallback` + same pattern at any other `stt.speech.microsoft.com` REST site
- Modify `~/.config/speech-to-cli/config.json` (user disk, not git) — `wyoming_host`, ports, voice
- Modify `README.md` / `CLAUDE.md` — document the feature and env lever

---

### Task 1: `wyoming.py` — protocol client + circuit breaker

**Files:** Create `wyoming.py`

- [ ] **Step 1: Write the module**

```python
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

# -- Azure circuit breaker ---------------------------------------------------

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

# -- Wire helpers -------------------------------------------------------------

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

# -- Public API ----------------------------------------------------------------

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
```

- [ ] **Step 2: Live round-trip sanity against the LAN host** (host from env: run with WYOMING_HOST set; config lands in Task 2)

```bash
cd ~/Projects/speech-to-cli && python3 -c "
import os, wyoming
WYOMING_HOST = os.environ['WYOMING_HOST']
r, w, c, pcm = wyoming.synthesize(WYOMING_HOST, 10200, 'Hello world, the realm endures.')
print('TTS', r, w, c, len(pcm), 'bytes'); assert len(pcm) > 10000
text = wyoming.transcribe(WYOMING_HOST, 10300, pcm, rate=r)
print('STT round-trip:', repr(text)); assert 'hello' in text.lower()
print('WYOMING-OK')
"
```
Expected: `WYOMING-OK` (round trip: Piper speaks → Parakeet reads it back)

- [ ] **Step 3: Commit**

```bash
python3 -c "import py_compile; py_compile.compile('wyoming.py', doraise=True)"
git add wyoming.py && git commit -m "feat: minimal Wyoming protocol client + Azure circuit breaker"
```

---

### Task 2: Config keys + TTS fallback in `speech_tts.tts()`

**Files:** Modify `speech_tts.py`, `~/.config/speech-to-cli/config.json` (not git)

- [ ] **Step 1: Add config keys to the user config** (jq-style merge via python, preserving everything else): `wyoming_host: "<your-wyoming-host>"`, `wyoming_tts_port: 10200`, `wyoming_stt_port: 10300`, `wyoming_tts_voice: ""`.

- [ ] **Step 2: Import + helper in `speech_tts.py`.** Add `import wyoming` near `import state`. Add above `def tts(`:

```python
def _tts_wyoming(text, proc, audio_level_cb=None, output_file=None,
                 progress_token=None):
    """Offline TTS via the configured Wyoming server.

    proc: an already-started Azure-rate player to discard, or None.
    Returns a result dict, or None when Wyoming is unconfigured/failed
    (caller then returns its original Azure error).
    """
    host = CONFIG.get("wyoming_host", "")
    if not host:
        return None
    try:
        rate, width, channels, pcm = wyoming.synthesize(
            host, int(CONFIG.get("wyoming_tts_port", 10200)), text,
            voice=CONFIG.get("wyoming_tts_voice") or None)
    except wyoming.WyomingError as e:
        print(f"[wyoming] TTS fallback failed: {e}", file=sys.stderr)
        return None
    if proc is not None:
        try:
            proc.stdin.close()
            proc.terminate()
        except Exception:
            pass
    proc = _start_player(rate)
    if proc is None:
        return {"error": "No audio player found — set 'player' in config"}
    register_proc(proc)
    play_speak()
    send_progress(progress_token, 5, 100, "🔊 Speaking (offline)...")
    try:
        lead_ms = _tts_lead_in_ms()
        proc.stdin.write(b"\x00" * (rate * width * lead_ms // 1000))
        n = 0
        for i in range(0, len(pcm), 16384):
            if is_cancelled():
                break
            while _pause_event.is_set() and not is_cancelled():
                time.sleep(0.05)
            chunk = pcm[i:i + 16384]
            proc.stdin.write(chunk)
            proc.stdin.flush()
            n += 1
            if audio_level_cb and n % 3 == 0:
                try:
                    audio_level_cb(min(rms_energy(chunk[:3200]) / 8000.0, 1.0))
                except Exception:
                    pass
        proc.stdin.close()
        while proc.poll() is None:
            if is_cancelled():
                proc.terminate()
                break
            time.sleep(0.1)
    except (BrokenPipeError, OSError):
        pass
    if output_file:
        try:
            with open(output_file, "wb") as f:
                f.write(struct.pack('<4sI4s4sIHHIIHH4sI',
                    b'RIFF', 36 + len(pcm), b'WAVE', b'fmt ', 16, 1, channels,
                    rate, rate * width * channels, width * channels,
                    width * 8, b'data', len(pcm)))
                f.write(pcm)
        except OSError:
            pass
    send_progress(progress_token, 100, 100, "✅ Done (offline)")
    return {"ok": True, "engine": "wyoming"}
```

(`sys` — check it's imported in speech_tts.py; add if missing. `struct`, `time`, `rms_energy`, `register_proc`, `_pause_event`, `is_cancelled`, `_start_player`, `send_progress`, `play_speak` all already imported per the file's header.)

- [ ] **Step 3: Hook `tts()`.** Replace this exact block:

```python
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
```

with:

```python
    # Circuit breaker open (or forced offline): skip Azure entirely
    if wyoming.skip_azure():
        result = _tts_wyoming(text, None, audio_level_cb=audio_level_cb,
                              output_file=output_file,
                              progress_token=progress_token)
        if result is not None:
            return result
        return {"error": "Azure marked down and offline fallback unavailable"}

    # Take pre-warmed player or start fresh (overlaps with TTS API latency)
    proc = _take_prewarmed_player(tts_rate) or _start_player(tts_rate)
    if proc is None:
        return {"error": "No audio player found — set 'player' in config"}

    # Fire TTS request (player is already waiting for stdin data)
    try:
        resp = get_http_session().post(url, headers=headers, data=ssml.encode("utf-8"), timeout=60, stream=True)
    except Exception as e:
        # Network-class failure — try the LAN Wyoming fallback
        wyoming.mark_azure_down()
        result = _tts_wyoming(text, proc, audio_level_cb=audio_level_cb,
                              output_file=output_file,
                              progress_token=progress_token)
        if result is not None:
            return result
        return {"error": f"TTS request failed: {e}"}
    if resp.status_code != 200:
        if resp.status_code >= 500:
            wyoming.mark_azure_down()
            result = _tts_wyoming(text, proc, audio_level_cb=audio_level_cb,
                                  output_file=output_file,
                                  progress_token=progress_token)
            if result is not None:
                return result
        proc.stdin.close()
        proc.wait()
        return {"error": f"Azure TTS error {resp.status_code}: {resp.text}"}
    wyoming.mark_azure_up()
```

- [ ] **Step 4: Validate — forced offline speaks in Piper's voice**

```bash
cd ~/Projects/speech-to-cli
python3 -c "import py_compile; py_compile.compile('speech_tts.py', doraise=True)"
SPEECH_FORCE_OFFLINE=1 python3 speak.py "The realm endures, even offline."
python3 speak.py -v en-US-AndrewNeural "And Azure has returned."
```
Expected: first plays in Piper (Cori) voice; second in Andrew (Azure, breaker not tripped by forced mode... note: forced mode must not set `_azure_down_until` — it doesn't; `skip_azure()` just reads the env).

- [ ] **Step 5: Commit** — `git add speech_tts.py && git commit -m "feat: TTS falls back to Wyoming/Piper when Azure is unreachable"`

---

### Task 3: STT fallback in `stt.py`

**Files:** Modify `stt.py`

- [ ] **Step 1: Import + helper.** Add `import wyoming` near the top imports. Add above `_rest_stt_fallback`:

```python
def _wyoming_stt(raw_data, _log):
    """Offline STT via the configured Wyoming server. Returns text or ""."""
    host = CONFIG.get("wyoming_host", "")
    if not host:
        return ""
    try:
        text = wyoming.transcribe(
            host, int(CONFIG.get("wyoming_stt_port", 10300)),
            raw_data, rate=16000)
        _log(f"Wyoming STT recovered: {repr(text[:100])}")
        return text
    except wyoming.WyomingError as e:
        _log(f"Wyoming STT failed: {e}")
        return ""
```

- [ ] **Step 2: Restructure `_rest_stt_fallback`** so Wyoming fires only on *network-class* Azure failure (never on Azure-reachable "NoMatch" — silence must stay silent):

At the top of the function body after `raw_data` is built and the empty check passes, add:

```python
    if wyoming.skip_azure():
        _log("Azure marked down — Wyoming STT direct")
        return _wyoming_stt(raw_data, _log)
    azure_reachable = False
```

In the existing `try:` block, set `azure_reachable = True` immediately after `if resp.status_code == 200:`; add `wyoming.mark_azure_up()` on the same branch; in the `else:` (HTTP error) branch add `if resp.status_code >= 500: wyoming.mark_azure_down()`. In the `except Exception as e:` branch add `wyoming.mark_azure_down()`. Replace the final `return ""` with:

```python
    if azure_reachable:
        return ""
    return _wyoming_stt(raw_data, _log)
```

- [ ] **Step 3: Same pattern at any other Azure REST STT site.** Run `grep -n "stt.speech.microsoft.com" stt.py` — for each additional site (e.g. the primary batch recognizer), apply the identical shape: breaker check before the call, `mark_azure_up/down` on outcome, `_wyoming_stt(raw_data, _log)` only on network-class failure. Reuse the helper; keep the raw 16 kHz PCM in scope for it.

- [ ] **Step 4: Validate — forced offline dictation round-trip**

```bash
cd ~/Projects/speech-to-cli
python3 -c "import py_compile; py_compile.compile('stt.py', doraise=True)"
SPEECH_FORCE_OFFLINE=1 python3 listen.py --markers --seconds 8
# JP (or the mic pass) speaks: transcript must come back via Parakeet.
# Headless proxy check (no mic needed): synthesize with Piper, feed PCM:
python3 -c "
import os, wyoming
WYOMING_HOST = os.environ['WYOMING_HOST']
r, w, c, pcm = wyoming.synthesize(WYOMING_HOST, 10200, 'testing offline dictation')
import audioop, sys
# resample 22050 -> 16000 for the ASR path shape used by _wyoming_stt
pcm16, _ = audioop.ratecv(pcm, 2, 1, r, 16000, None)
import stt
print(repr(stt._wyoming_stt(pcm16, print)))
"
```
Expected: transcript contains "testing offline dictation" (approximately).

- [ ] **Step 5: Commit** — `git add stt.py && git commit -m "feat: STT falls back to Wyoming/Parakeet on Azure network failure"`

---

### Task 4: gnome-speaks live validation + docs + ship

- [ ] **Step 1: Service-level forced-offline check** (transient env; gnome-speaks needs no code change):

```bash
systemctl --user set-environment SPEECH_FORCE_OFFLINE=1 && systemctl --user restart gnome-speaks.service && sleep 2
curl -s -X POST localhost:7710/speak -H 'Content-Type: application/json' -d '{"text":"Offline voice check via the queue."}'
sleep 5; journalctl --user -u gnome-speaks.service -n 20 --no-pager | grep -iE "wyoming|engine"
systemctl --user unset-environment SPEECH_FORCE_OFFLINE && systemctl --user restart gnome-speaks.service
curl -s -X POST localhost:7710/speak -H 'Content-Type: application/json' -d '{"text":"Azure voice restored."}'
```
Expected: first utterance in Piper's voice, second in the configured Azure voice; service stays `active` throughout.

- [ ] **Step 2: Docs.** README.md: new "Offline fallback (Wyoming)" section — config keys, breaker behavior, `SPEECH_FORCE_OFFLINE=1`, degradation notes (single Piper voice, no live partials). CLAUDE.md: update the faster-whisper line ("optional local STT" → superseded by Wyoming LAN fallback; faster-whisper remains unused).

- [ ] **Step 3: Ship.** Secrets/leak scan on the branch diff (grep the full patch history for internal hostnames, LAN IP prefixes, and site domains — **it must hit nothing**; keep live-check hostnames out of committed docs by writing `<wyoming-host>` in README examples). Push, PR, merge, back to main, restart service, update gnome-speaks CLAUDE.md external-deps note + memory `project_architecture.md` if stale.

---

## Self-review notes

- Spec coverage: client+breaker (T1), config keys (T2S1), TTS hook incl. breaker-skip/5xx/exception + player-rate restart + output_file (T2S2-3), STT hook with azure_reachable discipline (T3), force-offline lever used as the test seam throughout, engine logging (helper prints + gnome-speaks journal), docs + public-repo leak scan (T4). No gaps.
- The plan's validation commands use the real LAN hostname; T4S3 scrubs the committed plan/doc copies to `<wyoming-host>` before push — same amend-out discipline as the spellbook ship.
- Type consistency: `wyoming.synthesize → (rate, width, channels, pcm)` used identically in T1 test, T2 helper, T3 headless check; `_wyoming_stt(raw_data, _log)` matches T3 usage.
