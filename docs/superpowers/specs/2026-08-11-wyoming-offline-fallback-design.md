# Wyoming offline fallback — LAN STT/TTS when Azure is unreachable

## Problem

Every STT/TTS path in this engine layer requires Azure. When the WAN (or Azure)
is down, `tts()` returns an error dict and dictation dies. A LAN Wyoming stack
(Piper TTS + ONNX-ASR/Parakeet STT, plus openWakeWord) already runs on a home
server for other consumers — the engine should fall back to it automatically.
The previously-envisioned local option (`faster-whisper`) is not installed and
stays dead; the LAN stack supersedes it.

## Decisions (JP, 2026-08-11)

1. **Engine-layer fallback** (this repo): every consumer — gnome-speaks service,
   `speak.py`, `listen.py`, MCP — inherits it. gnome-speaks needs zero changes:
   its streaming worker already falls back into `_rest_stt_fallback`, which is
   one of the two hook points.
2. **Config-driven, disabled by default.** The Wyoming host lives only in
   `~/.config/speech-to-cli/config.json` (this repo is public; no LAN hosts in
   code): `wyoming_host` ("" = off), `wyoming_tts_port` (10200),
   `wyoming_stt_port` (10300), `wyoming_tts_voice` ("" = server default).
3. **Automatic on network-class failures only** (connect errors, timeouts,
   HTTP 5xx). Azure *content* errors (4xx, bad SSML) do not fall back.
4. **Circuit breaker:** module-level "Azure down until T" (60 s) so repeated
   offline calls skip the Azure timeout. `SPEECH_FORCE_OFFLINE=1` env forces
   the breaker open — the manual offline lever and the test seam.
5. **Out of scope:** katana-local engines (both hosts down = today's error),
   wake word (separate project), offline voice parity (everything speaks in
   Piper's voice), live partials offline (final transcript only).

## Components

### `wyoming.py` (new, stdlib-only)

Minimal Wyoming protocol client (JSONL event headers + binary payloads over
TCP). Lenient reader: header line may carry `data_length` (read that many bytes
as the data JSON) and/or `payload_length` (read payload bytes).

- `synthesize(host, port, text, voice=None, timeout=10.0) -> (rate, width, channels, pcm_bytes)`
  Sends `synthesize`; collects `audio-start` (format), `audio-chunk` payloads,
  `audio-stop`. Raises `WyomingError` on any failure.
- `transcribe(host, port, pcm, rate=16000, width=2, channels=1, timeout=20.0) -> str`
  Sends `transcribe`, `audio-start`, chunked `audio-chunk`s (~4 KB), `audio-stop`;
  reads until `transcript`; returns its `text`. Raises `WyomingError`.

### Circuit breaker (in `wyoming.py`)

- `azure_marked_down()` → True while `time.time() < _azure_down_until` or
  `SPEECH_FORCE_OFFLINE=1`.
- `mark_azure_down(cooldown=60)` / `mark_azure_up()`.
- `enabled()` → `CONFIG.get("wyoming_host")` non-empty (import-free check via a
  callable set by `state`, or read CONFIG directly — implementation picks the
  pattern matching existing modules).

### TTS hook (`speech_tts.tts()`)

- If `wyoming.enabled()` and `wyoming.azure_marked_down()`: skip Azure, go
  straight to Wyoming.
- Otherwise wrap the Azure POST: `requests` network exceptions and 5xx call
  `mark_azure_down()` and fall through to Wyoming (content 4xx returns the
  error as today). Success calls `mark_azure_up()`.
- Wyoming path: `synthesize(...)`, terminate/discard the pre-warmed Azure-rate
  player, `_start_player(wyoming_rate)`, stream PCM in ~16 KB chunks honoring
  `is_cancelled()` / pause event / `audio_level_cb` (reuse the existing
  download-thread pattern), honor `output_file` via `write_wav`. Returns
  `{"ok": True, "engine": "wyoming"}`; on `WyomingError` returns the original
  Azure error dict (contract unchanged). One INFO log line names the engine.

### STT hook (`stt.py`)

- `_rest_stt_fallback(raw_frames, ...)`: on Azure REST network failure (or
  breaker open), `wyoming.transcribe(b"".join(raw_frames))` → text. Covers the
  gnome-speaks streaming worker and the batch VAD path that already route
  through it.
- The primary batch REST call site(s) get the same wrap: network failure →
  mark down → Wyoming transcribe of the recorded PCM/WAV bytes.

## Error handling

Wyoming failures log at WARNING and surface the *original* Azure error (or a
`{"error": "offline fallback failed: …"}` when the breaker skipped Azure).
Short connect timeout (2 s) so a down LAN host doesn't stall calls. No retries
beyond the single fallback attempt.

## Validation (no test framework — live checks)

1. `python3 -c` round-trips against the configured host: `synthesize` returns
   nonzero PCM at a sane rate; `transcribe` on a synthesized "hello world" WAV
   returns text containing "hello".
2. `SPEECH_FORCE_OFFLINE=1 python3 speak.py "The realm endures"` → Piper voice.
3. `SPEECH_FORCE_OFFLINE=1 python3 listen.py --markers` → spoken phrase comes
   back transcribed by Parakeet.
4. gnome-speaks: restart service with `SPEECH_FORCE_OFFLINE=1` in its
   environment (systemd drop-in or transient env), `POST /speak` → Piper voice,
   dictation round-trip; then remove the env and confirm Azure resumes.
5. Breaker: with Azure key intact and force-offline off, calls use Azure
   (journal shows `engine=azure`).
