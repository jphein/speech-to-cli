# Azure Speech MCP Server (speech-to-cli)

MCP server providing voice I/O (STT + TTS) for AI CLI agents via Azure Speech Services.
Version 4.2.0. Python 3, threading-based, no Azure SDK — plain REST/WebSocket.

## Running
```bash
python3 mcp_speech.py          # MCP server over stdio (used by Claude Code, Copilot CLI, Gemini CLI)
python3 speech.py              # Standalone mic→text→clipboard
python3 tts.py "hello"         # Standalone text→speech
python3 voice_chat.py          # Standalone voice chat companion
SPEECH_DEBUG=1 python3 mcp_speech.py  # Debug mode → /tmp/speech-debug.log
```

## Module Structure & Import Graph
```
state.py (285L) — constants, CONFIG dict, mutable globals, small helpers
  ↑ imported by all modules
audio.py (691L) — audio I/O, device detection, chimes, colors, VAD, pre-warming
  ↑ imports state
stt.py (650L) — STT backends (streaming WS, VAD+REST, Whisper, fixed) + shared helpers
  ↑ imports state, audio
speech_tts.py (893L) — TTS, multi_speak, talk_fullduplex, get_voices
  ↑ imports state, audio, stt
mcp_speech.py (929L) — MCP tool schemas, request routing, stdio transport
  ↑ imports state, audio, stt, speech_tts
```
Standalone utilities (`speech.py` 119L, `tts.py` 97L, `voice_chat.py` 170L) are independent — they don't import the core modules.

**Import pattern**: `import state` at top. Reassignable globals via `state.X = val`. `CONFIG` dict imported by name (mutated in-place, never reassigned).

## Key Function Index
### state.py — Shared state & helpers
| Line | Function | Purpose |
|-----:|----------|---------|
| 121 | `_get_iso_timestamp()` | Cached ISO timestamp (refreshed every 500ms) |
| 156 | `load_config()` | Load config from `~/.config/speech-to-cli/config.json` + env vars |
| 207 | `is_cancelled()` | Check `_cancel_event` |
| 227 | `cancel_active()` | Kill tracked subprocesses, signal cancellation |
| 253 | `get_http_session()` | Reusable `requests.Session` for connection pooling |
| 261 | `send_progress(token, progress, total, description)` | MCP progress notification (ANSI in `description`, plain in `message`) |

### audio.py — Audio I/O, VAD, pre-warming
| Line | Function | Purpose |
|-----:|----------|---------|
| 70 | `detect_audio_output(sink_id)` | PipeWire: headphones vs speakers → half/full-duplex |
| 134 | `_refresh_audio_detection()` | Re-detect on sink change (~12ms, ~35ms on change) |
| 172 | `_build_player_cmd(tts_rate, target)` | Build aplay/pw-cat/ffplay command list |
| 200 | `_build_rec_cmd(max_seconds, raw)` | Build pw-record/arecord command |
| 335 | `_colorize(text, color_name)` | ANSI color wrapper (uses `_COLOR_MAP` dict at line 323) |
| 409 | `rms_energy(frame_bytes)` | RMS energy of 16-bit PCM frame |
| 418 | `calibrate_noise(proc, n_frames)` | Ambient noise estimation (cached 30s) |
| 441 | `is_speech_energy(chunk, vad, threshold)` | Combined VAD + energy gate |
| 538 | `record_with_vad(proc, max_seconds)` | Record with energy-gated VAD |
| 662 | `_prewarm_all()` | Pre-warm recorder + player + WS + TTS connection |
| 685 | `_schedule_warmup()` | Debounced warmup (collapses rapid calls) |

### stt.py — STT backends & shared helpers
| Line | Function | Purpose |
|-----:|----------|---------|
| 44 | `_get_stt_ws()` | Get/create persistent Azure STT WebSocket (cached 540s) |
| 85 | `_make_logger(tag)` | Debug logger factory → `/tmp/speech-debug.log` |
| 95 | `_check_end_word(text, end_word)` | Check if text ends with stop word |
| 105 | `_strip_end_word(text, end_word)` | Remove trailing end word |
| 136 | `_init_stt_ws_session(ws, request_id)` | Drain stale msgs, send config + WAV header |
| 162 | `_parse_ws_msg(msg, phrases, ...)` | Parse hypothesis/phrase/turn.end WS messages |
| 212 | `_rest_stt_fallback(raw_frames, _log)` | REST API fallback when WS fails |
| 260 | `stt_streaming(max_seconds, progress_token)` | Real-time STT via persistent Azure WS + VAD |
| 449 | `stt_vad(max_seconds, progress_token)` | Record with VAD, upload on silence |
| 504 | `stt_whisper(max_seconds, progress_token)` | Local faster-whisper (offline) |
| 552 | `stt_fixed(seconds, progress_token)` | Fixed-duration recording fallback |
| 609 | `stt(seconds, mode, silence_timeout, ...)` | Main STT dispatcher (auto-selects mode) |

### speech_tts.py — TTS & full-duplex talk
| Line | Function | Purpose |
|-----:|----------|---------|
| 66 | `_build_ssml(text, voice, quality, ...)` | Build SSML payload for Azure TTS |
| 86 | `_prepare_tts(text, quality, speed, ...)` | Sanitize inputs, build SSML, return (ssml, rate, headers, url) |
| 106 | `_build_multi_voice_ssml(segments, quality)` | Build SSML with multiple `<voice>` tags for single-request multi-voice TTS |
| 108 | `multi_speak(segments, quality, progress_token)` | Parallel TTS fetch, sequential playback |
| 131 | `multi_speak_stream(segments, quality, progress_token)` | Single-request multi-voice TTS via SSML (faster than multi_speak) |
| 230 | `tts(text, quality, speed, voice, ...)` | Single-segment TTS with streaming playback |
| 358 | `talk_fullduplex(text, quality, ...)` | Speak + listen simultaneously (full-duplex) |
| 883 | `get_voices()` | Fetch available Azure voices list |

### mcp_speech.py — MCP protocol layer
| Line | Function | Purpose |
|-----:|----------|---------|
| 38 | `TOOLS` | Tool schema definitions (list of dicts) |
| 338 | `handle_request(req)` | Main JSON-RPC request router |
| 856 | `_stdin_reader()` | Background stdin reader (routes pause/resume/cancel urgently) |
| 900 | `main()` | Entry point: generate chimes, detect audio, start warmup, event loop |

## MCP Tools
| Tool | Purpose |
|------|---------|
| `listen` | Record → transcribe (STT only) |
| `speak` | TTS one-way (no mic) |
| `multi_speak` | Parallel TTS fetch, sequential playback (multi-voice) |
| `multi_speak_stream` | Single-request multi-voice TTS via SSML multi-voice tags (faster than multi_speak) |
| `talk` | Full-duplex speak+listen — PRIMARY voice conversation tool |
| `converse` | Listen-only with conversational context (for speak→converse loops) |
| `configure` | View/change settings at runtime (saved to disk) |
| `get_voices` | List Azure voices for current region |
| `pause` / `resume` | Pause/resume audio |

## Key Implementation Details
- TTS max: 5000 chars/segment. HD=48kHz, fast=24kHz
- WS cached 540s (`WS_IDLE_TIMEOUT`), reconnects on Azure InitialSilenceTimeout
- REST STT fallback when WS returns empty
- Noise floor cached 30s. Post-TTS threshold reset to 300 (avoids TTS bleed)
- Recording timer counts post-TTS frames only (TTS time excluded from max)
- Debounced `_schedule_warmup()` pre-warms recorder + player + WS + HTTP
- Cached ISO timestamps in hot loop (avoids 1000+ strftime/s)
- End word "over" stops recording immediately (configurable)
- `[AGENT HINTS]` in responses suggest config changes based on user behavior
- Subtitle colors: default, green, light_green, yellow, amber, rust, red, light_red, blue, light_blue, cyan, light_cyan, magenta, light_magenta, white, gray

### Dynamic Configurables (via `configure` tool)
| Setting | Default | Range/Notes |
|---------|---------|-------------|
| `talk_silence_timeout` | 4.0s | Silence cutoff in talk mode |
| `silence_timeout` | 3.0s | Silence cutoff in listen mode |
| `no_speech_timeout` | 7.0s | Max wait for any speech (1-30s) |
| `energy_multiplier` | 2.5 | Mic sensitivity, lower=more sensitive (0.5-20) |
| `end_word` | "over" | Word to immediately stop recording |
| `max_record_seconds` | 120 | Max recording duration (5-300) |
| `enable_echo_cancel` | false | [Experimental] PipeWire echo cancellation |
| `enable_barge_in` | false | [Experimental] User speech pauses TTS |
| `debug` | false | Logs to /tmp/speech-debug.log |

## Conventions for Editing
- **State access**: Always `import state` then `state.X = val` for reassignable globals. Never do `from state import X` for mutable scalars.
- **CONFIG dict**: `from state import CONFIG` is fine — it's mutated in-place, never reassigned.
- **Threading**: All audio I/O runs in daemon threads. Use `is_cancelled()` checks in loops.
- **Progress**: Call `send_progress(token, pct, 100, "description")` — Claude Code renders ANSI in `description`, Gemini CLI uses plain `message` field.
- **SSML safety**: All user-supplied voice/pitch/volume values go through `_sanitize_ssml_attr()`.
- **Error returns**: Tool handlers return `{"error": "msg"}` dicts; MCP layer wraps in JSON-RPC.
- **No tests**: This project has no test suite. Test manually by running `python3 mcp_speech.py` and sending JSON-RPC on stdin, or use the MCP tools via Claude Code.

## Debugging
```bash
SPEECH_DEBUG=1 python3 mcp_speech.py   # or configure(debug=true) at runtime
tail -f /tmp/speech-debug.log          # watch STT/TTS debug output
```

## Planned: multi_talk
Multi-voice SSML in one TTS request (keeps talk_fullduplex architecture intact).
Alternative: multi_speak → converse (works today but mic not recording during TTS).

## Pairing with cloud-chat-assistant
Use alongside [cloud-chat-assistant](../cloud-chat-assistant/) MCP server for multi-model voice:
1. `multi_chat` → parallel LLM queries  2. `multi_speak` → parallel TTS + sequential playback

### Voice Assignments (multi-agent)
| Agent | Voice | Color |
|-------|-------|-------|
| Claude | en-US-AvaNeural | amber |
| GPT-5.3 | en-US-DavisNeural | cyan |
| Llama 405B | en-US-AndrewNeural | yellow |
| DeepSeek R1 | en-US-BrianNeural | magenta |
| Phi-4 | en-US-JennyNeural | light_blue |
