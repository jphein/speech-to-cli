# Azure Speech MCP Server (speech-to-cli)

## What This Is
MCP server providing voice I/O (STT + TTS) for AI CLI agents via Azure Speech Services.
Version 4.1.0. Modular implementation across 5 files (~3100 lines total).

## Module Structure
| Module | Lines | Purpose |
|--------|------:|---------|
| `state.py` | ~270 | Shared mutable state, constants, config, cached timestamp |
| `audio.py` | ~690 | Audio I/O, device detection, chimes, colors, VAD, pre-warming |
| `stt.py` | ~485 | STT backends (streaming WS, VAD+REST, Whisper, fixed) |
| `speech_tts.py` | ~835 | TTS (Azure TTS, multi_speak, talk full-duplex) |
| `mcp_speech.py` | ~820 | MCP protocol: tool schemas, request routing, stdio transport |

**Import pattern**: `import state` at top of each module. Reassignable globals via `state.X = val`. Containers (`CONFIG` dict) imported by name since mutated in-place.

## Architecture
- **Runtime**: Python 3, threading-based concurrency, `requests` for HTTP
- **STT**: Azure WebSocket streaming with energy-gated VAD (primary), local Whisper fallback
- **TTS**: Azure REST API with SSML, HTTP connection pooling, streaming playback via aplay
- **Protocol**: MCP v2024-11-05 over stdio JSON-RPC 2.0
- **Config**: `~/.config/speech-to-cli/config.json` or env vars (`AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`)

## MCP Tools
| Tool | Purpose |
|------|---------|
| `listen` | Record from mic, return transcribed text (STT) |
| `speak` | Speak text aloud (single voice, single segment) |
| `multi_speak` | Speak multiple text+voice segments — TTS requests fire in parallel, playback is sequential |
| `talk` | Full-duplex speak+listen — PRIMARY tool for voice conversations |
| `converse` | Listen only, signals conversational intent (for speak→converse loops with tool calls in between) |
| `configure` | View/change settings (subtitle colors, chimes, audio devices, etc.) |
| `get_voices` | List available Azure voices |
| `pause` / `resume` | Pause/resume audio playback |

## Colored Subtitles (v4.1.0)
Live scrolling subtitles with ANSI color support:
- `subtitle_color_user`: Color for user's transcribed speech (default: `light_green`)
- `subtitle_color_tts`: Color for agent's TTS speech (default: `amber`)
- Per-call `subtitle_color` param on `speak`, `talk`, `multi_speak` segments
- Colors: default, green, light_green, yellow, amber, rust, red, light_red, blue, light_blue, cyan, light_cyan, magenta, light_magenta, white, gray

## Key Implementation Details
- TTS max chars: 5000 per segment (`_MAX_TTS_CHARS`)
- HD quality uses 48kHz, fast uses 24kHz
- Cancellation support via `_cancel_event` threading.Event
- WebSocket connections cached for 540s (`WS_IDLE_TIMEOUT`)
- Noise floor cached for 30s between calls
- Debounced connection pre-warming (`_schedule_warmup`)
- Cached ISO timestamps in STT hot loop (avoids 1000+ strftime calls)
- Progress notification dedup to reduce MCP notification spam
- Full-duplex plays a chime when TTS finishes to signal "your turn"
- Silence countdown indicator (○→◔→◑→◕→●) shows how close to cutoff during listening
- `talk_silence_timeout` default 4.0s (configurable) — time to think between phrases
- Dynamic agent hints: tool responses include `[AGENT HINTS]` suggesting config changes based on user behavior (repeated no-speech, short responses)
- Agents guided to pass per-call `silence_timeout` (2s for yes/no, 6-8s for open-ended)

## Pairing with azure-chat-assistant
For voice conversations with multiple Azure AI models, use alongside the
[azure-chat-assistant](../azure-chat-assistant/) MCP server. Optimized 2-call flow:
1. `multi_chat` (chat server) — parallel LLM queries
2. `multi_speak` (this server) — parallel TTS synthesis + sequential playback

### Voice Assignments (for multi-agent calls)
| Agent | Voice | Subtitle Color |
|-------|-------|---------------|
| Claude | en-US-AvaNeural | amber |
| GPT-5.3 | en-US-DavisNeural | cyan |
| Llama 405B | en-US-AndrewNeural | yellow |
| DeepSeek R1 | en-US-BrianNeural | magenta |
| Phi-4 | en-US-JennyNeural | light_blue |
