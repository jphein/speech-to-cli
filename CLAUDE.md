# Azure Speech MCP Server (speech-to-cli)

## What This Is
MCP server providing voice I/O (STT + TTS) for AI CLI agents via Azure Speech Services.
Version 4.0.0. Single-file implementation: `mcp_speech.py` (~2800 lines).

## Architecture
- **Runtime**: Python 3, threading-based concurrency, `requests` for HTTP
- **STT**: Azure WebSocket streaming with energy-gated VAD (primary), local Whisper fallback
- **TTS**: Azure REST API with SSML, HTTP connection pooling, streaming playback via mpv/aplay
- **Protocol**: MCP v2024-11-05 over stdio JSON-RPC 2.0
- **Config**: `~/.config/speech-to-cli/config.json` or env vars (`AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`)

## MCP Tools
| Tool | Purpose |
|------|---------|
| `listen` | Record from mic, return transcribed text (STT) |
| `speak` | Speak text aloud (single voice, single segment) |
| `multi_speak` | Speak multiple text+voice segments — TTS requests fire in parallel, playback is sequential |
| `converse` | Like listen, but signals conversational intent (agent will speak reply then listen again) |
| `configure` | View/change settings |
| `get_voices` | List available Azure voices |
| `pause` / `resume` | Pause/resume audio playback |

## multi_speak (v4.0.0)
Designed for multi-agent voice conversations. Takes an array of `{text, voice}` segments:
- All TTS requests are fired concurrently via threads
- Audio plays back-to-back sequentially
- Reduces N round trips to 1 MCP call

### Voice Assignments (for multi-agent calls)
| Agent | Voice |
|-------|-------|
| Claude | en-US-AvaNeural |
| GPT-5.3 | en-US-DavisNeural |
| Llama 405B | en-US-AndrewNeural |
| DeepSeek R1 | en-US-BrianNeural |

## Key Implementation Details
- TTS max chars: 5000 per segment (`_MAX_TTS_CHARS`)
- HD quality uses 48kHz, fast uses 24kHz
- Cancellation support via `_cancel_event` threading.Event
- WebSocket connections cached for 540s (`WS_IDLE_TIMEOUT`)
- Noise floor cached for 30s between calls
- Connection pre-warming on startup and after each speak

## Pairing with azure-chat-assistant
For voice conversations with multiple Azure AI models, use alongside the
[azure-chat-assistant](../azure-chat-assistant/) MCP server. Optimized 2-call flow:
1. `multi_chat` (chat server) — parallel LLM queries
2. `multi_speak` (this server) — parallel TTS synthesis + sequential playback
