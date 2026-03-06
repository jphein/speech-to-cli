# speech-to-cli

Voice interface for AI coding assistants — talk to your CLI agent and hear it respond, powered by Azure Speech Services. Works with **GitHub Copilot CLI**, **Claude Code**, and **Gemini CLI**.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![License](https://img.shields.io/badge/license-GPLv3-green)

## What it does

This project adds voice input and output to your terminal AI workflow via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/):

| Tool | Description |
|------|-------------|
| **MCP Server** (`mcp_speech.py`) | Integrates with Copilot CLI, Claude Code, and Gemini CLI via MCP — gives your AI `listen`, `speak`, and `converse` tools |
| **Voice Chat** (`voice_chat.py`) | Standalone voice chat companion — runs in a second terminal alongside Copilot CLI |
| **Speech-to-Text** (`speech.py`) | Simple mic → text → clipboard tool |
| **Text-to-Speech** (`tts.py`) | Simple text → speech tool (reads from args, stdin, or clipboard) |

## Features

- **Azure HD Voices**: Uses high-quality DragonHD voices for natural-sounding speech.
- **Thinking Hum**: A subtle 150Hz tone that loops while the AI is processing.
- **Visual Status**: Colorful progress bars with live VU meters and real-time subtitles (🎤/🧠/🔊).
- **Audio Feedback**: Configurable chimes for ready, processing, speak, and done states.
- **Low Latency**: Streaming playback and persistent connections for fast responses.
- **VAD**: Energy-gated voice activity detection that auto-calibrates to your environment.

## About GitHub Copilot CLI

[GitHub Copilot CLI](https://docs.github.com/copilot/concepts/agents/about-copilot-cli) brings agentic AI coding assistance directly to your terminal. It can edit files, run commands, search code, and interact with GitHub — all through natural language.

### Available models

Copilot CLI gives you access to multiple frontier models via the `/model` command:

| Model | Tier |
|-------|------|
| Claude Sonnet 4.5 | Standard (default) |
| Claude Sonnet 4 | Standard |
| Claude Opus 4.5 | Premium |
| Claude Opus 4.6 | Premium |
| Claude Haiku 4.5 | Fast |
| GPT-5.1 / 5.2 / 5.4 | Standard |
| GPT-5.1-Codex / 5.2-Codex / 5.3-Codex | Standard |
| GPT-5.1-Codex-Max | Standard |
| GPT-5 mini | Fast |
| Gemini 3 Pro (Preview) | Standard |

### Free for students

**GitHub Education** members get **Copilot Pro free for 1 year**, which includes Copilot CLI access. Sign up at [education.github.com](https://education.github.com/) with your school email — no credit card required. Each prompt uses one premium request from your monthly quota.

### Install Copilot CLI

```bash
curl -fsSL https://gh.io/copilot-install | bash
copilot          # launch and authenticate
```

## Quick start

### Prerequisites

- Linux with ALSA audio (`arecord`/`aplay`)
- Python 3.8+
- An [Azure Speech Services](https://azure.microsoft.com/en-us/products/ai-services/speech-services) API key (free tier: 5 hours STT + 500K characters TTS per month)

### Azure Speech HD voices

This project defaults to **Azure HD (DragonHD) voices** — specifically `en-US-Ava:DragonHDLatestNeural`. These are Azure's highest-quality neural voices with natural intonation, breathing, and expressiveness that sounds remarkably human. You can change the voice via the `AZURE_SPEECH_VOICE` env var or config file.

Browse all available voices in the [Azure Voice Gallery](https://speech.microsoft.com/portal/voicegallery). Look for voices tagged **HD** or **DragonHD** for the best quality.

### Azure for nonprofits and education

- **Nonprofits**: Microsoft offers up to **$3,500/year in free Azure credits** through [Azure for Nonprofits](https://nonprofit.microsoft.com/en-us/getting-started). This more than covers Speech Services usage for voice-enabled Copilot workflows.
- **Students**: The [Azure for Students](https://azure.microsoft.com/en-us/free/students/) program provides **$100 in free credits** with no credit card required — just verify with your school email.

### Install

```bash
git clone https://github.com/jphein/speech-to-cli.git
cd speech-to-cli
./install.sh
```

### Configure

Set your Azure credentials via environment variables:

```bash
export AZURE_SPEECH_KEY="your-key-here"
export AZURE_SPEECH_REGION="westus2"           # optional, default: westus2
export AZURE_SPEECH_VOICE="en-US-Ava:DragonHDLatestNeural"  # optional
```
Or create a JSON config file at `~/.config/speech-to-cli/config.json`. You can create the directory if it doesn't exist:

```bash
mkdir -p ~/.config/speech-to-cli
```

Example `config.json`:

```json
{
  "key": "your-azure-speech-key",
  "region": "westus2",
  "voice": "en-US-Ava:DragonHDLatestNeural",
  "fast_voice": "en-US-AvaNeural",
  "chime_hum": true,
  "visual_indicator": true
}
```

### Configuration Settings

| Key | Default | Description |
|-----|---------|-------------|
| `key` | None | Your Azure Speech Services API key. |
| `region` | `westus2` | The Azure region for your speech resource. |
| `voice` | `en-US-Ava:DragonHDLatestNeural` | Primary voice for high-quality (HD) synthesis. |
| `fast_voice` | `en-US-AvaNeural` | Low-latency voice for fast responses. |
| `chime_ready` | `true` | Play an ascending tone when the microphone opens. |
| `chime_processing`| `false` | Play a short "blip" when speech is recognized. |
| `chime_hum` | `false` | Start a looping 150Hz tone while thinking. |
| `chime_speak` | `false` | Play a descending tone before starting to speak. |
| `chime_done` | `false` | Play a double-tap tone when the AI is done talking. |
| `visual_indicator`| `true` | Show status icons (🎤/🧠) in the terminal. |
| `live_subtitles` | `true` | Show real-time partial transcription in progress bar. |
| `vu_meter` | `true` | Show live volume meter in progress bar. |

> ⚠️ **Never commit your API key.** Use environment variables or the config file (which is in your home directory, outside the repo). See [Security](#security) below.

## Usage

### MCP Server (recommended)

The MCP server works with any AI CLI that supports the Model Context Protocol. **When paired with a modern terminal AI like Gemini CLI, Claude Code, or Copilot CLI, it creates an incredibly seamless voice loop.** The AI automatically invokes the `listen` tool when it needs input, processes your request, and calls the `speak` tool to respond—all while providing rich, real-time terminal UI feedback (like live subtitles and VU meters) without you ever needing to type a command.

#### GitHub Copilot CLI

Add to your `~/.copilot/mcp.json`:

```json
{
  "mcpServers": {
    "azure-speech": {
      "command": "python3",
      "args": ["/path/to/speech-to-cli/mcp_speech.py"]
    }
  }
}
```

#### Claude Code

Option A — Use the included `.mcp.json` (auto-detected when working in the project directory).

Option B — Add globally via the CLI:

```bash
claude mcp add --transport stdio azure-speech -- python3 /path/to/speech-to-cli/mcp_speech.py
```

#### Gemini CLI

Install as a Gemini extension (uses the included `gemini-extension.json`):

```bash
gemini extensions install /path/to/speech-to-cli
```

Or add manually to your `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "azure-speech": {
      "command": "python3",
      "args": ["/path/to/speech-to-cli/mcp_speech.py"]
    }
  }
}
```

---

Restart your CLI — it will now have `listen`, `speak`, and `converse` tools available. Just say "listen" and your AI will record your voice, transcribe it, and respond. Ask it to "speak" and it'll read its response aloud. Use "converse" for a continuous voice chat loop.

**MCP Tools:**

| Tool | Parameters | Description |
|------|-----------|-------------|
| `listen` | `seconds` (1-30), `mode` (streaming/vad/whisper/fixed) | Records from mic, returns transcribed text |
| `speak` | `text` (required), `quality` (fast/hd) | Speaks text aloud via Azure TTS |
| `converse` | `seconds`, `mode` | Like `listen`, but signals conversational intent — Copilot will speak its reply then listen again |

**STT Modes** (auto-selected by default):

| Mode | Description |
|------|-------------|
| `streaming` | Real-time Azure WebSocket + energy-gated VAD (fastest, default) |
| `vad` | Record with VAD, upload on silence |
| `whisper` | Local transcription via faster-whisper (offline, no network) |
| `fixed` | Record for full duration, then upload (fallback) |

**TTS Quality:**

| Quality | Voice | Azure latency | Best for |
|---------|-------|--------------|----------|
| `fast` (default) | AvaNeural | ~120ms | Conversation, quick responses |
| `hd` | DragonHD | ~1200ms | High-quality narration |

### Voice Chat (standalone companion)

Run in a separate terminal alongside Copilot CLI:

```bash
python3 voice_chat.py
```

Flow:
1. Press **Enter** → records your voice → transcribes → copies to clipboard
2. Paste into Copilot CLI with **Ctrl+Shift+V**
3. Copy Copilot's response with **Ctrl+Shift+C**
4. Press **Enter** → speaks the response aloud → starts recording again

### Standalone tools

```bash
# Speech-to-text: record and transcribe to clipboard
python3 speech.py

# Text-to-speech: speak text aloud
python3 tts.py "Hello world"
echo "Hello" | python3 tts.py
python3 tts.py  # speaks clipboard contents
```

## How it works

- **Recording**: Uses `arecord` (ALSA) to capture 16kHz mono audio from the default input device
- **Voice Activity Detection**: Energy-gated VAD auto-calibrates to ambient noise, stops recording on silence (~400ms after speech ends)
- **Speech-to-Text**: Streams audio to Azure via persistent WebSocket for real-time recognition, with local Whisper fallback for offline use
- **Text-to-Speech**: Sends SSML to Azure TTS REST API with HTTP connection pooling; streams MP3 audio through mpv for immediate playback
- **Ready chime**: A short ascending tone plays before each recording so you know when to speak
- **Performance**: Connections are pre-warmed on startup; noise floor is cached between calls; response latency is ~275ms from end of speech to first audio byte
- **MCP Protocol**: Implements [MCP](https://modelcontextprotocol.io/) (v2024-11-05) over stdio JSON-RPC for direct Copilot CLI integration

No Azure SDK required — just plain REST/WebSocket API calls.

## Security

This project handles audio data and API credentials. Please review:

### API key management
- **Never hardcode your Azure key in source code or commit it to git.** The `.gitignore` includes `.env` to help prevent this.
- Store your key via environment variable (`AZURE_SPEECH_KEY`) or in the user-level config file (`~/.config/speech-to-cli/config.json`).
- Azure keys can be rotated at any time in the [Azure Portal](https://portal.azure.com/) → your Speech resource → Keys and Endpoint.
- Consider using a restricted key with only Speech Services access (not a broad subscription key).

### Audio and data privacy
- **Audio is recorded from your local microphone** and sent to Azure Speech Services for processing. No audio is stored locally after transcription (temp files are deleted immediately).
- **Azure processes your audio** to produce transcriptions. Review the [Azure AI Services data privacy policy](https://learn.microsoft.com/en-us/legal/cognitive-services/speech-service/speech-to-text/data-privacy-security) to understand how Microsoft handles your audio data.
- **Text sent to TTS** is transmitted to Azure for synthesis. The same privacy policies apply.
- **No data is sent anywhere other than Azure Speech Services** — there are no analytics, telemetry, or third-party services.

### Network security
- All Azure API calls use **HTTPS** (TLS encrypted in transit).
- Audio data and API keys are sent over encrypted connections only.

### MCP server scope
- The MCP server exposes three tools (`listen`, `speak`, `converse`). It cannot read files, execute commands, or access anything beyond the microphone and Azure API.
- The server communicates with Copilot CLI over local stdio only — no network listeners are opened.

### Recommendations
- Rotate your Azure key periodically.
- Use Azure's free tier to limit potential cost exposure from a leaked key.
- If running on a shared machine, be aware that other users with access to your environment variables or config file can read your API key.

## Legal

### Azure Speech Services
Use of Azure Speech Services is subject to the [Microsoft Azure terms of service](https://azure.microsoft.com/en-us/support/legal/) and the [Azure AI Services terms](https://www.microsoft.com/licensing/terms/productoffering/MicrosoftAzure). You are responsible for your own Azure usage and billing.

### GitHub Copilot
Use of GitHub Copilot CLI requires an active Copilot subscription and is subject to the [GitHub Copilot terms](https://docs.github.com/en/site-policy/github-terms/github-terms-for-additional-products-and-features#github-copilot). Copilot is free for verified students, teachers, and maintainers of popular open-source projects.

### This project
This project is independently developed and is **not affiliated with, endorsed by, or sponsored by Microsoft or GitHub**. It is a third-party integration that connects to their respective APIs.

Licensed under the GNU General Public License v3.0 (GPLv3) — see [LICENSE](LICENSE).

## License

GNU General Public License v3.0 (GPLv3) — see [LICENSE](LICENSE) for details.
