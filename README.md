# speech-to-cli

Voice interface for GitHub Copilot CLI — talk to Copilot and hear it respond using Azure Speech Services.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)

## What it does

This project provides three ways to add voice to your CLI workflow:

| Tool | Description |
|------|-------------|
| **MCP Server** (`mcp_speech.py`) | Integrates directly with Copilot CLI via MCP protocol — gives Copilot `listen` and `speak` tools |
| **Voice Chat** (`voice_chat.py`) | Standalone voice chat companion — runs in a second terminal alongside Copilot CLI |
| **Speech-to-Text** (`speech.py`) | Simple mic → text → clipboard tool |
| **Text-to-Speech** (`tts.py`) | Simple text → speech tool (reads from args, stdin, or clipboard) |

## Quick start

### Prerequisites

- Linux with ALSA audio (`arecord`/`aplay`)
- Python 3.8+
- An [Azure Speech Services](https://azure.microsoft.com/en-us/products/ai-services/speech-services) key (free tier works fine)

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

Or create a JSON config file at `~/.config/speech-to-cli/config.json`:

```json
{
  "key": "your-azure-speech-key",
  "region": "westus2",
  "voice": "en-US-Ava:DragonHDLatestNeural"
}
```

## Usage

### MCP Server (recommended with Copilot CLI)

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

Restart Copilot CLI — it will now have `listen` and `speak` tools available. Just say "listen" and Copilot will record your voice, transcribe it, and respond. Ask it to "speak" and it'll read its response aloud.

**MCP Tools:**

| Tool | Parameters | Description |
|------|-----------|-------------|
| `listen` | `seconds` (1-30, default 5) | Records from mic, returns transcribed text |
| `speak` | `text` (required) | Speaks text aloud via Azure TTS |

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

- **Recording**: Uses `arecord` (ALSA) to capture 16kHz mono WAV audio from the default input device
- **Speech-to-Text**: Sends audio to the [Azure STT REST API](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-speech-to-text-short)
- **Text-to-Speech**: Sends SSML to the [Azure TTS REST API](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech), plays back via `aplay`
- **MCP Protocol**: Implements the [Model Context Protocol](https://modelcontextprotocol.io/) (v2024-11-05) over stdio JSON-RPC for direct Copilot CLI integration

No Azure SDK required — just plain REST API calls.

## License

MIT
