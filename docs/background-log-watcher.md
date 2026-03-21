# Background Log Watcher Pattern

Use a Claude Code background agent with the `speak` MCP tool to monitor system logs in real-time and announce anomalies aloud via TTS.

## Overview

A background Claude Code agent tails a log source (e.g., `journalctl -k -f`), pattern-matches for concerning events, and calls the `speak` tool to deliver audio alerts — all while the main conversation continues working normally.

No HTTP API is involved. The agent calls `speak` as an MCP tool — the same mechanism Claude Code uses for all its tools. The speech-to-cli MCP server runs as a subprocess of Claude Code, communicating over stdio JSON-RPC.

The main agent can read the watcher's output file at any time to see what was detected and act on it autonomously.

## Architecture

```
┌──────────────────────┐
│  Main Claude Code    │  ← your conversation
│  conversation        │
│                      │
│  Agent tool call:    │
│    run_in_background │─────────┐
│    = true            │         │
└──────────────────────┘         │
                                 v
┌──────────────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Background Agent    │───>│ speech-to-cli   │───>│ Azure TTS    │
│  (separate Claude    │    │ MCP server      │    │ REST endpoint│
│   Code subprocess)   │    │ (stdio JSON-RPC)│    └──────┬───────┘
│                      │    │                 │           │
│  1. speak() announce │    │ mcp_speech.py   │    48kHz audio stream
│  2. journalctl -k -f │    │  handle_request │           │
│  3. sleep 30 + tail  │    │  → tts()        │           v
│  4. analyze output   │    │  → _build_ssml  │    ┌──────────────┐
│  5. speak() on error │    │  → Azure POST   │    │ PipeWire     │
│  6. loop forever     │    │  → pw-cat play  │    │ → speakers   │
└──────────┬───────────┘    └─────────────────┘    └──────────────┘
           │
           │ output file
           v
  /tmp/claude-*/tasks/<agent-id>.output
  (main agent reads this to see what was detected)
```

## Data Flow for a speak() Call

When the background agent decides to announce something, here's the exact chain:

```
1. Agent LLM generates tool_use:
   {"name": "mcp__speech-to-cli__speak",
    "input": {"text": "Critical: xHCI controller died", "voice": "en-US-DavisNeural"}}

2. Claude Code harness serializes to JSON-RPC over stdio:
   → {"method": "tools/call", "params": {"name": "speak", "arguments": {...}}}

3. mcp_speech.py receives on stdin, routes via handle_request():
   → speech_tts.tts(text, voice=..., quality="hd")

4. _prepare_tts() builds SSML XML payload

5. HTTP POST to Azure cognitiveservices TTS endpoint
   → Returns 48kHz PCM audio stream

6. Audio piped to pw-cat (PipeWire player)
   → Plays through speakers
```

The agent never touches HTTP directly. The MCP protocol abstracts the entire Azure TTS pipeline — the agent just calls `speak` like any other tool.

## How to Launch

From the main Claude Code conversation, use the Agent tool:

```
Agent tool call:
  description: "Watch [system] logs and speak alerts"
  name: "my-watcher"
  run_in_background: true
  prompt: |
    You are a [SYSTEM] log monitor. Your job is to tail [LOG_SOURCE]
    for [SPECIFIC EVENTS] and use the speech MCP tool to announce
    anything concerning.

    Steps:
    1. Use the speak tool to announce: "[System] log watcher online."
    2. Run [LOG_COMMAND] in the background via Bash
    3. Every ~30 seconds, check the output for anomalies
    4. On anomaly: announce via speak tool with a brief summary
    5. On normal: stay quiet — only speak when something is wrong
    6. Keep monitoring indefinitely

    Use the mcp__speech-to-cli__speak tool for announcements.
    Keep them short and clear.
```

### What the Agent Actually Does

The background agent is a separate Claude Code subprocess with its own context window. It has access to the same tools as the main conversation (Bash, MCP tools, etc.) but runs independently. Its internal loop:

1. **ToolSearch** for `"speak speech"` — MCP tools are deferred, so the agent fetches the schema first
2. **speak()** — announces itself: "USB log watcher online."
3. **Bash** with `run_in_background: true` — starts `journalctl -k -f` writing to a temp file
4. **Bash** with `sleep 30 && tail -10 <output-file>` — polls the journalctl output
5. **LLM reasoning** — reads the tail output, decides if anything is anomalous
6. **speak()** on anomaly — announces what it found, with context
7. **Goto 4** — loops indefinitely until killed

The polling pattern (`sleep 30 && tail`) is crude but effective. The Claude Code Bash tool has no streaming mode — foreground commands block until exit, background commands write to a file with no callback. So the agent must poll the output file periodically, even though the underlying `journalctl -f` is event-driven.

## Example: USB/xHCI Monitor

Used during the [kiyo-xhci-fix](https://github.com/jphein/kiyo-xhci-fix) project to monitor for USB controller crashes:

```
prompt: |
  You are a USB log monitor. Tail the kernel logs for USB-related
  events and use the speech MCP tool to announce any concerning
  USB events (errors, resets, disconnects, watchdog triggers,
  xhci failures).

  1. Announce: "USB log watcher online. Monitoring for anomalies."
  2. Run: journalctl -k -f --grep='usb|xhci|USB' --no-pager
  3. Check output every ~30 seconds
  4. Announce anomalies: "Warning: USB controller reset detected"
  5. Stay quiet on normal traffic
  6. Monitor indefinitely
```

### What It Caught (2026-03-21)

The agent ran for ~30 minutes during USB stress testing and caught a full xHCI controller death in real time:

1. **TRB warnings** — "WARN Event TRB for slot 16 ep 2 with no TDs queued?" (precursor, 13:17)
2. **Controller death** — "xHCI host controller not responding, assume dead" / "HC died; cleaning up" (13:30)
3. **Cascade disconnect** — 15 USB devices disconnected simultaneously across bus 1 and bus 2 (keyboard, mouse, Logitech receiver, Razer webcam, Bluetooth, audio, USB storage)

The agent's actual TTS announcement when the controller died:

> "Critical alert! The xHCI USB host controller has died. The kernel reports: host controller not responding, assume dead, cleaning up. All USB devices on both bus 1 and bus 2 have disconnected simultaneously. This includes keyboard, mouse, Logitech receiver, Razer webcam, USB storage, Bluetooth, and audio devices. Total of 15 device disconnects. You have lost all USB input. The earlier TRB warnings at 13:17 were a precursor to this failure."

It then continued polling for recovery events every 30 seconds.

## Reading Watcher Output

The main agent can check what the watcher reported:

```bash
# Read the agent's full transcript (JSONL)
cat /tmp/claude-*/tasks/<agent-id>.output

# Extract just the speak messages
grep -o '"text":"[^"]*"' /tmp/claude-*/tasks/<agent-id>.output | tail -10

# Or use the TaskOutput tool from within Claude Code
TaskOutput: agent-id
```

## Autonomous Response Loop

The powerful pattern: the main agent reads the watcher output and acts on it without human intervention.

```
┌─────────────┐    TTS alert     ┌──────┐
│  Background  │────────────────>│ User │ (hears it)
│  Watcher     │                 │      │
│              │    output file  │      │  "check the watcher"
│              │────────┐       └──┬───┘
└─────────────┘        │          │
                       v          v
              ┌─────────────────────┐
              │   Main Agent        │
              │                     │
              │  1. Reads output    │
              │  2. Investigates    │
              │  3. Fixes issue     │
              │  4. Deploys fix     │
              └─────────────────────┘
```

1. **Watcher detects anomaly** → speaks alert via TTS
2. **Main agent reads watcher output** → sees alert details
3. **Main agent investigates** → reads crash logs, checks dmesg
4. **Main agent fixes** → edits config, restarts service, deploys

The user hears the TTS alert and knows work is happening. No manual intervention needed.

**Caveat**: Background agents don't push notifications to the main agent — they only report back when they complete or are killed. The main agent must proactively read the output file (or the user can tell it to check).

## Adapting to Other Log Sources

| Use Case | Log Command | Watch For |
|----------|-------------|-----------|
| USB/xHCI | `journalctl -k -f --grep='usb\|xhci\|USB'` | `HC died`, `not responding`, `disconnect` |
| systemd service | `journalctl -u myservice -f` | `error`, `failed`, `timeout` |
| Nginx | `tail -f /var/log/nginx/error.log` | `502`, `upstream timed out` |
| Kubernetes | `kubectl logs -f deploy/myapp` | `OOMKilled`, `CrashLoopBackOff` |
| Docker | `docker logs -f container` | `ERROR`, `FATAL`, stack traces |
| Build system | `tail -f build.log` | `FAILED`, `error:`, non-zero exit |
| Network/firewall | `logread -f` (OpenWrt) | `DROP`, `REJECT`, zone violations |

## Tips

- **Be specific in the prompt** about what's anomalous vs. normal — reduces false positive announcements
- **Stay quiet on normal** — constant "all clear" announcements are annoying
- **Short announcements** — "USB controller reset on bus 2" not a full paragraph (though the agent will sometimes give detailed context on critical events, which is useful)
- **The agent needs ToolSearch first** — MCP tools are deferred, so the prompt should reference the tool name (`mcp__speech-to-cli__speak`) but the agent will fetch the schema itself
- **Background agents don't notify the main agent** until they complete — since watchers run indefinitely, the main agent must proactively read the output file
- **The user is the bridge** — they hear the TTS and can tell the main agent "check the watcher" until the main agent learns to check periodically
- **Poll interval tradeoff** — 30 seconds is a good default; shorter catches events faster but burns more LLM tokens on empty polls
- **Voice choice** — `en-US-DavisNeural` works well for alerts (clear, authoritative); consider `en-US-AvaNeural` for softer notifications
