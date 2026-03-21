# Background Log Watcher Pattern

Use a Claude Code background agent with the `speak` MCP tool to monitor system logs in real-time and announce anomalies aloud via TTS.

## Overview

A background Claude Code agent tails a log source (e.g., `journalctl -k -f`), pattern-matches for concerning events, and calls the `speak` tool to deliver audio alerts — all while the main conversation continues working normally.

The main agent can read the watcher's output file at any time to see what was detected and act on it autonomously.

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────┐
│  Main Claude Code   │────>│  Background Agent     │────>│ speech-to-  │
│  conversation       │     │  (log-watcher)        │     │ cli MCP     │
│                     │     │                       │     │ server      │
│  - continues work   │     │  - tails log source   │     │             │
│  - reads watcher    │<──┐ │  - pattern matches    │──┐  │  speak()    │
│    output on demand  │   │ │  - speaks alerts      │  │  └──────┬──────┘
└─────────────────────┘   │ └──────────────────────┘  │         │
                          │                            │         v
                          │   output file              │  ┌─────────────┐
                          └───(/tmp/claude-*/tasks/    │  │ Azure TTS   │
                               <agent-id>.output)      │  │ via speakers│
                                                       │  └─────────────┘
                                                       └── announces
                                                           anomalies only
```

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

## Example: USB/xHCI Monitor

Used during the kiyo-xhci-fix project to monitor for USB controller crashes:

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

What it caught in practice:
- TRB warnings on xHCI controller during stress testing
- Device resets on bus 1 (Logitech, keyboard, mouse)
- Full "HC died" controller death with cascade disconnect
- Successful recovery after xHCI rebind

## Reading Watcher Output

The main agent can check what the watcher reported:

```bash
# Extract speech messages from watcher output
grep -o '"text":"[^"]*"' /tmp/claude-*/tasks/<agent-id>.output | tail -10
```

## Autonomous Response Loop

The powerful pattern: the main agent reads the watcher output and acts on it without human intervention.

1. **Watcher detects anomaly** → speaks alert via TTS
2. **Main agent reads watcher output** → sees alert details
3. **Main agent investigates** → reads crash logs, checks dmesg
4. **Main agent fixes** → edits config, restarts service, deploys

The user hears the TTS alert and knows work is happening. No manual intervention needed.

## Adapting to Other Log Sources

| Use Case | Log Command | Watch For |
|----------|-------------|-----------|
| USB/xHCI | `journalctl -k -f` | `HC died`, `not responding`, `disconnect` |
| systemd service | `journalctl -u myservice -f` | `error`, `failed`, `timeout` |
| Nginx | `tail -f /var/log/nginx/error.log` | `502`, `upstream timed out` |
| Kubernetes | `kubectl logs -f deploy/myapp` | `OOMKilled`, `CrashLoopBackOff` |
| Docker | `docker logs -f container` | `ERROR`, `FATAL`, stack traces |
| Build system | `tail -f build.log` | `FAILED`, `error:`, non-zero exit |

## Tips

- **Be specific in the prompt** about what's anomalous vs. normal — reduces false positive announcements
- **Stay quiet on normal** — constant "all clear" announcements are annoying
- **Short announcements** — "USB controller reset on bus 2" not a full paragraph
- **Background agents don't notify the main agent** until they complete — since watchers run indefinitely, the main agent must proactively read the output file
- **The user is the bridge** — they hear the TTS and can tell the main agent "check the watcher" until the main agent learns to check periodically
