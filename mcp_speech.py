#!/usr/bin/env python3
"""
Azure Speech MCP Server — compatible with Copilot CLI, Claude Code, and Gemini CLI.

Provides 'listen' (STT), 'speak' (TTS), 'talk' (full-duplex), and 'converse' tools
so AI assistants can have voice conversations.

Module structure:
  state.py       — Shared mutable state, constants, config, small helpers
  audio.py       — Audio I/O, device detection, chimes, UI, VAD, pre-warming
  stt.py         — Speech-to-text backends (streaming WS, VAD+REST, Whisper, fixed)
  speech_tts.py  — Text-to-speech (Azure TTS, multi_speak, talk_fullduplex)
  mcp_speech.py  — MCP protocol layer (this file): tool schemas, request routing, stdio
"""

import json
import os
import subprocess
import sys
import threading

import state
from state import (CONFIG, HAS_WS, DEFAULTS_PATH, _SCRIPT_DIR,
                   _cancel_event, cancel_active, pause_active, resume_active,
                   send_progress, _stdout_lock, _request_queue, _request_cond)
from audio import (_generate_chimes, _refresh_audio_detection, _prewarm_all,
                   _prewarm_recorder, _schedule_warmup,
                   has_echo_cancel, _COLOR_MAP,
                   _build_player_cmd, _build_rec_cmd)
from stt import stt, _get_stt_ws, _invalidate_stt_ws
from speech_tts import tts, multi_speak, multi_speak_stream, talk_fullduplex, get_voices


# ---------------------------------------------------------------------------
# MCP Tool schemas
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "listen",
        "description": (
            "Listen through the microphone and return what the user said as text. "
            "Stops automatically when the user finishes speaking. "
            "This is listen-only — if you also need to speak, use 'talk' instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Max recording duration in seconds (default 120).",
                    "default": 120,
                    "minimum": 1,
                    "maximum": 30,
                },
                "mode": {
                    "type": "string",
                    "description": "STT mode: 'streaming' (fastest), 'vad', 'whisper', 'fixed'.",
                    "enum": ["streaming", "vad", "whisper", "fixed"],
                },
                "silence_timeout": {
                    "type": "number",
                    "description": "Seconds of silence before stopping (default 0.8).",
                },
                "vad_aggressiveness": {
                    "type": "integer",
                    "description": "VAD level 0-3 (3 is most aggressive, default 3).",
                    "minimum": 0,
                    "maximum": 3,
                },
                "energy_multiplier": {
                    "type": "number",
                    "description": "Energy threshold multiplier for noise gating (default 2.5).",
                },
            },
        },
    },
    {
        "name": "speak",
        "description": (
            "Say something out loud (text-to-speech) WITHOUT listening for a reply. "
            "Use this for one-way announcements or final messages. "
            "If you want to speak AND hear the user's response, use 'talk' instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to speak aloud",
                },
                "quality": {
                    "type": "string",
                    "description": "Voice quality: 'fast' or 'hd'.",
                    "enum": ["fast", "hd"],
                    "default": "fast",
                },
                "voice": {
                    "type": "string",
                    "description": "Azure voice name (e.g. 'en-US-AvaNeural').",
                },
                "speed": {
                    "type": "number",
                    "description": "Playback speed multiplier (default 1.0).",
                    "default": 1.0,
                },
                "pitch": {
                    "type": "string",
                    "description": "Pitch: 'high', 'low', '+20%', or 'default'.",
                },
                "volume": {
                    "type": "string",
                    "description": "Volume: 'loud', 'soft', '+10%', or 'default'.",
                },
                "subtitle_color": {
                    "type": "string",
                    "description": "Override subtitle color for this speak call. Useful for giving each voice its own color in multi-agent conversations.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "multi_speak",
        "description": (
            "Speak multiple text segments with different voices in one call. "
            "Azure TTS requests are fired in parallel for speed, then audio plays back-to-back. "
            "Use this for multi-agent conversations to avoid multiple round trips."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to speak."},
                            "voice": {"type": "string", "description": "Azure voice name."},
                            "subtitle_color": {"type": "string", "description": "Subtitle color for this segment."},
                        },
                        "required": ["text"],
                    },
                    "description": "List of {text, voice, subtitle_color} segments to speak back-to-back.",
                },
                "quality": {
                    "type": "string",
                    "enum": ["fast", "hd"],
                    "default": "fast",
                },
            },
            "required": ["segments"],
        },
    },
    {
        "name": "multi_speak_stream",
        "description": (
            "Stream multi-voice TTS in a SINGLE Azure request using SSML multi-voice tags. "
            "MUCH FASTER than multi_speak — one API call instead of N calls. "
            "Ideal for multi_chat responses: each model's response plays with its own voice, "
            "switching seamlessly within a single audio stream. "
            "Voice assignments: gpt-5.3-chat→DavisNeural, Claude→AvaNeural, "
            "Llama→AndrewNeural, DeepSeek→BrianNeural, Phi→JennyNeural, Gemini→AriaNeural."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to speak."},
                            "voice": {"type": "string", "description": "Azure voice name (e.g., en-US-DavisNeural)."},
                        },
                        "required": ["text"],
                    },
                    "description": "List of {text, voice} segments. All synthesized in ONE request.",
                },
                "quality": {
                    "type": "string",
                    "enum": ["fast", "hd"],
                    "default": "fast",
                },
            },
            "required": ["segments"],
        },
    },
    {
        "name": "get_voices",
        "description": "List available Azure Speech voices for the current region.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "converse",
        "description": (
            "Listen to the user through their microphone and return what they said as text. "
            "Use this to START a voice conversation. After getting the user's words, respond "
            "with 'speak' (to say something) then 'converse' again (to listen for their reply). "
            "This speak\u2192converse loop lets you call other tools between turns \u2014 research, "
            "edit files, query APIs \u2014 then speak results and listen for the next instruction. "
            "Use 'talk' only when you need a fast atomic speak+listen with no tools in between."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Max recording duration in seconds (default 120).",
                    "default": 120,
                },
                "mode": {
                    "type": "string",
                    "description": "STT mode (streaming, vad, whisper, fixed).",
                },
                "silence_timeout": {
                    "type": "number",
                    "description": "Seconds of silence before stopping.",
                },
            },
        },
    },
    {
        "name": "configure",
        "description": (
            "View or change audio settings on the fly. Call with no arguments to see current settings "
            "(includes detected audio output device and echo cancel status). "
            "Pass any setting as a key-value pair to update it. Changes are saved to disk and take "
            "effect immediately. half_duplex defaults to 'auto' which detects headphones vs speakers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Azure Speech Services API key. Overrides AZURE_SPEECH_KEY env var."},
                "region": {"type": "string", "description": "Azure region for STT (e.g. westus2, eastus). Overrides AZURE_SPEECH_REGION env var."},
                "tts_region": {"type": "string", "description": "Azure region for TTS only (e.g. eastus for DragonHD voices). Falls back to main region if not set."},
                "tts_key": {"type": "string", "description": "Azure Speech API key for TTS region. Falls back to main key if not set."},
                "player": {"type": "string", "description": "Audio player: aplay, pw-play, pw-cat, ffplay, or auto."},
                "recorder": {"type": "string", "description": "Audio recorder: pw-record, arecord, or auto."},
                "mic_source": {"type": "string", "description": "PipeWire node name for mic input, or 'null' for default."},
                "speaker_sink": {"type": "string", "description": "PipeWire node name for speaker output, or 'null' for default."},
                "silence_timeout": {"type": "number", "description": "Silence timeout for listen tool (seconds)."},
                "talk_silence_timeout": {"type": "number", "description": "Silence timeout for talk tool (seconds)."},
                "max_record_seconds": {"type": "integer", "description": "Max recording duration after TTS finishes (default 120)."},
                "end_word": {"type": "string", "description": "Word that immediately stops recording when said at end of sentence (default 'over'). Set to empty string to disable."},
                "voice": {"type": "string", "description": "Azure TTS voice name."},
                "fast_voice": {"type": "string", "description": "Azure TTS voice name for fast quality."},
                "bt_profile": {"type": "string", "description": "Bluetooth profile: a2dp (output only, hi-fi) or hfp (mic+speaker, lower quality)."},
                "half_duplex": {"type": "string", "description": "Controls echo avoidance. 'auto' (default): auto-detects headphones vs speakers. 'true': force half-duplex (speakers). 'false': force full-duplex (headphones/earbuds)."},
                "chime_ready": {"type": "boolean", "description": "Play chime when mic starts listening."},
                "chime_processing": {"type": "boolean", "description": "Play chime when processing starts."},
                "chime_speak": {"type": "boolean", "description": "Play chime before TTS speaks."},
                "chime_done": {"type": "boolean", "description": "Play chime when done."},
                "chime_hum": {"type": "boolean", "description": "Play ambient hum while waiting."},
                "chime_barge_in": {"type": "boolean", "description": "Play chime on barge-in detection."},
                "visual_indicator": {"type": "boolean", "description": "Show visual indicators in terminal."},
                "live_subtitles": {"type": "boolean", "description": "Show live subtitles during TTS."},
                "subtitle_color_user": {"type": "string", "description": "Color for user's transcribed speech subtitles. Options: default, green, light_green, yellow, amber, red, blue, cyan, magenta, white, gray, light_red, light_blue, light_cyan, light_magenta."},
                "subtitle_color_tts": {"type": "string", "description": "Color for agent's TTS speech subtitles. Options: default, green, light_green, yellow, amber, red, blue, cyan, magenta, white, gray, light_red, light_blue, light_cyan, light_magenta."},
                "vu_meter": {"type": "boolean", "description": "Show VU meter animation during audio."},
                "enable_pause": {"type": "boolean", "description": "Allow pausing/resuming playback and recording."},
                "enable_echo_cancel": {"type": "boolean", "description": "[Experimental] Use PipeWire echo cancellation nodes if available (default true)."},
                "enable_barge_in": {"type": "boolean", "description": "[Experimental] Allow user speech to pause TTS (barge-in, default false)."},
                "no_speech_timeout": {"type": "number", "description": "Seconds to wait for any speech before giving up (default 7)."},
                "energy_multiplier": {"type": "number", "description": "Energy threshold multiplier for noise gating (default 2.5). Lower = more sensitive."},
                "barge_in_frames": {"type": "integer", "description": "[Experimental] Speech frames needed to trigger barge-in (default 3)."},
                "barge_in_silence": {"type": "number", "description": "[Experimental] Silence seconds to resume TTS after barge-in (default 1.0)."},
                "debug": {"type": "boolean", "description": "Write debug logs to /tmp/speech-debug.log."},
            },
        },
    },
    {
        "name": "talk",
        "description": (
            "Say something out loud and listen for the user's reply. This is the PRIMARY tool for voice conversations. "
            "CONVERSATION RULES: "
            "(1) ALWAYS call 'talk' again after getting a result \u2014 NEVER drop to text mid-conversation. "
            "(2) If '(no speech detected)', call 'talk' again with a short prompt \u2014 do NOT give up. "
            "(3) Keep messages short and conversational (1-3 sentences). Long messages delay the user's reply. "
            "(4) Treat this like a phone call \u2014 the user is talking to a mic. Half-duplex auto-detects speakers vs headphones. "
            "(5) Only use 'speak' for a true final goodbye. "
            "(6) When the user types 'talk' in chat, start the voice conversation immediately. "
            "TOKEN ECONOMY: "
            "(A) Keep 'text' under 2 sentences \u2014 speech is slower than reading, so less is more. "
            "(B) Never echo back what the user said \u2014 they already know. "
            "(C) Do NOT output any text to the chat between talk calls \u2014 the user is listening via audio. "
            "(D) Skip preamble like 'Sure!' or 'Great question!' \u2014 just answer directly. "
            "TYPED INPUT: If this call is cancelled and the user typed a message, respond to their text normally \u2014 do not call talk again unless they ask. "
            "SWITCHING MODES: If the user asks you to do something that requires calling other tools (search, edit, "
            "query APIs, read files), switch to the speak\u2192converse pattern: call 'speak' with your response, do your "
            "tool calls, then call 'speak' with results and 'converse' to listen. Switch back to 'talk' for quick "
            "back-and-forth that doesn't need tool calls in between. "
            "DYNAMIC TUNING: "
            "(1) For yes/no questions, pass silence_timeout=2. For open-ended questions, pass silence_timeout=6-8. "
            "(2) If you get '(no speech detected)' twice, call configure(talk_silence_timeout=N) to increase the default. "
            "(3) You can toggle chimes mid-conversation: configure(chime_ready=true/false). "
            "(4) Watch for [AGENT HINTS] in responses \u2014 they suggest configuration changes based on the user's behavior."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to speak aloud before listening.",
                },
                "quality": {
                    "type": "string",
                    "description": "Voice quality: 'fast' or 'hd'.",
                    "enum": ["fast", "hd"],
                    "default": "fast",
                },
                "speed": {
                    "type": "number",
                    "description": "Playback speed multiplier (default 1.0).",
                    "default": 1.0,
                },
                "voice": {
                    "type": "string",
                    "description": "Azure voice name.",
                },
                "pitch": {
                    "type": "string",
                    "description": "Pitch: 'high', 'low', '+20%', or 'default'.",
                },
                "volume": {
                    "type": "string",
                    "description": "Volume: 'loud', 'soft', '+10%', or 'default'.",
                },
                "seconds": {
                    "type": "integer",
                    "description": "Max recording duration in seconds (default 120).",
                    "default": 120,
                },
                "mode": {
                    "type": "string",
                    "description": "STT mode (streaming, vad, whisper, fixed).",
                },
                "silence_timeout": {
                    "type": "number",
                    "description": "Seconds of silence before stopping.",
                },
                "subtitle_color": {
                    "type": "string",
                    "description": "Override subtitle color for TTS portion. Useful for giving each voice its own color.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "pause",
        "description": (
            "Pause whatever is currently happening \u2014 freezes audio playback or recording in place. "
            "Call 'resume' to pick up exactly where it left off. "
            "Use this when the user says 'pause', 'hold on', 'wait', or 'stop for a sec'."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "resume",
        "description": (
            "Resume after a pause \u2014 unfreezes audio playback or recording from where it stopped. "
            "Call this when the user says 'resume', 'continue', 'go ahead', or 'unpause'."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

def handle_request(req):
    method = req.get("method")
    params = req.get("params", {})
    req_id = req.get("id")
    progress_token = params.get("_meta", {}).get("progressToken")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": True}},
                "serverInfo": {"name": "azure-speech", "version": "4.2.0"},
            },
        }
    elif method == "notifications/initialized":
        return None
    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})
        # Auto-refresh audio detection synchronously (~12ms, ~35ms on change)
        if tool_name in ("listen", "converse", "speak", "talk"):
            _refresh_audio_detection()
        if tool_name in ("listen", "converse"):
            result = stt(
                seconds=args.get("seconds"),
                mode=args.get("mode"),
                silence_timeout=args.get("silence_timeout"),
                vad_aggressiveness=args.get("vad_aggressiveness"),
                energy_multiplier=args.get("energy_multiplier"),
                progress_token=progress_token
            )
            if result.get("cancelled"):
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": "(cancelled \u2014 if the user typed a message, respond to that instead of calling talk)"}]},
                }
            text = result.get("text", result.get("error", ""))
            content_text = text or "(no speech detected)"
            _schedule_warmup()

            # --- Dynamic agent hints for listen/converse ---
            listen_hints = []
            if not text:
                state._consecutive_no_speech += 1
                if state._consecutive_no_speech >= 2:
                    cur_st = args.get("silence_timeout") or CONFIG.get("silence_timeout", 3.0)
                    listen_hints.append(f"No speech {state._consecutive_no_speech}x in a row. "
                                        f"Consider passing silence_timeout={min(cur_st + 2, 10):.0f} on your next call.")
            else:
                state._consecutive_no_speech = 0

            if tool_name == "converse":
                content_text += "\n\n[Voice conversation active \u2014 call 'speak' to reply, then call 'converse' to listen again. You may call other tools (search, edit, query) between speak and converse. Keep spoken replies short (1-3 sentences). For quick back-and-forth with no tools needed, switch to 'talk'. Use 'speak' for a final goodbye with no reply needed.]"
            if listen_hints:
                content_text += "\n\n[AGENT HINTS: " + " ".join(listen_hints) + "]"
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": content_text}]},
            }
        elif tool_name == "multi_speak":
            segments = args.get("segments", [])
            if not segments:
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": "Error: 'segments' is required."}]},
                }
            quality = args.get("quality", "fast")
            if quality not in ("fast", "hd"):
                quality = "fast"
            result = multi_speak(segments, quality=quality, progress_token=progress_token)
            if result.get("cancelled"):
                msg = "(cancelled)"
            else:
                count = result.get("spoken", 0)
                msg = f"Spoke {count} segment{'s' if count != 1 else ''} aloud." if count else result.get("error", "Failed")
            _schedule_warmup()
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": msg}]},
            }
        elif tool_name == "multi_speak_stream":
            segments = args.get("segments", [])
            if not segments:
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": "Error: 'segments' is required."}]},
                }
            quality = args.get("quality", "fast")
            if quality not in ("fast", "hd"):
                quality = "fast"
            result = multi_speak_stream(segments, quality=quality, progress_token=progress_token)
            if result.get("cancelled"):
                msg = "(cancelled)"
            elif result.get("error"):
                msg = result["error"]
            else:
                count = result.get("spoken", 0)
                msg = f"Streamed {count} voice{'s' if count != 1 else ''} in single request."
            _schedule_warmup()
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": msg}]},
            }
        elif tool_name == "speak":
            speak_text = args.get("text", "")
            if not speak_text or not isinstance(speak_text, str):
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": "Error: 'text' is required."}]},
                }
            quality = args.get("quality", "fast")
            if quality not in ("fast", "hd"):
                quality = "fast"
            speed_val = args.get("speed", 1.0)
            if not isinstance(speed_val, (int, float)):
                speed_val = 1.0
            speed_val = max(0.5, min(float(speed_val), 3.0))
            result = tts(
                speak_text,
                quality=quality,
                voice=args.get("voice"),
                speed=speed_val,
                pitch=args.get("pitch", "default"),
                volume=args.get("volume", "default"),
                progress_token=progress_token,
                subtitle_color=args.get("subtitle_color"),
            )
            if result.get("cancelled"):
                msg = "(cancelled)"
            else:
                msg = "Spoke the text aloud." if result.get("spoken") else result.get("error", "Failed")
                _schedule_warmup()
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": msg}]},
            }
        elif tool_name == "talk":
            speak_text = args.get("text", "")
            if not speak_text or not isinstance(speak_text, str):
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": "Error: 'text' is required."}]},
                }
            quality = args.get("quality", "fast")
            if quality not in ("fast", "hd"):
                quality = "fast"
            speed_val = args.get("speed", 1.0)
            if not isinstance(speed_val, (int, float)):
                speed_val = 1.0
            speed_val = max(0.5, min(float(speed_val), 3.0))

            # Use half-duplex (speak then listen) when configured
            if not CONFIG.get("half_duplex", False):
                result = talk_fullduplex(
                    speak_text,
                    quality=quality,
                    speed=speed_val,
                    voice=args.get("voice"),
                    pitch=args.get("pitch", "default"),
                    volume=args.get("volume", "default"),
                    seconds=args.get("seconds"),
                    mode=args.get("mode"),
                    silence_timeout=args.get("silence_timeout"),
                    progress_token=progress_token,
                    subtitle_color=args.get("subtitle_color"),
                )
                if result.get("cancelled"):
                    return {
                        "jsonrpc": "2.0", "id": req_id,
                        "result": {"content": [{"type": "text", "text": "(cancelled \u2014 if the user typed a message, respond to that instead of calling talk)"}]},
                    }
                user_said = result.get("text", "")
                content_text = user_said or "(no speech detected)"
                _audio_hint = "on earbuds" if CONFIG.get("_detected_output") == "headphones" else "on speakers"

                # --- Dynamic agent hints ---
                hints = []
                cur_silence = args.get("silence_timeout") or CONFIG.get("talk_silence_timeout", 4.0)
                if user_said:
                    state._consecutive_no_speech = 0
                    word_count = len(user_said.split())
                    if word_count <= 4:
                        state._consecutive_short_response += 1
                        if state._consecutive_short_response >= 2:
                            hints.append(f"User gave {state._consecutive_short_response} short replies in a row. "
                                         f"They may be getting cut off. Consider passing silence_timeout={min(cur_silence + 2, 10):.0f} on your next call.")
                    else:
                        state._consecutive_short_response = 0
                    content_text += f"\n\n[Call 'talk' now. Keep reply under 2 sentences. No chat text \u2014 user is {_audio_hint}. If you need to call tools first, switch to speak\u2192converse pattern.]"
                else:
                    state._consecutive_no_speech += 1
                    if state._consecutive_no_speech >= 2:
                        hints.append(f"No speech {state._consecutive_no_speech}x in a row. "
                                     f"Consider: configure(talk_silence_timeout={min(cur_silence + 2, 10):.0f}) to give the user more time, "
                                     f"or ask if they're still there.")
                    content_text += "\n\n[No speech \u2014 call 'talk' with a short check-in. Don't drop to text.]"

                if hints:
                    content_text += "\n\n[AGENT HINTS: " + " ".join(hints) + "]"

                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": content_text}]},
                }

            # Fallback: sequential speak then listen
            tts_result = tts(
                speak_text,
                quality=quality,
                voice=args.get("voice"),
                speed=speed_val,
                pitch=args.get("pitch", "default"),
                volume=args.get("volume", "default"),
                progress_token=progress_token,
                subtitle_color=args.get("subtitle_color"),
            )
            if tts_result.get("cancelled"):
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": "(cancelled \u2014 if the user typed a message, respond to that instead of calling talk)"}]},
                }
            if not tts_result.get("spoken"):
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": tts_result.get("error", "TTS failed")}]},
                }
            _prewarm_recorder()
            try:
                if HAS_WS:
                    _get_stt_ws()
            except Exception:
                pass
            stt_result = stt(
                seconds=args.get("seconds"),
                mode=args.get("mode"),
                silence_timeout=args.get("silence_timeout"),
                progress_token=progress_token,
            )
            if isinstance(stt_result, dict) and stt_result.get("cancelled"):
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": "(cancelled \u2014 if the user typed a message, respond to that instead of calling talk)"}]},
                }
            user_said = stt_result.get("text", stt_result.get("error", "")) if isinstance(stt_result, dict) else str(stt_result)
            content_text = user_said or "(no speech detected)"
            _schedule_warmup()
            _audio_hint2 = "on earbuds" if CONFIG.get("_detected_output") == "headphones" else "on speakers"

            # --- Dynamic agent hints (half-duplex path) ---
            hd_hints = []
            hd_cur_silence = args.get("silence_timeout") or CONFIG.get("talk_silence_timeout", 4.0)
            if user_said:
                state._consecutive_no_speech = 0
                word_count = len(user_said.split())
                if word_count <= 4:
                    state._consecutive_short_response += 1
                    if state._consecutive_short_response >= 2:
                        hd_hints.append(f"User gave {state._consecutive_short_response} short replies. "
                                        f"Consider silence_timeout={min(hd_cur_silence + 2, 10):.0f}.")
                else:
                    state._consecutive_short_response = 0
                content_text += f"\n\n[RESPOND NOW: call 'talk' with a short spoken reply. Do NOT type text to the user \u2014 they are {_audio_hint2}. Keep it conversational, 1-3 sentences. If you need to call tools first, switch to speak\u2192converse pattern.]"
            else:
                state._consecutive_no_speech += 1
                if state._consecutive_no_speech >= 2:
                    hd_hints.append(f"No speech {state._consecutive_no_speech}x. "
                                    f"Consider: configure(talk_silence_timeout={min(hd_cur_silence + 2, 10):.0f}).")
                content_text += "\n\n[No speech detected \u2014 the user may still be there. Call 'talk' again with a brief check-in like 'Hey, are you still there?' Do NOT drop to text.]"

            if hd_hints:
                content_text += "\n\n[AGENT HINTS: " + " ".join(hd_hints) + "]"
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": content_text}]},
            }
        elif tool_name == "configure":
            # Handle Bluetooth profile switching
            bt_profile = args.pop("bt_profile", None)
            bt_msg = ""
            if bt_profile:
                try:
                    dev_id = None
                    wp_out = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=3)
                    for obj in json.loads(wp_out.stdout):
                        props = obj.get("info", {}).get("props", {})
                        if "Galaxy" in str(props.get("device.description", "")):
                            dev_id = obj.get("id")
                            profiles = {p["name"]: p["index"]
                                        for p in obj.get("info", {}).get("params", {}).get("EnumProfile", [])}
                            break
                    if dev_id and profiles:
                        if bt_profile == "hfp":
                            idx = profiles.get("headset-head-unit-msbc") or profiles.get("headset-head-unit")
                        else:
                            idx = profiles.get("a2dp-sink-sbc_xq") or profiles.get("a2dp-sink")
                        if idx is not None:
                            subprocess.run(["wpctl", "set-profile", str(dev_id), str(idx)], timeout=5)
                            bt_msg = f"Bluetooth profile \u2192 {bt_profile} (index {idx}). "
                        else:
                            bt_msg = f"Profile '{bt_profile}' not found. "
                    else:
                        bt_msg = "No Bluetooth audio device found. "
                except Exception as e:
                    bt_msg = f"Bluetooth profile switch failed: {e}. "

            # Update config settings
            settable = {"key", "region", "tts_region", "tts_key",
                        "player", "recorder", "mic_source", "speaker_sink",
                        "silence_timeout", "talk_silence_timeout", "no_speech_timeout",
                        "max_record_seconds", "energy_multiplier", "end_word", "voice", "fast_voice",
                        "half_duplex", "chime_ready", "chime_processing", "chime_speak",
                        "chime_done", "chime_hum", "chime_barge_in", "visual_indicator",
                        "live_subtitles", "subtitle_color_user", "subtitle_color_tts",
                        "vu_meter", "enable_pause",
                        "enable_echo_cancel", "enable_barge_in",
                        "barge_in_frames", "barge_in_silence", "debug"}
            updated = []
            for k, v in args.items():
                if k in settable:
                    if v == "null" or v is None:
                        CONFIG[k] = None
                    elif k in ("key", "region", "tts_region", "tts_key"):
                        CONFIG[k] = str(v) if v else None
                        state._http_session = None
                        if k in ("key", "region"):
                            _invalidate_stt_ws()
                    elif k == "end_word":
                        CONFIG[k] = str(v).strip().lower() if v else ""
                    elif k == "half_duplex":
                        if isinstance(v, str) and v.lower() == "auto":
                            state._half_duplex_setting = "auto"
                            state._last_detected_sink_id = None
                            _refresh_audio_detection()
                        else:
                            state._half_duplex_setting = v
                            CONFIG[k] = v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
                    elif k in ("subtitle_color_user", "subtitle_color_tts"):
                        color_val = str(v).lower().strip()
                        if color_val not in _COLOR_MAP:
                            continue  # Ignore invalid color names
                        CONFIG[k] = color_val
                    elif k in ("chime_ready", "chime_processing", "chime_speak",
                              "chime_done", "chime_hum", "chime_barge_in", "visual_indicator",
                              "live_subtitles", "vu_meter", "enable_pause",
                              "enable_echo_cancel", "enable_barge_in", "debug"):
                        CONFIG[k] = v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
                        if k == "chime_hum" and not CONFIG[k]:
                            from audio import stop_hum
                            stop_hum()  # Kill running hum immediately
                        if k == "enable_echo_cancel":
                            state._has_echo_cancel = None  # force re-detection
                    elif k in ("silence_timeout", "talk_silence_timeout", "barge_in_silence"):
                        CONFIG[k] = max(0.1, min(float(v), 10.0))
                    elif k == "no_speech_timeout":
                        val = max(1.0, min(float(v), 30.0))
                        state.NO_SPEECH_TIMEOUT = val
                        CONFIG[k] = val
                    elif k == "energy_multiplier":
                        val = max(0.5, min(float(v), 20.0))
                        state.ENERGY_THRESHOLD_MULTIPLIER = val
                        CONFIG[k] = val
                    elif k == "max_record_seconds":
                        CONFIG[k] = max(5, min(int(v), 300))
                    elif k == "barge_in_frames":
                        CONFIG[k] = max(1, min(int(v), 20))
                    else:
                        CONFIG[k] = v
                    updated.append(f"{k}={CONFIG[k]}")

            # Save to disk
            if updated:
                cfg_path = DEFAULTS_PATH
                os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
                disk_cfg = {}
                if os.path.exists(cfg_path):
                    with open(cfg_path) as f:
                        disk_cfg = json.load(f)
                for k, v in args.items():
                    if k in settable:
                        if k == "half_duplex" and isinstance(v, str) and v.lower() == "auto":
                            disk_cfg[k] = "auto"
                        else:
                            disk_cfg[k] = CONFIG[k]
                with open(cfg_path, "w") as f:
                    json.dump(disk_cfg, f, indent=4)

            # Build response
            if updated or bt_msg:
                text = bt_msg + ("Updated: " + ", ".join(updated) if updated else "")
            else:
                key_display = "***" + CONFIG.get("key", "")[-4:] if CONFIG.get("key") else "NOT SET"
                sections = [
                    ("Credentials", []),
                    ("Audio", ["player", "recorder", "mic_source", "speaker_sink"]),
                    ("Voice", ["voice", "fast_voice"]),
                    ("Timing", ["silence_timeout", "talk_silence_timeout", "no_speech_timeout",
                               "max_record_seconds", "energy_multiplier", "end_word"]),
                    ("Mode", ["half_duplex", "enable_pause"]),
                    ("Experimental", ["enable_echo_cancel", "enable_barge_in",
                                      "barge_in_frames", "barge_in_silence", "debug"]),
                    ("Chimes", ["chime_ready", "chime_processing", "chime_speak",
                                "chime_done", "chime_hum", "chime_barge_in"]),
                    ("UI", ["visual_indicator", "live_subtitles", "subtitle_color_user",
                            "subtitle_color_tts", "vu_meter"]),
                ]
                lines = []
                for label, keys in sections:
                    lines.append(f"[{label}]")
                    if label == "Credentials":
                        lines.append(f"  key: {key_display}")
                        lines.append(f"  region: {CONFIG.get('region')}")
                        continue
                    for k in keys:
                        lines.append(f"  {k}: {CONFIG.get(k)}")
                # Resolve effective player/recorder
                _player = CONFIG.get("player", "auto")
                _recorder = CONFIG.get("recorder", "auto")
                if _player == "auto":
                    _eff_player = _build_player_cmd(24000)[0][0] if _build_player_cmd(24000) else "unknown"
                else:
                    _eff_player = _player
                if _recorder == "auto":
                    _eff_recorder = _build_rec_cmd()[0] if _build_rec_cmd() else "unknown"
                else:
                    _eff_recorder = _recorder
                det_type = CONFIG.get("_detected_output", "unknown")
                det_info = CONFIG.get("_detected_output_info", {})
                det_desc = det_info.get("description", "unknown")
                lines.append(f"[Detected]")
                lines.append(f"  player_binary: {_eff_player}")
                lines.append(f"  recorder_binary: {_eff_recorder}")
                lines.append(f"  output_device: {det_desc} ({det_type})")
                lines.append(f"  echo_cancel: {has_echo_cancel()}")
                text = "Current settings:\n" + "\n".join(lines)
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": text.strip()}]},
            }

        elif tool_name == "get_voices":
            voices = get_voices()
            if isinstance(voices, dict) and "error" in voices:
                text = voices["error"]
            else:
                lines = [f"{v['ShortName']} ({v['Gender']}, {v['LocaleName']})" for v in voices[:50]]
                text = "Available voices (first 50):\n" + "\n".join(lines)
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": text}]},
            }
        else:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }
    elif method == "resources/list":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "resources": [
                    {
                        "uri": "speech://readme",
                        "name": "Speech-to-CLI Documentation",
                        "description": "The complete README and documentation for the speech-to-cli MCP server, including features, usage, and configuration.",
                        "mimeType": "text/markdown"
                    },
                    {
                        "uri": "speech://config-schema",
                        "name": "Configuration Schema",
                        "description": "The current configuration settings loaded by the MCP server.",
                        "mimeType": "application/json"
                    }
                ]
            }
        }
    elif method == "resources/read":
        uri = params.get("uri")
        if uri == "speech://readme":
            try:
                readme_path = os.path.join(_SCRIPT_DIR, "README.md")
                with open(readme_path, "r") as f:
                    content = f.read()
            except Exception as e:
                content = f"Error reading README: {e}"
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "text/markdown",
                            "text": content
                        }
                    ]
                }
            }
        elif uri == "speech://config-schema":
            safe_config = {k: ("***" if k == "key" else v) for k, v in CONFIG.items()}
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(safe_config, indent=2)
                        }
                    ]
                }
            }
        else:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32602, "message": f"Resource not found: {uri}"}
            }
    elif method in ("resources/templates/list", "prompts/list", "completion/complete"):
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    elif method == "notifications/cancelled":
        cancel_active()
        return None
    else:
        if req_id is not None:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }
        return None


# ---------------------------------------------------------------------------
# Stdio transport
# ---------------------------------------------------------------------------

def _write_response(resp):
    """Thread-safe write to stdout."""
    with _stdout_lock:
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


def _stdin_reader():
    """Read stdin in a background thread, routing urgent requests immediately."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = req.get("method")
        req_id = req.get("id")

        if method == "notifications/cancelled":
            cancel_active()
            continue

        if method == "tools/call":
            tool_name = req.get("params", {}).get("name")
            if tool_name == "pause":
                pause_active()
                _write_response({
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": "Paused. Call 'resume' to continue."}]},
                })
                continue
            elif tool_name == "resume":
                resume_active()
                _write_response({
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": "Resumed."}]},
                })
                continue

        with _request_cond:
            _request_queue.append(req)
            _request_cond.notify()

    with _request_cond:
        _request_queue.append(None)
        _request_cond.notify()


def main():
    # Validate Azure credentials at startup (warn but don't exit — some tools work without Azure)
    if not CONFIG.get("key") or not CONFIG.get("region"):
        print("ERROR: AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set. "
              "See README.md for setup.", file=sys.stderr)

    # Startup side effects (run once, not on import)
    _generate_chimes()
    _refresh_audio_detection()
    _schedule_warmup()

    reader = threading.Thread(target=_stdin_reader, daemon=True)
    reader.start()

    while True:
        with _request_cond:
            while not _request_queue:
                _request_cond.wait()
            req = _request_queue.pop(0)

        if req is None:
            break  # EOF

        _cancel_event.clear()
        state._active_request_id = req.get("id")

        resp = handle_request(req)
        if resp is not None:
            _write_response(resp)

        state._active_request_id = None


if __name__ == "__main__":
    main()
