"""Shared mutable state for the Azure Speech MCP server.

Every module does `import state` and accesses reassignable globals as `state.X`.
Containers (CONFIG dict, _active_procs list) and Events/Locks can also be
imported by name since they are mutated in-place, not reassigned.
"""

import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import requests

try:
    import webrtcvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False

try:
    import websocket
    HAS_WS = True
except ImportError:
    HAS_WS = False

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
    _whisper_model = None
except ImportError:
    HAS_WHISPER = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULTS_PATH = os.path.expanduser("~/.config/speech-to-cli/config.json")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHIME_PATH = os.path.join(_SCRIPT_DIR, "chime_ready.wav")
CHIME_PROCESSING = os.path.join(_SCRIPT_DIR, "chime_processing.wav")
CHIME_SPEAK = os.path.join(_SCRIPT_DIR, "chime_speak.wav")
CHIME_DONE = os.path.join(_SCRIPT_DIR, "chime_done.wav")
CHIME_HUM = os.path.join(_SCRIPT_DIR, "chime_hum.wav")
CHIME_PAUSE = os.path.join(_SCRIPT_DIR, "chime_pause.wav")
CHIME_RESUME = os.path.join(_SCRIPT_DIR, "chime_resume.wav")

# Audio settings
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = SAMPLE_RATE * 2 * FRAME_MS // 1000  # 960 bytes per 30ms frame

# Echo cancellation (PipeWire)
EC_SOURCE = "echo_cancel_source"  # cleaned mic (AEC output)
EC_SINK = "echo_cancel_sink"      # TTS audio routes here so AEC can subtract it

# VAD + energy gate settings (these may be temporarily overridden by stt())
SILENCE_TIMEOUT = 3.0
NO_SPEECH_TIMEOUT = 7.0
MIN_SPEECH_DURATION = 0.15
VAD_AGGRESSIVENESS = 3
ENERGY_CALIBRATION_FRAMES = 5
ENERGY_THRESHOLD_MULTIPLIER = 2.5
NOISE_CACHE_TTL = 30.0

WS_IDLE_TIMEOUT = 540  # Azure closes at ~600s

_MAX_TTS_CHARS = 5000
_SSML_SAFE_RE = re.compile(r'^[a-zA-Z0-9\-_.:+%() ]+$')

# ---------------------------------------------------------------------------
# Mutable globals
# ---------------------------------------------------------------------------

_has_echo_cancel = None  # lazy-detected

_cached_noise_threshold = None
_cached_noise_time = 0.0
_http_session = None
_persistent_ws = None
_persistent_ws_time = 0.0
_hum_proc = None

# Cancellation support
_cancel_event = threading.Event()
_active_request_id = None
_active_procs = []
_active_procs_lock = threading.Lock()
_pause_event = threading.Event()

# Pre-warmed recorder
_prewarmed_rec = None
_prewarmed_rec_lock = threading.Lock()
_warmup_pending = False
_warmup_lock = threading.Lock()

# Pre-warmed player
_prewarmed_player = None
_prewarmed_player_rate = 0
_prewarmed_player_lock = threading.Lock()

# TTY / terminal width caching
_tty_fd = None
_cached_tty_width = None
_cached_tty_width_time = 0.0

# ISO timestamp caching
_cached_iso_ts = ""
_cached_iso_ts_time = 0.0


def _get_iso_timestamp():
    """Cached ISO timestamp, refreshed every 500ms. Avoids 1000+ strftime calls per recording."""
    import state
    now = time.time()
    if now - state._cached_iso_ts_time > 0.5:
        state._cached_iso_ts = time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime(now))
        state._cached_iso_ts_time = now
    return state._cached_iso_ts


# STT lock
_stt_lock = threading.Lock()

# TTS timing
_last_tts_end = 0.0

# Audio detection
_half_duplex_setting = None  # set after load_config()
_last_detected_sink_id = None
_auto_detect_lock = threading.Lock()

# Stdout / request queue
_stdout_lock = threading.Lock()
_request_queue = []
_request_cond = threading.Condition()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config():
    cfg = {}
    if os.path.exists(DEFAULTS_PATH):
        with open(DEFAULTS_PATH) as f:
            cfg = json.load(f)
    return {
        "key": os.environ.get("AZURE_SPEECH_KEY") or cfg.get("key"),
        "region": os.environ.get("AZURE_SPEECH_REGION") or cfg.get("region", "westus2"),
        "voice": os.environ.get("AZURE_SPEECH_VOICE") or cfg.get("voice", "en-US-Ava:DragonHDLatestNeural"),
        "fast_voice": cfg.get("fast_voice", "en-US-AvaNeural"),
        # Audio device settings
        "player": cfg.get("player", "auto"),
        "recorder": cfg.get("recorder", "auto"),
        "mic_source": cfg.get("mic_source", None),
        "speaker_sink": cfg.get("speaker_sink", None),
        "silence_timeout": cfg.get("silence_timeout", SILENCE_TIMEOUT),
        "talk_silence_timeout": cfg.get("talk_silence_timeout", 4.0),
        # UI settings
        "chime_ready": cfg.get("chime_ready", True),
        "chime_processing": cfg.get("chime_processing", False),
        "chime_speak": cfg.get("chime_speak", False),
        "chime_done": cfg.get("chime_done", False),
        "chime_hum": cfg.get("chime_hum", False),
        "visual_indicator": cfg.get("visual_indicator", True),
        "live_subtitles": cfg.get("live_subtitles", True),
        "subtitle_color_user": cfg.get("subtitle_color_user", "default"),
        "subtitle_color_tts": cfg.get("subtitle_color_tts", "default"),
        "vu_meter": cfg.get("vu_meter", True),
        "barge_in_frames": cfg.get("barge_in_frames", 3),
        "barge_in_silence": cfg.get("barge_in_silence", 1.0),
        "chime_barge_in": cfg.get("chime_barge_in", True),
        "enable_pause": cfg.get("enable_pause", True),
        "enable_echo_cancel": cfg.get("enable_echo_cancel", False),
        "enable_barge_in": cfg.get("enable_barge_in", False),
        "half_duplex": cfg.get("half_duplex", "auto"),
    }


CONFIG = load_config()
_half_duplex_setting = CONFIG.get("half_duplex", "auto")


# ---------------------------------------------------------------------------
# Small cross-cutting helpers
# ---------------------------------------------------------------------------

def is_cancelled():
    """Check if the current operation has been cancelled."""
    return _cancel_event.is_set()


def register_proc(proc):
    """Track a subprocess so it can be killed on cancellation."""
    with _active_procs_lock:
        _active_procs.append(proc)


def unregister_proc(proc):
    """Stop tracking a subprocess."""
    with _active_procs_lock:
        try:
            _active_procs.remove(proc)
        except ValueError:
            pass


def cancel_active():
    """Kill all tracked subprocesses and signal cancellation."""
    _cancel_event.set()
    _pause_event.clear()  # unpause first so loops can exit
    with _active_procs_lock:
        for proc in _active_procs:
            try:
                proc.terminate()
            except Exception:
                pass


def pause_active():
    """Pause playback by setting the pause event (data feed stops, no SIGSTOP)."""
    if not CONFIG.get("enable_pause", True):
        return
    _pause_event.set()


def resume_active():
    """Resume playback by clearing the pause event."""
    if not CONFIG.get("enable_pause", True):
        return
    _pause_event.clear()


def get_http_session():
    """Reuse HTTP session for connection pooling (saves ~150ms per TTS call)."""
    import state
    if state._http_session is None:
        state._http_session = requests.Session()
    return state._http_session


def send_progress(token, progress, total=None, description=None):
    """Send an MCP progress notification if a token is provided."""
    if token is None:
        return
    msg = {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {
            "progressToken": token,
            "progress": progress,
        }
    }
    if total is not None:
        msg["params"]["total"] = total
    if description:
        # Gemini CLI uses the 'message' field to show status text
        msg["params"]["message"] = description
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()
