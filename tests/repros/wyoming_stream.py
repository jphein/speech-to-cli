"""Repro for #19: offline TTS must start playing before audio-stop.

Usage: python3 tests/repros/wyoming_stream.py [/path/to/speech-to-cli]
(default: this checkout). Exit 0 = all checks pass. Fails on main before the
fix: the first PCM write lands AFTER audio-stop and _mark_tts_end() is never
called. Fake Wyoming server on loopback, fake player, no audio, no network,
nothing written to the user config (state dir is per-PID and removed).
"""
import io
import json
import os
import socket
import sys
import threading
import time

SVC = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SVC)
os.environ.setdefault("XDG_STATE_HOME", os.path.join(os.path.dirname(__file__), "state-%d" % os.getpid()))

import state  # noqa: E402
import wyoming  # noqa: E402
import speech_tts  # noqa: E402

CHUNK_GAP = 0.15
N_CHUNKS = 6
CHUNK = b"\x01\x02" * 2000  # 4000 B


def _send(sock, etype, data=None, payload=b""):
    hdr = {"type": etype, "data": data or {}}
    if payload:
        hdr["payload_length"] = len(payload)
    sock.sendall(json.dumps(hdr).encode() + b"\n")
    if payload:
        sock.sendall(payload)


class FakeServer:
    def __init__(self, gap=CHUNK_GAP, n=N_CHUNKS, stall_before_stop=0.0, no_chunks=False):
        self.gap, self.n, self.stall, self.no_chunks = gap, n, stall_before_stop, no_chunks
        self.stop_at = None
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        conn, _ = self.srv.accept()
        f = conn.makefile("rb")
        f.readline()  # synthesize
        _send(conn, "audio-start", {"rate": 22050, "width": 2, "channels": 1})
        if not self.no_chunks:
            for _ in range(self.n):
                time.sleep(self.gap)
                _send(conn, "audio-chunk", {"rate": 22050, "width": 2, "channels": 1}, CHUNK)
        if self.stall:
            time.sleep(self.stall)
        self.stop_at = time.monotonic()
        _send(conn, "audio-stop", {})
        try:
            conn.close()
        except OSError:
            pass


class FakeStdin(io.RawIOBase):
    def __init__(self, delay=0.0):
        self.writes = []
        self.delay = delay
        self.closed_at = None

    def writable(self):
        return True

    def write(self, b):
        if self.delay:
            time.sleep(self.delay)
        self.writes.append((time.monotonic(), bytes(b)))
        return len(b)

    def flush(self):
        pass

    def close(self):
        self.closed_at = time.monotonic()
        super().close()


class FakeProc:
    def __init__(self, delay=0.0):
        self.stdin = FakeStdin(delay)
        self.terminated = False

    def poll(self):
        return 0 if self.stdin.closed else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def patch_player(proc):
    speech_tts._take_prewarmed_player = lambda rate: None
    speech_tts._start_player = lambda rate, target=None: proc
    speech_tts.register_proc = lambda p: None
    speech_tts.play_speak = lambda: None
    speech_tts.send_progress = lambda *a, **k: None
    speech_tts.is_cancelled = lambda: False


# --- 1. playback starts before audio-stop ------------------------------------
srv = FakeServer()
state.CONFIG["wyoming_host"] = "127.0.0.1"
state.CONFIG["wyoming_tts_port"] = srv.port
state.CONFIG["wyoming_tts_voice"] = ""
state._last_tts_end = 0.0
proc = FakeProc()
patch_player(proc)
t0 = time.monotonic()
res = speech_tts._tts_wyoming("hello there", None)
check(res == {"ok": True, "engine": "wyoming"}, f"result ok: {res}")
pcm_writes = [t for t, b in proc.stdin.writes if b != b"\x00" * len(b)]
check(len(pcm_writes) == N_CHUNKS, f"{len(pcm_writes)} PCM writes reached the player (expect {N_CHUNKS})")
first = pcm_writes[0] - t0 if pcm_writes else None
stop = (srv.stop_at - t0) if srv.stop_at else None
check(first is not None and srv.stop_at is not None and pcm_writes[0] < srv.stop_at,
      f"first PCM write at {first and round(first*1000)} ms, audio-stop at {stop and round(stop*1000)} ms -- playback began before synthesis finished")
check(first is not None and first < CHUNK_GAP * 2.5, "first write landed within ~2 chunk gaps of the request")
check(state._last_tts_end != 0.0, "_mark_tts_end() was called on the Wyoming path")
check(proc.stdin.writes and proc.stdin.writes[0][1] == b"\x00" * (22050 * 2 * 200 // 1000),
      "cold-device 200 ms lead-in written once (first offline utterance)")

# --- 2. second utterance within 10 s: no lead-in ---------------------------
srv = FakeServer(n=2)
state.CONFIG["wyoming_tts_port"] = srv.port
proc = FakeProc()
patch_player(proc)
speech_tts._tts_wyoming("again", None)
zeros = [b for t, b in proc.stdin.writes if b and b == b"\x00" * len(b)]
check(not zeros, "warm device: no 200 ms zero lead-in on the next offline utterance")

# --- 3. output_file WAV contains every byte ----------------------------------
srv = FakeServer(n=3)
state.CONFIG["wyoming_tts_port"] = srv.port
proc = FakeProc()
patch_player(proc)
out = os.path.join(os.environ["XDG_STATE_HOME"], "out.wav")
os.makedirs(os.path.dirname(out), exist_ok=True)
speech_tts._tts_wyoming("save me", None, output_file=out)
data = open(out, "rb").read()
check(data[:4] == b"RIFF" and data[44:] == CHUNK * 3, f"WAV written with all {len(CHUNK)*3} PCM bytes")

# --- 4. slow consumer does not trip the deadline -----------------------------
srv = FakeServer(gap=0.0, n=8)
gen = wyoming.synthesize_stream("127.0.0.1", srv.port, "slow", timeout=0.5)
fmt = next(gen)
got = 0
try:
    for chunk in gen:
        time.sleep(0.12)  # 8 x 0.12 = 0.96 s of consumer time > 0.5 s timeout
        got += len(chunk)
    check(got == len(CHUNK) * 8, "consumer slower than timeout: stream completed, no false 'timed out'")
except wyoming.WyomingError as e:
    check(False, f"consumer slower than timeout raised: {e}")

# --- 5. a stalled server still times out ------------------------------------
srv = FakeServer(gap=0.0, n=1, stall_before_stop=1.2)
gen = wyoming.synthesize_stream("127.0.0.1", srv.port, "stall", timeout=0.5)
next(gen)
try:
    list(gen)
    check(False, "stalled server: expected WyomingError")
except wyoming.WyomingError as e:
    check(True, f"stalled server raised WyomingError: {e}")

# --- 6. empty stream -> WyomingError, and _tts_wyoming returns None (fallback)
srv = FakeServer(no_chunks=True)
state.CONFIG["wyoming_tts_port"] = srv.port
proc = FakeProc()
patch_player(proc)
res = speech_tts._tts_wyoming("empty", None)
check(res is None, f"no audio at all -> None so the caller may fall back to Azure: {res}")

# --- 7. collect-all wrapper unchanged for tts.py / speak.py -----------------
srv = FakeServer(gap=0.0, n=4)
rate, width, channels, pcm = wyoming.synthesize("127.0.0.1", srv.port, "collect")
check((rate, width, channels) == (22050, 2, 1) and pcm == CHUNK * 4, "synthesize() still returns (rate, width, channels, pcm)")

# --- 8. cancel mid-stream closes the socket promptly --------------------------
srv = FakeServer(gap=0.2, n=10)
state.CONFIG["wyoming_tts_port"] = srv.port
proc = FakeProc()
patch_player(proc)
calls = {"n": 0}


def cancel_after_two():
    calls["n"] += 1
    return calls["n"] > 2


speech_tts.is_cancelled = cancel_after_two
t0 = time.monotonic()
speech_tts._tts_wyoming("cancel me", None)
dt = time.monotonic() - t0
check(dt < 1.0, f"cancel mid-stream returned in {dt:.2f}s (server had 2 s of chunks left)")

import shutil
shutil.rmtree(os.environ["XDG_STATE_HOME"], ignore_errors=True)
print("\n%d failure(s)" % len(fails))
sys.exit(1 if fails else 0)
