"""Repro for #19: offline TTS must start playing before audio-stop.

Usage: python3 tests/repros/wyoming_stream.py [/path/to/speech-to-cli]
(default: this checkout). Exit 0 = all checks pass. Fails on main before the
fix: the first PCM write lands AFTER audio-stop and _mark_tts_end() is never
called. Fake Wyoming server on loopback, fake player, no audio, no network,
nothing written to the user config.

Scratch: a PRIVATE per-PID directory under <this repo>/tmp/repros/, assigned
to XDG_STATE_HOME unconditionally and removed in a `finally`. The rmtree only
ever touches that local path, never the environment variable (#20 review: a
`setdefault` followed by `rmtree(os.environ[...])` deleted a PRESET
XDG_STATE_HOME -- where gnome-speaks keeps chronicle.jsonl).
`wyoming_stream_canary.py` is the control for that.

Also covers the #20 review findings: a mid-stream RST surfaces as
WyomingError (never a bare OSError), the player is released on every exit
path, and a prewarmed player at the WRONG rate does not delay the first PCM
write (its wait() happens off-thread).
"""
import io
import json
import os
import shutil
import socket
import struct
import sys
import threading
import time

SVC = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SVC)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH_ROOT = os.path.join(REPO_ROOT, "tmp", "repros")
STATE_DIR = os.path.join(SCRATCH_ROOT, "wyoming-stream-%d" % os.getpid())
os.makedirs(STATE_DIR, exist_ok=False)
os.environ["XDG_STATE_HOME"] = STATE_DIR  # assigned, never setdefault
os.environ.pop("SPEECH_FORCE_OFFLINE", None)

import state  # noqa: E402
import wyoming  # noqa: E402
import audio  # noqa: E402
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
    def __init__(self, gap=CHUNK_GAP, n=N_CHUNKS, stall_before_stop=0.0, no_chunks=False,
                 rst_after=None, rate=22050):
        self.gap, self.n, self.stall, self.no_chunks = gap, n, stall_before_stop, no_chunks
        self.rst_after, self.rate = rst_after, rate
        self.stop_at = None
        self.start_at = None
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        conn, _ = self.srv.accept()
        try:
            self._serve(conn)
        except OSError:
            # The client hung up first (cancel mid-stream) -- expected, not noise.
            pass

    def _serve(self, conn):
        f = conn.makefile("rb")
        f.readline()  # synthesize
        fmt = {"rate": self.rate, "width": 2, "channels": 1}
        self.start_at = time.monotonic()
        _send(conn, "audio-start", fmt)
        if not self.no_chunks:
            for i in range(self.n):
                time.sleep(self.gap)
                _send(conn, "audio-chunk", fmt, CHUNK)
                if self.rst_after is not None and i + 1 == self.rst_after:
                    # Let the client drain what was sent, then RESET the
                    # connection: SO_LINGER(on, 0) + close() emits RST, and the
                    # peer's next read raises ConnectionResetError.
                    time.sleep(0.3)
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    conn.close()
                    return
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


class SlowOldPlayer(FakeProc):
    """A prewarmed player whose teardown takes WAIT_S -- like aplay/pw-play
    releasing the sink (164-214 ms measured in the #20 review)."""
    WAIT_S = 0.3

    def __init__(self):
        super().__init__()
        self.waited_at = None

    def wait(self, timeout=None):
        time.sleep(self.WAIT_S if timeout is None else min(timeout, self.WAIT_S))
        self.waited_at = time.monotonic()
        return 0

    def kill(self):
        pass


fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def patch_player(proc):
    speech_tts._take_prewarmed_player = lambda rate: None
    speech_tts._start_player = lambda rate, target=None: proc
    speech_tts.register_proc = lambda p: None
    speech_tts.unregister_proc = lambda p: None
    speech_tts.play_speak = lambda: None
    speech_tts.send_progress = lambda *a, **k: None
    speech_tts.is_cancelled = lambda: False


def main():
    # --- 1. playback starts before audio-stop --------------------------------
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
    check(proc.stdin.closed and not proc.terminated, "healthy path: player got EOF and was NOT terminated")

    # --- 2. second utterance within 10 s: no lead-in -------------------------
    srv = FakeServer(n=2)
    state.CONFIG["wyoming_tts_port"] = srv.port
    proc = FakeProc()
    patch_player(proc)
    speech_tts._tts_wyoming("again", None)
    zeros = [b for t, b in proc.stdin.writes if b and b == b"\x00" * len(b)]
    check(not zeros, "warm device: no 200 ms zero lead-in on the next offline utterance")

    # --- 3. output_file WAV contains every byte --------------------------------
    srv = FakeServer(n=3)
    state.CONFIG["wyoming_tts_port"] = srv.port
    proc = FakeProc()
    patch_player(proc)
    out = os.path.join(STATE_DIR, "out.wav")
    speech_tts._tts_wyoming("save me", None, output_file=out)
    data = open(out, "rb").read()
    check(data[:4] == b"RIFF" and data[44:] == CHUNK * 3, f"WAV written with all {len(CHUNK)*3} PCM bytes")

    # --- 4. slow consumer does not trip the deadline ---------------------------
    srv = FakeServer(gap=0.0, n=8)
    gen = wyoming.synthesize_stream("127.0.0.1", srv.port, "slow", timeout=0.5)
    next(gen)
    got = 0
    try:
        for chunk in gen:
            time.sleep(0.12)  # 8 x 0.12 = 0.96 s of consumer time > 0.5 s timeout
            got += len(chunk)
        check(got == len(CHUNK) * 8, "consumer slower than timeout: stream completed, no false 'timed out'")
    except wyoming.WyomingError as e:
        check(False, f"consumer slower than timeout raised: {e}")

    # --- 5. a stalled server still times out ----------------------------------
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
    check(proc.stdin.closed, "empty stream: player released (stdin closed) all the same")

    # --- 7. collect-all wrapper unchanged for tts.py / speak.py ---------------
    srv = FakeServer(gap=0.0, n=4)
    rate, width, channels, pcm = wyoming.synthesize("127.0.0.1", srv.port, "collect")
    check((rate, width, channels) == (22050, 2, 1) and pcm == CHUNK * 4, "synthesize() still returns (rate, width, channels, pcm)")

    # --- 8. cancel mid-stream closes the socket promptly ------------------------
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

    # --- 9. mid-stream RST: WyomingError, never a bare OSError; player released
    srv = FakeServer(gap=0.0, n=4, rst_after=2)
    gen = wyoming.synthesize_stream("127.0.0.1", srv.port, "rst", timeout=5.0)
    next(gen)
    try:
        list(gen)
        check(False, "RST mid-stream: expected WyomingError, stream ended cleanly instead")
    except wyoming.WyomingError as e:
        check(True, f"RST mid-stream raised WyomingError: {e}")
    except OSError as e:
        check(False, f"RST mid-stream escaped as bare {type(e).__name__}: {e}")

    srv = FakeServer(gap=0.0, n=4, rst_after=2)
    state.CONFIG["wyoming_tts_port"] = srv.port
    proc = FakeProc()
    patch_player(proc)
    res = speech_tts._tts_wyoming("rst", None)
    heard = sum(len(b) for t, b in proc.stdin.writes if b != b"\x00" * len(b))
    check(isinstance(res, dict) and "error" in res and heard > 0,
          f"RST after {heard} B were heard -> error dict (no Azure re-speak, no false ok): {res}")
    check(proc.stdin.closed and proc.poll() is not None,
          "RST mid-stream: player got EOF and exited (not orphaned on stdin)")

    # --- 10. a prewarmed player at the WRONG rate must not delay the first write
    old = SlowOldPlayer()
    with state._prewarmed_player_lock:
        state._prewarmed_player = old
        state._prewarmed_player_rate = 24000
    srv = FakeServer(gap=0.0, n=4)
    state.CONFIG["wyoming_tts_port"] = srv.port
    proc = FakeProc()
    patch_player(proc)
    speech_tts._take_prewarmed_player = audio._take_prewarmed_player  # the real one
    speech_tts._tts_wyoming("mismatch", None)
    pcm_writes = [t for t, b in proc.stdin.writes if b != b"\x00" * len(b)]
    lat = (pcm_writes[0] - srv.start_at) if pcm_writes and srv.start_at else None
    print(f"INFO first-byte latency audio-start -> first PCM write with a 24 kHz prewarm in the way: "
          f"{lat is not None and round(lat*1000)} ms (old player's teardown takes {round(SlowOldPlayer.WAIT_S*1000)} ms)")
    check(lat is not None and lat < 0.1,
          f"mismatched prewarmed player did not stall the first write ({lat is not None and round(lat*1000)} ms < 100 ms)")
    check(old.stdin.closed, "mismatched prewarmed player was EOF'd immediately")
    deadline = time.monotonic() + 1.5
    while old.waited_at is None and time.monotonic() < deadline:
        time.sleep(0.02)
    check(old.waited_at is not None, "mismatched prewarmed player was still reaped (wait() ran off-thread)")
    check(state._prewarmed_player is None, "prewarm slot is empty after the discard")

    # --- 11. the prewarm rate follows the backend ------------------------------
    # getattr: on the pre-fix tree neither symbol exists, and a control must
    # report the red, not crash before it.
    learned = getattr(state, "_wyoming_tts_rate", None)
    prewarm_rate = getattr(audio, "_prewarm_rate", lambda: None)
    check(learned == 22050, f"server rate learned from audio-start: {learned}")
    state.CONFIG["speech_backend"] = "local"
    check(prewarm_rate() == 22050, f"speech_backend=local -> prewarm at the server's rate ({prewarm_rate()})")
    srv = FakeServer(gap=0.0, n=1, rate=16000)
    state.CONFIG["wyoming_tts_port"] = srv.port
    proc = FakeProc()
    patch_player(proc)
    speech_tts._tts_wyoming("sixteen", None)
    check(prewarm_rate() == 16000, f"a different server rate is learned and prewarmed ({prewarm_rate()})")
    state.CONFIG["speech_backend"] = "azure"
    check(prewarm_rate() == 24000, f"speech_backend=azure -> 24 kHz prewarm as before ({prewarm_rate()})")


try:
    main()
finally:
    # Only ever remove the directory THIS process created, and only if it
    # still resolves inside the scratch root.
    real = os.path.realpath(STATE_DIR)
    root = os.path.realpath(SCRATCH_ROOT)
    assert real.startswith(root + os.sep), (real, root)
    shutil.rmtree(real, ignore_errors=True)
    try:
        os.rmdir(root)  # only if empty -- another PID's dir stays
    except OSError:
        pass

print("\n%d failure(s)" % len(fails))
sys.exit(1 if fails else 0)
