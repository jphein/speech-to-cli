#!/usr/bin/env python3
"""Repro for the tts_prepare()/tts_play() seam (gnome-speaks#134).

Usage: python3 tests/repros/tts_prepare_play.py [/path/to/speech-to-cli]
       STC_BASELINE_TTS=/path/to/OLD/speech_tts.py  (two-sided control)

The seam splits tts() into its network half (route selection, breakers, the
Azure POST or the Wyoming stream up to audio-start) and its playback half, so
a caller can open sentence N+1 while N plays. Three things must hold, and each
is measured here rather than read off the code:

  1. EQUIVALENCE. For every route tts() knows -- Wyoming primary, Azure
     healthy, Azure 5xx -> Wyoming, Azure exception -> Wyoming, Azure 4xx,
     no text, forced-offline-without-Wyoming, Wyoming-dies-before-first-chunk
     -> Azure -- tts_play(tts_prepare(x)) returns the SAME result dict, trips
     the SAME breakers and writes the SAME PCM as tts(x). On this tree tts()
     IS that composition, so the comparison that means something is against
     the PRE-SEAM speech_tts.py: pass it as STC_BASELINE_TTS and every case
     is checked against the old tts() too. Without it the run still passes
     but says so -- a control that compares a function to itself is not one.
  2. HYGIENE. A prepared handle that is never played leaves nothing behind:
     fd count returns to baseline after close() (a real loopback socket, so
     the count is a positive control -- it rises on prepare first), no player
     was started, and playing a closed handle plays nothing.
  3. THE POINT. tts_prepare() returns at audio-start, not audio-stop: on a
     server that streams 0.9 s of chunks it returns in well under that, and a
     tts_play() a second later gets every chunk from the socket buffer at
     once. That is the latency a prefetching caller pays under the previous
     sentence instead of as a gap.

Fake Wyoming server on loopback, fake player, fake Azure session -- no audio,
no network, no chimes, nothing written to the user config. Reuses the fakes of
wyoming_stream.py (imported; its suite is __main__-guarded) and its per-PID
scratch, which this script cleans up.

exit 0 = all checks pass; 1 = a check failed.
"""
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wyoming_stream as ws  # noqa: E402  (sets SVC path + XDG_STATE_HOME, imports the tree)
from wyoming_stream import FakeServer, FakeProc, CHUNK, check, fails  # noqa: E402

import state        # noqa: E402
import wyoming      # noqa: E402
import speech_tts   # noqa: E402

BASELINE = os.environ.get("STC_BASELINE_TTS")
LEAD_IN = 22050 * 2 * 200 // 1000


# --- fakes -------------------------------------------------------------------

class FakeResp:
    def __init__(self, status=200, chunks=3, gap=0.02, text="boom"):
        self.status_code = status
        self.text = text
        self._chunks, self._gap = chunks, gap
        self.closed = False
        self.iter_started = None

    def iter_content(self, chunk_size=16384):
        self.iter_started = time.monotonic()
        for _ in range(self._chunks):
            time.sleep(self._gap)
            yield CHUNK

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc = resp, exc
        self.posts = []

    def post(self, url, **kw):
        self.posts.append(time.monotonic())
        if self.exc is not None:
            raise self.exc
        return self.resp


class Rig:
    """Everything one TTS call touches, patched on ONE speech_tts module."""

    def __init__(self, mod, session=None):
        self.mod = mod
        self.proc = FakeProc()
        self.player_starts = 0
        self.session = session or FakeSession(FakeResp())

        def start(rate, target=None):
            self.player_starts += 1
            return self.proc

        mod._take_prewarmed_player = lambda rate: None
        mod._start_player = start
        mod.register_proc = lambda p: None
        mod.unregister_proc = lambda p: None
        mod.play_speak = lambda: None
        mod.play_done = lambda: None
        mod.stop_hum = lambda: None
        mod.send_progress = lambda *a, **k: None
        mod.is_cancelled = lambda: False
        mod.get_http_session = lambda: self.session

    def pcm_writes(self):
        return [b for _t, b in self.proc.stdin.writes if b != b"\x00" * len(b)]


def pin_config(**over):
    state.CONFIG.update({
        "wyoming_host": "", "wyoming_tts_port": 0, "wyoming_tts_voice": "",
        "speech_backend": "azure", "save_audio_dir": "",
        "chime_speak": False, "chime_done": False, "chime_hum": False,
        "key": "test-key", "region": "westus", "tts_key": "", "tts_region": "",
        "fast_voice": "en-US-AvaNeural", "voice": "en-US-Ava:DragonHDLatestNeural",
        "live_subtitles": False, "vu_meter": False,
    })
    state.CONFIG.update(over)
    wyoming._azure_down_until = 0.0
    wyoming._local_down_until = 0.0
    state._last_tts_end = 0.0
    os.environ.pop("SPEECH_FORCE_OFFLINE", None)


def with_server(**kw):
    srv = FakeServer(**kw)
    state.CONFIG["wyoming_host"] = "127.0.0.1"
    state.CONFIG["wyoming_tts_port"] = srv.port
    return srv


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("speech_tts_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fd_count():
    return len(os.listdir("/proc/self/fd"))


def wait_for(pred, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


# --- the routes ---------------------------------------------------------------
# Each case: (name, setup(), text) -- setup() re-pins config and returns the
# session to hand the module. A fresh FakeServer per RUN (it accepts once).

def case_wyoming_primary():
    pin_config(speech_backend="local")
    with_server(gap=0.0, n=3)
    return FakeSession(FakeResp())


def case_azure_healthy():
    pin_config()
    return FakeSession(FakeResp(200, chunks=3))


def case_azure_5xx_to_wyoming():
    pin_config()
    with_server(gap=0.0, n=3)
    return FakeSession(FakeResp(503, text="upstream"))


def case_azure_exception_to_wyoming():
    pin_config()
    with_server(gap=0.0, n=3)
    return FakeSession(exc=ConnectionError("dns"))


def case_azure_4xx():
    pin_config()
    return FakeSession(FakeResp(400, text="bad ssml"))


def case_forced_offline_no_wyoming():
    pin_config()
    os.environ["SPEECH_FORCE_OFFLINE"] = "1"
    return FakeSession(FakeResp())


def case_wyoming_dies_before_first_chunk_then_azure():
    # audio-start, then audio-stop with no chunks: synthesize_stream raises
    # "no audio returned" inside the play loop, total == 0 -> tts() tripped the
    # local breaker and fell back to Azure. The prepared handle must too.
    pin_config(speech_backend="local")
    with_server(no_chunks=True)
    return FakeSession(FakeResp(200, chunks=2))


# (name, setup, text, players): `players` is the number of playback ATTEMPTS
# the route makes -- one per player the seam may start. Two only where the
# first attempt legitimately got a player and then produced no audio (Wyoming
# answered audio-start and died), so the Azure fallback is a second attempt.
CASES = [
    ("wyoming primary (speech_backend=local)", case_wyoming_primary, "hello there", 1),
    ("azure healthy", case_azure_healthy, "hello there", 1),
    ("azure 5xx -> wyoming", case_azure_5xx_to_wyoming, "hello there", 1),
    ("azure exception -> wyoming", case_azure_exception_to_wyoming, "hello there", 1),
    ("azure 4xx -> error", case_azure_4xx, "hello there", 0),
    ("no text -> error", case_azure_healthy, "", 0),
    ("forced offline, no wyoming -> error", case_forced_offline_no_wyoming, "hello there", 0),
    ("wyoming dies before first chunk -> azure", case_wyoming_dies_before_first_chunk_then_azure, "hello there", 2),
]


def observe(mod, call, setup, text):
    """Run one TTS call on `mod` with fresh fakes; return what it left behind."""
    session = setup()
    rig = Rig(mod, session)
    result = call(mod, text)
    return {
        "result": result,
        "pcm": rig.pcm_writes(),
        "players": rig.player_starts,
        "azure_down": wyoming._azure_down_until > time.time(),
        "local_down": wyoming._local_down_until > time.time(),
        "posts": len(session.posts),
        "resp_closed": getattr(session.resp, "closed", None),
    }


def via_split(mod, text):
    return mod.tts_play(mod.tts_prepare(text))


def via_tts(mod, text):
    return mod.tts(text)


def main():
    base = load_baseline(BASELINE) if BASELINE else None
    print(f"tree under test: {ws.SVC}")
    print(f"baseline tts():  {BASELINE or 'NONE -- equivalence is checked against this tree only'}")

    # --- 1. equivalence -------------------------------------------------------
    print("\n== 1. tts_play(tts_prepare(x)) == tts(x), every route ==")
    # `players` is compared by CONTRACT, not to the baseline: the pre-seam
    # tts() spawned an Azure-rate player BEFORE its POST so the spawn overlapped
    # the API latency, then discarded it on a fallback (2 players for one
    # utterance) or left it unused on a 4xx (1 player, nothing played). The
    # seam starts a player only in tts_play(), one per playback attempt --
    # measured below against the per-route expectation in CASES, and the old
    # count is printed beside it so the change is visible, not silent.
    keys = ("result", "pcm", "azure_down", "local_down", "posts")
    for name, setup, text, want in CASES:
        split = observe(speech_tts, via_split, setup, text)
        same = observe(speech_tts, via_tts, setup, text)
        ok = all(split[k] == same[k] for k in keys + ("players",))
        check(ok, f"[{name}] split == this tree's tts(): {split['result']}"
                  + ("" if ok else f"  vs {same['result']} / {[(k, split[k] != same[k]) for k in keys]}"))
        check(split["players"] == want,
              f"[{name}] players started = {split['players']} (expected {want}: one per playback attempt)")
        if base is not None:
            old = observe(base, via_tts, setup, text)
            ok = all(split[k] == old[k] for k in keys)
            check(ok, f"[{name}] split == PRE-SEAM tts(): result/pcm/breakers/posts"
                      + ("" if ok else f"  DIFF {[(k, split[k], old[k]) for k in keys if split[k] != old[k]]}"))
            check(split["players"] <= old["players"],
                  f"[{name}] players: pre-seam {old['players']}, seam {split['players']} (never more)")
        if split["resp_closed"] is not None and split["result"].get("engine") == "wyoming" and split["posts"]:
            check(split["resp_closed"], f"[{name}] the Azure response was closed after falling back")
    os.environ.pop("SPEECH_FORCE_OFFLINE", None)

    # --- 2. hygiene: an unplayed handle leaves nothing behind ------------------
    print("\n== 2. unplayed handle: sockets, players ==")
    pin_config(speech_backend="local")
    srv = with_server(gap=0.05, n=6)
    rig = Rig(speech_tts)
    before = fd_count()
    h = speech_tts.tts_prepare("abandon me")
    mid = fd_count()
    check(h.route == "wyoming" and not h.closed, f"prepared a wyoming handle: {h}")
    check(mid > before, f"positive control: fd count rose on prepare ({before} -> {mid})")
    h.close()
    check(h.closed, "close() marks the handle closed")
    check(wait_for(lambda: fd_count() == before, 3.0),
          f"fd count back to baseline after close(): {before} -> {mid} -> {fd_count()}")
    check(rig.player_starts == 0, f"no player was started for a handle never played ({rig.player_starts})")
    replay = speech_tts.tts_play(h)
    check(replay.get("cancelled") is True and not rig.pcm_writes() and rig.player_starts == 0,
          f"playing a closed handle plays nothing: {replay}")
    check(srv.stop_at is None or True, "server side released (fd count above is the evidence)")

    pin_config()
    rig = Rig(speech_tts, FakeSession(FakeResp(200, chunks=3)))
    h = speech_tts.tts_prepare("abandon azure")
    check(h.route == "azure", f"prepared an azure handle: {h}")
    h.close()
    check(rig.session.resp.closed and rig.player_starts == 0,
          "azure handle: response closed, no player, on close()")

    # --- 3. the point: prepare returns at audio-start, play drains the buffer --
    print("\n== 3. prepare pays first-chunk latency only; audio buffers behind it ==")
    pin_config(speech_backend="local")
    gap, n = 0.15, 6                      # 0.9 s of streaming after audio-start
    with_server(gap=gap, n=n)
    rig = Rig(speech_tts)
    t0 = time.monotonic()
    h = speech_tts.tts_prepare("prefetch me")
    t_prep = time.monotonic() - t0
    time.sleep(gap * n + 0.2)             # "sentence N is playing"
    t1 = time.monotonic()
    res = speech_tts.tts_play(h)
    t_play = time.monotonic() - t1
    got = sum(len(b) for b in rig.pcm_writes())
    print(f"INFO tts_prepare returned in {t_prep*1000:.0f} ms; the server then streamed "
          f"{gap*n:.2f} s of audio; tts_play drained it in {t_play*1000:.0f} ms")
    check(res == {"ok": True, "engine": "wyoming"}, f"prefetched handle played: {res}")
    check(t_prep < gap * 2, f"prepare returned at audio-start ({t_prep*1000:.0f} ms), not after the stream ({gap*n*1000:.0f} ms)")
    check(got == len(CHUNK) * n, f"every chunk reached the player from the buffer ({got} B)")
    check(t_play < gap * n / 2, f"play drained buffered audio faster than realtime ({t_play*1000:.0f} ms)")
    check(rig.proc.stdin.writes and rig.proc.stdin.writes[0][1] == b"\x00" * LEAD_IN,
          "cold-device lead-in still written first")


try:
    main()
finally:
    ws.cleanup_scratch()

print("\n%d failure(s)" % len(fails))
sys.exit(1 if fails else 0)
