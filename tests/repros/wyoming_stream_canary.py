"""Control for the scratch-cleanup hazard in wyoming_stream.py (#20 review).

The old repro did `os.environ.setdefault("XDG_STATE_HOME", …)` and then
`shutil.rmtree(os.environ["XDG_STATE_HOME"])` -- with XDG_STATE_HOME PRESET it
deleted the caller's state dir (for gnome-speaks: chronicle.jsonl). This runs
the repro with XDG_STATE_HOME pointing at a directory holding a canary file
and requires the canary to survive.

Usage: python3 tests/repros/wyoming_stream_canary.py [/path/to/speech-to-cli]
Env:   WYOMING_STREAM_REPRO=/path/to/wyoming_stream.py  (default: the sibling)
       -- point it at the PRE-FIX repro to prove this control can fail.
Exit 0 = canary survived and the repro passed; 1 otherwise.
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
REPRO = os.environ.get("WYOMING_STREAM_REPRO", os.path.join(HERE, "wyoming_stream.py"))
SCRATCH_ROOT = os.path.join(REPO_ROOT, "tmp", "repros")
MINE = os.path.join(SCRATCH_ROOT, "wyoming-stream-canary-%d" % os.getpid())
PRESET = os.path.join(MINE, "preset-state")
CANARY = os.path.join(PRESET, "gnome-speaks", "chronicle.jsonl")

os.makedirs(os.path.dirname(CANARY), exist_ok=False)
with open(CANARY, "w") as f:
    f.write('{"canary": true}\n')

rc = 1
try:
    env = dict(os.environ, XDG_STATE_HOME=PRESET)
    cmd = [sys.executable, REPRO] + sys.argv[1:]
    child = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = child.stdout.strip().splitlines()[-1:] or ["<no output>"]
    fails = []
    if child.returncode != 0:
        fails.append(f"repro exited {child.returncode}: {tail[0]}")
    if not os.path.isfile(CANARY):
        fails.append(f"CANARY DELETED: {CANARY}")
    elif open(CANARY).read() != '{"canary": true}\n':
        fails.append("canary rewritten")
    if not os.path.isdir(PRESET):
        fails.append(f"preset XDG_STATE_HOME removed: {PRESET}")
    for f in fails:
        print("FAIL " + f)
    if not fails:
        print(f"PASS preset XDG_STATE_HOME survived the repro (canary intact), repro rc=0: {tail[0]}")
        rc = 0
finally:
    real, root = os.path.realpath(MINE), os.path.realpath(SCRATCH_ROOT)
    assert real.startswith(root + os.sep), (real, root)
    shutil.rmtree(real, ignore_errors=True)
    try:
        os.rmdir(root)
    except OSError:
        pass
sys.exit(rc)
