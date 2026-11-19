"""
Job queue for long render sessions.

Start it once (it keeps running after the Blender window is closed):

    blender -b --factory-startup --python tools/runner.py -- --daemon

or from Blender's Python console:

    exec(open("/path/to/eyeverse-x/tools/runner.py").read())

Then drop a JSON list of jobs into output/queue.json, for example

    [["build", "--python", "tools/build_scene.py"],
     ["official", "scene.blend", "--python", "tools/render_official.py"],
     ["sheet", "@py", "tools/contact_sheet.py"]]

Jobs run one after another and log to output/<name>.log; progress goes to
output/runner.log. Only the project's own scripts are accepted (see ALLOWED), every
argument is checked, and nothing is passed through a shell. On macOS each job runs
under `caffeinate -i` so the machine does not idle-sleep mid-render.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import bpy

try:
    _ROOT = Path(__file__).resolve().parent.parent
except NameError:  # exec() from the console
    _ROOT = Path(os.environ.get("EYEVERSE_ROOT", Path.home() / "projects" / "eyeverse-x"))
_QUEUE = _ROOT / "output" / "queue.json"
_LOG = _ROOT / "output" / "runner.log"
_busy = {"on": False}

BLENDER_SCRIPTS = {"tools/build_scene.py", "tools/render_official.py", "tools/generate_dataset.py",
                   "wrapper/eyes.py"}
PYTHON_SCRIPTS = {"tools/contact_sheet.py", "tools/report.py", "dashboard/build.py", "tools/package.py",
                  "tools/make_video.py", "tools/publish.py"}
SAFE_ARG = re.compile(r"^[A-Za-z0-9_.:+\-/=]{1,80}$")
NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
MAX_JOBS = 24


def _note(text):
    with _LOG.open("a") as f:
        f.write(time.strftime("%H:%M:%S ") + text + "\n")


def _safe_path_arg(arg):
    """Relative paths only, and they must stay inside the project."""
    if arg.startswith("/") or ".." in Path(arg).parts:
        return False
    return (_ROOT / arg).resolve().is_relative_to(_ROOT)


def validate(job):
    """Turn one queue entry into an argv list, or raise ValueError."""
    if not isinstance(job, list) or len(job) < 2 or not all(isinstance(a, str) for a in job):
        raise ValueError("a job is a list of strings")
    name, *args = job
    if not NAME.match(name):
        raise ValueError(f"bad job name {name!r}")
    for a in args[1:] if args[0] == "@py" else args:
        if not SAFE_ARG.match(a) or not _safe_path_arg(a):
            raise ValueError(f"argument not allowed: {a!r}")
    if args[0] == "@py":
        script = args[1]
        if script == "-m":
            if args[2:4] != ["pytest", "-q"]:
                raise ValueError("only `-m pytest -q` is allowed")
        elif script not in PYTHON_SCRIPTS:
            raise ValueError(f"python script not allowed: {script}")
        return name, [_python(), *args[1:]]
    if "--python" not in args:
        raise ValueError("blender jobs need --python <script>")
    script = args[args.index("--python") + 1]
    if script not in BLENDER_SCRIPTS:
        raise ValueError(f"blender script not allowed: {script}")
    head = args[:args.index("--python")]
    if head not in ([], ["scene.blend"]):
        raise ValueError("only scene.blend may be opened")
    return name, [bpy.app.binary_path, "--background", "--factory-startup", *args]


def _python():
    home = Path.home()
    for c in (str(home / "anaconda3/bin/python3"), "/opt/anaconda3/bin/python3",
              str(home / "opt/anaconda3/bin/python3"), "/opt/homebrew/bin/python3",
              "/usr/local/bin/python3", shutil.which("python3") or "", "/usr/bin/python3"):
        if c and Path(c).exists() and subprocess.run([c, "-c", "import numpy, PIL"],
                                                     capture_output=True).returncode == 0:
            return c
    return "python3"


def _run(jobs):
    _busy["on"] = True
    awake = ["caffeinate", "-i"] if shutil.which("caffeinate") else []
    try:
        for name, argv in jobs:
            stop = _ROOT / "output" / "stop"
            if stop.exists():
                stop.unlink()
                _note("stop requested; queue dropped")
                break
            _note(f"start {name}: {' '.join(argv[3:] if argv[1:2] == ['--background'] else argv[1:])}")
            t0 = time.time()
            with open(_ROOT / "output" / f"{name}.log", "w") as log:
                code = subprocess.call(awake + argv, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, PYTHONUNBUFFERED="1"))
            _note(f"end {name}: exit {code} in {time.time() - t0:.0f} s")
            if code != 0:
                _note("queue stopped")
                break
        _note("idle")
    finally:
        _busy["on"] = False


def _take():
    if _busy["on"] or not _QUEUE.exists():
        return None
    try:
        raw = json.loads(_QUEUE.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    _QUEUE.rename(_QUEUE.with_suffix(".taken"))
    if not isinstance(raw, list) or len(raw) > MAX_JOBS:
        _note("rejected queue: expected a list of at most %d jobs" % MAX_JOBS)
        return None
    try:
        return [validate(j) for j in raw]
    except ValueError as err:
        _note(f"rejected queue: {err}")
        return None


def _poll():
    jobs = _take()
    if jobs:
        threading.Thread(target=_run, args=(jobs,), daemon=True).start()
    return 2.0


def daemon():
    _note(f"runner ready (daemon, pid {os.getpid()})")
    while True:
        jobs = _take()
        if jobs:
            _run(jobs)
        time.sleep(2.0)


def detach():
    """Start the daemon as its own process so it outlives this Blender window."""
    subprocess.Popen([bpy.app.binary_path, "-b", "--factory-startup", "--python", str(_ROOT / "tools" / "runner.py"),
                      "--", "--daemon"], cwd=_ROOT, start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


_LOG.parent.mkdir(parents=True, exist_ok=True)
if "--daemon" in sys.argv:
    daemon()
elif bpy.app.background:
    pass
else:
    detach()
    _note("daemon started from the Blender window")
