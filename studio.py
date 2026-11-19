"""
Run the pipeline from Blender's own interface.

Open this file in Blender's Text Editor (Scripting tab -> Text -> Open) and press
Run Script. Each step runs as a separate background Blender or Python process;
progress streams into the "studio.log" text block.

Edit STEPS to choose what runs. Same steps as ./run.sh.
"""
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import bpy

STEPS = ["scene", "official", "sheet", "report", "dashboard"]
DATASET_ARGS = {}        # e.g. {"saccade": ["--seconds", "4"], "random": ["--count", "40"]}


def _root():
    for text in bpy.data.texts:
        if text.filepath and Path(bpy.path.abspath(text.filepath)).name == "studio.py":
            return Path(bpy.path.abspath(text.filepath)).resolve().parent
    return Path(__file__).resolve().parent


ROOT = _root()
LOG = ROOT / "output" / "logs" / "studio.log"
BLENDER = bpy.app.binary_path


def find_python():
    home = Path.home()
    candidates = [os.environ.get("PYTHON", ""), str(home / "anaconda3/bin/python3"),
                  "/opt/anaconda3/bin/python3", str(home / "opt/anaconda3/bin/python3"),
                  str(home / "miniconda3/bin/python3"), "/opt/homebrew/bin/python3",
                  "/usr/local/bin/python3", shutil.which("python3") or "", "/usr/bin/python3"]
    for c in candidates:
        if c and Path(c).exists():
            ok = subprocess.run([c, "-c", "import numpy, PIL"], capture_output=True).returncode == 0
            if ok:
                return c
    for c in candidates:
        if c and Path(c).exists():
            subprocess.run([c, "-m", "pip", "install", "--user", "-q", "numpy", "pillow", "scipy"])
            return c
    raise RuntimeError("no python3 found")


def commands():
    py = None
    blend = [BLENDER, "--background", "--factory-startup"]
    for step in STEPS:
        if step == "scene":
            yield step, blend + ["--python", "tools/build_scene.py", "--"]
        elif step == "official":
            yield step, blend + ["scene.blend", "--python", "tools/render_official.py", "--"]
        elif step.startswith("dataset:"):
            mode = step.split(":", 1)[1]
            yield step, blend + ["scene.blend", "--python", "tools/generate_dataset.py", "--",
                                 "--mode", mode, *DATASET_ARGS.get(mode, [])]
        else:
            py = py or find_python()
            script = {"sheet": "tools/contact_sheet.py", "report": "tools/report.py",
                      "dashboard": "dashboard/build.py", "package": "tools/package.py",
                      "video": "tools/make_video.py", "test": "-m"}[step]
            yield step, ([py, "-m", "pytest", "-q", "tests"] if step == "test" else [py, script])


def worker():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w") as log:
        log.write(f"root {ROOT}\nblender {bpy.app.version_string}\n")
        for step, cmd in commands():
            t0 = time.time()
            log.write(f"\n== {step} ==\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            code = proc.wait()
            log.write(f"== {step} exit {code} in {time.time() - t0:.0f} s ==\n")
            log.flush()
            if code != 0:
                log.write("stopped\n")
                break
        log.write("\nALL DONE\n")


def refresh():
    text = bpy.data.texts.get("studio.log") or bpy.data.texts.new("studio.log")
    if LOG.exists():
        lines = LOG.read_text(errors="replace").splitlines()
        text.from_string("\n".join(lines[-60:]))
    done = LOG.exists() and "ALL DONE" in LOG.read_text(errors="replace")
    return None if done else 1.0


if __name__ == "__main__":
    LOG.unlink(missing_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    bpy.app.timers.register(refresh, first_interval=1.0)
    print("studio started:", STEPS)
