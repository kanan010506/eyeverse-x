"""
Commit the project in logical steps and push it to GitHub.

    python3 tools/publish.py [remote-url]

Uses the git identity and credentials already configured on this machine.
Safe to re-run: steps with nothing new to commit are skipped.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "https://github.com/warismittal/eyeverse-x.git"

STEPS = [
    ("Gaze geometry and camera model", [
        "wrapper/__init__.py", "wrapper/config.py", "wrapper/gaze.py", "wrapper/camera.py",
        "requirements.txt", ".gitignore"]),
    ("Scene builder for the MakeHuman face, eyes and lighting", [
        "assets", "lab/__init__.py", "lab/textures.py", "lab/shading.py", "tools/__init__.py",
        "tools/build_scene.py"]),
    ("Eye wrapper with move/reset/render and scene ground truth", [
        "wrapper/eyes.py", "wrapper/truth.py", "wrapper/safety.py"]),
    ("Official clockwise sequence, direction check and contact sheet", [
        "tools/render_official.py", "tools/contact_sheet.py", "scene.blend", "renders",
        "metadata.json", "contact_sheet.png", "output/directions"]),
    ("Movement models, identities and dataset generator", [
        "lab/motion.py", "lab/identity.py", "tools/generate_dataset.py", "tools/make_video.py"]),
    ("Pupil detector, gaze reconstruction and quality checks", [
        "lab/detector.py", "lab/quality.py", "lab/heatmap.py", "tools/report.py", "output/report"]),
    ("Calibration dashboard", ["dashboard"]),
    ("Tests", ["tests"]),
    ("Run scripts, packaging and README", [
        "run.sh", "studio.py", "tools/runner.py", "tools/package.py", "tools/publish.py", "README.md",
        "SECURITY.md", "LICENSE"]),
    ("Pitch deck", ["docs"]),
    ("Lab datasets", ["output"]),
]


def git(*args, check=True):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    res = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, env=env)
    out = (res.stdout + res.stderr).strip()
    if out:
        print(f"$ git {' '.join(args)}\n{out}", flush=True)
    if check and res.returncode != 0:
        raise SystemExit(f"git {args[0]} failed ({res.returncode})")
    return res


def main(remote=REMOTE):
    if not (ROOT / ".git").exists():
        git("init", "-b", "main")
    if git("remote", check=False).stdout.split().count("origin") == 0:
        git("remote", "add", "origin", remote)
    if not git("config", "user.email", check=False).stdout.strip():
        git("config", "user.name", "Waris")
        git("config", "user.email", "warismittal40@gmail.com")
    for message, paths in STEPS:
        present = [p for p in paths if (ROOT / p).exists()]
        if not present:
            continue
        git("add", "--", *present)
        if git("diff", "--cached", "--quiet", check=False).returncode == 0:
            continue
        git("commit", "-q", "-m", message)
    git("add", "-A")
    if git("diff", "--cached", "--quiet", check=False).returncode != 0:
        git("commit", "-q", "-m", "Tidy up")
    git("log", "--oneline", "-n", "12")
    if git("push", "-u", "origin", "main", check=False).returncode != 0:
        # the repo was created with a starter README/licence: keep its history, keep our files
        git("fetch", "origin", "main")
        git("merge", "-q", "--allow-unrelated-histories", "-X", "ours", "-m", "Merge initial repository files",
            "origin/main")
        git("push", "-u", "origin", "main")
    print("pushed to", remote, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
