import sys
import types
from pathlib import Path

import pytest

from wrapper import config as cfg, gaze
from wrapper.safety import inside, safe_output

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("bad", [0.0, -1.0, 50.0, float("inf"), float("nan"), "far"])
def test_distance_is_bounded(bad):
    with pytest.raises(ValueError):
        gaze.fixation_point(0, 0, distance=bad)


def test_safe_output_rules(tmp_path):
    ok = safe_output(tmp_path / "a" / "b.png")
    assert ok.parent.exists()
    for bad in (tmp_path / "x.py", tmp_path / "x.png.sh", ROOT / "wrapper" / "eyes.png", ROOT / "tools" / "x.png"):
        with pytest.raises(ValueError):
            safe_output(bad)
    link = tmp_path / "link.png"
    link.symlink_to(tmp_path / "target.png")
    with pytest.raises(ValueError):
        safe_output(link)


def test_inside(tmp_path):
    assert inside(tmp_path / "a" / "b", tmp_path)
    assert not inside(tmp_path / ".." / "elsewhere", tmp_path)


@pytest.fixture()
def runner(monkeypatch):
    bpy = types.ModuleType("bpy")
    bpy.app = types.SimpleNamespace(binary_path="/blender", background=True)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(sys, "argv", ["test"])
    ns = {"__file__": str(ROOT / "tools" / "runner.py"), "__name__": "runner_under_test"}
    exec((ROOT / "tools" / "runner.py").read_text(), ns)
    return ns["validate"]


def test_runner_accepts_project_jobs(runner):
    name, argv = runner(["official", "scene.blend", "--python", "tools/render_official.py"])
    assert name == "official" and argv[:3] == ["/blender", "--background", "--factory-startup"]
    runner(["sheet", "@py", "tools/contact_sheet.py"])
    runner(["test", "@py", "-m", "pytest", "-q", "tests"])
    runner(["sac", "scene.blend", "--python", "tools/generate_dataset.py", "--", "--mode", "saccade"])


@pytest.mark.parametrize("job", [
    ["x", "@py", "-c", "import os"],
    ["x", "--python", "/tmp/evil.py"],
    ["x", "--python", "tools/../../evil.py"],
    ["x", "scene.blend", "--python", "tools/build_scene.py", "--", "--out", "../../etc/x"],
    ["x", "other.blend", "--python", "tools/build_scene.py"],
    ["x", "@py", "tools/report.py;rm -rf ~"],
    ["x", "@py", "-m", "pip", "install", "anything"],
    ["Bad Name", "@py", "tools/report.py"],
    "not a list",
])
def test_runner_rejects_everything_else(runner, job):
    with pytest.raises(ValueError):
        runner(job)
