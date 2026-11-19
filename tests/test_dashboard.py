import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from wrapper import camera, config as cfg, gaze

JS = Path(__file__).resolve().parent.parent / "dashboard" / "gaze.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_browser_maths_matches_python():
    K = camera.intrinsics()
    w2c = camera.world_to_camera(camera.pose())
    centres = {s: c.tolist() for s, c in gaze.eye_centres().items()}
    cases = [(0, 0, 0.35), (1, 0, 0.35), (-0.4, 0.9, 0.6), (0.7, -0.7, 0.2)]
    script = f"""
      const G = require({json.dumps(str(JS))});
      const opts = {{centres: {json.dumps(centres)}, radius: {cfg.EYE_RADIUS}, span: {cfg.FIXATION_RANGE},
                    K: {json.dumps(K.tolist())}, w2c: {json.dumps(w2c.tolist())}}};
      const out = {json.dumps(cases)}.map(([h, v, d]) => G.solve(h, v, {{...opts, distance: d}}));
      console.log(JSON.stringify(out));
    """
    res = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    for (h, v, d), js in zip(cases, res):
        py = gaze.solve(h, v, d)
        assert js["vergence"] == pytest.approx(py["vergence_deg"], abs=1e-9)
        for s in gaze.SIDES:
            assert np.allclose(js["eyes"][s]["euler"], py["eyes"][s]["euler"], atol=1e-12)
            uv = camera.project(py["eyes"][s]["pupil"], K, w2c)[0][0]
            assert np.allclose(js["eyes"][s]["uv"], uv, atol=1e-9)
