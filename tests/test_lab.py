import json

import numpy as np
from PIL import Image, ImageDraw

from lab import detector, heatmap, quality, textures
from wrapper import camera, gaze


def fake_record(h=0.4, v=-0.2):
    K = camera.intrinsics()
    w2c = camera.world_to_camera(camera.pose())
    sol = gaze.solve(h, v)
    rec = {"frame": 0, "theta": 0.0, "input": {"h": h, "v": v},
           "eye_rotation_L": [0, 0, 0], "eye_rotation_R": [0, 0, 0],
           "camera_matrix": camera.describe(K, w2c), "image": "img.png"}
    for s in gaze.SIDES:
        e = sol["eyes"][s]
        rec[f"pupil_center_2D_{s}"] = camera.project(e["pupil"], K, w2c)[0][0].tolist()
        rec[f"eye_center_{s}"] = e["centre"].tolist()
        rec[f"pupil_center_3D_{s}"] = e["pupil"].tolist()
        rec[f"gaze_vector_{s}"] = e["direction"].tolist()
    return rec


def draw(rec, path):
    img = Image.new("RGB", (1024, 1024), (200, 170, 150))
    d = ImageDraw.Draw(img)
    for s in gaze.SIDES:
        u, v = rec[f"pupil_center_2D_{s}"]
        d.ellipse([u - 60, v - 60, u + 60, v + 60], fill=(110, 70, 40))
        d.ellipse([u - 20, v - 20, u + 20, v + 20], fill=(5, 5, 5))
    img.save(path)


def test_detector_and_reconstruction(tmp_path):
    rec = fake_record()
    draw(rec, tmp_path / "img.png")
    res = detector.evaluate(rec, tmp_path / "img.png")
    for s in gaze.SIDES:
        assert res[s]["found"]
        assert res[s]["error_px"] < 1.0
        assert res[s]["angular_error_deg"] < 0.3


def test_schema_and_quality(tmp_path):
    rec = fake_record()
    draw(rec, tmp_path / "img.png")
    recs = []
    for k in range(10):
        r = json.loads(json.dumps(rec))
        r["frame"] = k
        r["input"] = {"h": 0.4, "v": -0.2 + k * 1e-3}
        recs.append(r)
    assert quality.check_schema(recs) == []
    broken = json.loads(json.dumps(recs))
    del broken[3]["pupil_center_2D_R"]
    assert any("record 3" in p for p in quality.check_schema(broken))
    qc = quality.validate([recs[0]], tmp_path)
    assert qc["stats"]["detector_agree"] == 1 and qc["stats"]["flagged"] == 0


def test_textures_are_valid():
    for c in textures.IRIS_PALETTES:
        img = textures.iris(c, seed=1, width=256, height=64)
        assert img.shape == (64, 256, 3) and img.min() >= 0 and img.max() <= 1
    s = textures.sclera(width=128, height=32)
    assert s.mean() > 0.5
    d = textures.skin_detail(size=128)
    assert d.shape == (128, 128, 3)


def test_heatmaps():
    t = heatmap.target_heatmap([(0, 0), (0.5, 0.5), (0.5, 0.5)], size=100)
    assert t.shape == (100, 100, 3) and t.dtype == np.uint8


def test_detector_ignores_eyes_outside_the_frame():
    grey = np.full((64, 64), 0.6)
    assert detector.detect_pupil(grey, (-200.0, 30.0), window=20) is None
    assert detector.detect_pupil(grey, (30.0, 400.0), window=20) is None
