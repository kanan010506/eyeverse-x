import numpy as np
import pytest

from wrapper import camera, config as cfg, gaze


def setup():
    K = camera.intrinsics()
    w2c = camera.world_to_camera(camera.pose())
    return K, w2c


def test_origin_projects_to_image_centre():
    K, w2c = setup()
    uv, z = camera.project([0, 0, 0], K, w2c)
    assert np.allclose(uv[0], [cfg.IMAGE_SIZE / 2] * 2)
    assert z[0] == pytest.approx(cfg.CAMERA_DISTANCE)


def test_axes_map_to_image():
    K, w2c = setup()
    right, _ = camera.project([0.01, 0, 0], K, w2c)
    up, _ = camera.project([0, 0, 0.01], K, w2c)
    assert right[0, 0] > cfg.IMAGE_SIZE / 2       # +X is image right
    assert up[0, 1] < cfg.IMAGE_SIZE / 2          # +Z is image up (v smaller)


def test_left_eye_on_image_right():
    K, w2c = setup()
    c = gaze.eye_centres()
    uvl, _ = camera.project(c["L"], K, w2c)
    uvr, _ = camera.project(c["R"], K, w2c)
    assert uvl[0, 0] > uvr[0, 0]


def test_both_eyes_fit_in_frame():
    K, w2c = setup()
    for side, c in gaze.eye_centres().items():
        for dx in (-cfg.EYE_RADIUS, cfg.EYE_RADIUS):
            uv, _ = camera.project(c + np.array([dx, 0, 0]), K, w2c)
            assert 40 < uv[0, 0] < cfg.IMAGE_SIZE - 40


def test_ray_back_projection_round_trip():
    K, w2c = setup()
    rng = np.random.default_rng(1)
    for _ in range(20):
        h, v = rng.uniform(-1, 1, 2)
        e = gaze.solve(h, v)["eyes"]["L"]
        uv, _ = camera.project(e["pupil"], K, w2c)
        o, d = camera.pixel_ray(uv[0], K, w2c)
        hit = camera.ray_sphere(o, d, e["centre"], cfg.EYE_RADIUS)
        assert np.allclose(gaze.normalise(hit - e["centre"]), e["direction"], atol=1e-9)


def test_pupil_travel_is_visible():
    K, w2c = setup()
    rest = camera.project(gaze.solve(0, 0)["eyes"]["L"]["pupil"], K, w2c)[0][0]
    for h, v in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        moved = camera.project(gaze.solve(h, v)["eyes"]["L"]["pupil"], K, w2c)[0][0]
        assert np.linalg.norm(moved - rest) > 12.0


def test_extra_camera_pose_looks_at_origin():
    m = camera.pose(offset=(0.1, 0.02, -0.05))
    w2c = camera.world_to_camera(m)
    uv, _ = camera.project([0, 0, 0], camera.intrinsics(), w2c)
    assert np.allclose(uv[0], [cfg.IMAGE_SIZE / 2] * 2, atol=1e-6)
