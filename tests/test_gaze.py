import math

import numpy as np
import pytest

from wrapper import config as cfg, gaze


def test_sequence_matches_brief():
    rows = gaze.sequence()
    assert len(rows) == 10
    assert rows[0]["h"] == 0.0 and rows[0]["v"] == 1.0
    for k, r in enumerate(rows):
        theta = 2 * math.pi * k / 10
        assert r["theta"] == pytest.approx(theta)
        assert r["h"] == pytest.approx(math.sin(theta), abs=1e-12)
        assert r["v"] == pytest.approx(math.cos(theta), abs=1e-12)
    assert rows[3]["h"] == pytest.approx(0.951, abs=5e-4) and rows[3]["v"] == pytest.approx(-0.309, abs=5e-4)


@pytest.mark.parametrize("bad", [1.2, -1.01, float("nan"), "x", None])
def test_inputs_are_checked(bad):
    with pytest.raises(ValueError):
        gaze.fixation_point(bad, 0.0)


def test_fixation_point_geometry():
    assert np.allclose(gaze.fixation_point(1, -1), [0.05, -0.35, -0.05])
    assert np.allclose(gaze.fixation_point(0, 0, distance=0.6), [0, -0.6, 0])


def test_eyes_converge_on_target():
    sol = gaze.solve(0.3, -0.7)
    for side in gaze.SIDES:
        e = sol["eyes"][side]
        to_target = gaze.normalise(sol["target"] - e["centre"])
        assert np.allclose(e["direction"], to_target)
        assert np.allclose(gaze.quat_matrix(e["quaternion"]) @ gaze.FORWARD, e["direction"])
        assert np.linalg.norm(e["pupil"] - e["centre"]) == pytest.approx(cfg.EYE_RADIUS)


def test_listing_law_axis_in_frontal_plane():
    for h, v in [(1, 1), (-0.6, 0.4), (0.2, -1)]:
        for e in gaze.solve(h, v)["eyes"].values():
            q = e["quaternion"]
            axis = np.array(q[1:])
            if np.linalg.norm(axis) > 1e-9:
                assert abs(np.dot(axis / np.linalg.norm(axis), gaze.FORWARD)) < 1e-9


def test_euler_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        e = rng.uniform(-0.6, 0.6, 3)
        m = gaze.euler_matrix_xyz(e)
        assert np.allclose(gaze.matrix_euler_xyz(m), e, atol=1e-9)


def test_rest_pose_vergence():
    sol = gaze.solve(0, 0)
    expected = 2 * math.degrees(math.atan((cfg.IPD / 2) / cfg.FIXATION_DISTANCE))
    assert sol["vergence_deg"] == pytest.approx(expected, rel=1e-9)
    assert gaze.solve(0, 0, distance=10.0)["vergence_deg"] < 0.4


def test_directions_are_signed_correctly():
    for side in gaze.SIDES:
        assert gaze.solve(1, 0)["eyes"][side]["yaw_deg"] > gaze.solve(0, 0)["eyes"][side]["yaw_deg"]
        assert gaze.solve(0, 1)["eyes"][side]["pitch_deg"] > 0
        assert gaze.solve(0, -1)["eyes"][side]["pitch_deg"] < 0


def test_pupil_diameter_tracks_light():
    dark, office, sun = (gaze.pupil_diameter(x) for x in (0.01, 100, 10000))
    assert dark > office > sun
    lo, hi = cfg.PUPIL_LIMITS_MM
    assert lo <= sun and dark <= hi
