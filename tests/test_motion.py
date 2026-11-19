import pytest

from lab import motion


def test_min_jerk_profile():
    assert motion.min_jerk(0) == 0 and motion.min_jerk(1) == pytest.approx(1)
    assert motion.min_jerk(0.5) == pytest.approx(0.5)
    xs = [motion.min_jerk(i / 100) for i in range(101)]
    assert all(b >= a for a, b in zip(xs, xs[1:]))


def test_main_sequence_duration():
    assert motion.saccade_duration(10) == pytest.approx(0.043)
    assert motion.saccade_duration(1) < motion.saccade_duration(15)


def test_saccade_path_reaches_each_target():
    pts = [(0, 0), (0.8, 0.5), (-0.6, -0.4)]
    rows = motion.saccade_path(pts, fps=120, drift=0.0, seed=3)
    events = {r["event"] for r in rows}
    assert events == {"fixation", "saccade"}
    assert all(-1 <= r["h"] <= 1 and -1 <= r["v"] <= 1 for r in rows)
    fix = [r for r in rows if r["event"] == "fixation"]
    assert (fix[-1]["h"], fix[-1]["v"]) == pytest.approx(pts[-1], abs=1e-9)
    ts = [r["t"] for r in rows]
    assert ts == sorted(ts)


def test_pursuit_follows_target():
    rows = motion.pursuit_path(duration=4, fps=60)
    late = rows[len(rows) // 2:]
    err = [motion.amplitude_deg((r["h"], r["v"]), (r["target_h"], r["target_v"])) for r in late]
    assert sum(err) / len(err) < 2.0
    assert any(r["event"] == "pursuit" for r in rows)


def test_blinks_do_not_move_gaze():
    rows = motion.saccade_path([(0, 0), (0.5, 0.5)], fps=60, fixation=(1.5, 1.5), drift=0.0)
    before = [(r["h"], r["v"]) for r in rows]
    motion.blink_track(rows, rate_per_min=60, seed=2)
    assert [(r["h"], r["v"]) for r in rows] == before
    assert max(r["lid"] for r in rows) == pytest.approx(1.0)
    assert all(0 <= r["lid"] <= 1 for r in rows)
