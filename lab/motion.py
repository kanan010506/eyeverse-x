"""
Eye movement timelines in normalised (h, v) space.

saccade_path   fixations joined by saccades that follow the main sequence
pursuit_path   smooth pursuit of a moving target with catch-up saccades
blink_track    eyelid closure over time

Every timeline is a list of dicts with t (s), h, v, event and lid (0 open, 1 shut).
Durations and gains come from the eye-movement literature:
    saccade duration  D = 2.2 A + 21 ms        (Carpenter 1988)
    profile           minimum jerk
    pursuit gain      ~0.9, 100 ms latency, catch-up when error > 2 deg
    blink             ~15 per minute, 100 ms close, 50 ms shut, 150 ms open
"""
import math
import random

from wrapper import config as cfg

DEG_PER_UNIT = math.degrees(math.atan(cfg.FIXATION_RANGE / cfg.FIXATION_DISTANCE))


def amplitude_deg(a, b):
    """Angular distance between two (h, v) points seen from the eye midpoint."""
    span, dist = cfg.FIXATION_RANGE, cfg.FIXATION_DISTANCE
    pa = (a[0] * span, -dist, a[1] * span)
    pb = (b[0] * span, -dist, b[1] * span)
    dot = sum(x * y for x, y in zip(pa, pb))
    na = math.sqrt(sum(x * x for x in pa))
    nb = math.sqrt(sum(x * x for x in pb))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (na * nb)))))


def saccade_duration(amp_deg):
    return (2.2 * amp_deg + 21.0) / 1000.0


def min_jerk(s):
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def _clip(x):
    return max(-1.0, min(1.0, x))


def saccade_path(points, fps=60, fixation=(0.18, 0.40), drift=0.004, seed=0):
    """Fixate each point for a random dwell, then jump to the next one."""
    rng = random.Random(seed)
    dt = 1.0 / fps
    rows, t = [], 0.0
    cur = points[0]
    for nxt in list(points[1:]) + [None]:
        dwell = rng.uniform(*fixation)
        end = t + dwell
        # fixational drift: a slow random walk around the target
        dh = dv = 0.0
        while t < end:
            dh += rng.gauss(0.0, drift)
            dv += rng.gauss(0.0, drift)
            dh *= 0.9
            dv *= 0.9
            rows.append({"t": t, "h": _clip(cur[0] + dh), "v": _clip(cur[1] + dv), "event": "fixation"})
            t += dt
        if nxt is None:
            break
        dur = saccade_duration(amplitude_deg(cur, nxt))
        start = t
        while t < start + dur:
            s = min_jerk(min(1.0, (t - start) / dur))
            rows.append({
                "t": t,
                "h": cur[0] + (nxt[0] - cur[0]) * s,
                "v": cur[1] + (nxt[1] - cur[1]) * s,
                "event": "saccade",
            })
            t += dt
        cur = nxt
    return [dict(r, lid=0.0) for r in rows]


def random_scan(n=8, seed=0):
    rng = random.Random(seed)
    return [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]


def lissajous(t, ax=0.9, ay=0.7, fx=0.23, fy=0.31):
    return ax * math.sin(2 * math.pi * fx * t), ay * math.sin(2 * math.pi * fy * t + 0.6)


def pursuit_path(duration=6.0, fps=60, target=lissajous, gain=0.9, latency=0.1, catch_up_deg=2.0):
    """Eye follows a moving target with lag and reduced gain; large errors trigger catch-up."""
    dt = 1.0 / fps
    rows = []
    eye = list(target(0.0))
    jump = None
    t = 0.0
    while t <= duration:
        goal = target(max(0.0, t - latency))
        prev_goal = target(max(0.0, t - latency - dt))
        if jump is None:
            # smooth phase: velocity matching at reduced gain plus slow position correction
            for i in (0, 1):
                eye[i] += gain * (goal[i] - prev_goal[i]) + 0.04 * (goal[i] - eye[i])
            event = "pursuit"
            if amplitude_deg(eye, target(t)) > catch_up_deg:
                aim = target(t + 0.05)
                jump = (t, list(eye), aim, saccade_duration(amplitude_deg(eye, aim)))
        if jump is not None:
            t0, start, aim, dur = jump
            s = min_jerk(min(1.0, (t - t0) / dur))
            eye = [start[i] + (aim[i] - start[i]) * s for i in (0, 1)]
            event = "catch-up"
            if t - t0 >= dur:
                jump = None
        th, tv = target(t)
        rows.append({
            "t": t, "h": _clip(eye[0]), "v": _clip(eye[1]), "event": event,
            "target_h": th, "target_v": tv, "lid": 0.0,
        })
        t += dt
    return rows


def blink_profile(s):
    """Lid closure for a blink, s in [0, 1] across 300 ms."""
    close, shut = 100 / 300, 150 / 300
    if s < close:
        return min_jerk(s / close)
    if s < shut:
        return 1.0
    return 1.0 - min_jerk((s - shut) / (1.0 - shut))


def blink_track(rows, rate_per_min=15.0, length=0.30, seed=1):
    """Overlay blinks on an existing timeline. Gaze is left untouched during a blink."""
    if not rows:
        return rows
    rng = random.Random(seed)
    end = rows[-1]["t"]
    starts, t = [], rng.uniform(0.3, 1.2)
    while t < end:
        starts.append(t)
        t += rng.expovariate(rate_per_min / 60.0) + length
    for r in rows:
        lid = 0.0
        for s in starts:
            if s <= r["t"] < s + length:
                lid = blink_profile((r["t"] - s) / length)
        r["lid"] = max(r.get("lid", 0.0), lid)
        if lid > 0.0 and r["event"] == "fixation":
            r["event"] = "blink"
    return rows
