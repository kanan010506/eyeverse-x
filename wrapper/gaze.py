"""
Gaze geometry: fixation targets, eye rotations, pupil positions.

No Blender imports here, so the same numbers are used by the scene, the tests,
the report and the dashboard.

Each eye turns about its own centre towards the fixation point. The rotation
is the shortest arc from the primary direction (0, -1, 0) to the line of
sight, which keeps the rotation axis in the frontal plane (Listing's law), so
the eyes never pick up false torsion on oblique gaze.
"""
import math

import numpy as np

from . import config as cfg

FORWARD = np.array([0.0, -1.0, 0.0])
SIDES = ("L", "R")


def check_input(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(value) or abs(value) > 1.0 + 1e-9:
        raise ValueError(f"{name} must lie in [-1, 1], got {value}")
    return min(1.0, max(-1.0, value))


def sequence(n=cfg.SEQUENCE_LENGTH):
    """Official clockwise sampling: theta_k = 2*pi*k/n, h = sin, v = cos, frame 0 at the top."""
    rows = []
    for k in range(n):
        theta = 2.0 * math.pi * k / n
        h, v = math.sin(theta), math.cos(theta)
        # snap float dust so frame 0 is exactly (0, 1)
        h = 0.0 if abs(h) < 1e-12 else h
        v = 0.0 if abs(v) < 1e-12 else v
        rows.append({"frame": k, "theta": theta, "h": h, "v": v})
    return rows


def eye_centres(ipd=cfg.IPD):
    half = ipd / 2.0
    return {"L": np.array([half, 0.0, 0.0]), "R": np.array([-half, 0.0, 0.0])}


def check_distance(distance):
    try:
        distance = float(distance)
    except (TypeError, ValueError):
        raise ValueError(f"distance must be a number, got {distance!r}") from None
    lo, hi = cfg.DISTANCE_LIMITS
    if not math.isfinite(distance) or not lo <= distance <= hi:
        raise ValueError(f"fixation distance must lie in [{lo}, {hi}] m, got {distance}")
    return distance


def fixation_point(h, v, distance=cfg.FIXATION_DISTANCE, span=cfg.FIXATION_RANGE):
    h = check_input("h", h)
    v = check_input("v", v)
    distance = check_distance(distance)
    return np.array([h * span, -distance, v * span])


def normalise(vec):
    vec = np.asarray(vec, dtype=float)
    length = np.linalg.norm(vec)
    if length < 1e-12:
        raise ValueError("zero-length vector")
    return vec / length


# -- rotations --------------------------------------------------------------
def arc_quaternion(a, b):
    """Unit quaternion (w, x, y, z) taking unit vector a onto unit vector b by the shortest arc."""
    a, b = normalise(a), normalise(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot < -0.999999:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 0.0, 1.0])
        axis = normalise(axis)
        return np.array([0.0, *axis])
    axis = np.cross(a, b)
    q = np.array([1.0 + dot, *axis])
    return q / np.linalg.norm(q)


def quat_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_euler_xyz(m):
    """Blender 'XYZ' Euler angles (R = Rz @ Ry @ Rx), radians."""
    sy = -m[2, 0]
    sy = min(1.0, max(-1.0, sy))
    ry = math.asin(sy)
    if abs(sy) < 0.999999:
        rx = math.atan2(m[2, 1], m[2, 2])
        rz = math.atan2(m[1, 0], m[0, 0])
    else:  # gimbal lock, not reachable inside the +/-5 cm range
        rx = math.atan2(-m[1, 2], m[1, 1])
        rz = 0.0
    return np.array([rx, ry, rz])


def euler_matrix_xyz(e):
    rx, ry, rz = e
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry),
                              math.sin(ry), math.cos(rz), math.sin(rz))
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def direction_angles(d):
    """Horizontal and vertical angle of a gaze vector in degrees (+ = image right / up)."""
    d = normalise(d)
    yaw = math.degrees(math.atan2(d[0], -d[1]))
    pitch = math.degrees(math.asin(np.clip(d[2], -1.0, 1.0)))
    return yaw, pitch


# -- eyes ---------------------------------------------------------------------
def aim(centre, target, radius=cfg.EYE_RADIUS):
    """Everything about one eye looking at one point."""
    centre = np.asarray(centre, dtype=float)
    direction = normalise(np.asarray(target, dtype=float) - centre)
    q = arc_quaternion(FORWARD, direction)
    m = quat_matrix(q)
    yaw, pitch = direction_angles(direction)
    return {
        "centre": centre,
        "direction": direction,
        "quaternion": q,
        "euler": matrix_euler_xyz(m),
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "pupil": centre + direction * radius,
    }


def solve(h, v, distance=cfg.FIXATION_DISTANCE, centres=None, radius=cfg.EYE_RADIUS):
    """Binocular solution for normalised input (h, v)."""
    centres = centres or eye_centres()
    target = fixation_point(h, v, distance)
    eyes = {side: aim(centres[side], target, radius) for side in SIDES}
    return {
        "h": check_input("h", h),
        "v": check_input("v", v),
        "distance": float(distance),
        "target": target,
        "eyes": eyes,
        "vergence_deg": vergence(eyes["L"]["direction"], eyes["R"]["direction"]),
    }


def vergence(dir_l, dir_r):
    c = float(np.clip(np.dot(normalise(dir_l), normalise(dir_r)), -1.0, 1.0))
    return math.degrees(math.acos(c))


def angle_between(a, b):
    c = float(np.clip(np.dot(normalise(a), normalise(b)), -1.0, 1.0))
    return math.degrees(math.acos(c))


# -- pupil -------------------------------------------------------------------
def pupil_diameter(luminance):
    """Moon & Spencer (1944): D = 4.9 - 3 tanh(0.4 log10 L), L in cd/m^2, D in mm."""
    luminance = max(float(luminance), 1e-4)
    d = 4.9 - 3.0 * math.tanh(0.4 * math.log10(luminance))
    lo, hi = cfg.PUPIL_LIMITS_MM
    return min(hi, max(lo, d))
