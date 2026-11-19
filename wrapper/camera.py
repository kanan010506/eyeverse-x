"""
Pinhole camera model shared by Blender code, tests and analysis tools.

Camera frame follows OpenCV: x right, y down, z forward.
Pixel coordinates start at the top-left corner of the image (u right, v down),
so the centre of the first pixel is (0.5, 0.5).
Square pixels, no skew, no distortion, principal point at the image centre.
"""
import math

import numpy as np

from . import config as cfg

# Blender cameras look down their local -Z with +Y up.
BLENDER_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def intrinsics(lens=cfg.LENS_MM, sensor=cfg.SENSOR_MM, size=cfg.IMAGE_SIZE):
    f = lens / sensor * size
    c = size / 2.0
    return np.array([[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]])


def pose(distance=cfg.CAMERA_DISTANCE, offset=(0.0, 0.0, 0.0), look_at=(0.0, 0.0, 0.0)):
    """Blender matrix_world of a camera at (0, -distance, 0) + offset aimed at look_at."""
    eye = np.array([0.0, -distance, 0.0]) + np.asarray(offset, dtype=float)
    forward = np.asarray(look_at, dtype=float) - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    m = np.eye(4)
    # Blender camera axes: +X right, +Y up, -Z forward
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, -forward, eye
    return m


def world_to_camera(camera_world):
    """4x4 OpenCV extrinsic from a Blender camera matrix_world."""
    return BLENDER_TO_CV @ np.linalg.inv(np.asarray(camera_world, dtype=float))


def project(points, K, w2c):
    """Nx3 world points -> (Nx2 pixels, N depths in metres)."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    pc = p @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    if np.any(z <= 0):
        raise ValueError("point behind the camera")
    uv = np.stack([K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * pc[:, 1] / z + K[1, 2]], axis=1)
    return uv, z


def pixel_ray(uv, K, w2c):
    """World-space origin and unit direction of the ray through pixel uv."""
    u, v = uv
    d_cam = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    r = w2c[:3, :3]
    origin = -r.T @ w2c[:3, 3]
    d = r.T @ d_cam
    return origin, d / np.linalg.norm(d)


def ray_sphere(origin, direction, centre, radius):
    """Nearest positive hit of a ray on a sphere, or None."""
    oc = np.asarray(origin) - np.asarray(centre)
    b = float(np.dot(oc, direction))
    c = float(np.dot(oc, oc)) - radius * radius
    disc = b * b - c
    if disc < 0:
        return None
    t = -b - math.sqrt(disc)
    if t < 0:
        t = -b + math.sqrt(disc)
    return None if t < 0 else np.asarray(origin) + t * np.asarray(direction)


def describe(K, w2c, size=cfg.IMAGE_SIZE, lens=cfg.LENS_MM, sensor=cfg.SENSOR_MM):
    """camera_matrix block written into every metadata record."""
    fov = 2.0 * math.degrees(math.atan(sensor / (2.0 * lens)))
    return {
        "K": np.round(K, 6).tolist(),
        "world_to_camera": np.round(w2c, 9).tolist(),
        "image_size": [size, size],
        "lens_mm": lens,
        "sensor_mm": sensor,
        "fov_deg": round(fov, 4),
        "convention": "OpenCV (x right, y down, z forward); pixel origin at top-left corner",
    }
