"""
Image-side checks: find the pupils in a render and turn them back into 3D gaze.

detect_pupil()     dark-blob detector inside a window around the projected eye centre
reconstruct()      pixel -> camera ray -> eyeball sphere -> gaze direction

Needs numpy and Pillow only.
"""
import math

import numpy as np
from PIL import Image

from wrapper import camera, gaze


def load_grey(path):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _components(mask):
    """4-connected components; returns a label image and the number of labels."""
    try:
        from scipy import ndimage
        return ndimage.label(mask)
    except ImportError:
        pass
    labels = np.zeros(mask.shape, dtype=np.int32)
    n = 0
    h, w = mask.shape
    for y0, x0 in zip(*np.nonzero(mask)):
        if labels[y0, x0]:
            continue
        n += 1
        stack = [(y0, x0)]
        labels[y0, x0] = n
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not labels[yy, xx]:
                    labels[yy, xx] = n
                    stack.append((yy, xx))
    return labels, n


def _circle_fit(xs, ys):
    a = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    b = xs * xs + ys * ys
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0], sol[1]
    return cx, cy, float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))


def detect_pupil(grey, around, window=90, pupil_px=None):
    """Pupil centre near `around` (u, v) from the image alone.

    Dark compact blob -> its boundary -> least-squares circle, ignoring boundary pixels that
    touch a bright corneal reflection (those edges belong to the glint, not the pupil)."""
    h, w = grey.shape
    u0, v0 = int(round(around[0])), int(round(around[1]))
    x0, x1 = max(0, u0 - window), min(w, u0 + window)
    y0, y1 = max(0, v0 - window), min(h, v0 + window)
    # an eye outside the frame (side cameras) has nothing to find; a negative end
    # would otherwise wrap around as a numpy slice
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    roi = grey[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = (xx - around[0]) ** 2 + (yy - around[1]) ** 2 < window ** 2
    vals = roi[disk]
    lo = float(np.percentile(vals, 1.0))
    thr = lo + 0.3 * (float(np.percentile(vals, 30)) - lo)
    mask = (roi < thr) & disk
    try:
        from scipy import ndimage
        mask = ndimage.binary_opening(mask, iterations=1)
    except ImportError:
        pass
    labels, n = _components(mask)
    best, best_score = None, -1.0
    for k in range(1, n + 1):
        blob = labels == k
        area = int(blob.sum())
        if area < 20:
            continue
        ys, xs = np.nonzero(blob)
        size_fit = 1.0
        if pupil_px:
            size_fit = math.exp(-abs(math.log(max(area, 1) / (math.pi * pupil_px ** 2))))
        darkness = 1.0 - float(roi[ys, xs].mean())
        dist = math.hypot(xs.mean() + x0 - around[0], ys.mean() + y0 - around[1]) / window
        score = area ** 0.5 * darkness * size_fit * (1.0 - 0.5 * dist)
        if score > best_score:
            best_score, best = score, blob
    if best is None:
        return None
    # boundary pixels whose outside neighbours are iris, not glint
    pad = np.pad(best, 1)
    inner = pad[1:-1, 1:-1]
    edge = inner & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    ey, ex = np.nonzero(edge)
    bright = np.zeros(len(ex), dtype=bool)
    glint = float(np.percentile(vals, 30)) * 1.6 + 0.08
    for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)):
        yy2 = np.clip(ey + dy, 0, roi.shape[0] - 1)
        xx2 = np.clip(ex + dx, 0, roi.shape[1] - 1)
        bright |= roi[yy2, xx2] > glint
    keep = ~bright
    ys, xs = np.nonzero(best)
    area = len(xs)
    cx, cy, r = xs.mean(), ys.mean(), math.sqrt(area / math.pi)
    method = "centroid"
    if keep.sum() >= 12:
        fx, fy, fr = _circle_fit(ex[keep].astype(float), ey[keep].astype(float))
        if pupil_px is None or 0.5 * pupil_px < fr < 1.6 * pupil_px:
            cx, cy, r, method = fx, fy, fr, "edge circle"
    return {"uv": [float(cx + x0 + 0.5), float(cy + y0 + 0.5)], "area": area, "radius_px": round(r, 2),
            "edge_points": int(keep.sum()), "method": method}


def reconstruct(uv, K, w2c, centre, radius):
    """Gaze direction implied by a pupil pixel, given the calibrated eyeball sphere."""
    origin, d = camera.pixel_ray(uv, np.asarray(K), np.asarray(w2c))
    hit = camera.ray_sphere(origin, d, centre, radius)
    if hit is None:
        return None
    return gaze.normalise(hit - np.asarray(centre))


def eye_seed(record, side):
    """Calibration-only prior: the projected eyeball centre, and the expected iris and
    pupil radii in pixels. No ground-truth pupil position is used."""
    K = np.array(record["camera_matrix"]["K"])
    w2c = np.array(record["camera_matrix"]["world_to_camera"])
    centre = np.array(record[f"eye_center_{side}"])
    uv, depth = camera.project(centre, K, w2c)
    radius = float(np.linalg.norm(np.array(record[f"pupil_center_3D_{side}"]) - centre))
    px_per_m = K[0, 0] / float(depth[0] - radius)
    k = radius / 0.012
    iris_px = 0.49 * radius * px_per_m
    pupil_px = record.get("pupil_diameter_mm", 4.0) / 2000.0 * k * px_per_m
    return uv[0], iris_px, pupil_px


def evaluate(record, image_path):
    """Detected vs ground-truth pupils and reconstructed vs true gaze for one record."""
    grey = load_grey(image_path)
    K = np.array(record["camera_matrix"]["K"])
    w2c = np.array(record["camera_matrix"]["world_to_camera"])
    scale = grey.shape[1] / record["camera_matrix"]["image_size"][0]
    out = {}
    for side in gaze.SIDES:
        seed, iris_px, pupil_px = eye_seed(record, side)
        found = detect_pupil(grey, seed * scale, window=max(8, int(iris_px * 0.95 * scale)),
                             pupil_px=pupil_px * scale)
        gt = np.array(record[f"pupil_center_2D_{side}"])
        if found is None:
            out[side] = {"found": False}
            continue
        uv = np.array(found["uv"]) / scale
        centre = np.array(record[f"eye_center_{side}"])
        radius = float(np.linalg.norm(np.array(record[f"pupil_center_3D_{side}"]) - centre))
        rec = reconstruct(uv, K, w2c, centre, radius)
        err_deg = gaze.angle_between(rec, record[f"gaze_vector_{side}"]) if rec is not None else None
        out[side] = {
            "found": True,
            "detected": [round(float(uv[0]), 3), round(float(uv[1]), 3)],
            "error_px": round(float(np.linalg.norm(uv - gt)), 3),
            "reconstructed_gaze": None if rec is None else [round(float(a), 6) for a in rec],
            "angular_error_deg": None if err_deg is None else round(err_deg, 4),
            "radius_px": found["radius_px"],
            "method": found["method"],
        }
    return out
