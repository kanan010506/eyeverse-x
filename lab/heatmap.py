"""
Gaze heatmaps.

target_heatmap()   where the eyes looked on the fixation plane (h, v space)
pupil_heatmap()    where the pupils sat in the image, blended over a base render
"""
import numpy as np
from PIL import Image

# perceptually even dark-to-bright ramp (inferno-like control points)
_RAMP = np.array([
    [0.00, 0.00, 0.02], [0.16, 0.04, 0.33], [0.42, 0.04, 0.43], [0.68, 0.17, 0.33],
    [0.89, 0.36, 0.13], [0.98, 0.65, 0.04], [0.99, 1.00, 0.64],
])


def colourise(x):
    x = np.clip(x, 0.0, 1.0)
    pos = x * (len(_RAMP) - 1)
    i = np.clip(pos.astype(int), 0, len(_RAMP) - 2)
    t = (pos - i)[..., None]
    return _RAMP[i] * (1 - t) + _RAMP[i + 1] * t


def splat(points, size, sigma, weights=None):
    """Sum of Gaussians at pixel positions (u, v) on a size x size grid."""
    grid = np.zeros((size, size))
    if len(points) == 0:
        return grid
    pts = np.asarray(points, dtype=float)
    w = np.ones(len(pts)) if weights is None else np.asarray(weights, dtype=float)
    ax = np.arange(size) + 0.5
    for (u, v), wt in zip(pts, w):
        gx = np.exp(-((ax - u) ** 2) / (2 * sigma * sigma))
        gy = np.exp(-((ax - v) ** 2) / (2 * sigma * sigma))
        grid += wt * np.outer(gy, gx)
    return grid


def target_heatmap(hv, size=360, sigma=9.0, dwell=None):
    """hv: list of (h, v) in [-1, 1]. Returns an RGB uint8 image (v up)."""
    pts = [((h + 1) / 2 * size, (1 - (v + 1) / 2) * size) for h, v in hv]
    g = splat(pts, size, sigma, dwell)
    g = g / g.max() if g.max() > 0 else g
    img = colourise(g ** 0.6)
    # faint grid at +/-0.5 and 0
    for f in (0.25, 0.5, 0.75):
        k = int(f * size)
        img[k, :, :] = img[k, :, :] * 0.6 + 0.12
        img[:, k, :] = img[:, k, :] * 0.6 + 0.12
    return (img * 255).astype(np.uint8)


def pupil_heatmap(base_path, uv, sigma=4.0, alpha=0.75):
    base = np.asarray(Image.open(base_path).convert("RGB"), dtype=float) / 255.0
    size = base.shape[0]
    g = splat(uv, size, sigma)
    g = g / g.max() if g.max() > 0 else g
    heat = colourise(g ** 0.5)
    a = (np.clip(g * 3.0, 0, 1) * alpha)[..., None]
    out = base * 0.55 * (1 - a) + heat * a
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)
