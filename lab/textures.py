"""
Procedural texture maps for the eyes and skin (numpy only, runs inside Blender too).

Iris and sclera maps are stored in polar layout so the shader can stretch the
iris when the pupil changes size:
    x (columns)  azimuth, 0..2*pi around the optical axis
    y (rows)     radius, pupil margin at row 0 -> limbus at the last row

All maps return float32 RGB(A) arrays in [0, 1], origin at the bottom row the
way Blender stores image pixels.
"""
import numpy as np

IRIS_PALETTES = {
    # (deep stroma, mid stroma, highlights, collarette)
    "brown": ((0.06, 0.028, 0.010), (0.20, 0.095, 0.035), (0.42, 0.24, 0.09), (0.36, 0.20, 0.07)),
    "dark_brown": ((0.05, 0.025, 0.012), (0.16, 0.08, 0.035), (0.30, 0.17, 0.07), (0.24, 0.12, 0.05)),
    "hazel": ((0.12, 0.08, 0.03), (0.35, 0.28, 0.10), (0.55, 0.50, 0.22), (0.50, 0.30, 0.10)),
    "green": ((0.06, 0.10, 0.06), (0.22, 0.34, 0.20), (0.45, 0.58, 0.36), (0.45, 0.35, 0.12)),
    "blue": ((0.07, 0.12, 0.20), (0.25, 0.40, 0.58), (0.55, 0.70, 0.85), (0.48, 0.52, 0.48)),
    "grey": ((0.12, 0.13, 0.14), (0.33, 0.36, 0.38), (0.62, 0.65, 0.66), (0.45, 0.44, 0.40)),
}


def _rng(seed):
    return np.random.default_rng(seed)


def periodic_noise(shape, sy, sx, rng):
    """Smooth noise that wraps on both axes; sy, sx are blur lengths in pixels per axis."""
    h, w = shape
    white = rng.standard_normal(shape)
    fy = np.fft.fftfreq(h)[:, None] * sy
    fx = np.fft.fftfreq(w)[None, :] * sx
    spectrum = np.fft.fft2(white) * np.exp(-(np.pi ** 2) * (fx * fx + fy * fy))
    out = np.real(np.fft.ifft2(spectrum))
    out -= out.mean()
    return out / (out.std() + 1e-9)


def _lerp(a, b, t):
    t = np.clip(t, 0.0, 1.0)[..., None]
    return np.asarray(a) * (1.0 - t) + np.asarray(b) * t


def iris(colour="brown", seed=0, width=1024, height=256):
    """Polar iris map: radial stroma fibres, collarette, crypts, limbal ring."""
    rng = _rng(seed)
    deep, mid, light, collar = (np.array(c) for c in IRIS_PALETTES[colour])
    r = np.linspace(0.0, 1.0, height)[:, None] * np.ones((1, width))

    # radial fibres: high frequency in azimuth, very low along the radius
    fibres = periodic_noise((height, width), 60.0, 2.2, rng)
    fine = periodic_noise((height, width), 14.0, 1.2, rng)
    blotch = periodic_noise((height, width), 30.0, 60.0, rng)

    tone = 0.5 + 0.22 * fibres + 0.12 * fine + 0.10 * blotch
    base = _lerp(deep, mid, tone * 1.1)
    base = _lerp(base, light, np.clip(tone - 0.55, 0.0, 1.0) * 2.2)

    # collarette: a wavy brighter band about a third of the way out
    wobble = 0.03 * periodic_noise((1, width), 1.0, 40.0, rng)
    band = np.exp(-((r - (0.32 + wobble)) / 0.05) ** 2)
    base = _lerp(base, collar, band * 0.55)

    # crypts: darker pits just outside the collarette
    crypt_field = periodic_noise((height, width), 7.0, 9.0, rng)
    crypts = np.clip(crypt_field - 1.4, 0.0, 1.0) * np.exp(-((r - 0.48) / 0.18) ** 2)
    base *= (1.0 - 0.6 * crypts)[..., None]

    # contraction furrows near the limbus
    furrows = 0.5 + 0.5 * np.cos(r * np.pi * 9.0 + 0.5 * blotch)
    base *= (1.0 - 0.08 * furrows * (r > 0.65))[..., None]

    # pupillary ruff and limbal ring
    ruff = np.exp(-(r / 0.035) ** 2)
    base = _lerp(base, deep * 0.4, ruff)
    limbus = np.clip((r - 0.80) / 0.20, 0.0, 1.0) ** 1.6
    base = _lerp(base, deep * 0.25, limbus)

    return np.clip(base, 0.0, 1.0).astype(np.float32)


def sclera(seed=0, width=1024, height=256):
    """Polar sclera map from the limbus (row 0) to the back of the globe (last row)."""
    rng = _rng(seed + 101)
    t = np.linspace(0.0, 1.0, height)[:, None] * np.ones((1, width))
    tint = periodic_noise((height, width), 20.0, 40.0, rng)
    white = np.array([0.86, 0.82, 0.78])
    base = white * (1.0 + 0.025 * tint[..., None])
    # slightly yellower and darker towards the canthi
    base *= (1.0 - 0.10 * t)[..., None]
    base[..., 2] *= 1.0 - 0.06 * t

    veins = np.zeros((height, width))
    for scale, weight in ((40.0, 1.0), (16.0, 0.6)):
        n = periodic_noise((height, width), scale * 2.0, scale, rng)
        veins += weight * np.clip(1.0 - np.abs(n) * 9.0, 0.0, 1.0)
    veins *= np.clip((t - 0.12) / 0.5, 0.0, 1.0) ** 1.2
    red = np.array([0.55, 0.12, 0.10])
    base = _lerp(base, red, veins * 0.35)
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def skin_detail(seed=0, size=1024):
    """Tileable pore/micro-variation map; R = albedo variation, G = pores, B = roughness."""
    rng = _rng(seed + 7)
    blotch = periodic_noise((size, size), 40.0, 40.0, rng)
    pores = periodic_noise((size, size), 2.0, 2.0, rng)
    pores = np.clip(-pores - 1.2, 0.0, 1.0)
    rough = periodic_noise((size, size), 8.0, 8.0, rng)
    out = np.stack([
        0.5 + 0.18 * blotch,
        pores,
        0.5 + 0.2 * rough,
    ], axis=-1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def with_alpha(rgb):
    alpha = np.ones(rgb.shape[:2] + (1,), dtype=np.float32)
    return np.concatenate([rgb, alpha], axis=-1)


def flatten_iris(polar, pupil_frac=0.33, size=512, sclera_map=None):
    """Front view of an iris map, for previews and the dashboard."""
    h, w = polar.shape[:2]
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    rad = np.sqrt(x * x + y * y)
    az = (np.arctan2(x, y) % (2 * np.pi)) / (2 * np.pi)
    col = np.clip((az * w).astype(int), 0, w - 1)
    rn = (rad - pupil_frac) / (1.0 - pupil_frac)
    row = np.clip((rn * (h - 1)).astype(int), 0, h - 1)
    img = polar[row, col]
    img[rad < pupil_frac] = 0.012
    if sclera_map is not None:
        srow = np.clip(((rad - 1.0) / 0.4 * (h - 1)).astype(int), 0, h - 1)
        img = np.where((rad >= 1.0)[..., None], sclera_map[srow, col], img)
    else:
        img[rad >= 1.0] = 1.0
    return img
