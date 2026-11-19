"""
Appearance identities and scene randomisation (Blender only).

An identity changes what the face looks like without touching the calibrated
geometry: iris colour and pattern, skin tone, brow colour and density.
A scene sample adds lighting, exposure, camera jitter, pupil size, lid closure,
glasses and cornea refraction on top of an identity.
"""
import math
import random

import bpy
import numpy as np

from lab import shading, textures

PRESETS = {
    "aarav": {"iris": "dark_brown", "skin": "brown", "brow": (0.018, 0.012, 0.009), "seed": 21},
    "mei": {"iris": "brown", "skin": "light", "brow": (0.020, 0.013, 0.010), "seed": 34},
    "lena": {"iris": "blue", "skin": "light", "brow": (0.10, 0.065, 0.035), "seed": 55},
    "omar": {"iris": "hazel", "skin": "tan", "brow": (0.022, 0.015, 0.010), "seed": 89},
    "zuri": {"iris": "dark_brown", "skin": "dark", "brow": (0.012, 0.009, 0.008), "seed": 144},
    "sofia": {"iris": "green", "skin": "fair", "brow": (0.06, 0.040, 0.022), "seed": 233},
}
LIGHTS = ("Key", "Fill", "Catch", "Top")
_base = {}


def _remember():
    if _base:
        return
    for name in LIGHTS:
        light = bpy.data.objects.get(name)
        if light:
            _base[name] = (light.data.energy, tuple(light.data.color), tuple(light.location))
    _base["exposure"] = bpy.context.scene.view_settings.exposure


def apply(identity):
    """identity: dict with iris, skin, brow, seed (see PRESETS)."""
    shading.replace_pixels(bpy.data.images["Iris"], textures.iris(identity["iris"], seed=identity["seed"]))
    nodes = bpy.data.materials["Skin"].node_tree.nodes
    tone = identity["skin"]
    if isinstance(tone, str):
        tone = shading.SKIN_TONES[tone]
    nodes["SkinTone"].outputs[0].default_value = (*tone, 1.0)
    nodes["BrowColour"].outputs[0].default_value = (*identity["brow"], 1.0)


def studio():
    """The identity scene.blend was built with."""
    import json
    info = json.loads(bpy.context.scene.get("eyeverse", "{}"))
    return {"iris": info.get("iris", "brown"), "skin": info.get("skin", "medium"),
            "brow": (0.030, 0.019, 0.012), "seed": 11}


def kelvin_rgb(k):
    """Rough blackbody tint, normalised to the green channel."""
    t = k / 100.0
    r = 1.0 if t <= 66 else min(1.0, 1.292936 * (t - 60) ** -0.1332047592)
    g = (0.39008 * math.log(t) - 0.63184) if t <= 66 else 1.129891 * (t - 60) ** -0.0755148492
    b = 1.0 if t >= 66 else (0.0 if t <= 19 else 0.543206789 * math.log(t - 10) - 1.19625409)
    g = max(min(g, 1.0), 1e-3)
    return (min(r / g, 2.0), 1.0, max(b, 0.0) / g)


def random_identity(rng):
    skin = np.array(shading.SKIN_TONES[rng.choice(list(shading.SKIN_TONES))])
    skin = np.clip(skin * rng.uniform(0.9, 1.1, 3), 0.02, 0.95)
    brow = np.array((0.025, 0.017, 0.011)) * rng.uniform(0.5, 3.5)
    return {"iris": rng.choice(list(textures.IRIS_PALETTES)), "skin": tuple(skin),
            "brow": tuple(brow), "seed": int(rng.integers(0, 10_000))}


def randomise_scene(eyes, rng, strength=1.0):
    """Domain randomisation. Returns a dict describing everything that was changed."""
    _remember()
    scene = bpy.context.scene
    info = {}
    kelvin = float(rng.uniform(3200, 7500))
    tint = kelvin_rgb(kelvin)
    for name in LIGHTS:
        light = bpy.data.objects.get(name)
        if light is None:
            continue
        energy, colour, loc = _base[name]
        light.data.energy = energy * float(rng.uniform(0.45, 1.7) ** strength)
        light.data.color = tuple(c * t for c, t in zip(colour, tint))
        light.location = tuple(np.array(loc) + rng.normal(0, 0.05 * strength, 3))
        light.rotation_euler = (light.location.copy() * -1).to_track_quat("-Z", "Y").to_euler()
    scene.view_settings.exposure = _base["exposure"] + float(rng.normal(0, 0.35 * strength))
    info["kelvin"] = round(kelvin)
    info["exposure"] = round(scene.view_settings.exposure, 3)

    luminance = float(10 ** rng.uniform(-1.0, 3.5))
    info["luminance_cd_m2"] = round(luminance, 3)
    info["pupil_mm"] = round(eyes.set_pupil(luminance=luminance), 3)

    closure = 0.0
    if rng.random() < 0.18:
        closure = float(rng.uniform(0.2, 0.75))
    eyes.set_lids(closure)
    info["lid_closure"] = round(closure, 3)

    wear = rng.random() < 0.2
    for name in ("Glasses.Frame", "Glasses.Lens"):
        obj = bpy.data.objects.get(name)
        if obj:
            obj.hide_render = not wear
    info["glasses"] = bool(wear)

    refract = 1.0 if rng.random() < 0.3 else 0.0
    for side in ("L", "R"):
        cornea = bpy.data.objects.get(f"Cornea.{side}")
        if cornea:
            cornea["cornea_refraction"] = refract
            cornea.update_tag()
    info["cornea_refraction"] = bool(refract)
    return info


def restore(eyes):
    """Back to the calibrated studio state used for the official renders."""
    _remember()
    for name in LIGHTS:
        light = bpy.data.objects.get(name)
        if light is None:
            continue
        energy, colour, loc = _base[name]
        light.data.energy, light.data.color, light.location = energy, colour, loc
        light.rotation_euler = (light.location.copy() * -1).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.view_settings.exposure = _base["exposure"]
    for name in ("Glasses.Frame", "Glasses.Lens"):
        obj = bpy.data.objects.get(name)
        if obj:
            obj.hide_render = True
    for side in ("L", "R"):
        cornea = bpy.data.objects.get(f"Cornea.{side}")
        if cornea:
            cornea["cornea_refraction"] = 0.0
    apply(studio())
    eyes.reset()


def jitter_camera(cam, base_matrix, rng, metres=0.004, degrees=0.6):
    from mathutils import Euler, Matrix, Vector
    m = Matrix(base_matrix)
    shift = Vector(rng.normal(0, metres, 3))
    rot = Euler(tuple(math.radians(a) for a in rng.normal(0, degrees, 3))).to_matrix().to_4x4()
    loc = m.translation + shift
    cam.matrix_world = Matrix.Translation(loc) @ rot @ m.to_3x3().to_4x4()


def seeded(seed):
    return np.random.default_rng(seed), random.Random(seed)
