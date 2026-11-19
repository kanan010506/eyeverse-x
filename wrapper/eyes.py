"""
Eye control for scene.blend.

Inside Blender (scene.blend open):

    import eyes                      # or: from wrapper import eyes
    eyes.move(0.5, -0.2)             # h, v in [-1, 1]
    eyes.render("renders/look.png")
    eyes.reset()

From a shell:

    blender -b scene.blend -P wrapper/eyes.py -- --move 1 0 --out renders/right.png
    blender -b scene.blend -P wrapper/eyes.py -- --sequence     # 10 frames + metadata.json

h = +1 looks towards image right, v = +1 looks up. The fixation point is
(h * 5 cm, 35 cm forward, v * 5 cm) from the eye midpoint and both eyes converge
on it. The eyeballs are children of the eye.L / eye.R bones; move() poses the
bones, so the armature is the single source of truth for the gaze.

Extras on the Eyes object (not used by the official renders):
    look(left=(h, v), right=(h, v))     independent eyes
    move(h, v, distance=0.6)            vergence for any fixation distance
    set_pupil(mm) / set_pupil(luminance=cd_m2)
    set_lids(closure)                   0 open .. 1 shut, gaze unchanged
"""
import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import bpy  # noqa: E402
from mathutils import Matrix, Quaternion, Vector  # noqa: E402

from wrapper import config as cfg, gaze  # noqa: E402
from wrapper.safety import safe_output  # noqa: E402


def use_gpu(scene=None):
    """CPU by default (reproducible everywhere); EYEVERSE_DEVICE=GPU picks the best GPU backend."""
    scene = scene or bpy.context.scene
    if scene.render.engine != "CYCLES":
        return None
    if os.environ.get("EYEVERSE_DEVICE", "CPU").upper() != "GPU":
        scene.cycles.device = "CPU"
        return None
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError):
        return None
    for kind in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        devices = [d for d in prefs.devices if d.type == kind]
        if devices:
            for d in prefs.devices:
                d.use = d.type in (kind, "CPU") if kind == "METAL" else d.type == kind
            scene.cycles.device = "GPU"
            return kind
    scene.cycles.device = "CPU"
    return None


class Eyes:
    def __init__(self):
        self.scene = bpy.context.scene
        self.device = use_gpu(self.scene)
        objects = bpy.data.objects
        self.rig = objects.get(cfg.RIG_NAME)
        self.target = objects.get(cfg.TARGET_NAME)
        self.globes = {s: objects.get(cfg.EYEBALL_NAMES[s]) for s in gaze.SIDES}
        missing = [n for n, o in [(cfg.RIG_NAME, self.rig), (cfg.TARGET_NAME, self.target),
                                  *[(cfg.EYEBALL_NAMES[s], g) for s, g in self.globes.items()]] if o is None]
        if missing:
            raise RuntimeError(f"scene is missing {missing}; open scene.blend or run tools/build_scene.py")
        self.bones = {s: self.rig.pose.bones[cfg.EYE_BONES[s]] for s in gaze.SIDES}
        for pb in self.bones.values():
            pb.rotation_mode = "QUATERNION"
        self.lids = {s: (objects.get(f"LidUpper.{s}"), objects.get(f"LidLower.{s}")) for s in gaze.SIDES}
        self.distance = cfg.FIXATION_DISTANCE
        self.h = self.v = 0.0
        self.pupil_mm = cfg.PUPIL_DIAMETER_MM
        self.closure = 0.0
        self._clear_pose()
        self.centres = {s: self.centre(s) for s in gaze.SIDES}
        self.radius = float(self.globes["L"].get("radius", cfg.EYE_RADIUS))
        self.reset()

    # -- geometry readback ------------------------------------------------------
    def centre(self, side):
        bpy.context.view_layer.update()
        return Vector(self.globes[side].matrix_world.translation)

    def gaze_vector(self, side):
        """Current line of sight read back from the posed eyeball (its local -Y axis)."""
        bpy.context.view_layer.update()
        m = self.globes[side].matrix_world.to_3x3().normalized()
        return -Vector(m.col[1]).normalized()

    def rotation(self, side):
        """World-frame rotation of one eye relative to straight ahead (3x3)."""
        bpy.context.view_layer.update()
        return self.globes[side].matrix_world.to_3x3().normalized() @ self._rest[side].inverted()

    def pupil_point(self, side):
        return self.centre(side) + self.gaze_vector(side) * self.radius

    # -- posing -----------------------------------------------------------------
    def _clear_pose(self):
        for pb in self.bones.values():
            pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        self._rest = {s: self.globes[s].matrix_world.to_3x3().normalized() for s in gaze.SIDES}

    def _turn(self, side, world_q):
        """Pose an eye bone so the eye rotates by world_q about its own centre."""
        pb = self.bones[side]
        rest = pb.bone.matrix_local.to_3x3().normalized()
        arm = self.rig.matrix_world.to_3x3().normalized()
        r = Quaternion(world_q).to_matrix()
        pb.rotation_quaternion = (rest.inverted() @ arm.inverted() @ r @ arm @ rest).to_quaternion()

    def _aim(self, side, point):
        c = self.centres[side]
        q = gaze.arc_quaternion(gaze.FORWARD, [point[i] - c[i] for i in range(3)])
        self._turn(side, q)

    def move(self, h, v, distance=None):
        """Both eyes fixate (h * 5 cm, distance, v * 5 cm); returns the analytic solution."""
        distance = self.distance if distance is None else float(distance)
        target = gaze.fixation_point(h, v, distance)
        self.h, self.v, self.distance = float(h), float(v), distance
        self.target.location = tuple(target)
        for side in gaze.SIDES:
            self._aim(side, target)
        bpy.context.view_layer.update()
        self._place_lids()
        self._place_overlay()
        return gaze.solve(h, v, distance, centres={s: tuple(c) for s, c in self.centres.items()},
                          radius=self.radius)

    def look(self, left=(0.0, 0.0), right=(0.0, 0.0), distance=None):
        """Independent gaze: each eye gets its own (h, v) target on the fixation plane."""
        distance = self.distance if distance is None else float(distance)
        for side, hv in (("L", left), ("R", right)):
            self._aim(side, gaze.fixation_point(hv[0], hv[1], distance))
        mid = [(left[i] + right[i]) / 2.0 for i in (0, 1)]
        self.h, self.v = mid
        self.target.location = tuple(gaze.fixation_point(mid[0], mid[1], distance))
        bpy.context.view_layer.update()
        self._place_lids()
        self._place_overlay()

    def reset(self):
        self.distance = cfg.FIXATION_DISTANCE
        self.set_pupil(cfg.PUPIL_DIAMETER_MM)
        self.set_lids(0.0)
        return self.move(0.0, 0.0)

    # -- pupil, lids, overlay -----------------------------------------------------
    def set_pupil(self, mm=None, luminance=None):
        for name, val in (("mm", mm), ("luminance", luminance)):
            if val is not None and (not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0):
                raise ValueError(f"{name} must be a non-negative number, got {val!r}")
        if mm is None:
            mm = gaze.pupil_diameter(luminance if luminance is not None else 100.0)
        lo, hi = cfg.PUPIL_LIMITS_MM
        self.pupil_mm = min(hi, max(lo, float(mm)))
        for g in self.globes.values():
            g["pupil_ratio"] = (self.pupil_mm / 2000.0) / cfg.EYE_RADIUS
            g.update_tag()
        return self.pupil_mm

    def set_lids(self, closure):
        closure = float(closure)
        if not math.isfinite(closure):
            raise ValueError("closure must be a finite number")
        self.closure = min(1.0, max(0.0, closure))
        self._place_lids()

    def _place_lids(self):
        for side, (upper, lower) in self.lids.items():
            if upper is None or lower is None:
                continue
            top, bottom = upper["margin_deg"], lower["margin_deg"]
            # follow vertical gaze a little, as real lids do
            follow = 0.6 * math.degrees(math.asin(max(-1.0, min(1.0, self.gaze_vector(side).z))))
            meet = bottom + 0.3 * (top - bottom)
            u = top + follow + (meet - top - follow) * self.closure
            l_ = bottom + 0.3 * follow + (meet - bottom - 0.3 * follow) * self.closure
            upper.rotation_euler = (math.radians(upper["edge_deg"] - u), 0.0, 0.0)
            lower.rotation_euler = (math.radians(lower["edge_deg"] - l_), 0.0, 0.0)

    def _place_overlay(self):
        target = Vector(self.target.location)
        for side in gaze.SIDES:
            ray = bpy.data.objects.get(f"GazeRay.{side}")
            if ray is None:
                continue
            c = self.centres[side]
            d = target - c
            ray.location = c
            ray.rotation_mode = "QUATERNION"
            ray.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
            ray.scale = (1.0, 1.0, d.length)
        marker = bpy.data.objects.get("FixationMarker")
        if marker is not None:
            marker.location = target

    # -- output -----------------------------------------------------------------
    def render(self, path=None, camera=None):
        if path is None:
            path = cfg.RENDER_DIR / f"gaze_h{self.h:+.2f}_v{self.v:+.2f}.png"
        path = safe_output(path)
        scene = self.scene
        keep = scene.camera
        if camera is not None:
            cam = bpy.data.objects.get(camera)
            if cam is None or cam.type != "CAMERA":
                raise ValueError(f"no camera called {camera!r} in the scene")
            scene.camera = cam
        scene.render.filepath = str(path)
        try:
            bpy.ops.render.render(write_still=True)
        finally:
            scene.camera = keep
        if not path.exists():
            raise RuntimeError(f"render did not produce {path}")
        return path

    def record(self):
        from wrapper import truth
        return truth.record(self)


# -- module-level API --------------------------------------------------------------
_eyes = None


def get():
    global _eyes
    if _eyes is None or _eyes.rig is None or _eyes.rig.name not in bpy.data.objects:
        _eyes = Eyes()
    return _eyes


def move(h, v, distance=None):
    return get().move(h, v, distance)


def reset():
    return get().reset()


def render(path=None):
    return get().render(path)


def _cli():
    import argparse
    import json

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="eyes.py")
    p.add_argument("--move", nargs=2, type=float, metavar=("H", "V"))
    p.add_argument("--distance", type=float)
    p.add_argument("--out")
    p.add_argument("--sequence", action="store_true", help="render the 10 official frames + metadata.json")
    opt = p.parse_args(argv)
    if opt.sequence:
        from tools import render_official
        render_official.run()
        return
    e = get()
    if opt.move:
        e.move(*opt.move, distance=opt.distance)
    out = e.render(opt.out)
    print(json.dumps(e.record(), indent=2))
    print("saved", out)


if __name__ == "__main__":
    _cli()
