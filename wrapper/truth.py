"""
Ground truth read back from the live scene.

Pupil centres are where each posed eyeball's optical axis leaves the globe,
projected through the active camera. Every projection is cross-checked
against Blender's own world_to_camera_view; a disagreement above TOLERANCE_PX
raises instead of writing a bad record.
"""
import math

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from wrapper import camera, config as cfg, gaze

TOLERANCE_PX = 0.01
RING_SAMPLES = 12


def camera_model(scene=None, cam=None):
    scene = scene or bpy.context.scene
    cam = cam or scene.camera
    size = scene.render.resolution_x
    if size != scene.render.resolution_y:
        raise RuntimeError("expected a square render")
    K = camera.intrinsics(cam.data.lens, cam.data.sensor_width, size)
    w2c = camera.world_to_camera(np.array(cam.matrix_world))
    return K, w2c, size, cam


def _blender_uv(scene, cam, size, point):
    co = world_to_camera_view(scene, cam, Vector(point))
    return np.array([co.x * size, (1.0 - co.y) * size])


def _visible(scene, origin, point, ignore):
    """True when nothing opaque sits between the camera and the point."""
    dg = bpy.context.evaluated_depsgraph_get()
    o = Vector(origin)
    goal = Vector(point)
    for _ in range(8):
        d = goal - o
        dist = d.length
        hit, loc, _n, _i, obj, _m = scene.ray_cast(dg, o, d.normalized(), distance=dist - 1e-5)
        if not hit:
            return True
        if obj is None or obj.hide_render or obj.name.startswith(ignore):
            o = loc + d.normalized() * 1e-5
            continue
        return False
    return False


def pupil_visibility(eyes, side, scene, cam_origin):
    """Fraction of the pupil disc (centre + rim samples) seen by the camera."""
    c = eyes.centre(side)
    g = eyes.gaze_vector(side)
    a = g.cross(Vector((0, 0, 1)))
    if a.length < 1e-6:
        a = g.cross(Vector((1, 0, 0)))
    a.normalize()
    b = g.cross(a).normalized()
    ang = math.asin(min(1.0, eyes.pupil_mm / 2000.0 / cfg.EYE_RADIUS))
    pts = [c + g * eyes.radius]
    for k in range(RING_SAMPLES):
        phi = 2 * math.pi * k / RING_SAMPLES
        d = (g * math.cos(ang) + (a * math.cos(phi) + b * math.sin(phi)) * math.sin(ang)).normalized()
        pts.append(c + d * eyes.radius * 1.0005)
    seen = [_visible(scene, cam_origin, p, ("Cornea", "Eyeball")) and _front_facing(c, p, cam_origin)
            for p in pts]
    return sum(seen) / len(seen), bool(seen[0])


def _front_facing(centre, point, cam_origin):
    n = (point - centre).normalized()
    return n.dot((Vector(cam_origin) - point).normalized()) > 0.0


def record(eyes, scene=None, cam=None):
    scene = scene or bpy.context.scene
    K, w2c, size, cam = camera_model(scene, cam)
    out, worst = {}, 0.0
    cam_origin = cam.matrix_world.translation
    for side in gaze.SIDES:
        p = np.array(eyes.pupil_point(side))
        uv, depth = camera.project(p, K, w2c)
        worst = max(worst, float(np.abs(uv[0] - _blender_uv(scene, cam, size, p)).max()))
        rot = np.array(eyes.rotation(side))
        vis, centre_seen = pupil_visibility(eyes, side, scene, cam_origin)
        out[side] = {
            "uv": [round(float(uv[0, 0]), 4), round(float(uv[0, 1]), 4)],
            "euler": [round(float(a), 6) for a in gaze.matrix_euler_xyz(rot)],
            "gaze": [round(float(a), 6) for a in eyes.gaze_vector(side)],
            "centre": [round(float(a), 6) for a in eyes.centre(side)],
            "pupil3d": [round(float(a), 6) for a in p],
            "depth": round(float(depth[0]), 6),
            "visible": round(vis, 3),
            "centre_seen": centre_seen,
        }
    if worst > TOLERANCE_PX:
        raise RuntimeError(f"projection cross-check failed: {worst:.4f} px")
    target = [round(float(a), 6) for a in eyes.target.location]
    analytic = gaze.solve(eyes.h, eyes.v, eyes.distance,
                          centres={s: tuple(eyes.centres[s]) for s in gaze.SIDES}, radius=eyes.radius)
    aim_err = max(gaze.angle_between(out[s]["gaze"], analytic["eyes"][s]["direction"]) for s in gaze.SIDES)
    return {
        "input": {"h": round(eyes.h, 6), "v": round(eyes.v, 6)},
        "eye_rotation_L": out["L"]["euler"],
        "eye_rotation_R": out["R"]["euler"],
        "pupil_center_2D_L": out["L"]["uv"],
        "pupil_center_2D_R": out["R"]["uv"],
        "camera_matrix": camera.describe(K, w2c, size, cam.data.lens, cam.data.sensor_width),
        "fixation_point": target,
        "fixation_distance": round(eyes.distance, 6),
        "gaze_vector_L": out["L"]["gaze"],
        "gaze_vector_R": out["R"]["gaze"],
        "eye_center_L": out["L"]["centre"],
        "eye_center_R": out["R"]["centre"],
        "pupil_center_3D_L": out["L"]["pupil3d"],
        "pupil_center_3D_R": out["R"]["pupil3d"],
        "pupil_diameter_mm": round(eyes.pupil_mm, 3),
        "pupil_visibility_L": out["L"]["visible"],
        "pupil_visibility_R": out["R"]["visible"],
        "vergence_deg": round(gaze.vergence(out["L"]["gaze"], out["R"]["gaze"]), 4),
        "lid_closure": round(eyes.closure, 3),
        "aim_error_deg": round(aim_err, 6),
        "projection_check_px": round(worst, 6),
        "camera": cam.name,
    }
