"""
Build scene.blend from the MakeHuman/MPFB face in assets/face_source.blend.

    blender --background --python tools/build_scene.py -- [--out scene.blend] [--no-calibrate]

What it does
  1. appends the Human mesh and Human.rig armature, strips MPFB helper geometry
  2. scales the head so the eye centres sit 63 mm apart, eye midpoint at the origin
  3. fits an eyeball into each socket and parents it to the eye.L / eye.R bone
  4. builds procedural iris, sclera, cornea, skin, brows and lashes
  5. sets up the locked 40 cm camera, constant lighting and the render settings
  6. adds optional props used by the lab tools (lid shells, glasses, extra
     cameras, gaze-ray overlay), all hidden from the official render
  7. auto-levels exposure once on a test render, then saves with packed images
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wrapper import camera, config as cfg  # noqa: E402
from lab import shading, textures  # noqa: E402

LOG = {}


def say(*parts):
    print("[build]", *parts, flush=True)


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--face", default=str(cfg.FACE_SOURCE))
    p.add_argument("--out", default=str(cfg.SCENE_FILE))
    p.add_argument("--iris", default="brown", choices=sorted(textures.IRIS_PALETTES))
    p.add_argument("--skin", default="medium", choices=sorted(shading.SKIN_TONES))
    p.add_argument("--no-calibrate", action="store_true", help="skip the exposure auto-level")
    p.add_argument("--no-lashes", action="store_true")
    p.add_argument("--debug", action="store_true", help="print lid geometry diagnostics")
    return p.parse_args(argv)


# -- scene helpers -------------------------------------------------------------
def collection(name, parent=None, hidden=False):
    col = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if col.name not in (parent or bpy.context.scene.collection).children:
        (parent or bpy.context.scene.collection).children.link(col)
    col.hide_render = hidden
    return col


def put(obj, col=None):
    (col or bpy.context.scene.collection).objects.link(obj)
    return obj


def mesh_object(name, bm, material=None, col=None, smooth=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        for poly in me.polygons:
            poly.use_smooth = True
    if material is not None:
        me.materials.append(material)
    return put(bpy.data.objects.new(name, me), col)


# -- face import ---------------------------------------------------------------
def import_face(path):
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.objects = list(src.objects)
    objs = [o for o in dst.objects if o is not None]
    for o in objs:
        put(o)
    rig = bpy.data.objects.get(cfg.RIG_NAME) or next(o for o in objs if o.type == "ARMATURE")
    meshes = [o for o in objs if o.type == "MESH"]
    face = bpy.data.objects.get(cfg.FACE_NAME) or max(meshes, key=lambda o: len(o.data.vertices))
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)
    bpy.context.view_layer.update()
    # The MPFB eye proxy matches the unmorphed base mesh, not this face, so it is only
    # logged for reference; the rig's eye pivots and the socket itself decide the eyeball.
    proxy = next((o for o in meshes if o is not face and "eye.L" in o.vertex_groups), None)
    LOG["eye_proxy"] = proxy.name if proxy else None
    bake_shape_keys(face)
    for o in meshes:
        if o is not face:
            say("removing extra mesh", o.name)
            bpy.data.objects.remove(o, do_unlink=True)
    LOG["face_vertices_in"] = len(face.data.vertices)
    LOG["vertex_groups"] = [g.name for g in face.vertex_groups][:80]
    return rig, face


def fit_sphere(pts):
    pts = np.asarray(pts, dtype=float)
    a = np.c_[2 * pts, np.ones(len(pts))]
    b = (pts ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    c = sol[:3]
    return c, float(np.sqrt(sol[3] + c @ c))


def fit_eye_bones(rig, proxy):
    """Fit a sphere to each MPFB eyeball and move the eye bone heads onto the centres."""
    mw = proxy.matrix_world
    fits = {}
    for side in ("L", "R"):
        gi = proxy.vertex_groups[cfg.EYE_BONES[side]].index
        pts = [mw @ v.co for v in proxy.data.vertices
               if any(g.group == gi and g.weight > 0.9 for g in v.groups)]
        centre, radius = fit_sphere(pts)
        fits[side] = (Vector(centre), radius)
        say(f"eye sphere {side}: {len(pts)} verts, r={radius:.5f}, centre={tuple(round(c, 5) for c in centre)}")
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    inv = rig.matrix_world.inverted()
    for side, (centre, _) in fits.items():
        eb = rig.data.edit_bones[cfg.EYE_BONES[side]]
        offset = eb.tail - eb.head
        eb.head = inv @ centre
        eb.tail = eb.head + offset
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    LOG["proxy_eye_radius"] = {s: round(r, 6) for s, (_, r) in fits.items()}
    return fits


def bake_shape_keys(face):
    """Fold the face's morph targets into the mesh so later vertex edits take effect."""
    keys = face.data.shape_keys
    if keys is None:
        return
    n = len(keys.key_blocks)
    dg = bpy.context.evaluated_depsgraph_get()
    mixed = face.evaluated_get(dg).to_mesh()
    arm = [m for m in face.modifiers if m.type == "ARMATURE"]
    coords = None
    if not arm:
        coords = [v.co.copy() for v in mixed.vertices]
    face.evaluated_get(dg).to_mesh_clear()
    if coords is None:
        # armature at rest adds nothing, but read the plain shape-key mix to be safe
        for m in arm:
            m.show_viewport = False
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        mixed = face.evaluated_get(dg).to_mesh()
        coords = [v.co.copy() for v in mixed.vertices]
        face.evaluated_get(dg).to_mesh_clear()
        for m in arm:
            m.show_viewport = True
    face.shape_key_clear()
    for v, co in zip(face.data.vertices, coords):
        v.co = co
    face.data.update()
    LOG["shape_keys_baked"] = n
    say(f"baked {n} shape keys into the face")


def strip_helpers(face, keep_above_z):
    """Delete MPFB helper geometry (eye spheres, joint cubes, tights...) and the lower body."""
    groups = {g.index: g.name.lower() for g in face.vertex_groups}
    body = next((i for i, n in groups.items() if n == "body"), None)
    helper_ids = {i for i, n in groups.items()
                  if any(t in n for t in ("helper", "joint", "cube", "tights", "skirt", "hair", "tongue"))}
    doomed = set()
    mw = face.matrix_world
    for v in face.data.vertices:
        weights = {g.group: g.weight for g in v.groups}
        if body is not None and weights.get(body, 0.0) < 0.5:
            doomed.add(v.index)
        elif any(weights.get(i, 0.0) > 0.1 for i in helper_ids):
            doomed.add(v.index)
        elif (mw @ v.co).z < keep_above_z:
            doomed.add(v.index)
    for mod in list(face.modifiers):
        if mod.type == "MASK":
            face.modifiers.remove(mod)
    for name in cfg.EYE_BONES.values():
        group = face.vertex_groups.get(name)
        if group is not None:
            face.vertex_groups.remove(group)
    if doomed:
        bm = bmesh.new()
        bm.from_mesh(face.data)
        bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.verts[i] for i in sorted(doomed)], context="VERTS")
        bm.to_mesh(face.data)
        bm.free()
    LOG["face_vertices_kept"] = len(face.data.vertices)
    say("face vertices", LOG["face_vertices_in"], "->", LOG["face_vertices_kept"])


def bone_head(rig, name):
    return rig.matrix_world @ rig.data.bones[name].head_local


def normalise_head(rig, face):
    """Scale and move every root object so the eye centres are +/-IPD/2 on X at the origin."""
    left, right = bone_head(rig, cfg.EYE_BONES["L"]), bone_head(rig, cfg.EYE_BONES["R"])
    tail = rig.matrix_world @ rig.data.bones[cfg.EYE_BONES["L"]].tail_local
    looks = (tail - left).normalized()
    turn = Matrix.Identity(4)
    if looks.y > 0:  # face built looking +Y: spin it round
        turn = Matrix.Rotation(math.pi, 4, "Z")
    if (turn @ left).x < (turn @ right).x:
        raise RuntimeError("eye.L is not on the subject's left (+X); check the face source")
    ipd = (left - right).length
    s = cfg.IPD / ipd
    mid = (turn @ ((left + right) / 2.0))
    fix = Matrix.Scale(s, 4) @ Matrix.Translation(-mid) @ turn
    for obj in (rig, face):
        if obj.parent is None:
            obj.matrix_world = fix @ obj.matrix_world
    bpy.context.view_layer.update()
    LOG["source_ipd_m"] = round(ipd, 6)
    LOG["scale"] = round(s, 6)
    say(f"source IPD {ipd * 1000:.2f} mm, scale x{s:.4f}")
    return {side: bone_head(rig, cfg.EYE_BONES[side]) for side in ("L", "R")}


def socket_fit(face, centre, radius_hint=None):
    """Eyelid margin around one eye: radius for the globe and the lid opening angles."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = face.evaluated_get(dg)
    me = ev.to_mesh()
    mw = ev.matrix_world
    pts = np.array([mw @ v.co for v in me.vertices])
    ev.to_mesh_clear()
    rel = pts - np.array(centre)
    dist = np.linalg.norm(rel, axis=1)
    fwd = -rel[:, 1] / np.maximum(dist, 1e-9)
    near = (dist < 0.022) & (fwd > math.cos(math.radians(70)))
    if near.sum() < 12:
        raise RuntimeError(f"no eyelid vertices found near {tuple(centre)}")
    rim = rel[near]
    rd = dist[near]
    if radius_hint:
        radius = float(radius_hint)
    else:
        radius = float(np.clip(np.percentile(rd, 3) - 0.0003, 0.0095, 0.0165))
    elev = np.degrees(np.arctan2(rim[:, 2], -rim[:, 1]))
    azim = np.degrees(np.arctan2(rim[:, 0], -rim[:, 1]))
    close = rd < radius + 0.0025
    margin = rim[close]
    centre_band = close & (np.abs(azim) < 12)
    upper = elev[centre_band & (elev > 0)]
    lower = elev[centre_band & (elev < 0)]
    fit = {
        "radius": radius,
        "upper_deg": float(upper.min()) if upper.size else 22.0,
        "lower_deg": float(lower.max()) if lower.size else -18.0,
        "margin_points": len(margin),
        "rim": margin.tolist(),
    }
    say(f"socket fit r={radius * 1000:.2f} mm upper={fit['upper_deg']:.1f} lower={fit['lower_deg']:.1f}"
        f" margin pts={len(margin)}")
    return fit


# -- eyes ----------------------------------------------------------------------
def eyeball(side, rig, centre, radius, mats, col):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=128, v_segments=64, radius=radius)
    # sphere pole on -Y = optical axis
    bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(math.radians(90), 3, "X"), verts=bm.verts)
    globe = mesh_object(cfg.EYEBALL_NAMES[side], bm, mats["eye"], col)
    # iris, pupil and cornea keep adult proportions relative to the globe
    k = radius / cfg.EYE_RADIUS
    globe["pupil_ratio"] = (cfg.PUPIL_DIAMETER_MM / 2000.0) / cfg.EYE_RADIUS
    globe["iris_ratio"] = cfg.IRIS_RADIUS / cfg.EYE_RADIUS
    globe["radius"] = radius
    globe["anatomy_scale"] = k

    # cornea: a 7.8 mm cap (scaled) whose apex stands ~1.4 mm proud of the globe
    rc = cfg.CORNEA_RADIUS * k
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=96, v_segments=48, radius=rc)
    bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(math.radians(90), 3, "X"), verts=bm.verts)
    offset = radius + 0.0014 * k - rc
    keep_cos = math.cos(math.asin(min(1.0, (cfg.IRIS_RADIUS + 0.0012) * k / rc)))
    drop = [v for v in bm.verts if (-v.co.y / rc) < keep_cos]
    bmesh.ops.delete(bm, geom=drop, context="VERTS")
    cornea = mesh_object(f"Cornea.{side}", bm, mats["cornea"], col)
    cornea["cornea_refraction"] = 0.0
    cornea.parent = globe
    cornea.location = (0.0, -offset, 0.0)

    globe.parent = rig
    globe.parent_type = "BONE"
    globe.parent_bone = cfg.EYE_BONES[side]
    bpy.context.view_layer.update()
    globe.matrix_world = Matrix.Translation(centre)
    bpy.context.view_layer.update()
    return globe


def lashes(side, centre, fit, mat, col, rig, seed):
    """Upper and lower lashes as tapered curve strands rooted on the visible lid edge."""
    rng = np.random.default_rng(seed)
    data = bpy.data.curves.new(f"Lashes.{side}", "CURVE")
    data.dimensions = "3D"
    data.bevel_depth = 0.00006
    data.bevel_resolution = 1
    data.use_fill_caps = True
    c = np.array(centre)
    roots = fit.get("lash_roots") or {}
    for upper, count, length, lift in ((True, 95, 0.0078, 0.9), (False, 34, 0.0032, -0.6)):
        hits = sorted(roots.get("upper" if upper else "lower", []))
        if len(hits) < 4:
            continue
        az = np.array([h[0] for h in hits])
        xyz = np.array([h[1] for h in hits])
        lo, hi = az[0] + 3.0, az[-1] - 3.0
        sign = 1.0 if upper else -1.0
        for i in range(count):
            t = (i + rng.uniform(0.0, 0.7)) / count
            a = lo + (hi - lo) * t
            root = np.array([np.interp(a, az, xyz[:, k]) for k in range(3)])
            radial = root - c
            radial[2] = 0.0
            radial /= max(np.linalg.norm(radial), 1e-9)
            # lashes leave the lid towards the viewer, fanning slightly with the lid
            out = np.array([0.0, -1.0, 0.0]) * 0.8 + radial * 0.2
            out /= np.linalg.norm(out)
            root = root + np.array([0.0, -0.0003, 0.0])
            up = np.array([0.0, 0.0, sign])
            ln = length * (0.45 + 0.55 * math.sin(math.pi * t)) * rng.uniform(0.8, 1.1)
            # leave the lid almost straight out, then curl away from the eye
            p1 = root + (out * 0.9 + up * 0.1) * ln * 0.5
            p2 = p1 + (out * 0.45 + up * (0.45 + 0.3 * abs(lift))) * ln * 0.5
            spline = data.splines.new("NURBS")
            spline.points.add(2)
            for k, (pt, rad) in enumerate(((root, 1.0), (p1, 0.7), (p2, 0.15))):
                spline.points[k].co = (*pt, 1.0)
                spline.points[k].radius = rad
            spline.use_endpoint_u = True
            spline.order_u = 3
    data.materials.append(mat)
    obj = put(bpy.data.objects.new(f"Lashes.{side}", data), col)
    obj.parent = rig
    obj.matrix_parent_inverse = rig.matrix_world.inverted()
    return obj


def lid_edge(az_deg, upper, front_deg=None, arch_deg=cfg.LID_ARCH_DEG):
    """Elevation (deg) of a resting lid edge at a given azimuth.

    Each lid is a spherical cap whose axis leans back by arch_deg, so its edge is a circle
    on the globe: highest straight ahead for the upper lid, lowest for the lower one."""
    if front_deg is None:
        front_deg = cfg.LID_UPPER_DEG if upper else cfg.LID_LOWER_DEG
    front = front_deg if upper else -front_deg
    beta = math.radians(arch_deg)
    cos_alpha = math.cos(math.radians(90.0 + arch_deg - front))
    az = math.radians(az_deg)
    a, b = math.cos(beta), -math.cos(az) * math.sin(beta)
    r = math.hypot(a, b)
    el = math.degrees(math.asin(max(-1.0, min(1.0, cos_alpha / r))) - math.atan2(b, a))
    return el if upper else -el


def lid_span():
    """Azimuth range (deg) where the resting lids leave the eye open."""
    for az in np.arange(0.0, 80.0, 0.5):
        if lid_edge(az, True) - lid_edge(az, False) < 1.0:
            return float(min(az, 56.0))
    return 56.0


def visible_margin(centre, radius, cam=(0.0, -cfg.CAMERA_DISTANCE, 0.0)):
    """Lid edge as the camera sees it: for each azimuth, walk up (and down) the globe from
    the equator and stop at the first point the skin hides from the camera.
    Returns {"upper": [(az, el)], "lower": [(az, el)]} in degrees. Call before the eyes exist."""
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    cam = Vector(cam)
    c = Vector(centre)
    out = {"upper": [], "lower": [], "hits": {"upper": [], "lower": []}}
    for az in np.arange(-64.0, 64.1, 2.0):
        a = math.radians(az)
        for name, sign in (("upper", 1.0), ("lower", -1.0)):
            for el in np.arange(0.0, 80.0, 0.5):
                e = math.radians(sign * el)
                n = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
                p = c + n * radius
                to = cam - p
                if n.dot(to) <= 0:          # past the globe's silhouette
                    break
                dist = to.length
                hit, loc, *_ = scene.ray_cast(dg, cam, (p - cam).normalized(), distance=dist - 0.0002)
                if hit:
                    out[name].append((float(az), float(sign * el)))
                    out["hits"][name].append((float(az), tuple(loc)))
                    break
    return out


def margin_fit(margin):
    """Smooth quadratic el(az) per lid and the azimuth range where the eye is open."""
    fit = {}
    for name in ("upper", "lower"):
        pts = np.array(margin[name])
        if len(pts) < 8:
            return None
        fit[name] = np.polyfit(pts[:, 0], pts[:, 1], 2)
    up = dict(margin["upper"])
    lo = dict(margin["lower"])
    open_az = [az for az in up if az in lo and up[az] - lo[az] > 2.0]
    fit["span"] = (float(min(open_az)), float(max(open_az))) if open_az else (-40.0, 40.0)
    return fit


def spherical(p):
    d = np.linalg.norm(p, axis=-1)
    az = np.degrees(np.arctan2(p[..., 0], -p[..., 1]))
    el = np.degrees(np.arcsin(np.clip(p[..., 2] / np.maximum(d, 1e-9), -1, 1)))
    return az, el, d


def reshape_lids(face, centre, radius, fit):
    """Slide the lid margins along the globe to the resting arch (cfg.LID_*), fading out
    into the surrounding skin. Distances from the eye centre are kept, so the lids stay on
    the eyeball and nothing pokes through."""
    me = face.data
    mw = face.matrix_world
    inv = mw.inverted()
    c = np.array(centre)
    moved = 0
    for v in me.vertices:
        p = np.array(mw @ v.co) - c
        d = float(np.linalg.norm(p))
        if d > radius + 0.012 or d < radius - 0.0005 or p[1] > 0:
            continue
        az = math.degrees(math.atan2(p[0], -p[1]))
        el = math.degrees(math.asin(max(-1.0, min(1.0, p[2] / d))))
        lo, hi = fit["span"]
        if not (lo - 4.0 <= az <= hi + 4.0):
            continue
        upper = el >= 0
        m = float(np.polyval(fit["upper" if upper else "lower"], az))
        target = lid_edge(az, upper)
        shift = (m - target) if upper else (target - m)
        shift = max(-14.0, min(16.0, shift))
        half = max(abs(lo), abs(hi))
        shift *= 1.0 - _smooth(0.45 * half, 0.9 * half, abs(az))
        if abs(shift) < 0.5:
            continue
        beyond = (el - m) if upper else (m - el)
        if beyond < -4.0:
            continue
        w_el = 1.0 - _smooth(6.0, 40.0, beyond)
        w_d = 1.0 - _smooth(radius + 0.003, radius + 0.012, d)
        w = w_el * w_d
        if w <= 0:
            continue
        new_el = el - shift * w if upper else el + shift * w
        r_xy = d * math.cos(math.radians(new_el))
        q = np.array([r_xy * math.sin(math.radians(az)), -r_xy * math.cos(math.radians(az)),
                      d * math.sin(math.radians(new_el))])
        v.co = inv @ Vector(q + c)
        moved += 1
    me.update()
    return moved


def _smooth(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def lid_curve(fit, upper):
    if fit.get("margin"):
        coef = fit["margin"]["upper" if upper else "lower"]
        lo, hi = fit["margin"]["span"]
        return (lambda x: float(np.polyval(coef, x))), lo + 2.0, hi - 2.0
    lim = lid_span() - 3.0
    return (lambda x: lid_edge(x, upper)), -lim, lim


def lid_shell(side, centre, fit, mat, col, rig):
    """Resting lids as arched spherical caps over the globe; they also close for blinks."""
    r = fit["radius"] + 0.0006
    parts = []
    for upper in (True, False):
        front = cfg.LID_UPPER_DEG if upper else cfg.LID_LOWER_DEG
        sign = 1.0 if upper else -1.0
        beta = math.radians(cfg.LID_ARCH_DEG)
        axis = Vector((0.0, math.sin(beta), sign * math.cos(beta)))
        alpha = math.radians(90.0 + cfg.LID_ARCH_DEG - sign * front)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=96, v_segments=64, radius=r)
        drop = [v for v in bm.verts if v.co.normalized().dot(axis) < math.cos(alpha) or v.co.y > 0.004]
        bmesh.ops.delete(bm, geom=drop, context="VERTS")
        name = f"Lid{'Upper' if upper else 'Lower'}.{side}"
        obj = mesh_object(name, bm, mat, col)
        rim = obj.modifiers.new("Rim", "SOLIDIFY")
        rim.thickness = 0.0009
        rim.offset = 1.0
        rim.use_rim = True
        obj.location = centre
        obj.parent = rig
        obj.matrix_parent_inverse = rig.matrix_world.inverted()
        obj["edge_deg"] = front
        obj["margin_deg"] = front
        obj.hide_render = True
        parts.append(obj)
    return parts


def glasses(col, mats):
    """Thin rectangular frames with clear lenses, 14 mm in front of the corneas."""
    bm = bmesh.new()
    for side in (1, -1):
        cx = side * cfg.IPD / 2
        ring = bmesh.ops.create_circle(bm, cap_ends=False, segments=48, radius=0.024)["verts"]
        for v in ring:
            v.co = Vector((cx + v.co.x * 1.15, -0.026, v.co.y * 0.78))
    frame = mesh_object("Glasses.Frame", bm, mats["frame"], col)
    frame.modifiers.new("Skin", "SKIN")
    for v in frame.data.skin_vertices[0].data:
        v.radius = (0.0011, 0.0016)
    frame.modifiers.new("Smooth", "SUBSURF").levels = 1
    bm = bmesh.new()
    for side in (1, -1):
        res = bmesh.ops.create_circle(bm, cap_ends=True, segments=48, radius=0.0235)
        for v in res["verts"]:
            v.co = Vector((side * cfg.IPD / 2 + v.co.x * 1.15, -0.0262, v.co.y * 0.78))
    lens = mesh_object("Glasses.Lens", bm, mats["lens"], col)
    lens.modifiers.new("Solid", "SOLIDIFY").thickness = 0.0015
    for o in (frame, lens):
        o.hide_render = True
    return frame, lens


def overlay(col, mats):
    """Gaze rays and fixation marker for the 3D visualisation render."""
    objs = []
    for side, mat in (("L", mats["ray_l"]), ("R", mats["ray_r"])):
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=0.0007, radius2=0.0007, depth=1.0)
        bmesh.ops.translate(bm, vec=(0, 0, 0.5), verts=bm.verts)
        objs.append(mesh_object(f"GazeRay.{side}", bm, mat, col))
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=0.004)
    objs.append(mesh_object("FixationMarker", bm, mats["target"], col))
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=10, y_segments=10, size=cfg.FIXATION_RANGE)
    bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(math.radians(90), 3, "X"), verts=bm.verts)
    plane = mesh_object("FixationPlane", bm, mats["plane"], col, smooth=False)
    plane.location = (0, -cfg.FIXATION_DISTANCE, 0)
    plane.display_type = "WIRE"
    objs.append(plane)
    for o in objs:
        o.hide_render = True
    return objs


# -- camera, lights, world -------------------------------------------------------
def add_camera(name, matrix, col=None, lens=cfg.LENS_MM):
    data = bpy.data.cameras.new(name)
    data.lens = lens
    data.sensor_width = cfg.SENSOR_MM
    data.sensor_fit = "HORIZONTAL"
    data.clip_start = 0.01
    data.clip_end = 5.0
    cam = put(bpy.data.objects.new(name, data), col)
    cam.matrix_world = Matrix([list(r) for r in matrix])
    return cam


def add_light(name, kind, location, target, power, size, colour=(1, 1, 1), glint=False):
    data = bpy.data.lights.new(name, kind)
    data.energy = power
    data.color = colour
    if kind == "AREA":
        data.shape = "RECTANGLE"
        data.size, data.size_y = size
    light = put(bpy.data.objects.new(name, data))
    light.location = location
    direction = Vector(target) - Vector(location)
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    # only the catch light shows up in the cornea; the others light the face without a glint
    light.visible_glossy = glint
    return light


def lighting():
    # soft key upper left of camera, fill right, small rectangular catch light by the lens
    add_light("Key", "AREA", (0.28, -0.62, 0.30), (0, 0, 0), 26.0, (0.45, 0.45), (1.0, 0.96, 0.92))
    add_light("Fill", "AREA", (-0.42, -0.50, 0.05), (0, 0, 0), 9.0, (0.6, 0.6), (0.92, 0.95, 1.0))
    add_light("Catch", "AREA", (0.03, -0.70, 0.11), (0, 0, 0), 3.0, (0.07, 0.045), glint=True)
    add_light("Top", "AREA", (0.0, 0.10, 0.45), (0, 0, 0), 6.0, (0.4, 0.3))
    world = bpy.data.worlds.new("Studio")
    bpy.context.scene.world = world
    try:
        world.use_nodes = True
    except (AttributeError, TypeError):
        pass
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.045, 0.047, 0.052, 1.0)
    bg.inputs["Strength"].default_value = 1.0


def render_settings(scene):
    scene.render.resolution_x = scene.render.resolution_y = cfg.IMAGE_SIZE
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.engine = "CYCLES"
    cyc = scene.cycles
    cyc.samples = 64
    cyc.use_adaptive_sampling = True
    cyc.adaptive_threshold = 0.02
    cyc.use_denoising = True
    cyc.seed = 7
    cyc.max_bounces = 8
    cyc.transparent_max_bounces = 16
    cyc.transmission_bounces = 8
    from wrapper.eyes import use_gpu
    say("cycles device", use_gpu(scene) or "CPU")
    vs = scene.view_settings
    for view in ("AgX", "Filmic", "Standard"):
        try:
            vs.view_transform = view
            break
        except TypeError:
            continue
    vs.exposure = 0.0
    vs.gamma = 1.0


def auto_exposure(scene, rounds=7, target=0.46):
    """Render small test frames and set the view exposure so the skin sits at a fixed level."""
    tmp = cfg.EXTRA_DIR / "_exposure.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    keep = (scene.render.resolution_percentage, scene.cycles.samples)
    scene.render.resolution_percentage = 25
    scene.cycles.samples = 24
    patches = [(0.50, 0.12), (0.50, 0.46), (0.22, 0.82), (0.78, 0.82)]
    for i in range(rounds):
        scene.render.filepath = str(tmp)
        t0 = time.time()
        say(f"exposure round {i}: rendering")
        bpy.ops.render.render(write_still=True)
        say(f"exposure round {i}: rendered in {time.time() - t0:.1f} s")
        img = bpy.data.images.load(str(tmp), check_existing=False)
        w, h = img.size
        px = np.array(img.pixels[:]).reshape(h, w, -1)[::-1, :, :3]
        bpy.data.images.remove(img)
        vals = []
        for u, v in patches:
            x0, y0 = int(u * w), int(v * h)
            patch = px[max(0, y0 - 3):y0 + 4, max(0, x0 - 3):x0 + 4]
            vals.append(float(patch.mean()))
        level = float(np.median(vals))
        step = math.log2(max(target, 1e-3) / max(level, 1e-3)) * 2.2
        scene.view_settings.exposure = float(np.clip(scene.view_settings.exposure + step, -6, 6))
        say(f"exposure round {i}: skin {level:.3f} -> exposure {scene.view_settings.exposure:+.2f}")
        if abs(level - target) < 0.02:
            break
    scene.render.resolution_percentage, scene.cycles.samples = keep
    LOG["exposure"] = scene.view_settings.exposure
    tmp.unlink(missing_ok=True)


# -- main ------------------------------------------------------------------------
def brow_shape():
    # inner head, arch peak and tail of the brow in head space (|x|, z), metres
    pts = np.array([[0.011, 0.0175], [0.040, 0.0255], [0.058, 0.0195]])
    a2, a1, a0 = np.polyfit(pts[:, 0], pts[:, 1], 2)
    return {
        "poly": (float(a0), float(a1), float(a2)),
        "x_inner": 0.011, "x_outer": 0.060,
        "half_inner": 0.0042, "taper": 0.050,
        "colour": (0.030, 0.019, 0.012), "density": 0.92,
    }


def main():
    opt = args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"

    rig, face = import_face(opt.face)
    raw_left = bone_head(rig, cfg.EYE_BONES["L"])
    strip_helpers(face, keep_above_z=raw_left.z - 0.32)
    centres = normalise_head(rig, face)
    head_col = collection("Head")
    for o in (rig, face):
        for c in list(o.users_collection):
            c.objects.unlink(o)
        head_col.objects.link(o)

    fits = {side: socket_fit(face, centres[side]) for side in ("L", "R")}
    radius = round((fits["L"]["radius"] + fits["R"]["radius"]) / 2.0, 5)
    LOG["eye_radius_m"] = radius
    if opt.debug:
        c = np.array(centres["L"])
        mw = face.matrix_world
        rows = []
        for v in face.data.vertices:
            q = np.array(mw @ v.co) - c
            dd = np.linalg.norm(q)
            az_ = math.degrees(math.atan2(q[0], -q[1]))
            el_ = math.degrees(math.asin(q[2] / dd))
            if abs(az_) < 8 and q[1] < 0 and -80 < el_ < 80 and (dd - radius) < 0.03:
                rows.append((round(el_, 1), round((dd - radius) * 1000, 2), round(az_, 1)))
        for r_ in sorted(rows):
            say("vert el/d-R(mm)/az", r_)
        mv = visible_margin(centres["L"], radius)
        say("visible upper", [(a_, e_) for a_, e_ in mv["upper"] if abs(a_) <= 20])
        say("visible lower", [(a_, e_) for a_, e_ in mv["lower"] if abs(a_) <= 20])
    for side in ("L", "R"):
        before = margin_fit(visible_margin(centres[side], radius))
        if before is None:
            say(f"lid margin not found for {side}; lids left as modelled")
            continue
        n = reshape_lids(face, centres[side], radius, before)
        seen = visible_margin(centres[side], radius)
        after = margin_fit(seen) or before
        fits[side]["lash_roots"] = seen["hits"]
        fits[side]["margin"] = {k: (list(v) if k != "span" else v) for k, v in after.items()}
        say(f"lids {side}: margin {np.polyval(before['upper'], 0):.1f}/{np.polyval(before['lower'], 0):.1f} deg"
            f" -> {np.polyval(after['upper'], 0):.1f}/{np.polyval(after['lower'], 0):.1f} deg, {n} vertices moved")
        LOG.setdefault("lid_margin", {})[side] = {
            "before": [round(float(np.polyval(before[k], 0)), 2) for k in ("upper", "lower")],
            "after": [round(float(np.polyval(after[k], 0)), 2) for k in ("upper", "lower")],
        }
    LOG["lids"] = {s: {k: round(v, 2) for k, v in fits[s].items() if k.endswith("deg")} for s in fits}

    head_space = put(bpy.data.objects.new("HeadSpace", None), head_col)
    head_space.parent = rig
    head_space.matrix_parent_inverse = rig.matrix_world.inverted()
    head_space.matrix_world = Matrix.Identity(4)

    tex_dir = ROOT / "assets" / "textures"
    iris_img = shading.make_image("Iris", textures.iris(opt.iris, seed=11), tex_dir)
    sclera_img = shading.make_image("Sclera", textures.sclera(seed=11), tex_dir)
    detail_img = shading.make_image("SkinDetail", textures.skin_detail(seed=3), tex_dir, colorspace="Non-Color")
    eye_centres = {s: tuple(centres[s]) for s in ("L", "R")}
    skin = shading.skin_material("Skin", detail_img, head_space, eye_centres, radius, brow_shape())
    skin.node_tree.nodes["SkinTone"].outputs[0].default_value = (*shading.SKIN_TONES[opt.skin], 1.0)
    face.data.materials.clear()
    face.data.materials.append(skin)
    for poly in face.data.polygons:
        poly.use_smooth = True
    if not any(m.type == "SUBSURF" for m in face.modifiers):
        sub = face.modifiers.new("Subdivision", "SUBSURF")
        sub.levels, sub.render_levels = 1, 2

    mats = {
        "eye": shading.eye_material("Eye", iris_img, sclera_img),
        "cornea": shading.cornea_material("Cornea", cfg.CORNEA_IOR),
        "lash": shading.flat_material("Lash", (0.012, 0.009, 0.007), 0.45),
        "frame": shading.flat_material("GlassesFrame", (0.02, 0.02, 0.022), 0.25),
        "lens": shading.glass_material("GlassesLens"),
        "ray_l": shading.flat_material("RayL", (0.0, 0.6, 1.0), 0.5, emission=(0.0, 0.6, 1.0)),
        "ray_r": shading.flat_material("RayR", (1.0, 0.55, 0.0), 0.5, emission=(1.0, 0.55, 0.0)),
        "target": shading.flat_material("Target", (1.0, 0.2, 0.3), 0.4, emission=(1.0, 0.2, 0.3)),
        "plane": shading.flat_material("Plane", (0.5, 0.5, 0.5), 0.8),
    }

    eye_col = collection("Eyes")
    for side in ("L", "R"):
        eyeball(side, rig, centres[side], radius, mats, eye_col)
        lid_shell(side, centres[side], fits[side], skin, eye_col, rig)
        if not opt.no_lashes:
            lashes(side, centres[side], fits[side], mats["lash"], eye_col, rig, seed=5 if side == "L" else 6)

    props = collection("Props")
    glasses(props, mats)
    overlay(collection("Overlay"), mats)

    fixation = put(bpy.data.objects.new(cfg.TARGET_NAME, None))
    fixation.empty_display_type = "SPHERE"
    fixation.empty_display_size = 0.004
    fixation.location = (0.0, -cfg.FIXATION_DISTANCE, 0.0)

    rigs = collection("Cameras")
    cam = add_camera(cfg.CAMERA_NAME, camera.pose(), rigs)
    scene.camera = cam
    # extra viewpoints for the multi-camera study, all aimed at the eye midpoint at 40 cm
    for name, yaw, pitch in (("EyeCam.Left", -18, 0), ("EyeCam.Right", 18, 0),
                             ("EyeCam.Low", 0, -16), ("EyeCam.High", 0, 14)):
        d = cfg.CAMERA_DISTANCE
        cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        offset = (d * sy * cp, d - d * cy * cp, d * sp)
        add_camera(name, camera.pose(offset=offset), rigs)
    add_camera("OverlayCam", camera.pose(distance=0.0, offset=(0.40, -0.24, 0.12), look_at=(0, -0.17, 0)), rigs,
               lens=28.0)

    lighting()
    render_settings(scene)
    scene["eyeverse"] = json.dumps({
        "eye_radius": radius,
        "centres": {s: list(centres[s]) for s in centres},
        "lids": LOG["lids"],
        "iris": opt.iris,
        "skin": opt.skin,
    })
    if not opt.no_calibrate:
        auto_exposure(scene)

    out = Path(opt.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out), compress=True)
    report = cfg.EXTRA_DIR / "build_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    LOG["blender"] = bpy.app.version_string
    report.write_text(json.dumps({k: v for k, v in LOG.items() if k != "rim"}, indent=2))
    say("saved", out)


if __name__ == "__main__":
    main()
