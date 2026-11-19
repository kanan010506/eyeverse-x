"""
Lab datasets on top of the calibrated scene. Nothing here touches renders/ or metadata.json.

    blender -b scene.blend -P tools/generate_dataset.py -- --mode MODE [options]

modes
  random      domain-randomised stills: identity, light, exposure, pupil, lids,
              glasses, cornea refraction, camera jitter, gaze and fixation distance
  saccade     a scan path of fixations and main-sequence saccades, with blinks
  pursuit     smooth pursuit of a Lissajous target with catch-up saccades
  identities  every preset identity at rest and in the four cardinal directions
  cameras     the same gaze set seen from the five-camera rig
  vergence    near-to-far fixation sweep at a fixed direction
  rays        gaze-ray overlay renders of the official sequence (3D visualisation)

Output: output/<mode>/images/*.png and output/<mode>/labels.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from lab import identity, motion  # noqa: E402
from wrapper import config as cfg, eyes, gaze  # noqa: E402
from wrapper.safety import inside  # noqa: E402

CAMERAS = ("EyeCam", "EyeCam.Left", "EyeCam.Right", "EyeCam.Low", "EyeCam.High")


def options():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["random", "saccade", "pursuit", "identities", "cameras", "vergence", "rays"])
    p.add_argument("--count", type=int, default=60)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--size", type=int, default=512, help="output resolution (official renders use 1024)")
    p.add_argument("--engine", choices=["eevee", "cycles"], default="eevee")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default=None)
    opt = p.parse_args(argv)
    limits = {"count": (1, 5000), "seconds": (0.5, 60.0), "fps": (1, 60), "size": (128, 2048),
              "samples": (1, 1024)}
    for key, (lo, hi) in limits.items():
        val = getattr(opt, key)
        if not lo <= val <= hi:
            p.error(f"--{key} must lie in [{lo}, {hi}], got {val}")
    if opt.out:
        if not inside(opt.out, cfg.EXTRA_DIR):
            p.error("--out must be inside output/")
    return opt


def set_engine(scene, engine, samples, size):
    scene.render.resolution_x = scene.render.resolution_y = size
    if engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        eyes.use_gpu(scene)
        return
    for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = name
            break
        except TypeError:
            continue
    ev = scene.eevee
    for attr, val in (("taa_render_samples", samples), ("use_raytracing", True),
                      ("use_shadows", True), ("use_gtao", True)):
        if hasattr(ev, attr):
            setattr(ev, attr, val)


class Writer:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        self.rows = []

    def shot(self, e, name, extra=None, camera=None):
        path = self.root / "images" / f"{name}.png"
        e.render(path, camera=camera)
        cam = bpy.data.objects[camera] if camera else None
        from wrapper import truth
        rec = truth.record(e, cam=cam)
        rec["image"] = f"images/{name}.png"
        if extra:
            rec.update(extra)
        self.rows.append(rec)
        return rec

    def save(self, meta):
        meta = dict(meta, frames=len(self.rows), created=time.strftime("%Y-%m-%d %H:%M:%S"))
        (self.root / "labels.json").write_text(json.dumps({"meta": meta, "frames": self.rows}, indent=1))
        print(f"wrote {len(self.rows)} frames to {self.root}", flush=True)


def in_frame(e, cam, margin=24):
    from wrapper import truth
    rec = truth.record(e, cam=cam)
    w, h = rec["camera_matrix"]["image_size"]
    pad = margin * w / 1024
    return all(pad <= rec[f"pupil_center_2D_{s}"][0] < w - pad and pad <= rec[f"pupil_center_2D_{s}"][1] < h - pad
               for s in ("L", "R"))


def run_random(e, w, opt):
    rng = np.random.default_rng(opt.seed)
    cams = {name: bpy.data.objects[name].matrix_world.copy() for name in CAMERAS if name in bpy.data.objects}
    for i in range(opt.count):
        ident = identity.random_identity(rng)
        identity.apply(ident)
        info = identity.randomise_scene(e, rng)
        cam_name = str(rng.choice(list(cams)))
        cam = bpy.data.objects[cam_name]
        # draw camera jitter and gaze together until both pupils land well inside
        # the frame; a sample with an eye cut off is redrawn, never rendered
        for attempt in range(20):
            identity.jitter_camera(cam, cams[cam_name], rng)
            distance = float(rng.uniform(0.25, 1.2))
            h, v = (float(x) for x in rng.uniform(-1, 1, 2))
            e.move(h, v, distance=distance)
            if in_frame(e, cam):
                break
            cam.matrix_world = cams[cam_name]
        else:
            e.move(0.0, 0.0)
        info.update(identity=ident, camera=cam_name, redraws=attempt)
        w.shot(e, f"{i:05d}", {"sample": info}, camera=cam_name)
        cam.matrix_world = cams[cam_name]
        print(f"random {i + 1}/{opt.count}", flush=True)
    identity.restore(e)


def run_timeline(e, w, rows):
    for i, r in enumerate(rows):
        e.move(r["h"], r["v"])
        e.set_lids(r.get("lid", 0.0))
        extra = {"t": round(r["t"], 4), "event": r["event"]}
        if "target_h" in r:
            extra["target"] = [round(r["target_h"], 4), round(r["target_v"], 4)]
        w.shot(e, f"{i:05d}", extra)
        if i % 10 == 0:
            print(f"frame {i + 1}/{len(rows)}", flush=True)
    e.reset()


def run_identities(e, w):
    for name, ident in identity.PRESETS.items():
        identity.apply(ident)
        for label, h, v in (("rest", 0, 0), ("left", -1, 0), ("right", 1, 0), ("up", 0, 1), ("down", 0, -1)):
            e.move(h, v)
            w.shot(e, f"{name}_{label}", {"identity": name, "pose": label})
    identity.apply(identity.studio())
    e.reset()


def run_cameras(e, w):
    poses = (("rest", 0, 0), ("left", -1, 0), ("right", 1, 0), ("up", 0, 1), ("down", 0, -1))
    for cam in CAMERAS:
        if cam not in bpy.data.objects:
            continue
        for label, h, v in poses:
            e.move(h, v)
            w.shot(e, f"{cam.replace('.', '_')}_{label}", {"pose": label}, camera=cam)
    e.reset()


def run_vergence(e, w):
    for i, d in enumerate((0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.0, 2.0, 6.0)):
        e.move(0.0, 0.0, distance=d)
        w.shot(e, f"{i:02d}_{int(d * 100):04d}cm", {"distance_m": d})
    e.reset()


def run_rays(e, w):
    for name in ("GazeRay.L", "GazeRay.R", "FixationMarker", "FixationPlane"):
        obj = bpy.data.objects.get(name)
        if obj:
            obj.hide_render = False
    try:
        for row in gaze.sequence():
            e.move(row["h"], row["v"])
            w.shot(e, f"rays_{row['frame']:02d}", {"frame": row["frame"]}, camera="OverlayCam")
    finally:
        for name in ("GazeRay.L", "GazeRay.R", "FixationMarker", "FixationPlane"):
            obj = bpy.data.objects.get(name)
            if obj:
                obj.hide_render = True
        e.reset()


def main():
    opt = options()
    scene = bpy.context.scene
    set_engine(scene, opt.engine, opt.samples, opt.size)
    e = eyes.get()
    e.reset()
    out = Path(opt.out) if opt.out else cfg.EXTRA_DIR / opt.mode
    w = Writer(out)
    t0 = time.time()
    if opt.mode == "random":
        run_random(e, w, opt)
    elif opt.mode == "saccade":
        path = motion.saccade_path(motion.random_scan(10, seed=opt.seed), fps=opt.fps, seed=opt.seed)
        path = [r for r in path if r["t"] <= opt.seconds]
        run_timeline(e, w, motion.blink_track(path, seed=opt.seed))
    elif opt.mode == "pursuit":
        run_timeline(e, w, motion.blink_track(
            motion.pursuit_path(duration=opt.seconds, fps=opt.fps), rate_per_min=10, seed=opt.seed))
    elif opt.mode == "identities":
        run_identities(e, w)
    elif opt.mode == "cameras":
        run_cameras(e, w)
    elif opt.mode == "vergence":
        run_vergence(e, w)
    elif opt.mode == "rays":
        run_rays(e, w)
    w.save({"mode": opt.mode, "engine": opt.engine, "size": opt.size, "seed": opt.seed,
            "fps": opt.fps if opt.mode in ("saccade", "pursuit") else None,
            "seconds": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
