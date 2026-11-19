"""
Official Track 1 output: ten clockwise renders, metadata.json and the direction check.

    blender -b scene.blend -P tools/render_official.py [-- --skip-directions]

renders/frame_00.png .. frame_09.png    theta_k = 2*pi*k/10, h = sin, v = cos, frame 0 at the top
metadata.json                          one record per frame
output/directions/*.png + .json        reset, left, right, up, down with measured pupil shifts
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wrapper import config as cfg, eyes, gaze  # noqa: E402

DIRECTIONS = [("reset", 0.0, 0.0), ("left", -1.0, 0.0), ("right", 1.0, 0.0),
              ("up", 0.0, 1.0), ("down", 0.0, -1.0)]
MIN_SHIFT_PX = 3.0


def sequence(e):
    records = []
    for row in gaze.sequence():
        e.move(row["h"], row["v"])
        name = f"frame_{row['frame']:02d}.png"
        e.render(cfg.RENDER_DIR / name)
        rec = {"frame": row["frame"], "theta": round(row["theta"], 6)}
        rec.update(e.record())
        rec["image"] = f"renders/{name}"
        records.append(rec)
        print(f"frame {row['frame']:02d}  h={row['h']:+.3f} v={row['v']:+.3f}  "
              f"L={rec['pupil_center_2D_L']}  R={rec['pupil_center_2D_R']}", flush=True)
    cfg.METADATA_FILE.write_text(json.dumps(records, indent=2))
    return records


def direction_check(e):
    out_dir = cfg.EXTRA_DIR / "directions"
    rows = {}
    for name, h, v in DIRECTIONS:
        e.move(h, v)
        e.render(out_dir / f"{name}.png")
        rows[name] = e.record()
    base = rows["reset"]
    verdict = {}
    for name, h, v in DIRECTIONS[1:]:
        shifts = {}
        ok = True
        for side in gaze.SIDES:
            du = rows[name][f"pupil_center_2D_{side}"][0] - base[f"pupil_center_2D_{side}"][0]
            dv = rows[name][f"pupil_center_2D_{side}"][1] - base[f"pupil_center_2D_{side}"][1]
            shifts[side] = [round(du, 3), round(dv, 3)]
            # image v grows downwards, so looking up means a negative dv
            main = du * h if h else -dv * v
            ok &= main >= MIN_SHIFT_PX
        verdict[name] = {"input": [h, v], "shift_px": shifts, "pass": bool(ok)}
    # reset must bring both pupils back to the rest position from anywhere
    e.move(0.8, -0.6)
    e.reset()
    back = e.record()
    drift = max(abs(back[f"pupil_center_2D_{s}"][i] - base[f"pupil_center_2D_{s}"][i])
                for s in gaze.SIDES for i in (0, 1))
    verdict["reset"] = {"input": [0.0, 0.0], "return_error_px": round(drift, 6), "pass": drift < 0.05}
    report = {"min_shift_px": MIN_SHIFT_PX, "directions": verdict,
              "passed": sum(d["pass"] for d in verdict.values()), "of": len(verdict)}
    (out_dir / "directions.json").write_text(json.dumps(report, indent=2))
    print(f"direction check: {report['passed']}/{report['of']} passed", flush=True)
    return report


def run(skip_directions=False):
    t0 = time.time()
    e = eyes.get()
    e.reset()
    sequence(e)
    if not skip_directions:
        direction_check(e)
    e.reset()
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    run(skip_directions="--skip-directions" in argv)
