"""
Calibration and dataset report.

    python3 tools/report.py

Reads metadata.json, the official renders and any lab datasets in output/, and writes
    output/report/calibration_report.json
    output/report/calibration_report.md
    output/report/heatmap_*.png, quality_*.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lab import detector, heatmap, quality  # noqa: E402
from wrapper import camera, config as cfg, gaze  # noqa: E402

OUT = cfg.EXTRA_DIR / "report"


def expected_pixels(rec):
    """Pupil pixels predicted from the analytic model alone (no Blender)."""
    centres = {s: tuple(rec[f"eye_center_{s}"]) for s in gaze.SIDES}
    radius = float(np.linalg.norm(np.array(rec["pupil_center_3D_L"]) - np.array(rec["eye_center_L"])))
    sol = gaze.solve(rec["input"]["h"], rec["input"]["v"], rec.get("fixation_distance", cfg.FIXATION_DISTANCE),
                     centres=centres, radius=radius)
    K = np.array(rec["camera_matrix"]["K"])
    w2c = np.array(rec["camera_matrix"]["world_to_camera"])
    return {s: camera.project(sol["eyes"][s]["pupil"], K, w2c)[0][0] for s in gaze.SIDES}, sol


def clockwise_ok(records):
    """Pupil displacement angle (from the mean position) must turn clockwise frame to frame."""
    pts = np.array([[(r["pupil_center_2D_L"][0] + r["pupil_center_2D_R"][0]) / 2,
                     (r["pupil_center_2D_L"][1] + r["pupil_center_2D_R"][1]) / 2] for r in records])
    rel = pts - pts.mean(axis=0)
    # image v points down; atan2(du, -dv) is 0 at the top and grows clockwise
    ang = np.unwrap(np.arctan2(rel[:, 0], -rel[:, 1]))
    steps = np.diff(ang)
    return bool(np.all(steps > 0)), [round(math.degrees(a) % 360, 2) for a in ang]


def official():
    records = json.loads(cfg.METADATA_FILE.read_text())
    rows = []
    for rec in records:
        exp, sol = expected_pixels(rec)
        det = detector.evaluate(rec, ROOT / rec["image"])
        row = {"frame": rec["frame"], "h": rec["input"]["h"], "v": rec["input"]["v"],
               "aim_error_deg": rec.get("aim_error_deg"), "vergence_deg": rec.get("vergence_deg")}
        for s in gaze.SIDES:
            gt = np.array(rec[f"pupil_center_2D_{s}"])
            row[f"model_vs_scene_px_{s}"] = round(float(np.linalg.norm(exp[s] - gt)), 4)
            row[f"detected_vs_truth_px_{s}"] = det[s].get("error_px")
            row[f"gaze_reconstruction_deg_{s}"] = det[s].get("angular_error_deg")
            row[f"pupil_{s}"] = [round(float(a), 2) for a in gt]
            row[f"detected_{s}"] = det[s].get("detected")
        rows.append(row)

    def stat(key):
        vals = [r[f"{key}_{s}"] for r in rows for s in gaze.SIDES if r[f"{key}_{s}"] is not None]
        return {"mean": round(float(np.mean(vals)), 4), "max": round(float(np.max(vals)), 4)} if vals else None

    cw, angles = clockwise_ok(records)
    directions = cfg.EXTRA_DIR / "directions" / "directions.json"
    summary = {
        "schema_problems": quality.check_schema(records),
        "clockwise": cw,
        "pupil_angles_deg": angles,
        "model_vs_scene_px": stat("model_vs_scene_px"),
        "detected_vs_truth_px": stat("detected_vs_truth_px"),
        "gaze_reconstruction_deg": stat("gaze_reconstruction_deg"),
        "max_aim_error_deg": max(r["aim_error_deg"] or 0 for r in rows),
        "directions": json.loads(directions.read_text()) if directions.exists() else None,
        "camera": records[0]["camera_matrix"],
        "frames": rows,
    }
    return summary, records


def datasets():
    found = {}
    if not cfg.EXTRA_DIR.exists():
        return found
    for labels in sorted(cfg.EXTRA_DIR.glob("*/labels.json")):
        mode = labels.parent.name
        data = json.loads(labels.read_text())
        frames = data["frames"]
        qc = quality.validate(frames, labels.parent, detect=mode != "rays")
        (OUT / f"quality_{mode}.json").write_text(json.dumps(qc, indent=1))
        labels.with_name("labels_qc.json").write_text(json.dumps({"meta": data["meta"], "frames": frames}, indent=1))
        entry = {"frames": len(frames), "quality": qc["stats"]}
        if mode in ("saccade", "pursuit", "random"):
            hv = [(f["input"]["h"], f["input"]["v"]) for f in frames]
            Image.fromarray(heatmap.target_heatmap(hv)).save(OUT / f"heatmap_{mode}_target.png")
            base = labels.parent / frames[0]["image"]
            uv = [f[f"pupil_center_2D_{s}"] for f in frames for s in gaze.SIDES
                  if f.get("camera", "EyeCam") == "EyeCam"]
            scale = Image.open(base).size[0] / frames[0]["camera_matrix"]["image_size"][0]
            uv = [(u * scale, v * scale) for u, v in uv]
            if uv:
                Image.fromarray(heatmap.pupil_heatmap(base, uv)).save(OUT / f"heatmap_{mode}_pupils.png")
            entry["heatmaps"] = [p.name for p in OUT.glob(f"heatmap_{mode}_*.png")]
        found[mode] = entry
    return found


def markdown(summary, lab):
    s = summary
    lines = ["# Calibration report", ""]
    lines.append(f"- schema: {'ok' if not s['schema_problems'] else '; '.join(s['schema_problems'])}")
    lines.append(f"- clockwise order: {'ok' if s['clockwise'] else 'FAILED'}")
    if s["directions"]:
        d = s["directions"]
        lines.append(f"- direction check: {d['passed']}/{d['of']} passed (min shift {d['min_shift_px']} px)")
    lines.append(f"- analytic model vs scene: mean {s['model_vs_scene_px']['mean']} px, "
                 f"max {s['model_vs_scene_px']['max']} px")
    if s["detected_vs_truth_px"]:
        lines.append(f"- image detector vs ground truth: mean {s['detected_vs_truth_px']['mean']} px, "
                     f"max {s['detected_vs_truth_px']['max']} px")
    if s["gaze_reconstruction_deg"]:
        lines.append(f"- 3D gaze reconstructed from images: mean {s['gaze_reconstruction_deg']['mean']} deg, "
                     f"max {s['gaze_reconstruction_deg']['max']} deg")
    lines.append(f"- worst eye aim error: {s['max_aim_error_deg']} deg")
    lines += ["", "| frame | h | v | pupil L (px) | pupil R (px) | detect err L/R (px) | gaze err L/R (deg) |",
              "|---|---|---|---|---|---|---|"]
    for r in s["frames"]:
        lines.append(f"| {r['frame']:02d} | {r['h']:+.3f} | {r['v']:+.3f} | {r['pupil_L']} | {r['pupil_R']} | "
                     f"{r['detected_vs_truth_px_L']} / {r['detected_vs_truth_px_R']} | "
                     f"{r['gaze_reconstruction_deg_L']} / {r['gaze_reconstruction_deg_R']} |")
    if lab:
        lines += ["", "## Lab datasets", "", "| set | frames | flagged | detector median (px) |", "|---|---|---|---|"]
        for mode, e in lab.items():
            q = e["quality"]
            lines.append(f"| {mode} | {e['frames']} | {q['flagged']} | {q['detector_median_px']} |")
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, _ = official()
    lab = datasets()
    summary["datasets"] = lab
    (OUT / "calibration_report.json").write_text(json.dumps(summary, indent=1))
    (OUT / "calibration_report.md").write_text(markdown(summary, lab))
    print(markdown(summary, lab))


if __name__ == "__main__":
    main()
