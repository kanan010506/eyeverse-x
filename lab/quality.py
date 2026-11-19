"""
Dataset validation and per-frame confidence labels.

check_schema()     official metadata.json keys and types
validate()         per-frame checks on any labels list: files, bounds, visibility,
                   geometry consistency, detector agreement, duplicates
"""
from pathlib import Path

import numpy as np
from PIL import Image

from lab import detector

REQUIRED = {
    "frame": int,
    "theta": float,
    "input": dict,
    "eye_rotation_L": list,
    "eye_rotation_R": list,
    "pupil_center_2D_L": list,
    "pupil_center_2D_R": list,
    "camera_matrix": dict,
}


def check_schema(records, n_expected=10):
    problems = []
    if not isinstance(records, list):
        return ["metadata.json must be a list of records"]
    if len(records) != n_expected:
        problems.append(f"expected {n_expected} records, found {len(records)}")
    for i, rec in enumerate(records):
        for key, kind in REQUIRED.items():
            if key not in rec:
                problems.append(f"record {i}: missing {key}")
            elif kind is float and not isinstance(rec[key], (int, float)):
                problems.append(f"record {i}: {key} should be a number")
            elif kind is not float and not isinstance(rec[key], kind):
                problems.append(f"record {i}: {key} should be {kind.__name__}")
        if rec.get("frame") != i:
            problems.append(f"record {i}: frame index is {rec.get('frame')}")
        inp = rec.get("input", {})
        if set(inp) != {"h", "v"}:
            problems.append(f"record {i}: input must hold exactly h and v")
        for key in ("eye_rotation_L", "eye_rotation_R"):
            if len(rec.get(key, [])) != 3:
                problems.append(f"record {i}: {key} needs 3 angles")
        for key in ("pupil_center_2D_L", "pupil_center_2D_R"):
            if len(rec.get(key, [])) != 2:
                problems.append(f"record {i}: {key} needs [x, y]")
    return problems


def confidence(rec, det_side=None):
    """0..1 trust in the pupil label for one eye, plus a coarse class."""
    side = det_side or "L"
    vis = rec.get(f"pupil_visibility_{side}", 1.0)
    lid = rec.get("lid_closure", 0.0)
    score = vis * (1.0 - 0.8 * lid)
    if rec.get("sample", {}).get("glasses"):
        score *= 0.9
    label = "clear" if score > 0.85 else "partial" if score > 0.4 else "occluded"
    return round(float(score), 3), label


def validate(records, root, detect=True, max_px=3.0):
    root = Path(root)
    flagged, seen = [], {}
    stats = {"frames": len(records), "images_ok": 0, "in_bounds": 0, "visible": 0,
             "detector_checked": 0, "detector_agree": 0}
    errors = []
    for i, rec in enumerate(records):
        issues = []
        img_path = root / rec["image"]
        if not img_path.exists():
            issues.append("image missing")
        else:
            with Image.open(img_path) as im:
                size = im.size
            want = rec["camera_matrix"]["image_size"]
            if list(size) != list(want):
                issues.append(f"image is {size}, metadata says {want}")
            else:
                stats["images_ok"] += 1
        w, h = rec["camera_matrix"]["image_size"]
        inside = all(0 <= rec[f"pupil_center_2D_{s}"][0] < w and 0 <= rec[f"pupil_center_2D_{s}"][1] < h
                     for s in ("L", "R"))
        stats["in_bounds"] += inside
        if not inside:
            issues.append("pupil outside the image")
        if rec.get("aim_error_deg", 0.0) > 0.05:
            issues.append(f"eye aim off by {rec['aim_error_deg']:.3f} deg")
        if rec.get("projection_check_px", 0.0) > 0.01:
            issues.append("projection disagrees with Blender")
        labels = {}
        for s in ("L", "R"):
            labels[s] = confidence(rec, s)
        rec["confidence_L"], rec["label_L"] = labels["L"]
        rec["confidence_R"], rec["label_R"] = labels["R"]
        if min(labels["L"][0], labels["R"][0]) > 0.85:
            stats["visible"] += 1
        key = (round(rec["input"]["h"], 4), round(rec["input"]["v"], 4), rec.get("camera"), rec.get("identity"), rec.get("t"),
               round(rec.get("fixation_distance", 0.35), 3), str(rec.get("sample", "")))
        if key in seen:
            issues.append(f"duplicate of frame {seen[key]}")
        seen.setdefault(key, i)
        if detect and img_path.exists() and min(labels["L"][0], labels["R"][0]) > 0.85 \
                and not rec.get("sample", {}).get("cornea_refraction"):
            res = detector.evaluate(rec, img_path)
            stats["detector_checked"] += 1
            errs = [res[s]["error_px"] for s in ("L", "R") if res[s].get("found")]
            rec["detector"] = res
            if len(errs) == 2 and max(errs) <= max_px:
                stats["detector_agree"] += 1
            else:
                issues.append(f"detector disagrees ({errs} px)")
            errors.extend(errs)
        if issues:
            flagged.append({"frame": i, "image": rec["image"], "issues": issues})
    stats["detector_median_px"] = round(float(np.median(errors)), 3) if errors else None
    stats["flagged"] = len(flagged)
    return {"stats": stats, "flagged": flagged}
