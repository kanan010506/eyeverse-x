"""
Build dashboard/index.html from metadata.json, the renders and the reports in output/.

    python3 dashboard/build.py          then open dashboard/index.html

The page is static: data is embedded, images are linked relatively, no server needed.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wrapper import config as cfg  # noqa: E402

HERE = Path(__file__).resolve().parent


def read(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def rel(path):
    return "../" + str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def lab_sets():
    sets = []
    for labels in sorted(cfg.EXTRA_DIR.glob("*/labels.json")):
        data = read(labels)
        frames = data["frames"]
        step = max(1, len(frames) // 12)
        sample = [{"src": rel(labels.parent / f["image"]), "h": f["input"]["h"], "v": f["input"]["v"],
                   "event": f.get("event") or f.get("pose") or f.get("camera", ""),
                   "label": f.get("label_L", "")} for f in frames[::step][:12]]
        mode = labels.parent.name
        qc = read(cfg.EXTRA_DIR / "report" / f"quality_{mode}.json", {})
        heat = sorted(rel(p) for p in (cfg.EXTRA_DIR / "report").glob(f"heatmap_{mode}_*.png"))
        video = cfg.EXTRA_DIR / f"{mode}.mp4"
        sets.append({"mode": mode, "meta": data["meta"], "count": len(frames), "sample": sample,
                     "quality": qc.get("stats"), "flagged": (qc.get("flagged") or [])[:8], "heatmaps": heat,
                     "video": rel(video) if video.exists() else None})
    return sets


def main():
    frames = read(cfg.METADATA_FILE, [])
    for f in frames:
        f["src"] = rel(ROOT / f["image"])
    directions = read(cfg.EXTRA_DIR / "directions" / "directions.json")
    dir_imgs = {n: rel(cfg.EXTRA_DIR / "directions" / f"{n}.png")
                for n in ("reset", "left", "right", "up", "down")
                if (cfg.EXTRA_DIR / "directions" / f"{n}.png").exists()}
    data = {
        "frames": frames,
        "report": read(cfg.EXTRA_DIR / "report" / "calibration_report.json"),
        "build": read(cfg.EXTRA_DIR / "build_report.json"),
        "directions": directions,
        "directionImages": dir_imgs,
        "sheet": rel(ROOT / "contact_sheet.png") if (ROOT / "contact_sheet.png").exists() else None,
        "lab": lab_sets(),
        "config": {"span": cfg.FIXATION_RANGE, "distance": cfg.FIXATION_DISTANCE,
                   "camera": cfg.CAMERA_DISTANCE, "lens": cfg.LENS_MM, "size": cfg.IMAGE_SIZE},
    }
    if directions:
        shifts = [abs(v[0 if d["input"][0] else 1]) for d in directions["directions"].values()
                  if "shift_px" in d for v in d["shift_px"].values()]
        data["travel"] = sum(shifts) / len(shifts) if shifts else None
    html = (HERE / "template.html").read_text()
    # embedded as data, never as markup: "</" cannot close the script element
    payload = json.dumps(data).replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = html.replace("/*__DATA__*/null", payload)
    html = html.replace("/*__GAZE__*/", (HERE / "gaze.js").read_text())
    (HERE / "index.html").write_text(html)
    print("wrote", HERE / "index.html")


if __name__ == "__main__":
    main()
