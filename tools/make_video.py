"""
Turn a rendered timeline (saccade / pursuit) into an MP4, or build the demo reel.

    python3 tools/make_video.py saccade            -> output/saccade.mp4
    python3 tools/make_video.py demo               -> output/demo.mp4  (sequence + timelines)

Needs ffmpeg on PATH (brew install ffmpeg).
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.contact_sheet import ACCENT, INK, MUTED, font  # noqa: E402
from wrapper import config as cfg  # noqa: E402

SIZE = 1024


def caption(img, title, sub=None):
    img = img.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, SIZE - 74, SIZE, SIZE], fill=(10, 12, 16, 200))
    d.text((28, SIZE - 64), title, font=font(26, True), fill=INK)
    if sub:
        d.text((28, SIZE - 30), sub, font=font(17), fill=MUTED)
    d.rectangle([0, SIZE - 74, 6, SIZE], fill=ACCENT)
    return img


def encode(frames_dir, out, fps):
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found; install it with: brew install ffmpeg")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(Path(frames_dir) / "%05d.png"), "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "18", "-pix_fmt", "yuv420p", str(out)], check=True)
    print("saved", out)


def timeline(mode):
    src = cfg.EXTRA_DIR / mode
    data = json.loads((src / "labels.json").read_text())
    fps = data["meta"].get("fps") or 30
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(data["frames"]):
            sub = f"t {f['t']:.2f} s   {f['event']}   h {f['input']['h']:+.2f}  v {f['input']['v']:+.2f}"
            caption(Image.open(src / f["image"]), mode.capitalize(), sub).save(Path(tmp) / f"{i:05d}.png")
        encode(tmp, cfg.EXTRA_DIR / f"{mode}.mp4", fps)


def demo():
    """Official sequence held per frame, then any rendered timelines, at 30 fps."""
    fps, hold = 30, 30
    shots = []
    for rec in json.loads(cfg.METADATA_FILE.read_text()):
        sub = f"h {rec['input']['h']:+.3f}   v {rec['input']['v']:+.3f}   θ {rec['theta']:.3f}"
        shots += [(ROOT / rec["image"], f"Official frame {rec['frame']:02d}", sub)] * hold
    for name, label in (("left", "Left"), ("right", "Right"), ("up", "Up"), ("down", "Down"), ("reset", "Reset")):
        p = cfg.EXTRA_DIR / "directions" / f"{name}.png"
        if p.exists():
            shots += [(p, f"Direction check: {label}", "eyes.move(h, v) / eyes.reset()")] * hold
    for mode in ("saccade", "pursuit"):
        lab = cfg.EXTRA_DIR / mode / "labels.json"
        if lab.exists():
            for f in json.loads(lab.read_text())["frames"]:
                shots.append((cfg.EXTRA_DIR / mode / f["image"], mode.capitalize(),
                              f"t {f['t']:.2f} s   {f['event']}"))
    with tempfile.TemporaryDirectory() as tmp:
        cache = {}
        for i, (path, title, sub) in enumerate(shots):
            key = (path, title, sub)
            if key not in cache:
                cache = {key: caption(Image.open(path), title, sub)}
            cache[key].save(Path(tmp) / f"{i:05d}.png")
        encode(tmp, cfg.EXTRA_DIR / "demo.mp4", fps)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "demo"
    demo() if what == "demo" else timeline(what)
