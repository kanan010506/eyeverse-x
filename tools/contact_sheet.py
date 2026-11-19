"""
5 x 2 contact sheet of the ten official frames with their inputs.

    python3 tools/contact_sheet.py            -> contact_sheet.png
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wrapper import config as cfg  # noqa: E402

TILE = 400
PAD = 16
HEADER = 64
CAPTION = 46
INK = (236, 238, 242)
MUTED = (150, 156, 168)
ACCENT = (64, 156, 255)
BG = (14, 16, 20)


def font(size, bold=False):
    names = (["SF-Pro-Display-Semibold.otf", "HelveticaNeue.ttc", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"]
             if bold else ["SF-Pro-Text-Regular.otf", "HelveticaNeue.ttc", "Arial.ttf", "DejaVuSans.ttf"])
    dirs = ["/System/Library/Fonts", "/Library/Fonts", "/usr/share/fonts/truetype/dejavu", ""]
    for name in names:
        for d in dirs:
            try:
                return ImageFont.truetype(str(Path(d) / name) if d else name, size)
            except OSError:
                continue
    return ImageFont.load_default()


def arrow(draw, cx, cy, h, v, r=13):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=MUTED, width=1)
    ex, ey = cx + h * (r - 3), cy - v * (r - 3)
    draw.line([cx, cy, ex, ey], fill=ACCENT, width=2)
    draw.ellipse([ex - 2.5, ey - 2.5, ex + 2.5, ey + 2.5], fill=ACCENT)


def main(out=ROOT / "contact_sheet.png"):
    records = json.loads(cfg.METADATA_FILE.read_text())
    cols, rows = 5, 2
    w = cols * TILE + (cols + 1) * PAD
    h = HEADER + rows * (TILE + CAPTION) + (rows + 1) * PAD
    sheet = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(sheet)
    d.text((PAD + 2, 18), "EYEVERSE X  ·  official clockwise sequence", font=font(24, True), fill=INK)
    note = "θk = 2πk/10   h = sin θ   v = cos θ   fixation 35 cm   camera 40 cm"
    d.text((w - PAD - d.textlength(note, font=font(15)), 24), note, font=font(15), fill=MUTED)
    for i, rec in enumerate(records):
        r, c = divmod(i, cols)
        x = PAD + c * (TILE + PAD)
        y = HEADER + PAD + r * (TILE + CAPTION + PAD)
        tile = Image.open(ROOT / rec["image"]).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
        sheet.paste(tile, (x, y))
        h_, v_ = rec["input"]["h"], rec["input"]["v"]
        d.text((x + 2, y + TILE + 8), f"frame {rec['frame']:02d}", font=font(16, True), fill=INK)
        d.text((x + 2, y + TILE + 27), f"h {h_:+.3f}   v {v_:+.3f}", font=font(14), fill=MUTED)
        arrow(d, x + TILE - 18, y + TILE + 24, h_, v_)
    sheet.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
