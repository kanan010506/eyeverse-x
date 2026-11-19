"""
Build the submission bundle in the layout evaluators expect, then re-open it to check it.

    python3 tools/package.py [TIET_TheTietans]

TIET_TheTietans.zip
    scene.blend
    wrapper/          eyes.py + the modules it imports
    renders/          frame_00.png .. frame_09.png
    metadata.json
    README.md
    contact_sheet.png
    demo.mp4, TIET_TheTietans.pptx, output/directions/   when present
    SHA256SUMS
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lab import quality  # noqa: E402

WRAPPER = ("__init__.py", "config.py", "gaze.py", "camera.py", "eyes.py", "truth.py", "safety.py")
TOOLS = ("__init__.py", "render_official.py")


def main(name="TIET_TheTietans"):
    frames = [ROOT / "renders" / f"frame_{k:02d}.png" for k in range(10)]
    need = [ROOT / "scene.blend", ROOT / "metadata.json", ROOT / "README.md", *frames]
    missing = [str(p.relative_to(ROOT)) for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}. Run ./run.sh all first.")
    problems = quality.check_schema(json.loads((ROOT / "metadata.json").read_text()))
    if problems:
        raise SystemExit("metadata.json: " + "; ".join(problems))

    out = ROOT / "dist" / f"{name}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "scene.blend", "scene.blend")
        for f in WRAPPER:
            z.write(ROOT / "wrapper" / f, f"wrapper/{f}")
        for f in TOOLS:
            z.write(ROOT / "tools" / f, f"tools/{f}")
        for p in frames:
            z.write(p, f"renders/{p.name}")
        for f in ("metadata.json", "README.md", "SECURITY.md", "LICENSE", "contact_sheet.png", "requirements.txt"):
            if (ROOT / f).exists():
                z.write(ROOT / f, f)
        # optional extras: the README's direction-check images, the demo reel and the deck
        for p in sorted((ROOT / "output" / "directions").glob("*")):
            z.write(p, f"output/directions/{p.name}")
        for src, arc in (("output/eyeverse_demo.mp4", "demo.mp4"), ("docs/TIET_TheTietans.pptx", "TIET_TheTietans.pptx")):
            if (ROOT / src).exists():
                z.write(ROOT / src, arc)

    # checksums for every file, written last so the bundle can be verified after download
    with zipfile.ZipFile(out) as z:
        lines = [f"{hashlib.sha256(z.read(n)).hexdigest()}  {n}" for n in sorted(z.namelist())]
    with zipfile.ZipFile(out, "a", zipfile.ZIP_DEFLATED) as z:
        z.writestr("SHA256SUMS", "\n".join(lines) + "\n")

    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert z.testzip() is None
        sums = dict(line.split("  ", 1)[::-1] for line in z.read("SHA256SUMS").decode().splitlines())
        for n, digest in sums.items():
            if hashlib.sha256(z.read(n)).hexdigest() != digest:
                raise SystemExit(f"checksum mismatch for {n}")
    expected = {"scene.blend", "wrapper/eyes.py", "metadata.json", "README.md",
                *(f"renders/frame_{k:02d}.png" for k in range(10))}
    lost = expected - names
    if lost:
        raise SystemExit(f"bundle is missing {sorted(lost)}")
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, {len(names)} files)")


if __name__ == "__main__":
    main(*sys.argv[1:2])
