"""
Guardrails shared by the wrapper and the tools. Pure Python, no bpy.
"""
from pathlib import Path

from . import config as cfg

PROTECTED = ("wrapper", "tools", "lab", "assets", "dashboard", "tests")


def safe_output(path, suffix=".png", root=cfg.ROOT):
    """Resolve an output path: fixed suffix, never inside the source folders,
    never onto a symlink. Parent folders are created."""
    path = Path(path).expanduser()
    if path.suffix.lower() != suffix:
        raise ValueError(f"expected a {suffix} file, got {path.name!r}")
    if path.is_symlink():
        raise ValueError("refusing to overwrite a symlink")
    path = path.resolve()
    for name in PROTECTED:
        if path.is_relative_to(Path(root).resolve() / name):
            raise ValueError(f"refusing to write into the source folder {name}/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def inside(path, folder):
    """True when path resolves to somewhere under folder."""
    return Path(path).resolve().is_relative_to(Path(folder).resolve())
