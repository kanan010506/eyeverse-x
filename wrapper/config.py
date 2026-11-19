"""
Shared settings for the scene, the wrapper and the offline tools.

World frame (Blender): right-handed, Z up, metres.
    origin  midpoint between the two eyeball centres
    -Y      the direction the face looks
    +X      image right, which is the subject's left
The camera sits on the -Y axis and looks back at the face along +Y.

This file must stay importable without bpy.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# -- Track 1 calibration constraints ----------------------------------------
CAMERA_DISTANCE = 0.40          # m, locked camera in front of the eye midpoint
FIXATION_DISTANCE = 0.35        # m, fixation plane in front of the eye midpoint
FIXATION_RANGE = 0.05           # m, +/- motion bound on the fixation plane
IMAGE_SIZE = 1024               # px, square output
SEQUENCE_LENGTH = 10            # theta_k = 2*pi*k / 10

# 135 mm on a 36 mm sensor at 40 cm frames a 10.7 cm wide field: both eyes,
# brows and the nose bridge, so the pupil travel stays clearly visible.
LENS_MM = 135.0
SENSOR_MM = 36.0

# -- Eye anatomy (adult averages) ------------------------------------------
IPD = 0.063                     # m, interpupillary distance after scene scaling
EYE_RADIUS = 0.0120             # m, globe radius (24 mm axial length)
IRIS_RADIUS = 0.0059            # m, visible iris (11.8 mm diameter)
CORNEA_RADIUS = 0.0078          # m, anterior corneal curvature
CORNEA_IOR = 1.376
PUPIL_DIAMETER_MM = 4.0         # official renders use one fixed pupil size
PUPIL_LIMITS_MM = (2.0, 8.0)
DISTANCE_LIMITS = (0.10, 10.0)  # m, accepted fixation distances

# resting lid aperture: elevation of each lid edge straight ahead (deg), and how far the
# lid's own axis leans back, which arches the upper lid and cups the lower one
LID_UPPER_DEG = 22.0
LID_LOWER_DEG = -26.0
LID_ARCH_DEG = 55.0

# -- Scene objects ---------------------------------------------------------
RIG_NAME = "Human.rig"
FACE_NAME = "Human"
EYE_BONES = {"L": "eye.L", "R": "eye.R"}
EYEBALL_NAMES = {"L": "Eyeball.L", "R": "Eyeball.R"}
TARGET_NAME = "Fixation"
CAMERA_NAME = "EyeCam"

# -- Paths -------------------------------------------------------------------
def _path(env, default):
    return Path(os.environ.get(env, default))


SCENE_FILE = _path("EYEVERSE_SCENE", ROOT / "scene.blend")
FACE_SOURCE = _path("EYEVERSE_FACE", ROOT / "assets" / "face_source.blend")
RENDER_DIR = _path("EYEVERSE_RENDERS", ROOT / "renders")
METADATA_FILE = _path("EYEVERSE_METADATA", ROOT / "metadata.json")
EXTRA_DIR = _path("EYEVERSE_EXTRA", ROOT / "output")
