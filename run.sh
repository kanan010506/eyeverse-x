#!/usr/bin/env bash
# EYEVERSE X task runner. Set BLENDER if blender is not on PATH, e.g.
#   export BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${BLENDER:-}" ]]; then
  if command -v blender >/dev/null 2>&1; then BLENDER=blender
  elif [[ -x /Applications/Blender.app/Contents/MacOS/Blender ]]; then BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
  else echo "Blender not found. export BLENDER=/path/to/blender" >&2; exit 1; fi
fi
PY="${PYTHON:-python3}"
B() { "$BLENDER" --background --factory-startup "$@"; }

case "${1:-help}" in
  scene)      B --python tools/build_scene.py -- "${@:2}" ;;
  official)   B scene.blend --python tools/render_official.py -- "${@:2}" ;;
  sheet)      "$PY" tools/contact_sheet.py ;;
  dataset)    B scene.blend --python tools/generate_dataset.py -- "${@:2}" ;;
  report)     "$PY" tools/report.py ;;
  dashboard)  "$PY" dashboard/build.py && echo "open dashboard/index.html" ;;
  video)      "$PY" tools/make_video.py "${@:2}" ;;
  package)    "$PY" tools/package.py "${@:2}" ;;
  test)       "$PY" -m pytest -q tests ;;
  all)
    "$0" scene
    "$0" official
    "$0" sheet
    "$0" report
    "$0" dashboard ;;
  *)
    echo "usage: ./run.sh {scene|official|sheet|dataset|report|dashboard|video|package|test|all}" ;;
esac
