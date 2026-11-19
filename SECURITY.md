# Security and safety

EYEVERSE X generates training data that downstream gaze models will learn from, and it is meant
to run unattended on shared machines. The guardrails below keep bad input from producing bad
labels, keep the tools from writing where they should not, and make every output verifiable.

## What is protected

| Risk | Guardrail | Where |
|---|---|---|
| Out-of-range or non-numeric gaze input silently producing wrong labels | `h`, `v` must be finite and in `[-1, 1]`; fixation distance in `[0.10, 10]` m; pupil size and lid closure clamped and type-checked. Bad input raises `ValueError` instead of rendering. | `wrapper/gaze.py`, `wrapper/eyes.py` |
| Labels drifting from the image | Every pupil projection is cross-checked against Blender's own camera transform (0.01 px tolerance) and every eye's aim is re-measured from the posed scene. A mismatch stops the run. | `wrapper/truth.py` |
| Writing over source code or through symlinks | Renders must be `.png`, are never written inside `wrapper/`, `tools/`, `lab/`, `assets/`, `dashboard/` or `tests/`, and never onto a symlink. Dataset output is confined to `output/`. | `wrapper/safety.py`, `tools/generate_dataset.py` |
| Resource exhaustion | Dataset size, resolution, duration, frame rate and sample counts are capped. | `tools/generate_dataset.py` |
| Remote job execution | The render queue only runs the project's own scripts from an allowlist, checks every argument against a strict pattern, refuses absolute paths and `..`, only opens `scene.blend`, and never invokes a shell. Rejected queues are logged, not run. | `tools/runner.py` |
| Untrusted Blender content | All Blender jobs run with `--factory-startup` (no user add-ons or startup scripts) and `scene.blend` contains no text blocks or drivers that execute code. | `run.sh`, `tools/runner.py` |
| Tampered or corrupted bundle | The submission zip carries `SHA256SUMS` for every file and is re-opened and verified after it is written. | `tools/package.py` |
| Dashboard injection | Data is embedded as JSON with `</` escaped, text is inserted with `textContent` or escaped, and a Content-Security-Policy limits the page to local files plus the font CDN. | `dashboard/build.py`, `dashboard/template.html` |
| Privacy | No real person is captured: the face is a CC0 parametric model and all eye images are synthetic, which is the point of the project. | — |

## Verifying a bundle

```bash
unzip TIET_TheTietans.zip -d bundle && cd bundle
shasum -a 256 -c SHA256SUMS
```

## Tests

`tests/test_security.py` covers the input bounds, the path rules and the job allowlist (shell
injection, path traversal, foreign `.blend` files, arbitrary `python -c` and `pip`).

## Reporting

Please open a GitHub issue marked **security**, or email the team, rather than posting details
publicly.
