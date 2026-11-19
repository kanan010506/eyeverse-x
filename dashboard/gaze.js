// Gaze maths for the browser preview. Mirrors wrapper/gaze.py and wrapper/camera.py
// so the live sliders show exactly what Blender would render.
(function (root) {
  const FORWARD = [0, -1, 0];

  const sub = (a, b) => a.map((x, i) => x - b[i]);
  const add = (a, b) => a.map((x, i) => x + b[i]);
  const scale = (a, s) => a.map((x) => x * s);
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => scale(a, 1 / Math.hypot(...a));

  function fixationPoint(h, v, distance, span) {
    return [h * span, -distance, v * span];
  }

  function arcQuat(a, b) {
    a = norm(a); b = norm(b);
    const d = Math.max(-1, Math.min(1, dot(a, b)));
    const ax = cross(a, b);
    const q = [1 + d, ...ax];
    const n = Math.hypot(...q);
    return q.map((x) => x / n);
  }

  function quatMatrix([w, x, y, z]) {
    return [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ];
  }

  function eulerXYZ(m) {
    const sy = Math.max(-1, Math.min(1, -m[2][0]));
    return [Math.atan2(m[2][1], m[2][2]), Math.asin(sy), Math.atan2(m[1][0], m[0][0])];
  }

  function project(p, K, w2c) {
    const c = [0, 1, 2].map((i) => dot(w2c[i].slice(0, 3), p) + w2c[i][3]);
    return [K[0][0] * c[0] / c[2] + K[0][2], K[1][1] * c[1] / c[2] + K[1][2]];
  }

  function solve(h, v, opts) {
    const t = fixationPoint(h, v, opts.distance, opts.span);
    const eyes = {};
    for (const side of ["L", "R"]) {
      const c = opts.centres[side];
      const d = norm(sub(t, c));
      const q = arcQuat(FORWARD, d);
      const pupil = add(c, scale(d, opts.radius));
      eyes[side] = {
        centre: c, direction: d, quaternion: q, euler: eulerXYZ(quatMatrix(q)), pupil,
        uv: opts.K ? project(pupil, opts.K, opts.w2c) : null,
        yaw: (Math.atan2(d[0], -d[1]) * 180) / Math.PI,
        pitch: (Math.asin(d[2]) * 180) / Math.PI,
      };
    }
    const vergence = (Math.acos(Math.max(-1, Math.min(1, dot(eyes.L.direction, eyes.R.direction)))) * 180) / Math.PI;
    return { target: t, eyes, vergence };
  }

  function pupilDiameter(luminance) {
    const d = 4.9 - 3 * Math.tanh(0.4 * Math.log10(Math.max(luminance, 1e-4)));
    return Math.min(8, Math.max(2, d));
  }

  const api = { solve, project, pupilDiameter, fixationPoint };
  if (typeof module !== "undefined") module.exports = api;
  else root.Gaze = api;
})(this);
