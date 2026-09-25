// Planar geometry helpers. Points are [x, y]; a homography is a row-major 9-array.

// Homography mapping the unit square (0,0)(1,0)(1,1)(0,1) onto quad q = [TL, TR, BR, BL].
export function squareToQuad(q) {
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = q;
  const dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3;
  const dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;
  let a, b, d, e, g, h;
  if (Math.abs(dx3) < 1e-9 && Math.abs(dy3) < 1e-9) {
    // parallelogram: plain affine mapping
    a = x1 - x0; b = x3 - x0; d = y1 - y0; e = y3 - y0; g = 0; h = 0;
  } else {
    const den = dx1 * dy2 - dx2 * dy1;
    g = (dx3 * dy2 - dx2 * dy3) / den;
    h = (dx1 * dy3 - dx3 * dy1) / den;
    a = x1 - x0 + g * x1; b = x3 - x0 + h * x3;
    d = y1 - y0 + g * y1; e = y3 - y0 + h * y3;
  }
  return [a, b, x0, d, e, y0, g, h, 1];
}

export function applyH(H, [x, y]) {
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}

// Convert a homography estimated at a scaled resolution (points multiplied by s) to full resolution.
export function applyScaledH(H, p, s) {
  const r = applyH(H, [p[0] * s, p[1] * s]);
  return [r[0] / s, r[1] / s];
}

// Sanity check so a bad tracking estimate can't collapse or flip the quad.
export function isSaneQuad(q) {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const [ax, ay] = q[i], [bx, by] = q[(i + 1) % 4], [cx, cy] = q[(i + 2) % 4];
    const cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx);
    if (!isFinite(cross) || cross === 0) return false;
    const s = Math.sign(cross);
    if (sign && s !== sign) return false;
    sign = s;
  }
  return true;
}

// Stricter than isSaneQuad: also rejects the twisted, stretched shapes a bad motion estimate produces
// (near-flat corners, opposite sides of wildly different length, corners far outside the picture).
export function isPlausibleQuad(q, vw, vh) {
  if (!isSaneQuad(q)) return false;
  if (q.some(([x, y]) => !(x > -3 * vw && x < 4 * vw && y > -3 * vh && y < 4 * vh))) return false;
  const len = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  for (let i = 0; i < 4; i++) {
    const p = q[i], a = q[(i + 3) % 4], b = q[(i + 1) % 4];
    const u = [a[0] - p[0], a[1] - p[1]], v = [b[0] - p[0], b[1] - p[1]];
    const deg = Math.acos((u[0] * v[0] + u[1] * v[1]) / (Math.hypot(...u) * Math.hypot(...v))) * 180 / Math.PI;
    if (!(deg > 40 && deg < 140)) return false;
  }
  const ratio = (a, b) => Math.max(a, b) / Math.max(1e-6, Math.min(a, b));
  return ratio(len(q[0], q[1]), len(q[3], q[2])) < 3 && ratio(len(q[0], q[3]), len(q[1], q[2])) < 3;
}

// Signed-area-free polygon area (shoelace).
export function quadArea(q) {
  let a = 0;
  q.forEach((p, i) => { const n = q[(i + 1) % 4]; a += p[0] * n[1] - n[0] * p[1]; });
  return Math.abs(a) / 2;
}

// Does the quad cover any part of the w x h picture? (samples a grid; the quad is convex)
export function quadInView(q, w, h) {
  const sign = (a, b, p) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
  for (let i = 0; i <= 8; i++) {
    for (let j = 0; j <= 8; j++) {
      const p = [w * i / 8, h * j / 8];
      const s = [0, 1, 2, 3].map((k) => sign(q[k], q[(k + 1) % 4], p));
      if (s.every((v) => v >= 0) || s.every((v) => v <= 0)) return true;
    }
  }
  return false;
}
