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
