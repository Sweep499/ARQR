// Corrects a drifted outline using the detector's pod mask. The mask exists on every check, even when the pod is
// cut off by the picture edge and its corners are off-screen, so it can pull the outline back onto the pod when
// motion tracking has slipped. It finds the small shift, scale and turn of the outline that makes its visible
// part overlap the mask best, and prefers the smallest such change, because the parts of the outline that are
// off-screen cannot be checked.

import { IN_W as W, IN_H as H } from './detector.js';

const LAMBDA = 1.0;        // how strongly a bigger change is discouraged (per unit of shift, scale or turn)
const MAX_SHIFT = 0.25;    // the largest correction: 25% of the picture, 40% in size, about 17 degrees of turn
const MAX_SCALE = 0.40;
const MAX_TURN = 0.30;

// Row-wise prefix sums of the mask so the overlap with a convex polygon costs one lookup per picture row.
function prefix(mask) {
  const pre = new Int32Array(H * (W + 1));
  let total = 0;
  for (let y = 0; y < H; y++) {
    let acc = 0;
    const o = y * (W + 1);
    for (let x = 0; x < W; x++) { if (mask[y * W + x]) acc++; pre[o + x + 1] = acc; }
    total += acc;
  }
  return { pre, total };
}

// IoU between the mask and the part of a convex quad (in mask coordinates) that lies inside the picture.
function iou(P, m) {
  let ymin = Infinity, ymax = -Infinity;
  for (const p of P) { ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]); }
  const y0 = Math.max(0, Math.floor(ymin)), y1 = Math.min(H - 1, Math.ceil(ymax));
  let inter = 0, area = 0;
  for (let y = y0; y <= y1; y++) {
    const yc = y + 0.5;
    let xl = Infinity, xr = -Infinity;
    for (let i = 0; i < 4; i++) {
      const a = P[i], b = P[(i + 1) % 4];
      if ((a[1] <= yc && b[1] > yc) || (b[1] <= yc && a[1] > yc)) {
        const x = a[0] + (yc - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
        if (x < xl) xl = x;
        if (x > xr) xr = x;
      }
    }
    if (xl === Infinity) continue;
    const l = Math.max(0, Math.min(W, Math.round(xl))), r = Math.max(0, Math.min(W, Math.round(xr)));
    if (r > l) { area += r - l; inter += m.pre[y * (W + 1) + r] - m.pre[y * (W + 1) + l]; }
  }
  const union = area + m.total - inter;
  return union > 0 ? inter / union : 0;
}

function move(P, c, dx, dy, ls, th) {
  const s = Math.exp(ls), co = Math.cos(th) * s, si = Math.sin(th) * s;
  return P.map(([x, y]) => [c[0] + co * (x - c[0]) - si * (y - c[1]) + dx, c[1] + si * (x - c[0]) + co * (y - c[1]) + dy]);
}

// -> { quad, before, after }   quad in video pixels; `before` and `after` are the overlaps with the mask
export function alignToMask(quad, mask, vw, vh) {
  const m = prefix(mask);
  if (m.total < 50) return { quad, before: 0, after: 0 };
  const P = quad.map(([x, y]) => [x * W / vw, y * H / vh]);
  const c = [P.reduce((a, p) => a + p[0], 0) / 4, P.reduce((a, p) => a + p[1], 0) / 4];
  const pen = (v) => LAMBDA * (Math.abs(v[0]) / W + Math.abs(v[1]) / H + Math.abs(v[2]) + Math.abs(v[3])) * 0.5;
  const score = (v) => iou(move(P, c, v[0], v[1], v[2], v[3]), m) - pen(v);
  const before = iou(P, m);
  let v = [0, 0, 0, 0], best = score(v);
  let step = [0.04 * W, 0.04 * H, 0.05, 0.04];
  for (let round = 0; round < 7; round++) {
    let improved = true, guard = 0;
    while (improved && guard++ < 6) {
      improved = false;
      for (let k = 0; k < 4; k++) {
        for (const sign of [1, -1]) {
          const t = v.slice(); t[k] += sign * step[k];
          if (Math.abs(t[0]) > MAX_SHIFT * W || Math.abs(t[1]) > MAX_SHIFT * H || Math.abs(t[2]) > MAX_SCALE || Math.abs(t[3]) > MAX_TURN) continue;
          const sc = score(t);
          if (sc > best + 1e-4) { best = sc; v = t; improved = true; }
        }
      }
    }
    step = step.map((x) => x / 2);
  }
  const moved = move(P, c, v[0], v[1], v[2], v[3]);
  return { quad: moved.map(([x, y]) => [x * vw / W, y * vh / H]), before, after: iou(moved, m) };
}
