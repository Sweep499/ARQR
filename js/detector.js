// Finds the pod's front frame automatically: a small segmentation model (models/pod_front.onnx, trained by
// tools/train/train.py) runs in the browser via onnxruntime-web, and the mask outline is reduced to the
// four corners. It only reports corners when the whole front is in view; when the front runs off the
// picture edge its corners are off-screen and cannot be recovered, so it returns { clipped: true }.

const ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
const IN_W = 256, IN_H = 448;               // model input (the same 9:16 shape it was trained on)
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const MIN_AREA = 0.08;                      // smallest accepted front, as a fraction of the picture
const MIN_SOLID = 0.85;                     // outline area / hull area; low means fragments or a leak
const EDGE = 2;                             // px from the picture edge that counts as touching it

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src; s.async = true;
    s.onload = resolve;
    s.onerror = () => reject(new Error('could not load ' + src));
    document.head.appendChild(s);
  });
}

// Resolves with a detector, or rejects if the runtime or model cannot load (the app then stays manual).
export async function loadDetector(cv, modelUrl = 'models/pod_front.onnx') {
  if (!window.ort) await loadScript(ORT_BASE + 'ort.min.js');
  const ort = window.ort;
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.proxy = true;                // run inference in a worker so the video overlay keeps moving
  const session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
  return new PodDetector(cv, ort, session);
}

export class PodDetector {
  constructor(cv, ort, session) {
    this.cv = cv;
    this.ort = ort;
    this.session = session;
    this.canvas = document.createElement('canvas');
    this.canvas.width = IN_W;
    this.canvas.height = IN_H;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
  }

  // -> null (no pod), { clipped: true } (pod found but not fully in view), or { quad, score } with the
  // corners [TL, TR, BR, BL] in the video's own pixel coordinates.
  async detect(video) {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return null;
    this.ctx.drawImage(video, 0, 0, IN_W, IN_H);
    const px = this.ctx.getImageData(0, 0, IN_W, IN_H).data;
    const plane = IN_W * IN_H;
    const input = new Float32Array(3 * plane);   // a fresh buffer each time: the worker takes ownership of it
    for (let i = 0; i < plane; i++) {
      for (let c = 0; c < 3; c++) input[c * plane + i] = (px[4 * i + c] / 255 - MEAN[c]) / STD[c];
    }
    const out = await this.session.run({ image: new this.ort.Tensor('float32', input, [1, 3, IN_H, IN_W]) });
    const logits = out.logits.data;
    const mask = new Uint8Array(plane);
    for (let i = 0; i < plane; i++) mask[i] = logits[i] > 0 ? 255 : 0;
    const r = this.outline(mask);
    if (!r) return null;
    if (r.clipped) return { clipped: true };
    const q = reduceToQuad(r.poly);
    if (!q) return null;
    return { quad: q.map(([x, y]) => [x * vw / IN_W, y * vh / IN_H]), score: r.solidity };
  }

  // Largest blob in the mask -> hull polygon, or null when it is too small or too ragged.
  outline(mask) {
    const cv = this.cv;
    const m = cv.matFromArray(IN_H, IN_W, cv.CV_8UC1, mask);
    const contours = new cv.MatVector(), hier = new cv.Mat();
    const hull = new cv.Mat(), ap = new cv.Mat();
    try {
      cv.morphologyEx(m, m, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5)));
      cv.findContours(m, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
      let best = null, bestArea = 0;
      for (let i = 0; i < contours.size(); i++) {
        const a = cv.contourArea(contours.get(i));
        if (a > bestArea) { bestArea = a; best = contours.get(i); }
      }
      if (!best || bestArea < MIN_AREA * IN_W * IN_H) return null;
      cv.convexHull(best, hull, false, true);
      const solidity = bestArea / Math.max(1, cv.contourArea(hull));
      if (solidity < MIN_SOLID) return null;
      cv.approxPolyDP(hull, ap, 0.004 * cv.arcLength(hull, true), true);
      const poly = [];
      for (let k = 0; k < ap.rows; k++) poly.push([ap.data32S[2 * k], ap.data32S[2 * k + 1]]);
      const clipped = poly.some(([x, y]) => x <= EDGE || y <= EDGE || x >= IN_W - 1 - EDGE || y >= IN_H - 1 - EDGE);
      return { poly, solidity, clipped };
    } finally {
      m.delete(); contours.delete(); hier.delete(); hull.delete(); ap.delete();
    }
  }
}

// Reduces a convex polygon to 4 corners [TL, TR, BR, BL] by repeatedly dropping the edge whose removal adds
// the least area, extending its two neighbours until they meet. (Same method as tools/prelabel/prelabel.py.)
export function reduceToQuad(poly) {
  const v = poly.map((p) => [p[0], p[1]]);
  if (v.length < 4) return null;
  while (v.length > 4) {
    const n = v.length;
    let best = null;
    for (let i = 0; i < n; i++) {
      const a = v[(i - 1 + n) % n], b = v[i], c = v[(i + 1) % n], d = v[(i + 2) % n];
      const r = [b[0] - a[0], b[1] - a[1]], u = [c[0] - d[0], c[1] - d[1]];  // extend a->b forwards and d->c backwards
      const den = r[0] * u[1] - r[1] * u[0];
      if (Math.abs(den) < 1e-9) continue;
      const w = [d[0] - a[0], d[1] - a[1]];
      const t = (w[0] * u[1] - w[1] * u[0]) / den, k = (w[0] * r[1] - w[1] * r[0]) / den;
      if (t < 1 || k < 1) continue;                                          // neighbours must diverge from the removed edge
      const pt = [a[0] + t * r[0], a[1] + t * r[1]];
      const cost = Math.abs((b[0] - pt[0]) * (c[1] - pt[1]) - (b[1] - pt[1]) * (c[0] - pt[0])) / 2;
      if (!best || cost < best.cost) best = { cost, i, pt };
    }
    if (!best) return null;
    v[best.i] = best.pt;
    v.splice((best.i + 1) % n, 1);
  }
  const sum = (p) => p[0] + p[1], diff = (p) => p[1] - p[0];
  const tl = v.reduce((a, b) => (sum(b) < sum(a) ? b : a));
  const br = v.reduce((a, b) => (sum(b) > sum(a) ? b : a));
  const tr = v.reduce((a, b) => (diff(b) < diff(a) ? b : a));
  const bl = v.reduce((a, b) => (diff(b) > diff(a) ? b : a));
  const q = [tl, tr, br, bl];
  return new Set(q).size === 4 ? q : null;
}
