// Frame-to-frame motion tracker: Lucas-Kanade optical flow over the whole frame plus a RANSAC
// homography, so a quad placed once keeps following the pod while the phone moves.
// Later a trained pod detector can call `app.setCorners()` to re-anchor and cancel drift.

import { applyH, isSaneQuad, quadArea } from './geometry.js';

const PROC_W = 320;
const MIN_POINTS = 70;
const MIN_RING_POINTS = 30;   // when following an outline, fewer points than this are enough to keep going
const FRAME_BAND = 0.86;      // the band of the outline that is tracked: from its edge in to this fraction of its size
const MIN_INLIERS = 12;      // fewer tracked points than this and the motion estimate is not trusted
const MIN_INLIER_RATIO = 0.35;

// A frame-to-frame motion is believable if it maps the picture onto a convex shape of about the same
// size that has not jumped far. A blank wall or a blur gives junk that fails this.
function plausibleMotion(H, w, h) {
  const c = [[0, 0], [w, 0], [w, h], [0, h]];
  const m = c.map((p) => applyH(H, p));
  if (m.some(([x, y]) => !Number.isFinite(x) || !Number.isFinite(y))) return false;
  if (!isSaneQuad(m)) return false;
  const ratio = quadArea(m) / (w * h);
  if (ratio < 0.6 || ratio > 1.6) return false;
  return m.every((p, i) => Math.hypot(p[0] - c[i][0], p[1] - c[i][1]) < 0.3 * w);
}

// Resolves with { cv }. The cv object is wrapped because opencv.js can expose a `then` method, and
// resolving a promise with such a thenable re-invokes it forever and freezes the page.
export function loadOpenCV() {
  return new Promise((resolve, reject) => {
    if (window.cv && window.cv.Mat) return resolve({ cv: window.cv });
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js';
    s.async = true;
    s.onerror = () => reject(new Error('OpenCV.js failed to load'));
    s.onload = () => {
      // depending on the build, cv is ready immediately, a thenable, or fires onRuntimeInitialized
      const done = (m) => { if (m && m.Mat) window.cv = m; resolve({ cv: window.cv }); };
      if (window.cv && window.cv.Mat) return done();
      if (window.cv && typeof window.cv.then === 'function') {
        // defer, so we have left cv.then before anything touches the cv object again
        return window.cv.then((m) => setTimeout(() => done(m), 0));
      }
      window.cv.onRuntimeInitialized = () => done();
    };
    document.head.appendChild(s);
  });
}

export class FlowTracker {
  constructor(cv) {
    this.cv = cv;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.prevGray = null;
    this.prevPts = null;
    this.scale = 1;
    this.band = FRAME_BAND;       // how far in from the outline's edge the tracked band reaches (1 = its whole area)
    this.mode = 'all';            // 'all' = features anywhere; 'frame' = features on the outline's frame only
    this.info = { mode: 'all', points: 0 };
  }

  // A mask that keeps features to the outline's frame (its edge band), then to its whole area, then to
  // nothing (the whole picture) if the outline is mostly off-screen. Points on the pod's frame move as one
  // plane; points elsewhere (the room, furniture seen through the glass) move differently as you walk.
  _mask(quad, level) {
    const cv = this.cv, w = this.canvas.width, h = this.canvas.height;
    const qs = quad.map(([x, y]) => [Math.max(-1e5, Math.min(1e5, x * this.scale)), Math.max(-1e5, Math.min(1e5, y * this.scale))]);
    const cx = qs.reduce((a, p) => a + p[0], 0) / 4, cy = qs.reduce((a, p) => a + p[1], 0) / 4;
    const poly = (k) => cv.matFromArray(4, 1, cv.CV_32SC2, qs.flatMap(([x, y]) => [Math.round(cx + (x - cx) * k), Math.round(cy + (y - cy) * k)]));
    const m = cv.Mat.zeros(h, w, cv.CV_8UC1);
    const outer = poly(1.03);
    cv.fillConvexPoly(m, outer, new cv.Scalar(255));
    outer.delete();
    if (level === 0) {
      const inner = poly(this.band);
      cv.fillConvexPoly(m, inner, new cv.Scalar(0));
      inner.delete();
    }
    return m;
  }

  reset() {
    if (this.prevGray) this.prevGray.delete();
    if (this.prevPts) this.prevPts.delete();
    this.prevGray = null;
    this.prevPts = null;
  }

  // Returns a homography (in downscaled coords) from the previous frame to this one, or null when the
  // motion could not be measured reliably (too few points, or an implausible result). With `quad` (the
  // outline in video pixels) the motion is measured on the outline's frame rather than on the whole room.
  step(video, quad = null) {
    const cv = this.cv;
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return null;
    this.scale = PROC_W / vw;
    const w = PROC_W, h = Math.round(vh * this.scale);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.reset();
    }
    this.ctx.drawImage(video, 0, 0, w, h);
    const rgba = cv.matFromImageData(this.ctx.getImageData(0, 0, w, h));
    const gray = new cv.Mat();
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    rgba.delete();

    let H = null;
    let keep = null;

    const mode = quad ? 'frame' : 'all';
    if (mode !== this.mode) {                 // the kind of point wanted has changed: start again from fresh points
      if (this.prevPts) this.prevPts.delete();
      this.prevPts = null;
      this.mode = mode;
    }

    if (this.prevGray && this.prevPts && this.prevPts.rows >= 8) {
      const next = new cv.Mat(), status = new cv.Mat(), err = new cv.Mat();
      cv.calcOpticalFlowPyrLK(this.prevGray, gray, this.prevPts, next, status, err, new cv.Size(21, 21), 3);
      const src = [], dst = [];
      for (let i = 0; i < status.rows; i++) {
        if (status.data[i] !== 1) continue;
        src.push(this.prevPts.data32F[2 * i], this.prevPts.data32F[2 * i + 1]);
        dst.push(next.data32F[2 * i], next.data32F[2 * i + 1]);
      }
      if (src.length >= 8) {
        const n = src.length / 2;
        const srcM = cv.matFromArray(n, 1, cv.CV_32FC2, src);
        const dstM = cv.matFromArray(n, 1, cv.CV_32FC2, dst);
        const mask = new cv.Mat();
        const Hm = cv.findHomography(srcM, dstM, cv.RANSAC, 3, mask);
        if (Hm && !Hm.empty()) {
          const Ha = Array.from(Hm.data64F);
          const inliers = [];
          for (let i = 0; i < n; i++) if (mask.data[i]) inliers.push(dst[2 * i], dst[2 * i + 1]);
          const count = inliers.length / 2;
          if (count >= MIN_INLIERS && count / n >= MIN_INLIER_RATIO && plausibleMotion(Ha, w, h)) {
            H = Ha;
            if (count >= 8) keep = cv.matFromArray(count, 1, cv.CV_32FC2, inliers);
          }
        }
        if (Hm) Hm.delete();
        srcM.delete(); dstM.delete(); mask.delete();
      }
      next.delete(); status.delete(); err.delete();
    }

    if (this.prevPts) this.prevPts.delete();
    if (keep && keep.rows >= (quad ? MIN_RING_POINTS : MIN_POINTS)) {
      this.prevPts = keep;
    } else {
      if (keep) keep.delete();
      let pts = new cv.Mat();
      if (quad) {
        for (let level = this.band >= 1 ? 1 : 0; level < 2; level++) {     // frame band first, then the whole outline area
          const mask = this._mask(quad, level);
          pts.delete(); pts = new cv.Mat();
          cv.goodFeaturesToTrack(gray, pts, 250, 0.01, 8, mask);
          mask.delete();
          if (pts.rows >= MIN_RING_POINTS) break;
        }
        if (pts.rows < 12) { pts.delete(); pts = new cv.Mat(); cv.goodFeaturesToTrack(gray, pts, 250, 0.01, 8); }
      } else {
        cv.goodFeaturesToTrack(gray, pts, 250, 0.01, 8);
      }
      this.prevPts = pts;
    }
    this.info = { mode: this.mode, points: this.prevPts.rows };
    if (this.prevGray) this.prevGray.delete();
    this.prevGray = gray;
    return H;
  }
}
