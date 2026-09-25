// Frame-to-frame motion tracker: Lucas-Kanade optical flow over the whole frame plus a RANSAC
// homography, so a quad placed once keeps following the pod while the phone moves.
// Later a trained pod detector can call `app.setCorners()` to re-anchor and cancel drift.

const PROC_W = 320;
const MIN_POINTS = 70;

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
  }

  reset() {
    if (this.prevGray) this.prevGray.delete();
    if (this.prevPts) this.prevPts.delete();
    this.prevGray = null;
    this.prevPts = null;
  }

  // Returns a homography (in downscaled coords) from the previous frame to this one, or null.
  step(video) {
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
          H = Array.from(Hm.data64F);
          const inliers = [];
          for (let i = 0; i < n; i++) if (mask.data[i]) inliers.push(dst[2 * i], dst[2 * i + 1]);
          if (inliers.length >= 8) keep = cv.matFromArray(inliers.length / 2, 1, cv.CV_32FC2, inliers);
        }
        if (Hm) Hm.delete();
        srcM.delete(); dstM.delete(); mask.delete();
      }
      next.delete(); status.delete(); err.delete();
    }

    if (this.prevPts) this.prevPts.delete();
    if (keep && keep.rows >= MIN_POINTS) {
      this.prevPts = keep;
    } else {
      if (keep) keep.delete();
      const pts = new cv.Mat();
      cv.goodFeaturesToTrack(gray, pts, 250, 0.01, 8);
      this.prevPts = pts;
    }
    if (this.prevGray) this.prevGray.delete();
    this.prevGray = gray;
    return H;
  }
}
