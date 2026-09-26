// Keeps an outline a physically possible view of the pod's front: a rectangle of known size (front_mm in
// data/pod.json) seen by a phone camera. A tracked or detected quad has 8 degrees of freedom, but a rigid
// rectangle seen from a camera has 6 (its position and turn), so fitting the pose throws away exactly the
// distortions that make an outline look twisted or stretched, and how badly the fit misses says how believable
// the quad was.

// Phone main cameras see about 65-70 degrees along the long side of the picture.
export function focalGuess(vw, vh) { return 0.75 * Math.max(vw, vh); }

// -> null (no valid pose) or { quad, err, tilt }
//    quad  the four corners [TL, TR, BR, BL] re-projected from the fitted pose, in the same pixel space
//    err   RMS distance between the given and the fitted corners, as a fraction of the picture diagonal
//    tilt  degrees between the pod's front and facing the camera squarely
export function fitRectangle(cv, quad, vw, vh, front, f = focalGuess(vw, vh)) {
  const W = front.width, H = front.height;
  const K = cv.matFromArray(3, 3, cv.CV_64F, [f, 0, vw / 2, 0, f, vh / 2, 0, 0, 1]);
  const obj = cv.matFromArray(4, 3, cv.CV_64F, [0, 0, 0, W, 0, 0, W, H, 0, 0, H, 0]);
  const img = cv.matFromArray(4, 1, cv.CV_64FC2, quad.flat());
  const dist = new cv.Mat(), rv = new cv.Mat(), tv = new cv.Mat(), out = new cv.Mat(), R = new cv.Mat();
  try {
    if (!cv.solvePnP(obj, img, K, dist, rv, tv, false, cv.SOLVEPNP_IPPE)) return null;
    if (!(tv.data64F[2] > 0)) return null;                    // the pod must be in front of the camera
    cv.projectPoints(obj, rv, tv, K, dist, out);
    const p = out.data64F, fitted = [];
    let se = 0;
    for (let i = 0; i < 4; i++) {
      fitted.push([p[2 * i], p[2 * i + 1]]);
      se += (p[2 * i] - quad[i][0]) ** 2 + (p[2 * i + 1] - quad[i][1]) ** 2;
    }
    if (!fitted.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y))) return null;
    cv.Rodrigues(rv, R);
    const nz = Math.abs(R.data64F[8]);                        // z component of the plane's normal in camera space
    return { quad: fitted, err: Math.sqrt(se / 4) / Math.hypot(vw, vh), tilt: Math.acos(Math.min(1, nz)) * 180 / Math.PI };
  } catch (e) {
    return null;
  } finally {
    K.delete(); obj.delete(); img.delete(); dist.delete(); rv.delete(); tv.delete(); out.delete(); R.delete();
  }
}
