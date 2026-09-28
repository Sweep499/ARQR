// The camera-pose math behind the 3D admin tools (tools/hotspots3d.html, admin3d/): once the pod's front
// rectangle gives a camera pose (the same solvePnP fit js/pose.js uses to keep the outline a real rectangle),
// that SAME pose locates the pod's own 3D coordinate frame in space, so any other 3D point in that frame -
// not just the four flat corners - can be projected to the right screen position with correct perspective.
// Convention: object points are [x_mm, y_mm, z_mm], x right and y down across the front (matching front_mm,
// same as js/pose.js), z_mm is depth BEHIND the front glass, into the pod (verified empirically: increasing
// object-space z increases camera-space depth for a camera standing in front of a real pod).
//
// Kept separate from js/pose.js on purpose: js/pose.js is used by the live app (js/app.js) and its behaviour
// must not change; this file is only imported by the 3D admin tools.

import { focalGuess } from './pose.js';
export { focalGuess };

// -> { rvec, tvec, K, f } (plain arrays, not cv.Mats: nothing here needs the caller to manage OpenCV memory)
// or null when the quad is not a believable view of a `front.width` x `front.height` rectangle.
export function solvePose(cv, quad, vw, vh, front, f = focalGuess(vw, vh)) {
  const W = front.width, H = front.height;
  const K = cv.matFromArray(3, 3, cv.CV_64F, [f, 0, vw / 2, 0, f, vh / 2, 0, 0, 1]);
  const obj = cv.matFromArray(4, 3, cv.CV_64F, [0, 0, 0, W, 0, 0, W, H, 0, 0, H, 0]);
  const img = cv.matFromArray(4, 1, cv.CV_64FC2, quad.flat());
  const dist = new cv.Mat(), rv = new cv.Mat(), tv = new cv.Mat();
  try {
    if (!cv.solvePnP(obj, img, K, dist, rv, tv, false, cv.SOLVEPNP_IPPE)) return null;
    if (!(tv.data64F[2] > 0)) return null;   // the pod must be in front of the camera
    return { rvec: Array.from(rv.data64F), tvec: Array.from(tv.data64F), K: Array.from(K.data64F), f };
  } catch (e) {
    return null;
  } finally {
    K.delete(); obj.delete(); img.delete(); dist.delete(); rv.delete(); tv.delete();
  }
}

// Rodrigues' rotation formula: a 3-vector (axis * angle) to a 3x3 rotation matrix, row-major. Kept as plain
// maths (no OpenCV call) so the projection functions below don't need a live `cv` at all.
function rodrigues([x, y, z]) {
  const th = Math.hypot(x, y, z);
  if (th < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const [ux, uy, uz] = [x / th, y / th, z / th], c = Math.cos(th), s = Math.sin(th), t = 1 - c;
  return [
    t * ux * ux + c, t * ux * uy - s * uz, t * ux * uz + s * uy,
    t * ux * uy + s * uz, t * uy * uy + c, t * uy * uz - s * ux,
    t * ux * uz - s * uy, t * uy * uz + s * ux, t * uz * uz + c,
  ];
}

// Projects 3D points (mm, in the pod's own frame) to screen pixels, given a pose from solvePose. A point
// behind the camera (should not happen for a point inside/near the pod with a valid pose) projects to null.
export function projectPoints(pose, points) {
  const R = rodrigues(pose.rvec), [tx, ty, tz] = pose.tvec, [fx, , cx, , fy, cy] = pose.K;
  return points.map(([x, y, z]) => {
    const cxp = R[0] * x + R[1] * y + R[2] * z + tx;
    const cyp = R[3] * x + R[4] * y + R[5] * z + ty;
    const czp = R[6] * x + R[7] * y + R[8] * z + tz;
    if (!(czp > 1e-6)) return null;
    return [fx * (cxp / czp) + cx, fy * (cyp / czp) + cy];
  });
}
export function projectPoint(pose, point) { return projectPoints(pose, [point])[0]; }

// The inverse problem, used when placing a point: given a pixel clicked on a photo taken from this pose, and
// a chosen depth z_mm (measured or estimated - a single photo cannot reveal depth by itself, see the tool's
// own notes), returns the [x_mm, y_mm] on the plane at that depth whose projection is that pixel, or null
// when the viewing angle is too edge-on to that depth plane to solve it.
export function unprojectAtDepth(pose, sx, sy, z_mm) {
  const R = rodrigues(pose.rvec), [tx, ty, tz] = pose.tvec, [fx, , cx, , fy, cy] = pose.K;
  // the camera's own position, and the clicked pixel's ray direction, both in the pod's 3D frame
  const Cx = -(R[0] * tx + R[3] * ty + R[6] * tz), Cy = -(R[1] * tx + R[4] * ty + R[7] * tz), Cz = -(R[2] * tx + R[5] * ty + R[8] * tz);
  const dcx = (sx - cx) / fx, dcy = (sy - cy) / fy, dcz = 1;
  const dx = R[0] * dcx + R[3] * dcy + R[6] * dcz, dy = R[1] * dcx + R[4] * dcy + R[7] * dcz, dz = R[2] * dcx + R[5] * dcy + R[8] * dcz;
  if (Math.abs(dz) < 1e-9) return null;
  const t = (z_mm - Cz) / dz;
  if (t <= 0) return null;                 // the plane at this depth is behind the camera from this pixel
  return [Cx + t * dx, Cy + t * dy];
}
