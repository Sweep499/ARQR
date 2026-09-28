// 3D admin test view: runs the exact same pod detection, tracking and mask-alignment as the live 2D app
// (imported fresh from the shared js/ modules - nothing here shares state with, or modifies, the live app),
// but positions each dot with a real 3D depth instead of pinning it flat to the glass. Reads its own separate
// data file (data/pod-3d.json), published by tools/hotspots3d.html. Not linked from the public site.

import { isPlausibleQuad, quadInView, applyScaledH } from '../js/geometry.js';
import { loadOpenCV, FlowTracker } from '../js/tracker.js';
import { loadDetector } from '../js/detector.js';
import { alignToMask } from '../js/align.js';
import { CameraControl } from '../js/camera.js';
import { solvePose, projectPoints } from '../js/pose3d.js';

const $ = (id) => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const ctx = overlay.getContext('2d');
const hotspotLayer = $('hotspots');
const hint = $('hint');
const sheet = $('sheet');

const CORNER_NAMES = ['top-left', 'top-right', 'bottom-right', 'bottom-left'];
const HANDLE_RADIUS = 32;
const HIDE_AFTER = 400;
const DROP_AFTER = 1500;
const OFF_DROP = 1500;
const AGREE_MIN = 0.4;
const ABSENT_FRAC = 0.02;
const AGREE_STRIKES = 2;
const ALIGN_MIN_GAIN = 0.02;
const ALIGN_MIN_MASK = 0.02;
const ALIGN_WEIGHT = 0.7;
const REDETECT_MS = 500;

const params = new URLSearchParams(location.search);
const debug = params.has('debug');
const NODROP = params.has('nodrop');
const devSrc = params.get('src');

let pod3d = null;         // { front_mm, points } from data/pod-3d.json
let quad = null;
let placing = [];
let dragging = -1;
let tracker = null;
let cvRef = null;
let camera = null;
let detector = null;
let detecting = false;
let lastDetect = 0;
let candidate = null;
let lostSince = 0;
let offSince = 0;
let strikes = 0;
let manual = false;
let holdHintUntil = 0;
let hotspotEls = [];
let activeEl = null;
let hintTimer = 0;
let pose = null;          // solved camera pose for the current quad, recomputed each frame
if (debug) window.podDebug3d = { get quad() { return quad; }, set quad(q) { quad = q; manual = false; lostSince = 0; offSince = 0; }, get tracker() { return tracker; }, get pod3d() { return pod3d; }, get pose() { return pose; }, set detector(d) { detector = d; } };

// ---------- start-up ----------

$('startBtn').addEventListener('click', start);
$('resetBtn').addEventListener('click', () => beginPlacing());
$('sheetClose').addEventListener('click', closeSheet);
window.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSheet(); });

async function start() {
  const err = $('error');
  err.hidden = true;
  try {
    pod3d = await (await fetch(new URL('../data/pod-3d.json', import.meta.url))).json();
    if (!pod3d.front_mm || !(pod3d.front_mm.width > 0) || !(pod3d.front_mm.height > 0)) throw new Error('data/pod-3d.json has no valid front_mm; set it in tools/hotspots3d.html first.');
    await openVideoSource();
  } catch (e) {
    err.textContent = friendlyError(e);
    err.hidden = false;
    return;
  }
  $('start').hidden = true;
  $('stage').hidden = false;
  buildHotspots();
  beginPlacing();
  loadOpenCV()
    .then(({ cv }) => {
      tracker = new FlowTracker(cv);
      cvRef = cv;
      loadDetector(cv, params.get('model') || undefined).then((d) => { detector = d; if (!quad && !placing.length) updatePlacingHint(); }).catch((e) => { console.warn('pod detector unavailable:', e); });
    })
    .catch(() => setHint('Live tracking could not load. The pod outline will stay where you placed it.', 6000));
  requestAnimationFrame(frame);
}

async function openVideoSource() {
  if (devSrc) {
    video.src = devSrc;
    video.loop = true;
    video.muted = true;
    await playVideo();
    return;
  }
  camera = new CameraControl(video);
  await camera.open();
  const wide = camera.preferredWide();
  if (wide) await camera.open(wide).catch(() => {});
  setUpCameraControls();
}

async function playVideo() {
  try {
    await video.play();
  } catch (e) {
    if (e?.name !== 'AbortError') throw e;
  }
}

// ---------- camera controls (same as the live app) ----------

function setUpCameraControls() {
  const many = camera.devices.length > 1;
  const zoom = camera.zoomRange();
  $('camBtn').hidden = !many && !zoom;
  if (!$('camBtn').hidden) buildCameraPanel();
}
function buildCameraPanel() {
  const list = $('camLenses');
  list.textContent = '';
  list.hidden = camera.devices.length <= 1;
  camera.devices.forEach((d, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = camera.labelFor(i);
    b.className = d.deviceId === camera.deviceId ? 'active' : '';
    b.addEventListener('click', () => switchLens(d.deviceId));
    list.appendChild(b);
  });
  const zoom = camera.zoomRange();
  const slider = $('camZoom');
  slider.hidden = !zoom;
  if (zoom) {
    slider.min = zoom.min; slider.max = zoom.max; slider.step = zoom.step; slider.value = zoom.value;
    $('camZoomLabel').textContent = zoom.min < 1 ? 'Zoom (below 1× is wider than normal)' : 'Zoom';
  }
}
async function switchLens(deviceId) {
  $('camPanel').hidden = true;
  const hadQuad = !!quad;
  try { await camera.open(deviceId); } catch (e) { setHint('Could not switch camera: ' + (e?.message || e)); return; }
  buildCameraPanel();
  if (hadQuad || placing.length) beginPlacing();
}
$('camBtn').addEventListener('click', () => { $('camPanel').hidden = !$('camPanel').hidden; if (!$('camPanel').hidden) buildCameraPanel(); });
$('camZoom').addEventListener('input', (e) => camera.setZoom(parseFloat(e.target.value)));
document.addEventListener('pointerdown', (e) => { if (!$('camPanel').hidden && !e.target.closest('#camPanel, #camBtn')) $('camPanel').hidden = true; });

function friendlyError(e) {
  if (e?.name === 'NotAllowedError') return 'Camera access was blocked. Allow the camera for this site in your browser settings and try again.';
  if (e?.name === 'NotFoundError') return 'No camera was found on this device.';
  return e?.message || 'Something went wrong starting the camera.';
}

// ---------- placing / dragging the outline (a manual fallback, same as the admin 2D page) ----------

function beginPlacing() {
  quad = null; placing = []; candidate = null; lostSince = 0; offSince = 0; strikes = 0; manual = false; pose = null;
  closeSheet();
  if (tracker) tracker.reset();
  updatePlacingHint();
}
function updatePlacingHint() {
  const n = placing.length;
  setHint(n === 0 && detector ? 'Looking for the pod. Step back until you can see its whole front, or tap its corners.' : `Tap the pod's ${CORNER_NAMES[n]} corner (${n + 1} of 4)`);
}
function setHint(text, ms = 0) {
  clearTimeout(hintTimer);
  hint.textContent = text;
  if (ms) hintTimer = setTimeout(() => { hint.textContent = ''; }, ms);
}

$('stage').addEventListener('pointerdown', (e) => {
  if (e.target.closest('button, aside, a')) return;
  const p = toVideo(e.clientX, e.clientY);
  if (!quad) {
    placing.push(p);
    if (placing.length === 4) { quad = placing; placing = []; manual = true; lostSince = 0; setHint('Tap a numbered dot for details. Drag a corner to fine-tune.', 7000); }
    else updatePlacingHint();
    return;
  }
  const s = quad.map((c) => toScreen(c));
  let best = -1, bestD = HANDLE_RADIUS;
  s.forEach(([x, y], i) => { const d = Math.hypot(x - e.clientX, y - e.clientY); if (d < bestD) { bestD = d; best = i; } });
  if (best >= 0) { dragging = best; manual = true; $('stage').setPointerCapture(e.pointerId); }
  else closeSheet();
});
$('stage').addEventListener('pointermove', (e) => { if (dragging >= 0) quad[dragging] = toVideo(e.clientX, e.clientY); });
const endDrag = () => { dragging = -1; };
$('stage').addEventListener('pointerup', endDrag);
$('stage').addEventListener('pointercancel', endDrag);

function layout() {
  const cw = overlay.clientWidth, ch = overlay.clientHeight;
  const vw = video.videoWidth || 1, vh = video.videoHeight || 1;
  const s = Math.max(cw / vw, ch / vh);
  return { s, ox: (cw - vw * s) / 2, oy: (ch - vh * s) / 2, cw, ch };
}
function toScreen([x, y]) { const { s, ox, oy } = layout(); return [x * s + ox, y * s + oy]; }
function toVideo(x, y) { const { s, ox, oy } = layout(); return [(x - ox) / s, (y - oy) / s]; }

// ---------- automatic pod detection (same logic as the live app; see its comments for why) ----------

const meanDist = (a, b) => a.reduce((s, p, i) => s + Math.hypot(p[0] - b[i][0], p[1] - b[i][1]), 0) / 4;

function dropQuad(why = 'Lost the pod.') {
  if (NODROP) return;
  beginPlacing();
  setHint(why + ' Point the camera at it and step back until the whole front is in view, or tap its corners.');
  holdHintUntil = performance.now() + 5000;
}

async function runDetector() {
  detecting = true;
  try {
    const r = await detector.detect(video);
    lastDetect = performance.now();
    if (debug) console.log('detect', video.currentTime.toFixed(1), r.quad ? 'quad ' + r.score.toFixed(2) : r.clipped ? 'clipped' : 'none');
    if (placing.length || dragging >= 0) return;
    const vw = video.videoWidth, vh = video.videoHeight;

    if (quad) {
      const absent = (r.maskFrac ?? 1) < ABSENT_FRAC;
      let a = absent || !r.mask ? 0 : detector.agreement(r.mask, quad, vw, vh);
      if (!manual && !absent && r.mask && dragging < 0 && a < 0.97 && r.maskFrac >= ALIGN_MIN_MASK) {
        const al = alignToMask(quad, r.mask, vw, vh);
        if (al.after - al.before >= ALIGN_MIN_GAIN && al.after >= 0.4) {
          quad = quad.map((p, i) => [p[0] + ALIGN_WEIGHT * (al.quad[i][0] - p[0]), p[1] + ALIGN_WEIGHT * (al.quad[i][1] - p[1])]);
          a = detector.agreement(r.mask, quad, vw, vh);
        }
      }
      strikes = absent || a < AGREE_MIN ? strikes + 1 : 0;
      if (strikes >= AGREE_STRIKES) { dropQuad(absent ? 'The pod went out of view.' : 'The outline drifted off the pod.'); return; }
    }

    if (r.quad) {
      const agrees = candidate && meanDist(candidate, r.quad) < 0.05 * vw;
      if (!agrees) { candidate = r.quad; return; }
      candidate = null;
      strikes = 0;
      if (!quad) { quad = r.quad; manual = false; setHint('Found the pod. Tap a dot for details. Drag a corner to fine-tune.', 7000); }
      else if (lostSince) { quad = r.quad; lostSince = 0; manual = false; setHint('Found the pod again.', 3000); }
      else if (!manual) {
        const d = meanDist(quad, r.quad);
        if (d > 0.06 * vw) quad = r.quad;
        else if (d > 0.005 * vw) quad = quad.map((p, i) => [p[0] + 0.35 * (r.quad[i][0] - p[0]), p[1] + 0.35 * (r.quad[i][1] - p[1])]);
      }
    } else {
      candidate = null;
      if (!quad) {
        const want = r.clipped ? 'I can see the pod but not all of its front. Step back a little, or tap its corners.' : 'Looking for the pod. Step back until you can see its whole front, or tap its corners.';
        if (hint.textContent !== want && performance.now() > holdHintUntil) setHint(want);
      }
    }
  } catch (e) {
    console.error(e);
    detector = null;
    if (!quad) updatePlacingHint();
  } finally {
    detecting = false;
  }
}

// ---------- hotspots and popup (3D: projected with the current pose, not a flat homography) ----------

function buildHotspots() {
  hotspotLayer.textContent = '';
  hotspotEls = pod3d.points.map((h, i) => {
    const b = document.createElement('button');
    b.className = 'hotspot3d';
    b.type = 'button';
    b.dataset.n = String(i + 1);
    b.setAttribute('aria-label', h.title);
    b.style.display = 'none';
    b.addEventListener('click', () => openSheet(h, b));
    hotspotLayer.appendChild(b);
    return b;
  });
  setDiag(`${pod3d.points.length} point${pod3d.points.length === 1 ? '' : 's'} loaded from data/pod-3d.json`);
}

function openSheet(h, el) {
  if (activeEl) activeEl.classList.remove('active');
  activeEl = el;
  el.classList.add('active');
  $('sheetTitle').textContent = h.title;
  $('sheetText').textContent = (h.text || '') + `  [depth ${h.z_mm ?? 0} mm]`;
  const link = $('sheetLink');
  link.hidden = !h.url;
  if (h.url) link.href = h.url;
  sheet.hidden = false;
}
function closeSheet() {
  sheet.hidden = true;
  if (activeEl) activeEl.classList.remove('active');
  activeEl = null;
}
function setDiag(text) { $('diag').textContent = text; }

// ---------- render loop ----------

function frame() {
  requestAnimationFrame(frame);
  const dpr = window.devicePixelRatio || 1;
  const w = overlay.clientWidth, h = overlay.clientHeight;
  if (overlay.width !== Math.round(w * dpr) || overlay.height !== Math.round(h * dpr)) { overlay.width = Math.round(w * dpr); overlay.height = Math.round(h * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  if (video.readyState >= 2 && tracker) {
    let H = null;
    try { H = tracker.step(video); } catch (e) { console.error(e); tracker = null; }
    if (quad && dragging < 0) {
      let moved = false;
      if (H) {
        const next = quad.map((p) => applyScaledH(H, p, tracker.scale));
        if (isPlausibleQuad(next, video.videoWidth, video.videoHeight)) { quad = next; moved = true; }
      }
      if (moved) { if (lostSince) { lostSince = 0; setHint(''); } }
      else if (!lostSince) lostSince = performance.now();
      else if (performance.now() - lostSince > DROP_AFTER) dropQuad();
    }
  }

  if (quad && dragging < 0) {
    if (quadInView(quad, video.videoWidth, video.videoHeight)) offSince = 0;
    else if (!offSince) offSince = performance.now();
    else if (performance.now() - offSince > OFF_DROP) dropQuad('The pod went out of view.');
  }

  const idle = !quad ? !placing.length : dragging < 0;
  if (idle && detector && !detecting && video.readyState >= 2 && performance.now() - lastDetect > (quad ? REDETECT_MS : 400)) runDetector();

  if (!quad) {
    drawPlacing();
    hotspotEls.forEach((el) => { el.style.display = 'none'; });
    pose = null;
    return;
  }

  const lost = lostSince && performance.now() - lostSince > HIDE_AFTER;
  if (lost && performance.now() > holdHintUntil) setHint('Lost track of the pod. Point the camera back at it.');
  const sq = quad.map(toScreen);
  drawQuad(sq, lost ? 0.3 : 1);
  if (lost) { hotspotEls.forEach((el) => { el.style.display = 'none'; }); pose = null; return; }

  pose = cvRef ? solvePose(cvRef, quad, video.videoWidth, video.videoHeight, pod3d.front_mm) : null;
  if (!pose) {
    hotspotEls.forEach((el) => { el.style.display = 'none'; });
    setDiag('No pose (front size invalid, or an implausible view)');
    return;
  }
  const proj = projectPoints(pose, pod3d.points.map((p) => [p.x_mm, p.y_mm, p.z_mm]));
  let shown = 0;
  proj.forEach((p, i) => {
    const el = hotspotEls[i];
    if (!p) { el.style.display = 'none'; return; }
    const [sx, sy] = toScreen(p);
    const visible = sx > -20 && sy > -20 && sx < w + 20 && sy < h + 20;
    el.style.display = visible ? '' : 'none';
    if (visible) { el.style.transform = `translate(${sx}px, ${sy}px)`; shown++; }
  });
  setDiag(`pose ok · ${shown}/${pod3d.points.length} dots on screen`);
}

function drawPlacing() {
  const pts = placing.map(toScreen);
  ctx.lineWidth = 3; ctx.strokeStyle = '#ffb400';
  ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
  pts.forEach(([x, y]) => dot(x, y, 7, '#ffb400'));
}
function drawQuad(sq, alpha = 1) {
  ctx.globalAlpha = alpha;
  ctx.lineWidth = 3; ctx.lineJoin = 'round'; ctx.strokeStyle = 'rgba(255, 180, 0, 0.95)';
  ctx.shadowColor = 'rgba(0, 0, 0, 0.6)'; ctx.shadowBlur = 6;
  ctx.beginPath(); sq.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); ctx.stroke();
  ctx.shadowBlur = 0;
  sq.forEach(([x, y]) => dot(x, y, 6, '#fff'));
  ctx.globalAlpha = 1;
}
function dot(x, y, r, color) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); }
