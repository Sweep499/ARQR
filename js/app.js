import { squareToQuad, applyH, applyScaledH, isSaneQuad } from './geometry.js';
import { loadOpenCV, FlowTracker } from './tracker.js';
import { loadDetector } from './detector.js';

const $ = (id) => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const ctx = overlay.getContext('2d');
const hotspotLayer = $('hotspots');
const hint = $('hint');
const sheet = $('sheet');

const CORNER_NAMES = ['top-left', 'top-right', 'bottom-right', 'bottom-left'];
const HANDLE_RADIUS = 32; // screen px within which a touch grabs a corner

const params = new URLSearchParams(location.search);
const debug = params.has('debug'); // ?debug logs each detector result to the console
const devSrc = params.get('src'); // ?src=dev/demo.mp4 plays a file instead of the camera (testing on a PC)

let config = null;
let quad = null;          // 4 corners in video pixels: TL, TR, BR, BL
let placing = [];         // corners tapped so far while placing
let dragging = -1;
let tracker = null;
let detector = null;      // finds the pod front automatically once its model has loaded
let detecting = false;
let lastDetect = 0;
let candidate = null;     // last detected quad, kept until a second detection agrees
let hotspotEls = [];
let activeEl = null;
let hintTimer = 0;

// ---------- start-up ----------

$('startBtn').addEventListener('click', start);
$('resetBtn').addEventListener('click', () => beginPlacing());
$('sheetClose').addEventListener('click', closeSheet);
window.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSheet(); });

async function start() {
  const err = $('error');
  err.hidden = true;
  try {
    config = await (await fetch('data/pod.json')).json();
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
      // the detector is optional: if it cannot load, tapping the corners still works
      loadDetector(cv).then((d) => { detector = d; if (!quad && !placing.length) updatePlacingHint(); }).catch((e) => console.warn('pod detector unavailable:', e));
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
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('This browser cannot open the camera. Try Safari or Chrome over https.');
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await playVideo();
}

// play() rejects with AbortError if the page is backgrounded mid-start; the autoplay attribute still resumes it.
async function playVideo() {
  try {
    await video.play();
  } catch (e) {
    if (e?.name !== 'AbortError') throw e;
  }
}

function friendlyError(e) {
  if (e?.name === 'NotAllowedError') return 'Camera access was blocked. Allow the camera for this site in your browser settings and try again.';
  if (e?.name === 'NotFoundError') return 'No camera was found on this device.';
  return e?.message || 'Something went wrong starting the camera.';
}

// ---------- placing / dragging the pod outline ----------

function beginPlacing() {
  quad = null;
  placing = [];
  candidate = null;
  closeSheet();
  if (tracker) tracker.reset();
  updatePlacingHint();
}

function updatePlacingHint() {
  const n = placing.length;
  if (n === 0 && detector) {
    setHint('Looking for the pod. Step back until you can see its whole front, or tap its corners.');
  } else {
    setHint(`Tap the pod's ${CORNER_NAMES[n]} corner (${n + 1} of 4)`);
  }
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
    if (placing.length === 4) {
      quad = placing;
      placing = [];
      setHint('Tap a numbered dot for details. Drag a corner to fine-tune.', 7000);
    } else {
      updatePlacingHint();
    }
    return;
  }
  const s = quad.map((c) => toScreen(c));
  let best = -1, bestD = HANDLE_RADIUS;
  s.forEach(([x, y], i) => {
    const d = Math.hypot(x - e.clientX, y - e.clientY);
    if (d < bestD) { bestD = d; best = i; }
  });
  if (best >= 0) {
    dragging = best;
    $('stage').setPointerCapture(e.pointerId);
  } else {
    closeSheet();
  }
});
$('stage').addEventListener('pointermove', (e) => {
  if (dragging >= 0) quad[dragging] = toVideo(e.clientX, e.clientY);
});
const endDrag = () => { dragging = -1; };
$('stage').addEventListener('pointerup', endDrag);
$('stage').addEventListener('pointercancel', endDrag);

// ---------- coordinate mapping (video is shown with object-fit: cover) ----------

function layout() {
  const cw = overlay.clientWidth, ch = overlay.clientHeight;
  const vw = video.videoWidth || 1, vh = video.videoHeight || 1;
  const s = Math.max(cw / vw, ch / vh);
  return { s, ox: (cw - vw * s) / 2, oy: (ch - vh * s) / 2, cw, ch };
}
function toScreen([x, y]) {
  const { s, ox, oy } = layout();
  return [x * s + ox, y * s + oy];
}
function toVideo(x, y) {
  const { s, ox, oy } = layout();
  return [(x - ox) / s, (y - oy) / s];
}

// ---------- automatic pod detection ----------

// Accepts a detection once two in a row agree (corners within 5% of the picture width), so a single
// wrong frame cannot place the outline.
async function runDetector() {
  detecting = true;
  try {
    const r = await detector.detect(video);
    lastDetect = performance.now();
    if (debug) console.log('detect', video.currentTime.toFixed(1), r ? (r.quad ? 'quad ' + r.score.toFixed(2) : 'clipped') : 'none');
    if (quad || placing.length) return;       // the user got there first
    if (r && r.quad) {
      const tol = 0.05 * video.videoWidth;
      if (candidate && candidate.every((p, i) => Math.hypot(p[0] - r.quad[i][0], p[1] - r.quad[i][1]) < tol)) {
        quad = r.quad;
        candidate = null;
        setHint('Found the pod. Tap a numbered dot for details. Drag a corner to fine-tune.', 7000);
      } else {
        candidate = r.quad;
      }
    } else {
      candidate = null;
      // tell the user why nothing happened, without rewriting the same hint every 400 ms
      const want = r && r.clipped
        ? 'I can see the pod but not all of its front. Step back a little, or tap its corners.'
        : 'Looking for the pod. Step back until you can see its whole front, or tap its corners.';
      if (hint.textContent !== want) setHint(want);
    }
  } catch (e) {
    console.error(e);
    detector = null;                          // give up on detection, manual placing still works
    updatePlacingHint();
  } finally {
    detecting = false;
  }
}

// ---------- hotspots and popup ----------

function buildHotspots() {
  hotspotLayer.textContent = '';
  hotspotEls = config.hotspots.map((h, i) => {
    const b = document.createElement('button');
    b.className = 'hotspot';
    b.type = 'button';
    b.dataset.n = String(i + 1);
    b.setAttribute('aria-label', h.title);
    b.style.display = 'none';
    b.addEventListener('click', () => openSheet(h, b));
    hotspotLayer.appendChild(b);
    return b;
  });
}

function openSheet(h, el) {
  if (activeEl) activeEl.classList.remove('active');
  activeEl = el;
  el.classList.add('active');
  $('sheetTitle').textContent = h.title;
  $('sheetText').textContent = h.text || '';
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

// ---------- render loop ----------

function frame() {
  requestAnimationFrame(frame);
  const dpr = window.devicePixelRatio || 1;
  const w = overlay.clientWidth, h = overlay.clientHeight;
  if (overlay.width !== Math.round(w * dpr) || overlay.height !== Math.round(h * dpr)) {
    overlay.width = Math.round(w * dpr);
    overlay.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  if (video.readyState >= 2 && tracker) {
    let H = null;
    try { H = tracker.step(video); } catch (e) { console.error(e); tracker = null; }
    if (H && quad && dragging < 0) {
      const next = quad.map((p) => applyScaledH(H, p, tracker.scale));
      if (isSaneQuad(next)) quad = next;
    }
  }

  if (!quad && !placing.length && detector && !detecting && video.readyState >= 2 && performance.now() - lastDetect > 400) {
    runDetector();
  }

  if (!quad) {
    drawPlacing();
    hotspotEls.forEach((el) => { el.style.display = 'none'; });
    return;
  }

  const sq = quad.map(toScreen);
  drawQuad(sq);
  const Hq = squareToQuad(quad);
  config.hotspots.forEach((hs, i) => {
    const [sx, sy] = toScreen(applyH(Hq, [hs.x, hs.y]));
    const el = hotspotEls[i];
    const visible = sx > -20 && sy > -20 && sx < w + 20 && sy < h + 20;
    el.style.display = visible ? '' : 'none';
    if (visible) el.style.transform = `translate(${sx}px, ${sy}px)`;
  });
}

function drawPlacing() {
  const pts = placing.map(toScreen);
  ctx.lineWidth = 3;
  ctx.strokeStyle = '#ffb400';
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
  pts.forEach(([x, y]) => dot(x, y, 7, '#ffb400'));
}

function drawQuad(sq) {
  ctx.lineWidth = 3;
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgba(255, 180, 0, 0.95)';
  ctx.shadowColor = 'rgba(0, 0, 0, 0.6)';
  ctx.shadowBlur = 6;
  ctx.beginPath();
  sq.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.stroke();
  ctx.shadowBlur = 0;
  sq.forEach(([x, y]) => dot(x, y, 6, '#fff'));
}

function dot(x, y, r, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}
