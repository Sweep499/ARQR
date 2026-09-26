import { squareToQuad, applyH, applyScaledH, isPlausibleQuad, quadInView } from './geometry.js';
import { loadOpenCV, FlowTracker } from './tracker.js';
import { loadDetector } from './detector.js';

// Two pages share this code. The admin page (index.html) draws the outline the app has found, so an
// administrator can check what it does. The user page (user/index.html, data-mode="user") tracks exactly
// the same way but draws no outline or corner handles, and does not ask people to tap corners.
const USER_MODE = document.documentElement.dataset.mode === 'user';
const TXT = USER_MODE ? {
  wait: 'Getting ready…',
  search: 'Point the camera at the pod and step back until you can see its whole front.',
  partial: 'Step back a little so the whole front of the pod is in view.',
  found: 'Tap a numbered dot for details.',
  foundAgain: '',
  lost: () => 'Point the camera back at the pod and step back until you can see its whole front.',
  lostTrack: 'Point the camera back at the pod.',
  noDetector: 'The pod finder could not start on this device.',
} : {
  wait: null,
  search: 'Looking for the pod. Step back until you can see its whole front, or tap its corners.',
  partial: 'I can see the pod but not all of its front. Step back a little, or tap its corners.',
  found: 'Found the pod. Tap a numbered dot for details. Drag a corner to fine-tune.',
  foundAgain: 'Found the pod again.',
  lost: (why) => why + ' Point the camera at it and step back until the whole front is in view, or tap its corners.',
  lostTrack: 'Lost track of the pod. Point the camera back at it.',
  noDetector: null,
};

const $ = (id) => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const ctx = overlay.getContext('2d');
const hotspotLayer = $('hotspots');
const hint = $('hint');
const sheet = $('sheet');

const CORNER_NAMES = ['top-left', 'top-right', 'bottom-right', 'bottom-left'];
const HANDLE_RADIUS = 32; // screen px within which a touch grabs a corner
const HIDE_AFTER = 400;   // ms without a trustworthy motion estimate before the outline is greyed out and the dots hidden
const DROP_AFTER = 1500;  // ms before the outline is dropped and the app goes back to looking for the pod
const OFF_DROP = 1500;    // ms the outline may stay completely out of view before it is dropped (a guess drifts while unseen)
const AGREE_MIN = 0.4;    // an outline whose overlap with the detector's mask stays below this is not on the pod
const ABSENT_FRAC = 0.02; // the detector's mask covers less of the picture than this: there is no pod in view
const AGREE_STRIKES = 2;  // consecutive re-checks (about REDETECT_MS apart) before dropping it
const REDETECT_MS = 500;  // how often the detector re-checks while an outline is showing

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
let lostSince = 0;        // when tracking last failed (0 = tracking is fine)
let offSince = 0;         // when the outline last left the picture entirely
let strikes = 0;          // consecutive re-checks where the detector's mask disagreed with the outline
let manual = false;       // the outline came from the user's taps or drags, so the detector must not nudge it
let holdHintUntil = 0;    // keep a status message on screen until then
let hotspotEls = [];
let activeEl = null;
let activeHotspot = null;  // the feature whose card is open
let viewerOpen = false;   // the in-app page viewer is showing
let hintTimer = 0;
if (debug) window.podDebug = { get quad() { return quad; }, get tracker() { return tracker; }, set detector(d) { detector = d; }, get lost() { return lostSince; } };

// ---------- start-up ----------

$('startBtn').addEventListener('click', start);
$('resetBtn').addEventListener('click', () => beginPlacing());
$('sheetClose').addEventListener('click', closeSheet);
window.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSheet(); });

async function start() {
  const err = $('error');
  err.hidden = true;
  try {
    config = await (await fetch(new URL('../data/pod.json', import.meta.url))).json();
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
      loadDetector(cv).then((d) => { detector = d; if (!quad && !placing.length) updatePlacingHint(); }).catch((e) => { console.warn('pod detector unavailable:', e); if (TXT.noDetector) setHint(TXT.noDetector); });
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
  lostSince = 0;
  offSince = 0;
  strikes = 0;
  manual = false;
  closeSheet();
  if (tracker) tracker.reset();
  updatePlacingHint();
}

function updatePlacingHint() {
  const n = placing.length;
  if (n === 0 && detector) {
    setHint(TXT.search);
  } else if (USER_MODE) {
    setHint(TXT.wait);
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
  if (e.target.closest('button, aside, a, #viewer')) return;
  if (USER_MODE) { closeSheet(); return; }   // nothing to place or drag: the outline is not shown
  const p = toVideo(e.clientX, e.clientY);
  if (!quad) {
    placing.push(p);
    if (placing.length === 4) {
      quad = placing;
      placing = [];
      manual = true;
      lostSince = 0;
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
    manual = true;
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

const meanDist = (a, b) => a.reduce((s, p, i) => s + Math.hypot(p[0] - b[i][0], p[1] - b[i][1]), 0) / 4;

// Tracking failed for too long: forget the outline and go back to looking for the pod.
function dropQuad(why = 'Lost the pod.') {
  beginPlacing();
  setHint(TXT.lost(why));
  holdHintUntil = performance.now() + 5000;
}

// Accepts a detection once two in a row agree (corners within 5% of the picture width), so a single
// wrong frame cannot place or move the outline. With an outline showing, a confirmed detection
// re-anchors it: it snaps at once if tracking was lost or the outline is well off, and otherwise nudges
// it 35% of the way, which cancels slow drift. An outline the user placed or dragged is left alone
// unless tracking has been lost.
async function runDetector() {
  detecting = true;
  try {
    const r = await detector.detect(video);
    lastDetect = performance.now();
    if (debug) console.log('detect', video.currentTime.toFixed(1), r.quad ? 'quad ' + r.score.toFixed(2) : r.clipped ? 'clipped' : 'none');
    if (placing.length || dragging >= 0) return;   // the user got there first
    const vw = video.videoWidth, vh = video.videoHeight;

    // an outline the detector does not support is wrong: either there is no pod in view any more, or the
    // pod is somewhere the outline is not (drifted). Two checks in a row (about a second) and it is deleted.
    if (quad) {
      const absent = (r.maskFrac ?? 1) < ABSENT_FRAC;
      const a = absent || !r.mask ? 0 : detector.agreement(r.mask, quad, vw, vh);
      if (debug) console.log(absent ? 'pod absent' : 'agreement ' + a.toFixed(2));
      strikes = absent || a < AGREE_MIN ? strikes + 1 : 0;
      if (strikes >= AGREE_STRIKES) { dropQuad(absent ? 'The pod went out of view.' : 'The outline drifted off the pod.'); return; }
    }

    if (r.quad) {
      const agrees = candidate && meanDist(candidate, r.quad) < 0.05 * vw;
      if (!agrees) { candidate = r.quad; return; }
      candidate = null;
      strikes = 0;
      if (!quad) {
        quad = r.quad;
        manual = false;
        setHint(TXT.found, 7000);
      } else if (lostSince) {
        quad = r.quad; lostSince = 0; manual = false;
        setHint(TXT.foundAgain, 3000);
      } else if (!manual) {
        const d = meanDist(quad, r.quad);
        if (d > 0.06 * vw) quad = r.quad;
        else if (d > 0.005 * vw) quad = quad.map((p, i) => [p[0] + 0.35 * (r.quad[i][0] - p[0]), p[1] + 0.35 * (r.quad[i][1] - p[1])]);
        if (debug) console.log('re-anchored, corner error', (d / vw * 100).toFixed(1) + '% of width');
      }
    } else {
      candidate = null;
      if (!quad) {
        // tell the user why nothing happened, without rewriting the same hint every 400 ms
        const want = r.clipped
          ? TXT.partial
          : TXT.search;
        if (hint.textContent !== want && performance.now() > holdHintUntil) setHint(want);
      }
    }
  } catch (e) {
    console.error(e);
    detector = null;                          // give up on detection, manual placing still works
    if (!quad) updatePlacingHint();
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
  activeHotspot = h;
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

// ---------- in-app page viewer ----------

// "Open feature page" shows the page over the camera view instead of leaving the app; Back to camera, the Back
// button on the phone, or Escape returns to it. The camera and tracking keep running underneath. Some sites
// refuse to be shown inside another page (silen.com does), so the bar always has "Open in browser", and a
// feature can set "embed": false in data/pod.json to open in a new tab straight away.
$('sheetLink').addEventListener('click', (e) => {
  const h = activeHotspot;
  if (!h || !h.url) return;
  e.preventDefault();
  if (h.embed === false) { window.open(h.url, '_blank', 'noopener'); return; }
  openViewer(h);
});
$('viewerClose').addEventListener('click', closeViewer);
window.addEventListener('popstate', () => { if (viewerOpen && !history.state?.viewer) hideViewer(); });
window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && viewerOpen) closeViewer(); });

// Closing pops the history entry openViewer added. If the person browsed inside the page, that Back only
// steps the page's own history, so the viewer is hidden regardless a moment later.
function closeViewer() {
  if (!viewerOpen) return;
  if (history.state?.viewer) history.back();
  setTimeout(() => { if (viewerOpen) hideViewer(); }, 150);
}

const IS_PHOTO = /\.(jpe?g|png|webp|gif|avif)(\?.*)?$/i;

function openViewer(h) {
  $('viewerTitle').textContent = h.title || '';
  $('viewerOpen').href = h.url;
  $('viewerLoading').textContent = 'Loading\u2026';
  $('viewerLoading').hidden = false;
  const f = $('viewerFrame'), img = $('viewerImg');
  const photo = IS_PHOTO.test(h.url);
  f.hidden = photo;
  img.hidden = !photo;
  $('viewer').hidden = false;
  viewerOpen = true;
  history.pushState({ viewer: true }, '');       // so the phone's Back button closes the viewer, not the app
  if (photo) {
    img.alt = h.title || '';
    img.onload = () => { $('viewerLoading').hidden = true; };
    img.onerror = () => { $('viewerLoading').textContent = 'The picture could not be loaded. Try "Open in browser".'; };
    img.src = h.url;
  } else {
    f.onload = () => { $('viewerLoading').hidden = true; };
    f.contentWindow.location.replace(h.url);     // replace, not src: setting src would add a history entry of its own
  }
  $('viewerClose').focus();
}

function hideViewer() {
  viewerOpen = false;
  $('viewer').hidden = true;
  const f = $('viewerFrame');
  f.onload = null;
  f.contentWindow.location.replace('about:blank'); // stops the page loading, playing sound or running
  const img = $('viewerImg');
  img.onload = img.onerror = null;
  img.removeAttribute('src');
  img.hidden = true;
  f.hidden = false;
  if (!$('sheet').hidden) $('sheetLink').focus();
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
    if (quad && dragging < 0) {
      let moved = false;
      if (H) {
        const next = quad.map((p) => applyScaledH(H, p, tracker.scale));
        if (isPlausibleQuad(next, video.videoWidth, video.videoHeight)) { quad = next; moved = true; }
      }
      if (moved) {
        if (lostSince) { lostSince = 0; setHint(''); }
      } else if (!lostSince) {
        lostSince = performance.now();
      } else if (performance.now() - lostSince > DROP_AFTER) {
        dropQuad();
      }
    }
  }

  if (quad && dragging < 0) {
    if (quadInView(quad, video.videoWidth, video.videoHeight)) offSince = 0;
    else if (!offSince) offSince = performance.now();
    else if (performance.now() - offSince > OFF_DROP) dropQuad('The pod went out of view.');
  }

  // look for the pod while there is no outline, and re-check now and then while there is one, so a
  // lost or drifted outline snaps back as soon as the whole front is in view
  const idle = !quad ? !placing.length : dragging < 0;
  if (idle && detector && !detecting && video.readyState >= 2 && performance.now() - lastDetect > (quad ? REDETECT_MS : 400)) {
    runDetector();
  }

  if (!quad) {
    if (!USER_MODE) drawPlacing();
    hotspotEls.forEach((el) => { el.style.display = 'none'; });
    return;
  }

  const lost = lostSince && performance.now() - lostSince > HIDE_AFTER;
  if (lost && performance.now() > holdHintUntil) setHint(TXT.lostTrack);
  const sq = quad.map(toScreen);
  if (!USER_MODE) drawQuad(sq, lost ? 0.3 : 1);
  if (lost) { hotspotEls.forEach((el) => { el.style.display = 'none'; }); return; }
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

function drawQuad(sq, alpha = 1) {
  ctx.globalAlpha = alpha;
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
  ctx.globalAlpha = 1;
}

function dot(x, y, r, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}
