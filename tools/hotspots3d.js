// 3D hotspot placement (admin testing only): like tools/hotspots.html, but each point has a real depth in
// millimetres behind the front glass, gathered and used with the same camera-pose maths that keeps the
// live app's outline a real rectangle (js/pose3d.js). A single photo can never prove a depth is correct -
// that needs a moving camera - so verify placements in admin3d/ before trusting them.
//
// Publishes to data/pod-3d.json, entirely separate from data/pod.json (the live 2D app's file).

import { loadOpenCV } from '../js/tracker.js';
import { loadDetector } from '../js/detector.js';
import { solvePose, projectPoints, unprojectAtDepth } from '../js/pose3d.js';
import { publishFile, PublishError } from './github-publish.js';

const $ = (id) => document.getElementById(id);
const view = $('view');
const ctx = view.getContext('2d');

const MARGIN = 0.25;
const HANDLE = 14;
const DOT_R = 12;
const MAX_SIDE = 1600;
const CORNERS = ['top-left', 'top-right', 'bottom-right', 'bottom-left'];
const DATA_FILE = '../data/pod-3d.json';
const SAVE_KEY = 'pod3d:hotspots:v1';
const PUB_KEY = 'pod3d:publish:v1';
const TOKEN_KEY = 'pod3d:token';

const S = {
  src: null, video: null,
  quad: null,           // [TL,TR,BR,BL] in src pixels
  placing: [],
  pose: null,           // solved camera pose for the current quad + front_mm; recomputed whenever either changes
  points: [],           // [{ title, text, url, embed, x_mm, y_mm, z_mm, _px }]  _px = last clicked pixel on the CURRENT photo (not exported)
  meta: { name: 'Pod (3D)', note: "x_mm/y_mm/z_mm: 3D position in the pod's own frame, millimetres. z is depth behind the front glass.", front_mm: null },
  sel: -1,
  drag: null,
  detector: null,
  loadingDetector: false,
  baseText: null,
};
window.podHotspots3d = S;

// ---------- start-up ----------

async function init() {
  try { S.baseText = await (await fetch(DATA_FILE)).text(); } catch { S.baseText = null; }
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(SAVE_KEY)); } catch { /* private mode */ }
  if (saved && Array.isArray(saved.points)) { S.points = saved.points; S.meta = { ...S.meta, ...saved.meta }; }
  else await reloadFromFile();
  renderFront();
  renderList();
  loadDetectorInBackground();
  loadDefaultPhoto();
}

async function reloadFromFile() {
  try {
    const j = await (await fetch(DATA_FILE)).json();
    S.meta = { name: j.name || S.meta.name, note: j.note || S.meta.note, front_mm: j.front_mm || null };
    S.points = (j.points || []).map((p) => ({ title: p.title || '', text: p.text || '', url: p.url || '', embed: p.embed !== false, x_mm: p.x_mm ?? null, y_mm: p.y_mm ?? null, z_mm: p.z_mm ?? 0, _px: null }));
  } catch (e) {
    setStatus('Could not read data/pod-3d.json: ' + e.message);
  }
}

function save() {
  try { localStorage.setItem(SAVE_KEY, JSON.stringify({ points: S.points.map(({ _px, ...p }) => p), meta: S.meta })); } catch { /* ignore */ }
}

async function loadDefaultPhoto() {
  try {
    const img = new Image();
    img.src = '../data/pod-photo.jpg';   // the same real photo the 2D tool defaults to
    await img.decode();
    setPicture(img, img.naturalWidth, img.naturalHeight);
  } catch { /* no default shipped, or it failed: stay on the "open a photo" empty state */ }
}

async function loadDetectorInBackground() {
  if (S.detector || S.loadingDetector) return;
  S.loadingDetector = true;
  try {
    const { cv } = await loadOpenCV();
    S.detector = await loadDetector(cv, '../models/pod_front.onnx');
    S.cv = cv;
    $('find').disabled = !S.src;
    if (S.src && !S.quad) findOutline();
  } catch (e) {
    console.warn('detector unavailable:', e);
    setStatus('Automatic outline is unavailable here; click the four corners instead.');
  } finally {
    S.loadingDetector = false;
  }
}

// ---------- opening a photo or video ----------

$('file').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  S.video = null;
  $('scrubGrp').hidden = true;
  if (f.type.startsWith('video/')) {
    const v = document.createElement('video');
    v.muted = true; v.playsInline = true; v.src = url;
    await new Promise((res, rej) => { v.onloadeddata = res; v.onerror = () => rej(new Error('could not decode this video')); }).catch((err) => setStatus(err.message));
    if (!v.videoWidth) return;
    S.video = v;
    $('scrub').max = String(v.duration);
    $('scrub').value = '0';
    $('scrubGrp').hidden = false;
    setPicture(v, v.videoWidth, v.videoHeight);
  } else {
    const img = new Image();
    img.src = url;
    await img.decode().catch(() => setStatus('Could not open that image.'));
    if (!img.naturalWidth) return;
    setPicture(img, img.naturalWidth, img.naturalHeight);
  }
});

function setPicture(source, w, h) {
  const k = Math.min(1, MAX_SIDE / Math.max(w, h));
  const c = document.createElement('canvas');
  c.width = Math.round(w * k); c.height = Math.round(h * k);
  c.getContext('2d').drawImage(source, 0, 0, c.width, c.height);
  c.videoWidth = c.width; c.videoHeight = c.height;
  S.src = c;
  S.quad = null; S.pose = null; S.placing = [];
  for (const p of S.points) p._px = null;   // a new photo invalidates any cached click rays from the old one
  $('empty').hidden = true;
  $('resetOutline').disabled = false;
  $('find').disabled = !S.detector;
  draw();
  setStatus(S.detector ? 'Looking for the pod...' : `Click the pod's top-left corner (1 of 4), or wait a moment for the automatic outline.`);
  if (S.detector) findOutline(); else loadDetectorInBackground();
}

$('scrub').addEventListener('input', async () => {
  if (!S.video) return;
  await new Promise((res) => { S.video.onseeked = res; S.video.currentTime = parseFloat($('scrub').value); });
  setPicture(S.video, S.video.videoWidth, S.video.videoHeight);
});

// ---------- the outline and its pose ----------

$('find').addEventListener('click', () => findOutline());
$('resetOutline').addEventListener('click', () => { S.quad = null; S.pose = null; S.placing = []; draw(); });

async function findOutline() {
  if (!S.detector || !S.src) return;
  setStatus('Looking for the pod...');
  try {
    const r = await S.detector.detect(S.src);
    if (r.quad) {
      S.quad = r.quad;
      recomputePose();
      setStatus(S.pose ? 'Found the outline. Drag a corner if it is off, then place the features.' : 'Found the outline, but the pod front size below looks wrong for it (or is not set).');
    } else if (r.clipped) {
      setStatus('The pod is not fully in the picture, so its corners are off-screen. Click the four corners instead.');
    } else {
      setStatus('No pod found. Click the four corners instead.');
    }
  } catch (e) {
    console.error(e);
    setStatus('Automatic outline failed: ' + e.message + '. Click the four corners instead.');
  }
  draw();
}

// Recomputes the camera pose from the current quad + front size. Called whenever either changes; every
// point's screen position and every click-to-place calculation depends on this being current.
function recomputePose() {
  const front = S.meta.front_mm;
  S.pose = (S.quad && S.cv && front && front.width > 0 && front.height > 0) ? solvePose(S.cv, S.quad, S.src.width, S.src.height, front) : null;
}

// ---------- geometry ----------

function layout() {
  const cw = view.clientWidth, ch = view.clientHeight;
  const w = S.src ? S.src.width : 1, h = S.src ? S.src.height : 1;
  const s = Math.min(cw / (w * (1 + 2 * MARGIN)), ch / (h * (1 + 2 * MARGIN)));
  return { s, ox: (cw - w * s) / 2, oy: (ch - h * s) / 2, cw, ch, w, h };
}
const toScreen = ([x, y], L = layout()) => [x * L.s + L.ox, y * L.s + L.oy];
const toImg = (x, y, L = layout()) => [(x - L.ox) / L.s, (y - L.oy) / L.s];

// ---------- drawing ----------

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const L = layout();
  if (view.width !== Math.round(L.cw * dpr) || view.height !== Math.round(L.ch * dpr)) {
    view.width = Math.round(L.cw * dpr);
    view.height = Math.round(L.ch * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0b0c0f';
  ctx.fillRect(0, 0, L.cw, L.ch);
  if (!S.src) return;
  ctx.drawImage(S.src, L.ox, L.oy, L.w * L.s, L.h * L.s);
  ctx.strokeStyle = 'rgba(255,255,255,.25)';
  ctx.lineWidth = 1;
  ctx.strokeRect(L.ox, L.oy, L.w * L.s, L.h * L.s);

  if (S.quad) {
    if ($('grid').checked && S.pose) drawGrid(L);
    const sq = S.quad.map((p) => toScreen(p, L));
    ctx.lineWidth = 2.5; ctx.lineJoin = 'round'; ctx.strokeStyle = '#ffb400';
    ctx.beginPath();
    sq.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath(); ctx.stroke();
    sq.forEach(([x, y], i) => cornerHandle(x, y, ['TL', 'TR', 'BR', 'BL'][i]));
    if (S.pose) {
      const pts = S.points.map((p) => (p.x_mm != null ? projectPoints(S.pose, [[p.x_mm, p.y_mm, p.z_mm]])[0] : null));
      S.points.forEach((p, i) => { if (pts[i]) dot(...toScreen(pts[i], L), i + 1, i === S.sel); });
    }
  } else if (S.placing.length) {
    const pts = S.placing.map((p) => toScreen(p, L));
    ctx.strokeStyle = '#ffb400'; ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
    pts.forEach(([x, y], i) => cornerHandle(x, y, ['TL', 'TR', 'BR', 'BL'][i]));
  }
}

// The front glass plane (z=0), divided into tenths, so a point meant to sit ON the glass can be lined up.
function drawGrid(L) {
  const front = S.meta.front_mm;
  ctx.strokeStyle = 'rgba(255,180,0,.35)'; ctx.lineWidth = 1;
  for (let k = 1; k < 10; k++) {
    const t = k / 10;
    const segs = [[[t * front.width, 0, 0], [t * front.width, front.height, 0]], [[0, t * front.height, 0], [front.width, t * front.height, 0]]];
    for (const [p0, p1] of segs) {
      const proj = projectPoints(S.pose, [p0, p1]);
      if (!proj[0] || !proj[1]) continue;
      const [a, b] = [toScreen(proj[0], L), toScreen(proj[1], L)];
      ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
    }
  }
}

function cornerHandle(x, y, label) {
  ctx.fillStyle = '#ffb400';
  ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#000'; ctx.lineWidth = 1; ctx.stroke();
  ctx.fillStyle = '#fff'; ctx.font = '600 12px system-ui'; ctx.textAlign = 'left';
  ctx.fillText(label, x + 9, y - 8);
}

function dot(x, y, n, selected) {
  ctx.fillStyle = '#ff6bd6';
  ctx.beginPath(); ctx.arc(x, y, DOT_R, 0, Math.PI * 2); ctx.fill();
  ctx.lineWidth = selected ? 3 : 1.5; ctx.strokeStyle = selected ? '#fff' : '#000'; ctx.stroke();
  ctx.fillStyle = '#1a0512'; ctx.font = '700 12px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(String(n), x, y + 0.5);
  ctx.textBaseline = 'alphabetic';
}

new ResizeObserver(draw).observe($('stage'));
$('grid').addEventListener('change', draw);

// ---------- clicking and dragging ----------

view.addEventListener('pointerdown', (e) => {
  if (!S.src) return;
  const r = view.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  const L = layout();

  if (!S.quad) {
    S.placing.push(toImg(x, y, L));
    if (S.placing.length === 4) { S.quad = S.placing; S.placing = []; recomputePose(); setStatus(S.pose ? 'Outline set. Drag a corner if it is off, then place the features.' : 'Outline set, but the pod front size is not set correctly below.'); }
    else setStatus(`Click the pod's ${CORNERS[S.placing.length]} corner (${S.placing.length + 1} of 4).`);
    draw();
    return;
  }
  let best = null, bd = HANDLE;
  if (S.pose) {
    S.points.forEach((p, i) => {
      if (p.x_mm == null) return;
      const proj = projectPoints(S.pose, [[p.x_mm, p.y_mm, p.z_mm]])[0];
      if (!proj) return;
      const [sx, sy] = toScreen(proj, L);
      const d = Math.hypot(sx - x, sy - y);
      if (d < bd) { bd = d; best = { type: 'dot', i }; }
    });
  }
  if (!best) {
    S.quad.forEach((p, i) => {
      const [sx, sy] = toScreen(p, L);
      const d = Math.hypot(sx - x, sy - y);
      if (d < bd) { bd = d; best = { type: 'corner', i }; }
    });
  }
  if (best) {
    S.drag = best;
    view.setPointerCapture(e.pointerId);
    if (best.type === 'dot') select(best.i);
    draw();
    return;
  }
  if (S.sel < 0) { setStatus('Select a feature in the list first, then click its spot on the picture.'); return; }
  placeSelected(toImg(x, y, L));
});

view.addEventListener('pointermove', (e) => {
  if (!S.drag) return;
  const r = view.getBoundingClientRect();
  const p = toImg(e.clientX - r.left, e.clientY - r.top);
  if (S.drag.type === 'corner') {
    S.quad[S.drag.i] = p;
    recomputePose();
  } else {
    placeAt(S.drag.i, p, false);
  }
  draw();
});
const endDrag = () => { if (S.drag) { S.drag = null; save(); renderList(); } };
view.addEventListener('pointerup', endDrag);
view.addEventListener('pointercancel', endDrag);

// Places (or re-places) point `i` at picture-space pixel `imgPx`, using its own depth to resolve the exact
// [x_mm, y_mm] via ray/plane intersection.
function placeAt(i, imgPx, announce = true) {
  if (!S.pose) { setStatus('Set the front size below first (needed to work out 3D positions).'); return false; }
  const xy = unprojectAtDepth(S.pose, imgPx[0], imgPx[1], S.points[i].z_mm ?? 0);
  if (!xy) { setStatus('Cannot place a point at that depth from this viewing angle; try a less edge-on reference photo.'); return false; }
  S.points[i].x_mm = xy[0]; S.points[i].y_mm = xy[1]; S.points[i]._px = imgPx;
  if (announce) setStatus(`Placed "${S.points[i].title || 'feature ' + (i + 1)}" at x ${xy[0].toFixed(0)} mm, y ${xy[1].toFixed(0)} mm, depth ${(S.points[i].z_mm ?? 0)} mm.`);
  return true;
}
function placeSelected(imgPx) {
  if (placeAt(S.sel, imgPx)) { save(); renderList(); draw(); }
}

// ---------- the feature list ----------

function renderList() {
  const list = $('list');
  list.textContent = '';
  S.points.forEach((p, i) => {
    const row = document.createElement('div');
    row.className = 'row' + (i === S.sel ? ' sel' : '');
    const n = document.createElement('span');
    n.className = 'n' + (p.x_mm == null ? ' off' : '');
    n.textContent = String(i + 1);
    const t = document.createElement('span');
    t.className = 't';
    t.textContent = p.title || '(untitled)';
    const xy = document.createElement('span');
    xy.className = 'xy';
    xy.textContent = p.x_mm == null ? 'not placed' : `${p.x_mm.toFixed(0)}, ${p.y_mm.toFixed(0)}, ${(p.z_mm ?? 0).toFixed(0)}mm`;
    row.append(n, t, xy);
    row.addEventListener('click', () => select(i));
    list.appendChild(row);
  });
  $('export').disabled = $('copy').disabled = $('publish').disabled = !S.points.length;
  $('del').disabled = S.sel < 0;
  const f = S.points[S.sel];
  for (const [id, key] of [['fTitle', 'title'], ['fText', 'text'], ['fUrl', 'url']]) {
    $(id).value = f ? f[key] : '';
    $(id).disabled = !f;
  }
  $('fEmbed').checked = f ? f.embed !== false : true;
  $('fEmbed').disabled = !f;
  $('fDepth').value = f ? (f.z_mm ?? 0) : 0;
  $('fDepth').disabled = !f;
  $('fX').value = f && f.x_mm != null ? Math.round(f.x_mm) : '';
  $('fY').value = f && f.y_mm != null ? Math.round(f.y_mm) : '';
  $('fX').disabled = $('fY').disabled = !f;
  $('fPxNote').textContent = f ? (f._px ? 'from a click on the current photo — editing depth recomputes x, y' : (f.x_mm != null ? 'from a saved position — click the photo to refresh' : 'not clicked yet')) : '';
}

function select(i) {
  S.sel = i;
  renderList();
  draw();
  const p = S.points[i];
  if (p && p.x_mm == null) setStatus(`Set a depth if you have one, then click where "${p.title || 'this feature'}" is on the picture.`);
}

for (const [id, key] of [['fTitle', 'title'], ['fText', 'text'], ['fUrl', 'url']]) {
  $(id).addEventListener('input', () => {
    if (S.sel < 0) return;
    S.points[S.sel][key] = $(id).value;
    save();
    if (key === 'title') {
      const t = $('list').children[S.sel]?.querySelector('.t');
      if (t) t.textContent = $(id).value || '(untitled)';
    }
  });
}
$('fEmbed').addEventListener('change', () => { if (S.sel >= 0) { S.points[S.sel].embed = $('fEmbed').checked; save(); } });

$('fDepth').addEventListener('change', () => {
  if (S.sel < 0) return;
  const p = S.points[S.sel];
  p.z_mm = parseFloat($('fDepth').value) || 0;
  if (p._px && S.pose) { placeAt(S.sel, p._px); renderList(); draw(); }   // re-solve x,y at the new depth from the same click
  else { save(); renderList(); draw(); }
});
for (const [id, key] of [['fX', 'x_mm'], ['fY', 'y_mm']]) {
  $(id).addEventListener('change', () => {
    if (S.sel < 0) return;
    const v = parseFloat($(id).value);
    S.points[S.sel][key] = Number.isFinite(v) ? v : null;
    S.points[S.sel]._px = null;   // typed by hand: no longer tied to a specific click, so depth edits won't move it
    save(); renderList(); draw();
  });
}

$('add').addEventListener('click', () => {
  S.points.push({ title: 'New feature', text: '', url: '', embed: true, x_mm: null, y_mm: null, z_mm: 0, _px: null });
  save();
  select(S.points.length - 1);
  $('fTitle').focus();
  $('fTitle').select();
});
$('del').addEventListener('click', () => {
  if (S.sel < 0) return;
  S.points.splice(S.sel, 1);
  S.sel = Math.min(S.sel, S.points.length - 1);
  save(); renderList(); draw();
});
$('reload').addEventListener('click', async () => {
  await reloadFromFile();
  S.sel = -1;
  renderFront();
  save(); renderList(); draw();
  setStatus('Loaded the points from data/pod-3d.json.');
});
$('clearAll').addEventListener('click', () => {
  if (S.points.length && !confirm('Remove all 3D points from this list? (data/pod-3d.json is not changed until you replace it.)')) return;
  S.points = []; S.sel = -1;
  save(); renderList(); draw();
});

// ---------- the pod front's real size ----------

function renderFront() {
  const f = S.meta.front_mm;
  $('fFrontW').value = f ? f.width : '';
  $('fFrontH').value = f ? f.height : '';
}
for (const id of ['fFrontW', 'fFrontH']) {
  $(id).addEventListener('change', () => {
    const w = parseFloat($('fFrontW').value), h = parseFloat($('fFrontH').value);
    S.meta.front_mm = w > 0 && h > 0 ? { width: w, height: h } : null;
    recomputePose();
    save(); draw();
  });
}

// ---------- export ----------

const slug = (t) => t.toLowerCase().normalize('NFKD').replace(/[^\w]+/g, '-').replace(/^-+|-+$/g, '');

function buildJson() {
  const used = new Set();
  const points = S.points.filter((p) => p.x_mm != null).map((p, i) => {
    let id = slug(p.title) || `point-${i + 1}`, k = 2;
    while (used.has(id)) id = `${slug(p.title) || 'point'}-${k++}`;
    used.add(id);
    return { id, title: p.title, text: p.text, url: p.url, ...(p.embed === false ? { embed: false } : {}), x_mm: +p.x_mm.toFixed(1), y_mm: +p.y_mm.toFixed(1), z_mm: +(p.z_mm ?? 0).toFixed(1) };
  });
  const front = S.meta.front_mm && S.meta.front_mm.width > 0 && S.meta.front_mm.height > 0 ? { front_mm: { width: S.meta.front_mm.width, height: S.meta.front_mm.height } } : {};
  return JSON.stringify({ name: S.meta.name, note: S.meta.note, ...front, points }, null, 2) + '\n';
}
S.buildJson = buildJson;

function unplaced() { return S.points.filter((p) => p.x_mm == null).length; }

$('export').addEventListener('click', () => {
  const n = unplaced();
  if (n && !confirm(`${n} feature${n > 1 ? 's are' : ' is'} not placed yet and will be left out. Download anyway?`)) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([buildJson()], { type: 'application/json' }));
  a.download = 'pod-3d.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  setStatus('Downloaded pod-3d.json. Replace data/pod-3d.json in the repo with it and push.');
});
$('copy').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(buildJson()); setStatus('Copied the JSON. Paste it over the contents of data/pod-3d.json.'); }
  catch { setStatus('Could not copy; use Download pod-3d.json instead.'); }
});

function setStatus(t) { $('status').textContent = t; }

init();

// ---------- publish to GitHub (separate token/prefs from the 2D tool) ----------

const pub = $('pub');

function defaultRepo() {
  const m = location.hostname.match(/^([^.]+)\.github\.io$/);
  const repo = location.pathname.split('/')[1];
  return m && repo ? `${m[1]}/${repo}` : 'Sweep499/ARQR';
}
function readPubPrefs() { try { return JSON.parse(localStorage.getItem(PUB_KEY)) || {}; } catch { return {}; } }
function readToken() { try { return localStorage.getItem(TOKEN_KEY) || sessionStorage.getItem(TOKEN_KEY) || ''; } catch { return ''; } }
function pubStatus(text, kind = '') { const el = $('pubStatus'); el.className = kind; el.textContent = text; return el; }

$('publish').addEventListener('click', () => {
  const prefs = readPubPrefs();
  $('pubRepo').value = prefs.repo || defaultRepo();
  $('pubBranch').value = prefs.branch || 'main';
  $('pubPath').value = prefs.path || 'data/pod-3d.json';
  $('pubMsg').value = prefs.msg || 'Update 3D feature points (admin testing)';
  const t = readToken();
  $('pubToken').value = t;
  try { $('pubRemember').checked = !!localStorage.getItem(TOKEN_KEY); } catch { $('pubRemember').checked = false; }
  $('pubForce').hidden = true;
  pubStatus(unplaced() ? `${unplaced()} feature${unplaced() > 1 ? 's are' : ' is'} not placed yet and will be left out.` : t ? '' : 'Paste a token to publish.');
  pub.showModal();
});
$('pubClose').addEventListener('click', () => pub.close());
$('pubForget').addEventListener('click', () => {
  try { localStorage.removeItem(TOKEN_KEY); sessionStorage.removeItem(TOKEN_KEY); } catch { /* ignore */ }
  $('pubToken').value = '';
  $('pubRemember').checked = false;
  pubStatus('The token was removed from this browser.');
});

async function runPublish(force) {
  const placed = S.points.filter((p) => p.x_mm != null).length;
  if (!placed) { pubStatus('Place at least one feature on the picture first.', 'err'); return; }
  const token = $('pubToken').value.trim();
  const prefs = { repo: $('pubRepo').value.trim(), branch: $('pubBranch').value.trim() || 'main', path: $('pubPath').value.trim() || 'data/pod-3d.json', msg: $('pubMsg').value.trim() || 'Update 3D feature points (admin testing)' };
  try {
    localStorage.setItem(PUB_KEY, JSON.stringify(prefs));
    localStorage.removeItem(TOKEN_KEY); sessionStorage.removeItem(TOKEN_KEY);
    (pub.querySelector('#pubRemember').checked ? localStorage : sessionStorage).setItem(TOKEN_KEY, token);
  } catch { /* private mode: the token just is not remembered */ }

  const text = buildJson();
  $('pubGo').disabled = $('pubForce').disabled = true;
  $('pubForce').hidden = true;
  pubStatus('Publishing...');
  try {
    const r = await publishFile({ token, repo: prefs.repo, branch: prefs.branch, path: prefs.path, text, message: prefs.msg, baseText: S.baseText, force });
    S.baseText = text;
    if (r.status === 'unchanged') {
      pubStatus('GitHub already has exactly these points. Nothing to publish.', 'ok');
    } else {
      const el = pubStatus(`Published ${placed} point${placed > 1 ? 's' : ''}. Commit `, 'ok');
      const a = document.createElement('a');
      a.href = r.commitUrl || '#'; a.target = '_blank'; a.rel = 'noopener';
      a.textContent = (r.commitSha || '').slice(0, 7) || 'view';
      el.append(a, '. The 3D test view uses the new points in about a minute.');
    }
  } catch (e) {
    pubStatus(e instanceof PublishError ? e.message : 'Publishing failed: ' + e.message, 'err');
    if (e instanceof PublishError && e.kind === 'changed') $('pubForce').hidden = false;
  } finally {
    $('pubGo').disabled = $('pubForce').disabled = false;
  }
}
$('pubGo').addEventListener('click', () => runPublish(false));
$('pubForce').addEventListener('click', () => runPublish(true));
