// Hotspot placement: open a photo or video of the pod front, let the detector find its outline (or click the
// four corners), then click where each feature is. Positions are stored as fractions of the front frame
// (x 0 = left edge, 1 = right edge; y 0 = top, 1 = bottom), exactly what data/pod.json and the app use.

import { squareToQuad, applyH, invertH } from '../js/geometry.js';
import { loadOpenCV } from '../js/tracker.js';
import { loadDetector } from '../js/detector.js';
import { publishFile, PublishError } from './github-publish.js';

const $ = (id) => document.getElementById(id);
const view = $('view');
const ctx = view.getContext('2d');

const MARGIN = 0.25;     // dark border around the picture, as a fraction of its size, for off-picture corners
const HANDLE = 14;       // css px within which a click grabs a corner or a dot
const DOT_R = 12;
const MAX_SIDE = 1600;   // the picture is scaled down to this before use (all coordinates are in that space)
const CORNERS = ['top-left', 'top-right', 'bottom-right', 'bottom-left'];
const SAVE_KEY = 'podhotspots:v1';
const PUB_KEY = 'podhotspots:publish:v1';    // repository, branch and file (never the token)
const TOKEN_KEY = 'podhotspots:token';

const S = {
  src: null,             // canvas holding the current picture
  video: null,           // set when a video was opened
  quad: null,            // [TL, TR, BR, BL] in src pixels
  placing: [],           // corners clicked so far
  hot: [],               // [{ title, text, url, x, y }]  x / y null = not placed yet
  meta: { name: 'Pod', note: 'x and y are positions on the pod\'s front frame: x 0 = left edge, x 1 = right edge; y 0 = top edge, y 1 = bottom edge.' },
  sel: -1,
  drag: null,            // { type: 'corner' | 'dot', i }
  detector: null,
  loadingDetector: false,
  baseText: null,        // data/pod.json as this page found it, to notice if GitHub's copy has changed since
};
window.podHotspots = S; // handy for debugging in the console

// ---------- start-up ----------

async function init() {
  try { S.baseText = await (await fetch('../data/pod.json')).text(); } catch { S.baseText = null; }
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(SAVE_KEY)); } catch { /* private mode */ }
  if (saved && Array.isArray(saved.hot)) { S.hot = saved.hot; S.meta = saved.meta || S.meta; }
  else await reloadFromFile();
  renderList();
  loadDetectorInBackground();
}

async function reloadFromFile() {
  try {
    const j = await (await fetch('../data/pod.json')).json();
    S.meta = { name: j.name || 'Pod', note: j.note || S.meta.note };
    S.hot = (j.hotspots || []).map((h) => ({ title: h.title || '', text: h.text || '', url: h.url || '', embed: h.embed !== false, x: h.x ?? null, y: h.y ?? null }));
  } catch (e) {
    setStatus('Could not read data/pod.json: ' + e.message);
  }
}

function save() {
  try { localStorage.setItem(SAVE_KEY, JSON.stringify({ hot: S.hot, meta: S.meta })); } catch { /* ignore */ }
}

async function loadDetectorInBackground() {
  if (S.detector || S.loadingDetector) return;
  S.loadingDetector = true;
  try {
    const { cv } = await loadOpenCV();
    S.detector = await loadDetector(cv, '../models/pod_front.onnx');
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
  c.videoWidth = c.width; c.videoHeight = c.height; // lets the detector treat it like a video frame
  S.src = c;
  S.quad = null;
  S.placing = [];
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

// ---------- the outline ----------

$('find').addEventListener('click', () => findOutline());
$('resetOutline').addEventListener('click', () => { S.quad = null; S.placing = []; draw(); });

async function findOutline() {
  if (!S.detector || !S.src) return;
  setStatus('Looking for the pod...');
  try {
    const r = await S.detector.detect(S.src);
    if (r.quad) {
      S.quad = r.quad;
      setStatus('Found the outline. Drag a corner if it is off, then place the features.');
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

// ---------- geometry ----------

function layout() {
  const cw = view.clientWidth, ch = view.clientHeight;
  const w = S.src ? S.src.width : 1, h = S.src ? S.src.height : 1;
  const s = Math.min(cw / (w * (1 + 2 * MARGIN)), ch / (h * (1 + 2 * MARGIN)));
  return { s, ox: (cw - w * s) / 2, oy: (ch - h * s) / 2, cw, ch, w, h };
}
const toScreen = ([x, y], L = layout()) => [x * L.s + L.ox, y * L.s + L.oy];
const toImg = (x, y, L = layout()) => [(x - L.ox) / L.s, (y - L.oy) / L.s];

// picture point -> fraction of the front frame, or the reverse
function toUnit(p) { return applyH(invertH(squareToQuad(S.quad)), p); }
function fromUnit(u) { return applyH(squareToQuad(S.quad), u); }

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
    if ($('grid').checked) drawGrid(L);
    const sq = S.quad.map((p) => toScreen(p, L));
    ctx.lineWidth = 2.5; ctx.lineJoin = 'round'; ctx.strokeStyle = '#ffb400';
    ctx.beginPath();
    sq.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath(); ctx.stroke();
    sq.forEach(([x, y], i) => cornerHandle(x, y, ['TL', 'TR', 'BR', 'BL'][i]));
    S.hot.forEach((h, i) => {
      if (h.x == null) return;
      const [x, y] = toScreen(fromUnit([h.x, h.y]), L);
      dot(x, y, i + 1, i === S.sel);
    });
  } else if (S.placing.length) {
    const pts = S.placing.map((p) => toScreen(p, L));
    ctx.strokeStyle = '#ffb400'; ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
    pts.forEach(([x, y], i) => cornerHandle(x, y, ['TL', 'TR', 'BR', 'BL'][i]));
  }
}

function drawGrid(L) {
  ctx.strokeStyle = 'rgba(255,180,0,.35)'; ctx.lineWidth = 1;
  for (let k = 1; k < 10; k++) {
    const t = k / 10;
    for (const seg of [[[t, 0], [t, 1]], [[0, t], [1, t]]]) {
      const [a, b] = seg.map((u) => toScreen(fromUnit(u), L));
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
  ctx.fillStyle = '#ffb400';
  ctx.beginPath(); ctx.arc(x, y, DOT_R, 0, Math.PI * 2); ctx.fill();
  ctx.lineWidth = selected ? 3 : 1.5; ctx.strokeStyle = selected ? '#fff' : '#000'; ctx.stroke();
  ctx.fillStyle = '#111'; ctx.font = '700 12px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
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
    if (S.placing.length === 4) { S.quad = S.placing; S.placing = []; setStatus('Outline set. Drag a corner if it is off, then place the features.'); }
    else setStatus(`Click the pod's ${CORNERS[S.placing.length]} corner (${S.placing.length + 1} of 4).`);
    draw();
    return;
  }
  // a dot first, then a corner, then "place the selected feature here"
  let best = null, bd = HANDLE;
  S.hot.forEach((h, i) => {
    if (h.x == null) return;
    const [sx, sy] = toScreen(fromUnit([h.x, h.y]), L);
    const d = Math.hypot(sx - x, sy - y);
    if (d < bd) { bd = d; best = { type: 'dot', i }; }
  });
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
    // the features keep their fractions, so they follow the corner as the outline changes
    S.quad[S.drag.i] = p;
  } else {
    setUnit(S.drag.i, toUnit(p));
  }
  draw();
});
const endDrag = () => { if (S.drag) { S.drag = null; save(); renderList(); } };
view.addEventListener('pointerup', endDrag);
view.addEventListener('pointercancel', endDrag);

function setUnit(i, [u, v]) {
  S.hot[i].x = Math.min(1, Math.max(0, u));
  S.hot[i].y = Math.min(1, Math.max(0, v));
}

function placeSelected(p) {
  const [u, v] = toUnit(p);
  setUnit(S.sel, [u, v]);
  const outside = u < -0.02 || u > 1.02 || v < -0.02 || v > 1.02;
  setStatus(outside ? 'That spot is outside the front frame, so it was moved to the nearest edge.' : `Placed "${S.hot[S.sel].title || 'feature ' + (S.sel + 1)}" at x ${S.hot[S.sel].x.toFixed(3)}, y ${S.hot[S.sel].y.toFixed(3)}.`);
  save(); renderList(); draw();
}

// ---------- the feature list ----------

function renderList() {
  const list = $('list');
  list.textContent = '';
  S.hot.forEach((h, i) => {
    const row = document.createElement('div');
    row.className = 'row' + (i === S.sel ? ' sel' : '');
    const n = document.createElement('span');
    n.className = 'n' + (h.x == null ? ' off' : '');
    n.textContent = String(i + 1);
    const t = document.createElement('span');
    t.className = 't';
    t.textContent = h.title || '(untitled)';
    const xy = document.createElement('span');
    xy.className = 'xy';
    xy.textContent = h.x == null ? 'not placed' : `${h.x.toFixed(2)}, ${h.y.toFixed(2)}`;
    row.append(n, t, xy);
    row.addEventListener('click', () => select(i));
    list.appendChild(row);
  });
  $('export').disabled = $('copy').disabled = $('publish').disabled = !S.hot.length;
  $('del').disabled = S.sel < 0;
  const f = S.hot[S.sel];
  for (const [id, key] of [['fTitle', 'title'], ['fText', 'text'], ['fUrl', 'url']]) {
    $(id).value = f ? f[key] : '';
    $(id).disabled = !f;
  }
  $('fEmbed').checked = f ? f.embed !== false : true;
  $('fEmbed').disabled = !f;
}

function select(i) {
  S.sel = i;
  renderList();
  draw();
  const h = S.hot[i];
  if (h && h.x == null) setStatus(`Click where "${h.title || 'this feature'}" is on the picture.`);
}

for (const [id, key] of [['fTitle', 'title'], ['fText', 'text'], ['fUrl', 'url']]) {
  $(id).addEventListener('input', () => {
    if (S.sel < 0) return;
    S.hot[S.sel][key] = $(id).value;
    save();
    if (key === 'title') {
      const t = $('list').children[S.sel]?.querySelector('.t');
      if (t) t.textContent = $(id).value || '(untitled)';
    }
  });
}

$('fEmbed').addEventListener('change', () => {
  if (S.sel < 0) return;
  S.hot[S.sel].embed = $('fEmbed').checked;
  save();
});

$('add').addEventListener('click', () => {
  S.hot.push({ title: 'New feature', text: '', url: '', embed: true, x: null, y: null });
  save();
  select(S.hot.length - 1);
  $('fTitle').focus();
  $('fTitle').select();
});
$('del').addEventListener('click', () => {
  if (S.sel < 0) return;
  S.hot.splice(S.sel, 1);
  S.sel = Math.min(S.sel, S.hot.length - 1);
  save(); renderList(); draw();
});
$('reload').addEventListener('click', async () => {
  await reloadFromFile();
  S.sel = -1;
  save(); renderList(); draw();
  setStatus('Loaded the features from data/pod.json.');
});
$('clearAll').addEventListener('click', () => {
  if (S.hot.length && !confirm('Remove all features from this list? (data/pod.json is not changed until you replace it.)')) return;
  S.hot = []; S.sel = -1;
  save(); renderList(); draw();
});

// ---------- export ----------

const slug = (t) => t.toLowerCase().normalize('NFKD').replace(/[^\w]+/g, '-').replace(/^-+|-+$/g, '');

function buildJson() {
  const used = new Set();
  const hotspots = S.hot.filter((h) => h.x != null).map((h, i) => {
    let id = slug(h.title) || `point-${i + 1}`, k = 2;
    while (used.has(id)) id = `${slug(h.title) || 'point'}-${k++}`;
    used.add(id);
    return { id, title: h.title, text: h.text, url: h.url, ...(h.embed === false ? { embed: false } : {}), x: +h.x.toFixed(3), y: +h.y.toFixed(3) };
  });
  return JSON.stringify({ name: S.meta.name, note: S.meta.note, hotspots }, null, 2) + '\n';
}
S.buildJson = buildJson;

function unplaced() { return S.hot.filter((h) => h.x == null).length; }

$('export').addEventListener('click', () => {
  const n = unplaced();
  if (n && !confirm(`${n} feature${n > 1 ? 's are' : ' is'} not placed yet and will be left out. Download anyway?`)) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([buildJson()], { type: 'application/json' }));
  a.download = 'pod.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  setStatus('Downloaded pod.json. Replace data/pod.json in the repo with it and push.');
});
$('copy').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(buildJson()); setStatus('Copied the JSON. Paste it over the contents of data/pod.json.'); }
  catch { setStatus('Could not copy; use Download pod.json instead.'); }
});

function setStatus(t) { $('status').textContent = t; }

init();


// ---------- publish to GitHub ----------

const pub = $('pub');

// on <owner>.github.io/<repo>/ the repository is the page's own; elsewhere fall back to the project's
function defaultRepo() {
  const m = location.hostname.match(/^([^.]+)\.github\.io$/);
  const repo = location.pathname.split('/')[1];
  return m && repo ? `${m[1]}/${repo}` : 'Sweep499/ARQR';
}
function readPubPrefs() {
  try { return JSON.parse(localStorage.getItem(PUB_KEY)) || {}; } catch { return {}; }
}
function readToken() {
  try { return localStorage.getItem(TOKEN_KEY) || sessionStorage.getItem(TOKEN_KEY) || ''; } catch { return ''; }
}
function pubStatus(text, kind = '') {
  const el = $('pubStatus');
  el.className = kind;
  el.textContent = text;
  return el;
}

$('publish').addEventListener('click', () => {
  const prefs = readPubPrefs();
  $('pubRepo').value = prefs.repo || defaultRepo();
  $('pubBranch').value = prefs.branch || 'main';
  $('pubPath').value = prefs.path || 'data/pod.json';
  $('pubMsg').value = prefs.msg || 'Update feature points (placement tool)';
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
  const placed = S.hot.filter((h) => h.x != null).length;
  if (!placed) { pubStatus('Place at least one feature on the picture first.', 'err'); return; }
  const token = $('pubToken').value.trim();
  const prefs = { repo: $('pubRepo').value.trim(), branch: $('pubBranch').value.trim() || 'main', path: $('pubPath').value.trim() || 'data/pod.json', msg: $('pubMsg').value.trim() || 'Update feature points (placement tool)' };
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
      const el = pubStatus(`Published ${placed} feature${placed > 1 ? 's' : ''}. Commit `, 'ok');
      const a = document.createElement('a');
      a.href = r.commitUrl || '#'; a.target = '_blank'; a.rel = 'noopener';
      a.textContent = (r.commitSha || '').slice(0, 7) || 'view';
      el.append(a, '. The live sites use the new points in about a minute.');
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
