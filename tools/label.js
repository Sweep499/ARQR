// Labelling tool for the pod-front detector: click the 4 corners of the pod's front frame on a few
// keyframes, propagate them to the frames in between with the app's own optical-flow tracker (forward
// from the previous key, backward from the next key, blended), then export quads, images and masks.
//
// Quads are [TL, TR, BR, BL] in video pixels and may lie outside the picture.

import { applyScaledH, isSaneQuad } from '../js/geometry.js';
import { loadOpenCV, FlowTracker } from '../js/tracker.js';

const $ = (id) => document.getElementById(id);
const video = $('video');
const view = $('view');
const ctx = view.getContext('2d');
const timeline = $('timeline');
const tctx = timeline.getContext('2d');

const MARGIN = 0.25;        // dark border around the picture, as a fraction of its size, for off-screen corners
const MAX_OUT = 1.0;        // propagation stops when a corner strays more than this many picture sizes outside the picture
const MAX_ONE_SIDED = 10;   // frames tracked from a single key (no second key to check against) before giving up
const HANDLE = 14;          // css px within which a click grabs a corner
const CORNERS = ['top-left', 'top-right', 'bottom-right', 'bottom-left'];
const COLORS = { key: '#ffb400', ok: '#3ecf8e', warn: '#ff6b4a', unv: '#5ea1ff', none: '#777' };

// labels[i] is null (unlabelled) or { kind: 'key' | 'auto' | 'none', quad, err }
//   key  = placed by hand; none = hand-marked "no pod front"; auto = propagated
//   err  = max corner disagreement (video px) between forward and backward tracking, null if only one side
const S = {
  step: 0.2, n: 0, cur: 0, labels: [], placing: [], drag: -1,
  busy: false, abort: false, name: '', size: 0, tracker: null, note: '',
};
window.podLabel = S; // handy for debugging in the console

// ---------- loading, persistence ----------

$('file').addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (f) loadVideo(URL.createObjectURL(f), f.name, f.size);
});
const srcParam = new URLSearchParams(location.search).get('src'); // ?src=../dataset/IMG_2822.MOV
if (srcParam) loadVideo(srcParam, srcParam.split('/').pop(), 0);

async function loadVideo(url, name, size) {
  video.src = url;
  await new Promise((res, rej) => {
    video.onloadeddata = res;
    video.onerror = () => rej(new Error('could not decode this video'));
  }).catch((e) => setStatus(e.message));
  if (!video.videoWidth) return;
  S.name = name; S.size = size;
  const saved = loadSaved();
  if (saved) $('step').value = S.step = saved.step;
  resetLabels();
  if (saved) applyKeys(saved.keys);
  $('empty').hidden = true;
  for (const id of ['none', 'clear', 'propagate', 'exportJson', 'exportDir']) $(id).disabled = false;
  await gotoFrame(0);
}

function storeKey() { return `podlabel:${S.name}:${S.size}`; }
function loadSaved() {
  try { return JSON.parse(localStorage.getItem(storeKey())); } catch { return null; }
}
function save() {
  try { localStorage.setItem(storeKey(), JSON.stringify({ step: S.step, keys: keyList() })); } catch { /* private mode etc. */ }
}
function keyList() {
  const out = [];
  S.labels.forEach((l, i) => { if (l && l.kind !== 'auto') out.push({ t: timeOf(i), quad: l.kind === 'none' ? null : l.quad }); });
  return out;
}
function applyKeys(keys) {
  for (const k of keys) {
    const i = Math.min(S.n - 1, Math.max(0, Math.round(k.t / S.step)));
    S.labels[i] = k.quad ? { kind: 'key', quad: k.quad, err: null } : { kind: 'none', quad: null, err: null };
  }
}
function resetLabels() {
  S.n = Math.floor(video.duration / S.step) + 1;
  S.labels = new Array(S.n).fill(null);
  S.cur = 0; S.placing = [];
}

$('step').addEventListener('change', async () => {
  const v = parseFloat($('step').value);
  if (!(v >= 0.05) || S.busy || !S.n) { $('step').value = S.step; return; }
  const keys = keyList(); // re-snap the hand labels to the new grid, drop propagated frames
  const t = timeOf(S.cur);
  S.step = v;
  resetLabels();
  applyKeys(keys);
  save();
  await gotoFrame(Math.min(S.n - 1, Math.round(t / S.step)));
});

$('importFile').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f || !S.n) return;
  try {
    const j = JSON.parse(await f.text());
    const keys = j.keys || (j.frames || []).filter((fr) => fr.kind !== 'auto').map((fr) => ({ t: fr.t, quad: fr.quad_px }));
    applyKeys(keys);
    save();
    draw(); drawTimeline();
  } catch (err) { setStatus('Could not read that labels file: ' + err.message); }
});

// ---------- frame navigation ----------

function timeOf(i) { return Math.min(i * S.step, Math.max(0, video.duration - 0.02)); }

function seek(t) {
  return new Promise((res) => {
    if (Math.abs(video.currentTime - t) < 1e-4) return res();
    const done = () => { video.removeEventListener('seeked', done); res(); };
    video.addEventListener('seeked', done);
    video.currentTime = t;
  });
}

async function gotoFrame(i) {
  S.note = '';
  S.cur = Math.max(0, Math.min(S.n - 1, i));
  S.placing = [];
  S.drag = -1;
  await seek(timeOf(S.cur));
  draw(); drawTimeline();
}

// ---------- geometry: canvas <-> video pixels ----------

function layout() {
  const cw = view.clientWidth, ch = view.clientHeight;
  const vw = video.videoWidth || 1, vh = video.videoHeight || 1;
  const s = Math.min(cw / (vw * (1 + 2 * MARGIN)), ch / (vh * (1 + 2 * MARGIN)));
  return { s, ox: (cw - vw * s) / 2, oy: (ch - vh * s) / 2, cw, ch, vw, vh };
}
const toScreen = ([x, y], L = layout()) => [x * L.s + L.ox, y * L.s + L.oy];
const toVideo = (x, y, L = layout()) => [(x - L.ox) / L.s, (y - L.oy) / L.s];

// ---------- drawing ----------

function labelColor(l, warnPx) {
  if (l.kind === 'key') return COLORS.key;
  if (l.kind === 'none') return COLORS.none;
  if (l.err == null) return COLORS.unv;
  return l.err > warnPx ? COLORS.warn : COLORS.ok;
}
const warnPx = () => (parseFloat($('warn').value) || 1.5) / 100 * (video.videoHeight || 1);

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
  if (!video.videoWidth) return;
  ctx.drawImage(video, L.ox, L.oy, L.vw * L.s, L.vh * L.s);
  ctx.strokeStyle = 'rgba(255,255,255,.25)';
  ctx.lineWidth = 1;
  ctx.strokeRect(L.ox, L.oy, L.vw * L.s, L.vh * L.s);

  const l = S.labels[S.cur];
  if (l && l.quad) drawQuad(l.quad.map((p) => toScreen(p, L)), labelColor(l, warnPx()), l.kind === 'auto');
  if (l && l.kind === 'none') {
    ctx.fillStyle = 'rgba(0,0,0,.55)';
    ctx.fillRect(L.ox, L.oy, L.vw * L.s, L.vh * L.s);
    ctx.fillStyle = '#ddd';
    ctx.font = '600 20px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('no pod front in this frame', L.ox + L.vw * L.s / 2, L.oy + L.vh * L.s / 2);
  }
  if (S.placing.length) {
    const pts = S.placing.map((p) => toScreen(p, L));
    ctx.strokeStyle = COLORS.key; ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
    pts.forEach(([x, y], i) => handle(x, y, i, COLORS.key));
  }
  updateStatus();
}

function drawQuad(sq, color, dashed) {
  ctx.lineWidth = 2.5; ctx.lineJoin = 'round'; ctx.strokeStyle = color;
  ctx.setLineDash(dashed ? [8, 5] : []);
  ctx.beginPath();
  sq.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath(); ctx.stroke();
  ctx.setLineDash([]);
  sq.forEach(([x, y], i) => handle(x, y, i, color));
}

function handle(x, y, i, color) {
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#000'; ctx.lineWidth = 1; ctx.stroke();
  ctx.fillStyle = '#fff';
  ctx.font = '600 12px system-ui';
  ctx.textAlign = 'left';
  ctx.fillText(['TL', 'TR', 'BR', 'BL'][i], x + 9, y - 8);
}

function drawTimeline() {
  const dpr = window.devicePixelRatio || 1;
  const w = timeline.clientWidth, h = timeline.clientHeight;
  if (timeline.width !== Math.round(w * dpr) || timeline.height !== Math.round(h * dpr)) {
    timeline.width = Math.round(w * dpr);
    timeline.height = Math.round(h * dpr);
  }
  tctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  tctx.fillStyle = '#101216';
  tctx.fillRect(0, 0, w, h);
  if (!S.n) return;
  const cw = w / S.n, wp = warnPx();
  S.labels.forEach((l, i) => {
    if (!l) return;
    tctx.fillStyle = labelColor(l, wp);
    tctx.fillRect(i * cw, l.kind === 'auto' ? 8 : 2, Math.max(1, cw - (cw > 3 ? 1 : 0)), l.kind === 'auto' ? h - 16 : h - 4);
  });
  tctx.fillStyle = '#fff';
  tctx.fillRect(S.cur * cw + cw / 2 - 1, 0, 2, h);
}

function updateStatus() {
  const l = S.labels[S.cur];
  let what = 'unlabelled';
  if (S.placing.length) what = `placing: click the ${CORNERS[S.placing.length]} corner (${S.placing.length + 1} of 4)`;
  else if (l?.kind === 'key') what = 'key';
  else if (l?.kind === 'none') what = 'no pod front';
  else if (l?.kind === 'auto') what = l.err == null ? 'propagated, unverified' : `propagated, disagreement ${l.err.toFixed(0)} px`;
  else what = 'unlabelled: click the top-left corner';
  const keys = S.labels.filter((x) => x && x.kind !== 'auto').length;
  const autos = S.labels.filter((x) => x?.kind === 'auto').length;
  setStatus(`frame ${S.cur + 1}/${S.n} · t=${timeOf(S.cur).toFixed(2)}s · ${what} · ${keys} keys, ${autos} propagated` + (S.note ? ` · ${S.note}` : ''));
}
function setStatus(t) { $('status').textContent = t; }

new ResizeObserver(() => { draw(); drawTimeline(); }).observe($('stage'));
new ResizeObserver(() => drawTimeline()).observe(timeline);
$('warn').addEventListener('change', drawTimeline);

// ---------- placing and dragging corners ----------

view.addEventListener('pointerdown', (e) => {
  if (S.busy || !S.n) return;
  const r = view.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  const L = layout();
  const l = S.labels[S.cur];

  if (l?.quad && !S.placing.length) {
    let best = -1, bd = HANDLE;
    l.quad.forEach((p, i) => {
      const [sx, sy] = toScreen(p, L);
      const d = Math.hypot(sx - x, sy - y);
      if (d < bd) { bd = d; best = i; }
    });
    if (best >= 0) {
      S.drag = best;
      view.setPointerCapture(e.pointerId);
      // editing a propagated frame turns it into a key
      if (l.kind === 'auto') { l.kind = 'key'; l.err = null; }
    }
    return;
  }
  if (l?.kind === 'none') return;
  S.placing.push(toVideo(x, y, L));
  if (S.placing.length === 4) {
    S.labels[S.cur] = { kind: 'key', quad: S.placing, err: null };
    S.placing = [];
    save(); drawTimeline();
  }
  draw();
});
view.addEventListener('pointermove', (e) => {
  if (S.drag < 0) return;
  const r = view.getBoundingClientRect();
  S.labels[S.cur].quad[S.drag] = toVideo(e.clientX - r.left, e.clientY - r.top);
  draw();
});
const endDrag = () => { if (S.drag >= 0) { S.drag = -1; save(); drawTimeline(); } };
view.addEventListener('pointerup', endDrag);
view.addEventListener('pointercancel', endDrag);

$('none').addEventListener('click', markNone);
$('clear').addEventListener('click', clearFrame);
function markNone() {
  if (S.busy) return;
  S.labels[S.cur] = S.labels[S.cur]?.kind === 'none' ? null : { kind: 'none', quad: null, err: null };
  S.placing = [];
  save(); draw(); drawTimeline();
}
function clearFrame() {
  if (S.busy) return;
  S.labels[S.cur] = null;
  S.placing = [];
  save(); draw(); drawTimeline();
}

timeline.addEventListener('pointerdown', (e) => {
  if (S.busy || !S.n) return;
  const r = timeline.getBoundingClientRect();
  const i = Math.floor(((e.clientX - r.left) / r.width) * S.n);
  gotoFrame(i);
});

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || S.busy || !S.n) return;
  const k = e.key;
  if (k === 'ArrowRight') gotoFrame(S.cur + (e.shiftKey ? 10 : 1));
  else if (k === 'ArrowLeft') gotoFrame(S.cur - (e.shiftKey ? 10 : 1));
  else if (k === ']' || k === '[') {
    const dir = k === ']' ? 1 : -1;
    for (let i = S.cur + dir; i >= 0 && i < S.n; i += dir) {
      if (S.labels[i] && S.labels[i].kind !== 'auto') { gotoFrame(i); break; }
    }
  } else if (k === 'n' || k === 'N') markNone();
  else if (k === 'Delete') clearFrame();
  else if (k === 'p' || k === 'P') propagate();
  else if (k === 'Escape') { S.placing = []; draw(); }
  else if (k === 'Backspace') { S.placing.pop(); draw(); }
  else return;
  e.preventDefault();
});

// ---------- propagation ----------

$('propagate').addEventListener('click', propagate);
$('stop').addEventListener('click', () => { S.abort = true; });

function setBusy(b) {
  S.busy = b;
  S.abort = false;
  for (const id of ['propagate', 'exportJson', 'exportDir', 'none', 'clear', 'file', 'importFile', 'step']) $(id).disabled = b;
  $('stop').disabled = !b;
}

// Tracks `quad` from frame `from` towards frame `to` (inclusive) one frame at a time. Returns a Map of
// frame -> quad and stops early where tracking is lost, the quad becomes implausible, or maxFrames is reached.
async function trackRun(from, to, quad, maxFrames = Infinity) {
  const dir = to > from ? 1 : -1;
  to = from + dir * Math.min(Math.abs(to - from), maxFrames);
  const out = new Map();
  await seek(timeOf(from));
  S.tracker.reset();
  S.tracker.step(video); // seeds the feature points
  let q = quad;
  for (let i = from + dir; dir > 0 ? i <= to : i >= to; i += dir) {
    if (S.abort) throw new Error('stopped');
    await seek(timeOf(i));
    const H = S.tracker.step(video);
    if (!H) break;
    const next = q.map((p) => applyScaledH(H, p, S.tracker.scale));
    const vw = video.videoWidth, vh = video.videoHeight;
    const plausible = next.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y)
      && x > -MAX_OUT * vw && x < (1 + MAX_OUT) * vw && y > -MAX_OUT * vh && y < (1 + MAX_OUT) * vh);
    if (!plausible || !isSaneQuad(next)) break;
    q = next;
    out.set(i, q);
    if ((i & 3) === 0) setStatus(`propagating… frame ${i + 1}/${S.n}`);
  }
  return out;
}

async function propagate() {
  if (S.busy || !S.n) return;
  const keys = [];
  S.labels.forEach((l, i) => { if (l && l.kind !== 'auto') keys.push(i); });
  if (!keys.some((i) => S.labels[i].quad)) { setStatus('Place at least one key first.'); return; }
  setBusy(true);
  const restore = S.cur;
  try {
    setStatus('loading OpenCV…');
    if (!S.tracker) S.tracker = new FlowTracker((await loadOpenCV()).cv);
    S.labels = S.labels.map((l) => (l && l.kind === 'auto' ? null : l));
    const put = (i, quad, err) => { S.labels[i] = { kind: 'auto', quad, err }; };
    const quadAt = (i) => S.labels[i].quad;

    // head: backward from the first key; tail: forward from the last key (one-sided, so unverified)
    const first = keys[0], last = keys[keys.length - 1];
    if (quadAt(first) && first > 0) for (const [i, q] of await trackRun(first, 0, quadAt(first), MAX_ONE_SIDED)) put(i, q, null);
    if (quadAt(last) && last < S.n - 1) for (const [i, q] of await trackRun(last, S.n - 1, quadAt(last), MAX_ONE_SIDED)) put(i, q, null);

    // between consecutive keys: forward from a, backward from b, blend by distance
    for (let k = 0; k + 1 < keys.length; k++) {
      const a = keys[k], b = keys[k + 1];
      if (b - a < 2) continue;
      const fwd = S.labels[a].quad ? await trackRun(a, b - 1, quadAt(a)) : new Map();
      const bwd = S.labels[b].quad ? await trackRun(b, a + 1, quadAt(b)) : new Map();
      for (let i = a + 1; i < b; i++) {
        const f = fwd.get(i), g = bwd.get(i);
        if (f && g) {
          const w = (i - a) / (b - a);
          const q = f.map((p, c) => [p[0] * (1 - w) + g[c][0] * w, p[1] * (1 - w) + g[c][1] * w]);
          put(i, q, Math.max(...f.map((p, c) => Math.hypot(p[0] - g[c][0], p[1] - g[c][1]))));
        } else if (f || g) put(i, f || g, null);
      }
    }
    S.tracker.reset();
    const autos = S.labels.filter((l) => l?.kind === 'auto');
    const bad = autos.filter((l) => l.err != null && l.err > warnPx()).length;
    setBusy(false);
    await gotoFrame(restore);
    S.note = (`propagated ${autos.length} frames (${bad} disagree, ${autos.filter((l) => l.err == null).length} unverified). Fix red spans by adding keys, then propagate again.`);
    draw();
  } catch (e) {
    S.tracker?.reset();
    setBusy(false);
    await gotoFrame(restore);
    setStatus(e.message === 'stopped' ? 'Stopped.' : 'Propagation failed: ' + e.message);
    if (e.message !== 'stopped') console.error(e);
  }
}

// ---------- export ----------

// Sutherland–Hodgman clip of a polygon to the picture rectangle.
function clipToFrame(poly, w, h) {
  const planes = [[1, 0, 0], [-1, 0, w], [0, 1, 0], [0, -1, h]]; // inside when nx*x + ny*y + c >= 0
  let out = poly;
  for (const [nx, ny, c] of planes) {
    const inp = out; out = [];
    inp.forEach((p, i) => {
      const q = inp[(i + 1) % inp.length];
      const dp = nx * p[0] + ny * p[1] + c, dq = nx * q[0] + ny * q[1] + c;
      if (dp >= 0) out.push(p);
      if ((dp >= 0) !== (dq >= 0)) { const t = dp / (dp - dq); out.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]); }
    });
    if (!out.length) break;
  }
  return out;
}
function area(poly) {
  let a = 0;
  poly.forEach((p, i) => { const q = poly[(i + 1) % poly.length]; a += p[0] * q[1] - q[0] * p[1]; });
  return Math.abs(a) / 2;
}

function exportRecords(includeUnverified) {
  const vw = video.videoWidth, vh = video.videoHeight, wp = warnPx();
  const recs = [];
  S.labels.forEach((l, i) => {
    if (!l) return;
    if (l.kind === 'auto' && (l.err == null ? !includeUnverified : l.err > wp)) return;
    const rec = { i, t: +timeOf(i).toFixed(4), kind: l.kind, err_px: l.err == null ? null : +l.err.toFixed(1) };
    if (l.quad) {
      const inside = l.quad.filter(([x, y]) => x >= 0 && y >= 0 && x <= vw && y <= vh).length;
      const clipped = clipToFrame(l.quad, vw, vh);
      Object.assign(rec, {
        quad_px: l.quad.map((p) => p.map((v) => +v.toFixed(1))),
        quad_norm: l.quad.map(([x, y]) => [+(x / vw).toFixed(5), +(y / vh).toFixed(5)]),
        corners_inside: inside,
        visible_frac: +(clipped.length ? area(clipped) / area(l.quad) : 0).toFixed(3),
      });
    } else rec.quad_px = null;
    recs.push(rec);
  });
  return recs;
}

function payload(recs) {
  return {
    version: 1,
    video: { name: S.name, width: video.videoWidth, height: video.videoHeight, duration: video.duration },
    step: S.step,
    corner_order: 'TL,TR,BR,BL of the pod front frame; video pixel coordinates, may lie outside the picture',
    keys: keyList(),
    frames: recs,
  };
}

$('exportJson').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(payload(exportRecords($('incUnv').checked)), null, 1)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = S.name.replace(/\.[^.]+$/, '') + '.labels.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

const toBlob = (canvas, type, q) => new Promise((res) => canvas.toBlob(res, type, q));
async function writeFile(dir, name, data) {
  const fh = await dir.getFileHandle(name, { create: true });
  const w = await fh.createWritable();
  await w.write(data);
  await w.close();
}

// Writes images/<stem>_<frame>.jpg, masks/<stem>_<frame>.png (white = pod front) and labels.json.
$('exportDir').addEventListener('click', async () => {
  if (!window.showDirectoryPicker) { setStatus('Folder export needs Chrome, Edge or Brave. Use "Export JSON" here instead.'); return; }
  let dir;
  try { dir = await window.showDirectoryPicker({ mode: 'readwrite' }); } catch { return; }
  const recs = exportRecords($('incUnv').checked);
  if (!recs.length) { setStatus('Nothing to export yet.'); return; }
  const restore = S.cur;
  setBusy(true);
  try {
    const imgDir = await dir.getDirectoryHandle('images', { create: true });
    const maskDir = await dir.getDirectoryHandle('masks', { create: true });
    const stem = S.name.replace(/\.[^.]+$/, '');
    const W = 640, H = Math.round(W * video.videoHeight / video.videoWidth);
    const c = document.createElement('canvas');
    c.width = W; c.height = H;
    const x = c.getContext('2d');
    const k = W / video.videoWidth;
    for (let n = 0; n < recs.length; n++) {
      if (S.abort) throw new Error('stopped');
      const r = recs[n];
      const name = `${stem}_${String(r.i).padStart(5, '0')}`;
      setStatus(`exporting ${n + 1}/${recs.length}…`);
      await seek(timeOf(r.i));
      x.drawImage(video, 0, 0, W, H);
      await writeFile(imgDir, name + '.jpg', await toBlob(c, 'image/jpeg', 0.92));
      x.fillStyle = '#000';
      x.fillRect(0, 0, W, H);
      if (r.quad_px) {
        x.fillStyle = '#fff';
        x.beginPath();
        r.quad_px.forEach(([px, py], j) => (j ? x.lineTo(px * k, py * k) : x.moveTo(px * k, py * k)));
        x.closePath(); x.fill();
      }
      await writeFile(maskDir, name + '.png', await toBlob(c, 'image/png'));
      r.image = `images/${name}.jpg`; r.mask = `masks/${name}.png`;
    }
    await writeFile(dir, `${stem}.labels.json`, JSON.stringify({ ...payload(recs), export_size: [W, H] }, null, 1));
    setBusy(false);
    await gotoFrame(restore);
    setStatus(`Exported ${recs.length} frames to the chosen folder.`);
  } catch (e) {
    setBusy(false);
    await gotoFrame(restore);
    setStatus(e.message === 'stopped' ? 'Export stopped.' : 'Export failed: ' + e.message);
  }
});
