// Tracking benchmark. Starts the outline at a hand-reviewed frame of a clip, steps through the clip and, at every
// later hand-reviewed frame, measures how well the outline overlaps the labelled pod front (intersection over union,
// counted inside the picture). Run it on the same clip with different URL switches to compare ways of tracking:
//
//   npm i puppeteer-core          # in this folder; Chrome must be installed
//   node serve.mjs &              # a static server with Range support for the repo root, on port 8766
//   node track_bench.mjs base     # the shipped behaviour;   noalign = without the mask alignment
//   node track_bench.mjs ring     # ?ring, ?rect and ?noalign are the switches in js/app.js
//
// It needs dataset/IMG_2823.MOV and dataset/labels/IMG_2823/ (git-ignored), and, to be fair to the detector,
// a detector that never saw that clip, trained with:  train.py --train IMG_2822 RENDER_360 RENDER_IMG NEG_REAL
// --out dev/noB.onnx  (the URL below points at it).
import puppeteer from 'puppeteer-core';
import fs from 'fs';
const H = process.env.HOME;
const mode = process.argv[2] || 'base';            // new | legacy | any query string like noring or norect&f=0.9
const T0 = +(process.argv[3] || 5.5), T1 = +(process.argv[4] || 24), STEP = +(process.argv[5] || 0.1);
const labels = JSON.parse(fs.readFileSync(`${H}/ARQR/dataset/labels/IMG_2823/IMG_2823.labels.json`, 'utf8')).frames;
const truth = {}; for (const f of labels) truth[f.t.toFixed(1)] = f.poly_px;
const start = labels.find((f) => Math.abs(f.t - T0) < 1e-6 && f.quad_px);
if (!start) throw new Error('no label quad at t=' + T0);
const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new' });
const page = await browser.newPage();
await page.setViewport({ width: 450, height: 800 });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`http://localhost:8766/?debug&nodrop&model=/dev/noB.onnx&${mode === 'base' ? '' : mode + '&'}src=dataset/IMG_2823.MOV`);
await page.click('#startBtn');
await page.waitForFunction(() => window.podDebug && podDebug.tracker && podDebug.front !== undefined);
await page.waitForFunction(() => !/Getting ready|Tap the pod's top-left/.test(document.getElementById('hint').textContent), { timeout: 60000 });
// no detector: measure the tracker alone
await page.evaluate(() => { const v = document.getElementById('video'); v.pause(); });   // the real detector runs
const iouJS = `(q, gt, W, Hh) => { const c = document.createElement('canvas'); c.width = 216; c.height = 384; const s = 216 / W; const draw = (poly) => { const x = c.getContext('2d'); x.clearRect(0,0,216,384); x.fillStyle = '#fff'; x.beginPath(); poly.forEach(([a,b],i) => i ? x.lineTo(a*s,b*s) : x.moveTo(a*s,b*s)); x.closePath(); x.fill(); return x.getImageData(0,0,216,384).data; }; const A = draw(q), B = draw(gt); let i = 0, u = 0; for (let k = 3; k < A.length; k += 4) { const a = A[k] > 127, b = B[k] > 127; if (a && b) i++; if (a || b) u++; } return u ? i / u : 0; }`;
await page.evaluate((t, q) => new Promise((res) => { const v = document.getElementById('video'); v.onseeked = () => res(); v.currentTime = t; }).then(() => { podDebug.quad = q; }), T0, start.quad_px);
await new Promise((r) => setTimeout(r, 600));
const rows = [];
for (let t = T0 + STEP; t <= T1 + 1e-6; t += STEP) {
  await page.evaluate((t) => new Promise((res) => { const v = document.getElementById('video'); v.onseeked = () => res(); v.currentTime = t; }), t);
  await new Promise((r) => setTimeout(r, 160));
  const key = t.toFixed(1);
  if (truth[key] && Math.abs(t * 2 - Math.round(t * 2)) < 1e-6) {
    const r = await page.evaluate(`(async () => { const q = podDebug.quad, vw = document.getElementById('video').videoWidth, vh = document.getElementById('video').videoHeight; const f = ${iouJS}; return { iou: f(q, ${JSON.stringify(truth[key])}, vw, vh), lost: !!podDebug.lost, pts: podDebug.tracker.info.points, mode: podDebug.tracker.info.mode }; })()`);
    rows.push({ t: +key, ...r });
  }
}
fs.writeFileSync(`bench_${mode.replace(/[^a-z0-9.]/gi, '_')}.json`, JSON.stringify(rows));
const mean = (a) => a.length ? a.reduce((s, x) => s + x, 0) / a.length : NaN;
const seg = (lo, hi) => rows.filter((r) => r.t > lo && r.t <= hi).map((r) => r.iou);
console.log(mode.padEnd(7), `n=${rows.length}`, '| mean IoU  0-1.5s:', mean(seg(T0, T0 + 1.5)).toFixed(3), ' 1.5-4s:', mean(seg(T0 + 1.5, T0 + 4)).toFixed(3), ' 4-8s:', mean(seg(T0 + 4, T0 + 8)).toFixed(3), ' 8s+:', mean(seg(T0 + 8, 99)).toFixed(3), '| frames IoU>0.7:', rows.filter((r) => r.iou > 0.7).length + '/' + rows.length, '| lost frames:', rows.filter((r) => r.lost).length);
await browser.close();
