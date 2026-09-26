import http from 'http'; import fs from 'fs'; import path from 'path';
const root = new URL('../..', import.meta.url).pathname;   // the repo root
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.onnx': 'application/octet-stream', '.MOV': 'video/quicktime', '.mp4': 'video/mp4' };
http.createServer((req, res) => {
  const f = path.join(root, decodeURIComponent(req.url.split('?')[0]).replace(/\/$/, '/index.html'));
  if (!f.startsWith(root) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); return res.end(); }
  const size = fs.statSync(f).size, type = types[path.extname(f)] || 'application/octet-stream', range = req.headers.range;
  if (range) {
    const [a, b] = range.replace('bytes=', '').split('-'); const s = +a, e = b ? +b : size - 1;
    res.writeHead(206, { 'Content-Range': `bytes ${s}-${e}/${size}`, 'Accept-Ranges': 'bytes', 'Content-Length': e - s + 1, 'Content-Type': type });
    fs.createReadStream(f, { start: s, end: e }).pipe(res);
  } else { res.writeHead(200, { 'Content-Length': size, 'Content-Type': type, 'Accept-Ranges': 'bytes' }); fs.createReadStream(f).pipe(res); }
}).listen(8766, () => console.log('listening'));
