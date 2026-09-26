"""Pre-label pod fronts in a video, so nobody has to click corners.

OWLv2 finds the black-framed glass panes by text, SAM 2.1 cuts out their union, and the mask outline is
written to <video>.proposals.json for review in tools/label.html ("Import proposals", then A = accept,
X = reject). Both models download from Hugging Face on first run and need no login.

    python prelabel.py ../../dataset/IMG_2822.MOV --step 0.5 --sheet

Per frame the JSON holds (all coordinates normalised to the frame, so any resolution works):
  poly_norm  outline of the SAM mask (convex hull, simplified); clipped by the picture edge when the
             front runs off-screen
  quad_norm  the outline reduced to 4 corners [TL, TR, BR, BL], only when it is not clipped
  clipped    the outline touches the picture edge (true corners are off-screen)
  box_fill   how well the mask fills the region the detector found (1 = fully)
  solidity   outline area / hull area (near 1 = one solid piece; low = leak or fragments)
  iou        overlap of quad_norm with poly_norm (1 = the 4-corner fit is the outline)
  auto_ok    detector confident, mask fills its box and, for a complete front, the quad fits the outline
"""
import argparse, json, os
import cv2, numpy as np, torch
from PIL import Image, ImageDraw
from transformers import Owlv2Processor, Owlv2ForObjectDetection, Sam2Processor, Sam2Model

QUERY = "a black framed glass door"
MIN_SCORE = 0.30      # OWLv2 confidence for a pane
SIBLING = 0.6         # also keep panes scoring at least this fraction of the best one
MODEL_W = 720         # frames are resized to this width for the models
OK_OWL, OK_IOU, OK_FILL, OK_SOLID, MIN_SOLID = 0.35, 0.95, 0.85, 0.90, 0.60
# auto_ok: OWL confidence, mask fills its box, mask is one solid piece, 4-corner fit matches the outline;
# proposals below MIN_SOLID are dropped as junk

dev = "mps" if torch.backends.mps.is_available() else "cpu"


def load_models():
    owl_p = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").eval()
    sam_p = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-small")
    sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-small").to(dev).eval()
    return owl_p, owl, sam_p, sam


def panes(m, im):
    owl_p, owl = m[0], m[1]
    inp = owl_p(text=[[QUERY]], images=im, return_tensors="pt")
    with torch.no_grad():
        out = owl(**inp)
    s = max(im.size)  # OWLv2 pads to a square
    r = owl_p.post_process_grounded_object_detection(out, threshold=MIN_SCORE, target_sizes=[(s, s)])[0]
    got = sorted(zip(r["scores"].tolist(), r["boxes"].tolist()), key=lambda t: -t[0])
    if not got:
        return []
    got = [(sc, b) for sc, b in got if sc >= SIBLING * got[0][0]]
    # a box that swallows two or more other panes is the whole pod body, not a pane: drop it
    def inside(a, b):  # is box a (mostly) inside box b
        w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
        return w > 0 and h > 0 and w * h >= 0.85 * (a[2] - a[0]) * (a[3] - a[1])
    return [(sc, b) for sc, b in got if sum(inside(o, b) for _, o in got if o is not b) < 2]


def add_rim(mask, gray, iters=None, dark=95):
    """SAM's mask stops at the inside of the black frame. Grow it outward over dark pixels only, so it
    reaches the frame's outer edge without spilling onto the white body or the floor."""
    if iters is None:
        # on a dark background the "dark frame" test would grow the mask into the background, so grow less
        ring = np.concatenate([gray[:10].ravel(), gray[-10:].ravel(), gray[:, :10].ravel(), gray[:, -10:].ravel()])
        iters = 22 if np.median(ring) > 60 else 6
    dark_px = (gray < dark).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cur = mask.copy()
    for _ in range(iters):
        cur = np.maximum(mask, cv2.dilate(cur, k) & (dark_px | mask))
    return cur


def segment(m, im, box):
    """-> (mask, box_fill) for the front inside `box`, or (None, 0).

    SAM proposes nested masks for a box. The right one is the largest that stays inside the (slightly
    expanded) box: it covers both panes, whereas a smaller one leaves part of the front out and its hull
    then cuts a diagonal. box_fill is how well the final mask's bounding box fills the box."""
    sam_p, sam = m[2], m[3]
    bw, bh = box[2] - box[0], box[3] - box[1]
    box = [max(0, box[0] - 0.03 * bw), max(0, box[1] - 0.02 * bh), min(im.width, box[2] + 0.03 * bw), min(im.height, box[3] + 0.02 * bh)]
    inp = sam_p(images=im, input_boxes=[[box]], return_tensors="pt").to(dev)
    with torch.no_grad():
        out = sam(**inp, multimask_output=True)
    masks = sam_p.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0][0]  # (3, H, W)
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    best = None
    for mk in masks:
        mk = (mk.numpy() > 0).astype(np.uint8)
        area = int(mk.sum())
        if area < 50 or mk[y0:y1, x0:x1].sum() / area < 0.97:
            continue
        if best is None or area > best[0]:
            best = (area, mk)
    if best is None:
        return None, 0.0
    mask = cv2.morphologyEx(best[1], cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))  # drop thin leaks
    gray = cv2.GaussianBlur(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY), (5, 5), 0)
    mask = add_rim(mask, gray)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None, 0.0
    ix = max(0, min(xs.max(), box[2]) - max(xs.min(), box[0])); iy = max(0, min(ys.max(), box[3]) - max(ys.min(), box[1]))
    inter = ix * iy
    union = (xs.max() - xs.min()) * (ys.max() - ys.min()) + (box[2] - box[0]) * (box[3] - box[1]) - inter
    return mask, float(inter / union) if union else 0.0


def reduce_to_quad(poly):
    """Reduce a convex polygon to 4 corners by repeatedly dropping the edge whose removal adds the least
    area, extending its two neighbours until they meet. Unlike picking 4 of the vertices, this keeps
    corners that the rounded frame corners and the mask's chamfers cut off."""
    v = [np.array(p, float) for p in poly]
    while len(v) > 4:
        n, best = len(v), None
        for i in range(n):
            a, b, c, d = v[(i - 1) % n], v[i], v[(i + 1) % n], v[(i + 2) % n]
            r, u = b - a, c - d                     # extend a->b forwards and d->c backwards
            den = r[0] * u[1] - r[1] * u[0]
            if abs(den) < 1e-9:
                continue
            w = d - a
            t, k = (w[0] * u[1] - w[1] * u[0]) / den, (w[0] * r[1] - w[1] * r[0]) / den
            if t < 1 or k < 1:                      # neighbours must diverge from the removed edge, not cross it
                continue
            pt = a + t * r
            cost = abs((b - pt)[0] * (c - pt)[1] - (b - pt)[1] * (c - pt)[0]) / 2
            if best is None or cost < best[0]:
                best = (cost, i, pt)
        if best is None:
            return None
        _, i, pt = best
        j = (i + 1) % n
        v[i] = pt
        del v[j]
    q = np.array(v)
    tl, br = q[np.argmin(q.sum(1))], q[np.argmax(q.sum(1))]
    dd = q[:, 1] - q[:, 0]
    return np.array([tl, q[np.argmin(dd)], br, q[np.argmax(dd)]])


def outline(mask):
    """-> (poly Nx2, quad 4x2 or None, solidity, clipped); solidity = outline area / hull area, low when
    the mask has a leak or is in pieces"""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None, None, 0.0, False
    c = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    poly = cv2.approxPolyDP(hull, 0.004 * peri, True)[:, 0, :].astype(float)
    h, w = mask.shape
    clipped = bool((poly[:, 0] <= 2).any() or (poly[:, 1] <= 2).any() or (poly[:, 0] >= w - 3).any() or (poly[:, 1] >= h - 3).any())
    quad = reduce_to_quad(poly) if len(poly) >= 4 and not clipped else None
    solidity = cv2.contourArea(c) / max(1.0, cv2.contourArea(hull))
    return poly, quad, float(solidity), clipped


def iou(poly_norm, quad_norm, n=400):
    """Overlap of two normalised polygons, by rasterising both."""
    a, b = np.zeros((n, n), np.uint8), np.zeros((n, n), np.uint8)
    cv2.fillPoly(a, [np.round(np.array(poly_norm) * n).astype(np.int32)], 1)
    cv2.fillPoly(b, [np.round(np.array(quad_norm) * n).astype(np.int32)], 1)
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def rescore(path):
    """Recompute quad, iou and auto_ok from the saved outlines (no model run)."""
    j = json.load(open(path))
    wh = np.array([j["video"]["width"], j["video"]["height"]], float)
    for r in j["frames"]:
        r["iou"] = None
        if r["poly_norm"] and not r["clipped"]:
            q = reduce_to_quad(np.array(r["poly_norm"]) * wh)
            r["quad_norm"] = None if q is None else (q / wh).round(5).tolist()
            if r["quad_norm"]:
                r["iou"] = round(iou(r["poly_norm"], r["quad_norm"]), 3)
        r["auto_ok"] = bool(r["poly_norm"] and r["owl"] >= OK_OWL and (r.get("box_fill") or 0) >= OK_FILL
                            and (r.get("solidity") or 0) >= OK_SOLID and (r["clipped"] or (r["iou"] or 0) >= OK_IOU))
    json.dump(j, open(path, "w"), indent=1)
    print(f"rescored {path}: {sum(r['auto_ok'] for r in j['frames'])} auto_ok of {len(j['frames'])} frames")


IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def frames(path, step):
    if path.lower().endswith(IMAGE_EXT):          # a still image counts as a one-frame video
        im = Image.open(path)
        if im.mode == "RGBA":                     # composite on white so transparent pixels are not black
            bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[3]); im = bg
        im = im.convert("RGB")
        yield 0.0, im.resize((MODEL_W, round(im.height * MODEL_W / im.width)), Image.LANCZOS), im.size
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 30
    t = 0.0
    while t < n / fps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, bgr = cap.read()
        if not ok:
            break
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        yield t, im.resize((MODEL_W, round(im.height * MODEL_W / im.width)), Image.LANCZOS), im.size
        t += step
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--rescore", metavar="JSON", help="recompute iou/auto_ok in an existing proposals file and exit")
    ap.add_argument("--step", type=float, default=0.5, help="seconds between sampled frames")
    ap.add_argument("--out", help="default: <video>.proposals.json next to the video")
    ap.add_argument("--sheet", action="store_true", help="also write contact sheets <out>.sheet_N.jpg")
    a = ap.parse_args()
    if a.rescore:
        return rescore(a.rescore)
    if not a.video:
        ap.error("give a video, or --rescore JSON")
    out = a.out or a.video + ".proposals.json"
    m = load_models()
    recs, tiles = [], []
    for t, im, (W, H) in frames(a.video, a.step):
        w, h = im.size
        rec = {"t": round(t, 3), "owl": None, "box_fill": None, "solidity": None, "iou": None, "clipped": None, "auto_ok": False, "quad_norm": None, "poly_norm": None}
        pn = panes(m, im)
        if pn:
            box = [max(0, min(b[0] for _, b in pn)), max(0, min(b[1] for _, b in pn)),
                   min(w, max(b[2] for _, b in pn)), min(h, max(b[3] for _, b in pn))]
            mask, fill = segment(m, im, box)
            poly, quad, solid, clipped = outline(mask) if mask is not None else (None, None, 0.0, False)
            rec.update(owl=round(pn[0][0], 3), clipped=clipped, box_fill=round(fill, 3), solidity=round(solid, 3))
            if poly is not None and solid >= MIN_SOLID:
                rec["poly_norm"] = (poly / [w, h]).round(5).tolist()
                if quad is not None and not clipped:
                    rec["quad_norm"] = (quad / [w, h]).round(5).tolist()
                    rec["iou"] = round(iou(rec["poly_norm"], rec["quad_norm"]), 3)
                # confident = a well-supported detection whose mask fills its box, and, when the front is
                # complete, whose 4-corner fit matches the outline
                rec["auto_ok"] = bool(pn[0][0] >= OK_OWL and fill >= OK_FILL and solid >= OK_SOLID and (clipped or rec["iou"] >= OK_IOU))
        recs.append(rec)
        print(f"t={t:6.2f} owl={rec['owl']} fill={rec['box_fill']} solid={rec['solidity']} iou={rec['iou']} clipped={rec['clipped']} ok={rec['auto_ok']}", flush=True)
        if a.sheet:
            d = ImageDraw.Draw(im)
            if rec["poly_norm"]:
                d.polygon([(x * w, y * h) for x, y in rec["poly_norm"]], outline="lime" if rec["auto_ok"] else "magenta", width=5)
            d.text((10, 10), f"t={t:.1f} ok={rec['auto_ok']}", fill="yellow")
            tiles.append(im.resize((240, 427)))
    json.dump({"video": {"name": os.path.basename(a.video), "width": W, "height": H}, "step": a.step,
               "query": QUERY, "frames": recs}, open(out, "w"), indent=1)
    print(f"wrote {out}: {sum(bool(r['poly_norm']) for r in recs)} proposals, {sum(r['auto_ok'] for r in recs)} auto_ok, of {len(recs)} frames")
    for k in range(0, len(tiles), 24):
        chunk = tiles[k:k + 24]
        sh = Image.new("RGB", (240 * 6, 427 * ((len(chunk) + 5) // 6)))
        for i, tl in enumerate(chunk):
            sh.paste(tl, (240 * (i % 6), 427 * (i // 6)))
        sh.save(f"{out}.sheet_{k // 24}.jpg")


if __name__ == "__main__":
    main()
