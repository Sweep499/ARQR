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
  iou        overlap of quad_norm with poly_norm (1 = the 4-corner fit is the outline)
  auto_ok    unclipped, quad fits the outline well and OWL is confident: safe to accept without a look
"""
import argparse, json, os
import cv2, numpy as np, torch
from PIL import Image, ImageDraw
from transformers import Owlv2Processor, Owlv2ForObjectDetection, Sam2Processor, Sam2Model

QUERY = "a black framed glass door"
MIN_SCORE = 0.30      # OWLv2 confidence for a pane
SIBLING = 0.6         # also keep panes scoring at least this fraction of the best one
MODEL_W = 720         # frames are resized to this width for the models
OK_OWL, OK_IOU = 0.35, 0.95   # auto_ok: OWL confidence and overlap of the 4-corner fit with the mask outline

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


def segment(m, im, box):
    """SAM proposes three nested masks for a box; take the one whose bounding box best fills the box, so
    a pair of panes gives the whole front (frame included) rather than one pane."""
    sam_p, sam = m[2], m[3]
    bw, bh = box[2] - box[0], box[3] - box[1]
    box = [max(0, box[0] - 0.03 * bw), max(0, box[1] - 0.02 * bh), min(im.width, box[2] + 0.03 * bw), min(im.height, box[3] + 0.02 * bh)]
    inp = sam_p(images=im, input_boxes=[[box]], return_tensors="pt").to(dev)
    with torch.no_grad():
        out = sam(**inp, multimask_output=True)
    masks = sam_p.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0][0]  # (3, H, W)
    best, best_s = None, -1.0
    for mk, pred in zip(masks, out.iou_scores.cpu().reshape(-1).tolist()):
        mk = (mk.numpy() > 0).astype(np.uint8)
        ys, xs = np.nonzero(mk)
        if not len(xs):
            continue
        ix = max(0, min(xs.max(), box[2]) - max(xs.min(), box[0])); iy = max(0, min(ys.max(), box[3]) - max(ys.min(), box[1]))
        inter = ix * iy
        union = (xs.max() - xs.min()) * (ys.max() - ys.min()) + (box[2] - box[0]) * (box[3] - box[1]) - inter
        score = inter / union + 0.05 * pred
        if score > best_s:
            best, best_s = mk, score
    if best is None:
        return np.zeros((im.height, im.width), np.uint8)
    return cv2.morphologyEx(best, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))  # drop thin leaks


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
    """-> (poly Nx2, quad 4x2 or None, fill, clipped)"""
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
    fill = 0.0
    return poly, quad, float(fill), clipped


def iou(poly_norm, quad_norm, n=400):
    """Overlap of two normalised polygons, by rasterising both."""
    a, b = np.zeros((n, n), np.uint8), np.zeros((n, n), np.uint8)
    cv2.fillPoly(a, [np.round(np.array(poly_norm) * n).astype(np.int32)], 1)
    cv2.fillPoly(b, [np.round(np.array(quad_norm) * n).astype(np.int32)], 1)
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def rescore(path):
    j = json.load(open(path))
    wh = np.array([j["video"]["width"], j["video"]["height"]], float)
    for r in j["frames"]:
        r["iou"] = r["auto_ok"] = None
        r.pop("fill", None)
        if r["poly_norm"] and not r["clipped"]:
            q = reduce_to_quad(np.array(r["poly_norm"]) * wh)
            r["quad_norm"] = None if q is None else (q / wh).round(5).tolist()
        if r["quad_norm"] and r["poly_norm"]:
            r["iou"] = round(iou(r["poly_norm"], r["quad_norm"]), 3)
            r["auto_ok"] = bool(r["owl"] >= OK_OWL and r["iou"] >= OK_IOU)
        else:
            r["auto_ok"] = False
    json.dump(j, open(path, "w"), indent=1)
    print(f"rescored {path}: {sum(r['auto_ok'] for r in j['frames'])} auto_ok of {len(j['frames'])} frames")


def frames(path, step):
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
        rec = {"t": round(t, 3), "owl": None, "iou": None, "clipped": None, "auto_ok": False, "quad_norm": None, "poly_norm": None}
        pn = panes(m, im)
        if pn:
            box = [max(0, min(b[0] for _, b in pn)), max(0, min(b[1] for _, b in pn)),
                   min(w, max(b[2] for _, b in pn)), min(h, max(b[3] for _, b in pn))]
            poly, quad, _, clipped = outline(segment(m, im, box))
            rec.update(owl=round(pn[0][0], 3), clipped=clipped)
            if poly is not None:
                rec["poly_norm"] = (poly / [w, h]).round(5).tolist()
                if quad is not None and not clipped:
                    rec["quad_norm"] = (quad / [w, h]).round(5).tolist()
                    rec["iou"] = round(iou(rec["poly_norm"], rec["quad_norm"]), 3)
                    rec["auto_ok"] = bool(pn[0][0] >= OK_OWL and rec["iou"] >= OK_IOU)
        recs.append(rec)
        print(f"t={t:6.2f} owl={rec['owl']} iou={rec['iou']} clipped={rec['clipped']} ok={rec['auto_ok']}", flush=True)
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
