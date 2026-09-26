"""Train the pod-front segmentation model on labels exported by tools/label.html.

    python train.py --train IMG_2822 --test IMG_2823          # cross-pod check (train on one pod, test on the other)
    python train.py --train IMG_2822 IMG_2823 --out ../../models/pod_front.onnx   # final model, exported for the browser

Labels are read from <labels>/<stem>/images and .../masks. The model is torchvision's LRASPP MobileNetV3
(ImageNet/COCO pretrained) with a 1-channel head; input is 256x448 RGB, output is a per-pixel logit for
"pod front frame".
"""
import argparse, glob, os, random
import cv2
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

W, H = 256, 448
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def load(labels, stems):
    """-> [(image, mask, is_render)]. Stems starting with RENDER are manufacturer renders (any aspect ratio)."""
    items = []
    for s in stems:
        for p in sorted(glob.glob(f"{labels}/{s}/images/*.jpg")):
            m = p.replace("/images/", "/masks/").replace(".jpg", ".png")
            items.append((Image.open(p).convert("RGB"), Image.open(m).convert("L"), s.startswith("RENDER")))
    return items


def empty_view(im, mk, tries=20):
    """A phone-shaped crop of a render that contains none of the pod (its mask is empty), or None if the
    picture leaves no room for one. Teaches the model that plain background is not a pod."""
    a, m = np.array(im), np.array(mk) > 127
    ys, xs = np.nonzero(m)
    hh, ww = m.shape
    bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = 0.08 * (bx1 - bx0), 0.08 * (by1 - by0)             # keep a margin around the pod
    for _ in range(tries):
        cw = random.uniform(0.15, 0.7) * ww
        ch = cw * H / W
        if ch > hh:
            cw, ch = hh * W / H * 0.98, hh * 0.98
        x0, y0 = random.uniform(0, ww - cw), random.uniform(0, hh - ch)
        if x0 + cw < bx0 - px or x0 > bx1 + px or y0 + ch < by0 - py or y0 > by1 + py:
            crop = a[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
            if crop.size:
                return Image.fromarray(cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)), Image.fromarray(np.zeros((H, W), np.uint8))
    return None


def portrait_view(im, mk, f, dx=0.0, dy=0.0):
    """A phone-shaped (9:16) view of a render in which the labelled front is `f` times the view's width
    (f < 1: the whole front is in view with room around it; f > 1: it runs off the sides). The picture is
    extended past its own edges by repeating the edge pixels."""
    a, m = np.array(im), np.array(mk) > 127
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    cw = max(bw / f, bh * W / (0.95 * H))          # source pixels across the view; also keep the front's height inside
    s = W / cw
    cx, cy = (x0 + x1) / 2 + dx * cw, (y0 + y1) / 2 + dy * cw
    M = np.float32([[s, 0, W / 2 - s * cx], [0, s, H / 2 - s * cy]])
    out = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    om = cv2.warpAffine(m.astype(np.uint8) * 255, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    return Image.fromarray(out), Image.fromarray(om)


class Data(torch.utils.data.Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train
        self.aug = T.Compose([
            T.RandomResizedCrop((H, W), scale=(0.5, 1.0), ratio=(W / H * 0.8, W / H * 1.25)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(0.4, 0.4, 0.4, 0.05),
        ])
        self.plain = T.Resize((H, W))
        self.aug_render = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(8), T.ColorJitter(0.4, 0.4, 0.4, 0.05)])
        self.norm = T.Compose([T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        im, mk, render = self.items[i]
        if render:
            # a random phone-style view of the render: usually the whole front in view, sometimes cut off
            view = empty_view(im, mk) if self.train and random.random() < 0.2 else None
            if view is None:
                f = random.uniform(0.5, 1.3) if self.train else 0.85
                d = (random.uniform(-0.08, 0.08), random.uniform(-0.08, 0.08)) if self.train else (0.0, 0.0)
                view = portrait_view(im, mk, f, *d)
            im, mk = view
            im, mk = tv_tensors.Image(im), tv_tensors.Mask(np.array(mk) > 127)
            if self.train:
                im, mk = self.aug_render(im, mk)
        else:
            im, mk = tv_tensors.Image(im), tv_tensors.Mask(np.array(mk) > 127)
            im, mk = (self.aug if self.train else self.plain)(im, mk)
        return self.norm(im), mk.float().unsqueeze(0) if mk.ndim == 2 else mk.float()


def make_model():
    m = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
    m.classifier.low_classifier = nn.Conv2d(m.classifier.low_classifier.in_channels, 1, 1)
    m.classifier.high_classifier = nn.Conv2d(m.classifier.high_classifier.in_channels, 1, 1)
    return m


class Wrapped(nn.Module):
    """model -> logits at input size (so the ONNX graph is image in, mask logits out)"""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x): return self.m(x)["out"]


def iou(logits, mask):
    p = (logits > 0).float()
    inter = (p * mask).sum((1, 2, 3)); union = ((p + mask) > 0).float().sum((1, 2, 3))
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union)).mean().item()   # both empty counts as a match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="../../dataset/labels")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", help="write the trained model here as ONNX")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render-repeat", type=int, default=2, help="how many times per epoch each render is drawn (each a new random view)")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    items = load(a.labels, a.train)
    items = [it for it in items for _ in range(a.render_repeat if it[2] else 1)]
    items += [it for it in items if not it[2] and not np.array(it[1]).any()] * 2     # real frames without the pod: 3x in total
    tr = torch.utils.data.DataLoader(Data(items, True), batch_size=8, shuffle=True, drop_last=True)
    tests = {}
    if a.test:
        allt = load(a.labels, a.test)
        for name, flag in (("real", False), ("render", True)):
            sub = [it for it in allt if it[2] == flag]
            if sub: tests[name] = torch.utils.data.DataLoader(Data(sub, False), batch_size=8)
    model = Wrapped(make_model()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=a.epochs * len(tr))
    for ep in range(a.epochs):
        model.train(); tot = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            lg = model(x)
            p = torch.sigmoid(lg)
            dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
            loss = F.binary_cross_entropy_with_logits(lg, y) + dice
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        if tests and (ep % 10 == 9 or ep == a.epochs - 1):
            model.eval(); res = {}
            with torch.no_grad():
                for name, dl in tests.items():
                    res[name] = np.mean([iou(model(x.to(dev)).cpu(), y) for x, y in dl])
            print(f"epoch {ep + 1:3d} loss {tot / len(tr):.3f}  held-out IoU " + "  ".join(f"{k} {v:.3f}" for k, v in res.items()), flush=True)
        elif ep % 10 == 9:
            print(f"epoch {ep + 1:3d} loss {tot / len(tr):.3f}", flush=True)
    if a.out:
        model.eval().cpu()
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        torch.onnx.export(model, torch.zeros(1, 3, H, W), a.out, input_names=["image"], output_names=["logits"], opset_version=17, dynamo=False)
        print("wrote", a.out, os.path.getsize(a.out) // 1024, "KB")


if __name__ == "__main__":
    main()
