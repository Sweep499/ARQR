"""Train the pod-front segmentation model on labels exported by tools/label.html.

    python train.py --train IMG_2822 --test IMG_2823          # cross-pod check (train on one pod, test on the other)
    python train.py --train IMG_2822 IMG_2823 --out ../../models/pod_front.onnx   # final model, exported for the browser

Labels are read from <labels>/<stem>/images and .../masks. The model is torchvision's LRASPP MobileNetV3
(ImageNet/COCO pretrained) with a 1-channel head; input is 256x448 RGB, output is a per-pixel logit for
"pod front frame".
"""
import argparse, glob, os, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

W, H = 256, 448
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def load(labels, stems):
    items = []
    for s in stems:
        for p in sorted(glob.glob(f"{labels}/{s}/images/*.jpg")):
            m = p.replace("/images/", "/masks/").replace(".jpg", ".png")
            items.append((Image.open(p).convert("RGB"), Image.open(m).convert("L")))
    return items


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
        self.norm = T.Compose([T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        im, mk = self.items[i]
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
    return (inter / union.clamp(min=1)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="../../dataset/labels")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", help="write the trained model here as ONNX")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    tr = torch.utils.data.DataLoader(Data(load(a.labels, a.train), True), batch_size=8, shuffle=True, drop_last=True)
    te = torch.utils.data.DataLoader(Data(load(a.labels, a.test), False), batch_size=8) if a.test else None
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
        if te and (ep % 10 == 9 or ep == a.epochs - 1):
            model.eval(); s = []
            with torch.no_grad():
                for x, y in te: s.append(iou(model(x.to(dev)).cpu(), y))
            print(f"epoch {ep + 1:3d} loss {tot / len(tr):.3f}  held-out IoU {np.mean(s):.3f}", flush=True)
        elif ep % 10 == 9:
            print(f"epoch {ep + 1:3d} loss {tot / len(tr):.3f}", flush=True)
    if a.out:
        model.eval().cpu()
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        torch.onnx.export(model, torch.zeros(1, 3, H, W), a.out, input_names=["image"], output_names=["logits"], opset_version=17, dynamo=False)
        print("wrote", a.out, os.path.getsize(a.out) // 1024, "KB")


if __name__ == "__main__":
    main()
