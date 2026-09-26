# Pod feature guide (web AR)

Scan a QR code, allow the camera, point it at a pod, and tap numbered dots to see what each part
is and open its feature page. Runs entirely in the browser as a static site, so it works on GitHub Pages.

## Status

Version 0. What works:

- Camera view (rear camera on phones) with permission prompt
- Pod outline found automatically: a small segmentation model (`models/pod_front.onnx`, run in the browser
  with onnxruntime-web) finds the pod's front frame and fits its four corners. It only places the outline
  when the **whole front is in view** (its corners are off-screen otherwise and cannot be recovered), and
  only after two detections in a row agree. Until then the hint says to step back, and tapping the four
  corners by hand still works at any time.
- The outline is then followed as the phone moves (optical flow via OpenCV.js), so you can walk closer.
  Corners can be dragged to fine-tune.
- Tracking is checked, so turning the camera away and back no longer leaves a mangled outline. While an
  outline is showing, the detector re-checks about twice a second: if it sees no pod, or sees the pod
  somewhere the outline is not, the outline is deleted after two checks in a row (about a second). It is
  also deleted after 1.5 s without a reliable motion estimate or entirely out of view, and unreliable or
  implausible motion estimates are ignored. Once the whole front is back in view a new outline is generated
  automatically (if only part of the front is in view it says to step back, or you can tap the corners).
  While an outline is showing, a confirmed whole-front detection also nudges it back onto the pod to cancel
  drift; an outline you placed or dragged yourself is left alone unless tracking is lost.

Not done yet: estimating corners that are off-screen, so the outline can be placed from a close-up view. The detector was trained on 65 labelled frames from two Silen pods (held-out IoU
about 0.82 to 0.84 when trained on one pod and tested on the other) and has not been tried on a clip that
shows a whole pod front from a distance, so record one and open the site with `?debug` to see what it reports.

## Train the detector

`tools/train/train.py` fine-tunes a small MobileNetV3 segmentation model on the labels exported by the
labeller and writes the ONNX file the app loads:

```
python tools/train/train.py --train IMG_2822 --test IMG_2823                       # cross-pod check
python tools/train/train.py --train IMG_2822 IMG_2823 --out models/pod_front.onnx  # final model
```

## Label pod fronts (training data for the detector)

### Pre-label automatically (no clicking)

`tools/prelabel/prelabel.py` finds the pod front with two pretrained models (OWLv2 for "black framed glass
door", SAM 2.1 for the outline) and writes `<video>.proposals.json`. Both download from Hugging Face on first
run, no login needed; Apple silicon uses the GPU.

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r tools/prelabel/requirements.txt
.venv/bin/python tools/prelabel/prelabel.py dataset/IMG_2822.MOV --step 0.5 --sheet
```

Then open `tools/label.html`, load the same video, set "Sample every" to the `--step` you used, and use
"Import proposals". Purple frames are proposals: `A` / Enter accepts and jumps to the next, `X` rejects.
"Accept confident" takes every `auto_ok` proposal at once (detector confident, mask fills its box, one solid
piece). On the two sample videos every confident outline I checked wrapped the whole front frame, but skim the
contact sheet (`--sheet`) first: green = confident, magenta = needs a look.
Fronts that run off the picture edge come as outlines only; they are valid segmentation masks but have no
corners. Expect to reject the close-ups where only part of a pane is found.

### Label by hand

`tools/label.html` is a standalone labelling page. Serve the folder as above, open
`http://localhost:8765/tools/label.html`, and pick a video with "Open video" (Chrome, Edge or Brave).

1. Click the four corners of the pod's front frame on a frame: top-left, top-right, bottom-right,
   bottom-left, as seen from the front. Corners may lie outside the picture; click in the dark margin.
2. Label a key every second or so (`N` marks a frame with no pod front). Use `[` `]` to jump between keys.
3. `P` propagates between keys with the same optical-flow tracker the app uses, forward from one key and
   backward from the next. Frames where the two directions disagree turn red; fix them by adding a key.
   Frames tracked from only one key (blue) are capped at 10 frames and left out of exports by default.
4. "Export frames + masks" writes `images/`, `masks/` (white = pod front) and `<video>.labels.json`
   (quads in pixels and normalised, plus `corners_inside` and `visible_frac` for filtering) to a folder.
   "Export JSON" writes only the labels. Labels autosave in the browser and can be re-imported.

## Place the feature dots

`tools/hotspots.html` (serve the folder as below, open `http://localhost:8765/tools/hotspots.html`) sets the
numbered dots without measuring anything by hand:

1. "Open photo or video" with a straight-on picture of the pod front, standing back so the whole front is in
   view. For a video, use the slider to pick a frame.
2. The outline is found automatically. If the whole front is not in view it says so; click the four corners
   instead (top-left, top-right, bottom-right, bottom-left) and drag them to fine-tune.
3. The list starts with what is in `data/pod.json`. Select a feature (or "+ Add feature"), then click its spot
   on the picture. Fill in its title, popup text and link on the right. Drag a dot to move it; "grid" shows
   the front frame divided into tenths.
4. "Download pod.json" (or "Copy JSON"), replace `data/pod.json` with it, and push.

The list autosaves in the browser. Dots follow the outline if you move a corner, so set the outline first.

## Edit the features

You can edit `data/pod.json` by hand, or use the placement page above. Each hotspot has:

| field | meaning |
|---|---|
| `title`, `text` | shown in the popup |
| `url` | opened by the "Open feature page" button (opens in a new tab; many sites block embedding) |
| `x`, `y` | position on the pod's front frame: `x` 0 = left edge, 1 = right edge; `y` 0 = top, 1 = bottom |

The `example.com` links are placeholders.

## Run locally

Camera access needs `https` or `localhost`:

```
python -m http.server 8765
```

then open http://localhost:8765/. To test on a PC without a camera, put a video in `dev/` and open
`http://localhost:8765/?src=dev/demo.mp4`.

## Publish

1. Create a GitHub repository and push this folder.
2. Settings, Pages, deploy from branch `main`, folder `/ (root)`.
3. The site is served at `https://<user>.github.io/<repo>/`, which is https, so the camera prompt works.
4. Make a QR code for that URL and print it next to the pod.

`dev/`, `dataset/` and `video/` are git-ignored, so raw showroom footage is not published.
