# Pod feature guide (web AR)

Scan a QR code, allow the camera, point it at a pod, and tap numbered dots to see what each part
is and open its feature page. Runs entirely in the browser as a static site, so it works on GitHub Pages.

## Status

Version 0. What works:

- Camera view (rear camera on phones) with permission prompt
- Pod outline placed by tapping the four corners of the pod's front frame once, then followed as the
  phone moves (optical flow via OpenCV.js). Corners can be dragged to fine-tune.
- Numbered feature dots pinned to the outline; tapping one opens a popup with a description and an
  "Open feature page" link
- All features are listed in `data/pod.json`

What is planned: a trained detector that finds the pod's four corners automatically, replacing the
tap-to-place step and cancelling tracking drift.

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

## Edit the features

Open `data/pod.json`. Each hotspot has:

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
