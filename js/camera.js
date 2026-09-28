// Live camera control: switching lens (for more field of view than the main camera gives) and hardware zoom.
// Browser support varies a lot by phone, so every control here is feature-detected and left out of the page
// entirely when the device does not offer it, rather than showing one that does nothing.

export class CameraControl {
  constructor(video) {
    this.video = video;
    this.stream = null;
    this.track = null;
    this.devices = [];      // every camera the browser reports, in whatever order/labels it gives them
    this.deviceId = null;
  }

  // Opens the named device (or the default "environment-facing" camera when deviceId is omitted), stopping
  // whatever was open before. Populates the device list on the first call, once the browser has granted
  // permission and so is willing to give real labels instead of "camera 1", "camera 2", ...
  async open(deviceId = null) {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser cannot open the camera. Try Safari or Chrome over https.');
    const constraints = deviceId
      ? { video: { deviceId: { exact: deviceId } }, audio: false }
      : { video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    this._stop();
    this.stream = stream;
    this.track = stream.getVideoTracks()[0];
    this.deviceId = this.track.getSettings?.().deviceId || deviceId;
    this.video.srcObject = stream;
    try {
      await this.video.play();
    } catch (e) {
      if (e?.name !== 'AbortError') throw e;    // rejects if the page was backgrounded mid-start; autoplay resumes it
    }
    if (!this.devices.length) {
      try {
        const all = await navigator.mediaDevices.enumerateDevices();
        this.devices = all.filter((d) => d.kind === 'videoinput');
      } catch { /* enumeration is optional: the lens picker just stays hidden */ }
    }
    return this;
  }

  _stop() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
  }

  // -> { min, max, step, value } for the hardware zoom range, or null when the device/browser has none. A
  // range whose min is below 1 means this phone can switch to a wider lens through the same control.
  zoomRange() {
    const c = this.track?.getCapabilities?.();
    if (!c?.zoom) return null;
    const cur = this.track.getSettings?.().zoom;
    return { min: c.zoom.min, max: c.zoom.max, step: c.zoom.step || (c.zoom.max - c.zoom.min) / 20 || 0.1, value: cur ?? c.zoom.min };
  }

  async setZoom(value) {
    if (!this.track) return;
    await this.track.applyConstraints({ advanced: [{ zoom: value }] }).catch(() => {});
  }

  // A human label for each device, falling back to a plain number when the browser gives no name (rare,
  // once permission is granted).
  labelFor(i) { return this.devices[i]?.label || `Camera ${i + 1}`; }

  // The device whose label most clearly says "wide" lens, if the phone has one and it isn't already open;
  // null when there is nothing to switch to, in which case the normal back camera already opened is used.
  preferredWide() {
    const score = (label) => {
      const l = (label || '').toLowerCase();
      if (/ultra[\s-]*wide/.test(l)) return 2;    // e.g. iPhone's "Back Ultra Wide Camera"
      if (/wide[\s-]*angle|\bwide\b/.test(l)) return 1;
      return 0;
    };
    let best = null, bestScore = 0;
    for (const d of this.devices) {
      const s = score(d.label);
      if (s > bestScore) { bestScore = s; best = d; }
    }
    return best && best.deviceId !== this.deviceId ? best.deviceId : null;
  }
}
