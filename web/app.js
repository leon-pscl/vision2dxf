// One script for every page; each init() returns early when its page is absent.

const FLASH_MS = 150; // the shutter flash, long enough for the subject to see

function el(id) {
  return document.getElementById(id);
}

function form(entries) {
  const data = new FormData();
  for (const [key, value] of Object.entries(entries)) data.append(key, value);
  return data;
}

async function post(url, data, output) {
  if (output) output.textContent = "working…";
  const response = await fetch(url, { method: "POST", body: data });
  const body = await response.json();
  if (!response.ok) {
    if (output) output.textContent = `${response.status}: ${body.detail ?? "failed"}`;
    return null;
  }
  return body;
}

function initCapture() {
  const page = el("capture");
  if (!page) return;

  const video = el("source");
  const preview = el("preview");
  const context = preview.getContext("2d");
  const passed = new Set();
  let sessionId = null;
  let rotationDegrees = 0;
  let cue = null; // {label, until} while a capture counts down
  let busy = false;
  let flashUntil = 0;

  fetch("/api/health")
    .then((response) => response.json())
    .then((health) => {
      rotationDegrees = ((health.camera_rotation % 360) + 360) % 360;
      const parts = [
        health.calibrated ? "lens corrected" : "lens distortion not corrected",
        health.weights ? "models loaded" : "model weights missing from models/",
      ];
      el("station-status").textContent = parts.join(" · ");
    })
    .catch((error) => {
      el("station-status").textContent = `backend unreachable: ${error.message}`;
    });

  // The browser is asked for a portrait stream, so the camera needs no physical
  // rotation and the frame arrives upright. A camera that ignores the request and
  // hands back a landscape stream still falls back to the configured rotation.
  function isUpright() {
    return video.videoHeight > video.videoWidth;
  }

  // The preview is turned to match, so what the operator sees is what gets measured.
  function rotation() {
    return isUpright() ? 0 : rotationDegrees;
  }

  function previewSize() {
    const turned = rotation() === 90 || rotation() === 270;
    return {
      width: turned ? video.videoHeight : video.videoWidth,
      height: turned ? video.videoWidth : video.videoHeight,
    };
  }

  function drawGuide() {
    const { width, height } = preview;
    if (!width || !height) return;

    const centre = width / 2;
    const top = height * 0.06;
    const floor = height * 0.96;
    const span = floor - top;
    context.strokeStyle = "rgba(0, 220, 0, 0.75)";
    context.setLineDash([12, 10]);
    context.lineWidth = 3;
    context.beginPath();
    context.ellipse(centre, top + span * 0.07, span * 0.05, span * 0.07, 0, 0, Math.PI * 2);
    context.moveTo(centre - span * 0.13, top + span * 0.17);
    context.lineTo(centre - span * 0.3, top + span * 0.5); // left arm, A-pose
    context.moveTo(centre + span * 0.13, top + span * 0.17);
    context.lineTo(centre + span * 0.3, top + span * 0.5);
    context.moveTo(centre - span * 0.13, top + span * 0.17);
    context.lineTo(centre - span * 0.1, top + span * 0.52);
    context.lineTo(centre - span * 0.08, floor);
    context.moveTo(centre + span * 0.13, top + span * 0.17);
    context.lineTo(centre + span * 0.1, top + span * 0.52);
    context.lineTo(centre + span * 0.08, floor);
    context.stroke();
    context.setLineDash([]);

    // A comfortable frame, looser than the gate: stand inside this and the
    // capture passes with room to spare.
    context.strokeStyle = "rgba(255, 200, 0, 0.6)";
    context.setLineDash([6, 8]);
    context.lineWidth = 2;
    const margin = height * 0.08;
    context.strokeRect(centre - width * 0.15, margin, width * 0.3, height - 2 * margin);
    context.setLineDash([]);

    if (cue) {
      const left = Math.max(1, Math.ceil((cue.until - performance.now()) / 1000));
      context.fillStyle = "rgba(0, 0, 0, 0.55)";
      context.fillRect(0, 0, width, height);
      context.fillStyle = "#fff";
      context.textAlign = "center";
      context.font = `${Math.round(height * 0.05)}px sans-serif`;
      context.fillText(cue.label, centre, height * 0.36);
      context.font = `${Math.round(height * 0.2)}px sans-serif`;
      context.fillText(String(left), centre, height / 2);
      context.textAlign = "start";
    }

    // The shutter, so the subject sees the shot was taken.
    if (performance.now() < flashUntil) {
      context.fillStyle = "rgba(255, 255, 255, 0.85)";
      context.fillRect(0, 0, width, height);
    }
  }

  function drawPreview() {
    if (video.videoWidth) {
      const { width, height } = previewSize();
      if (preview.width !== width) preview.width = width;
      if (preview.height !== height) preview.height = height;
      context.save();
      context.translate(width / 2, height / 2);
      context.rotate((rotation() * Math.PI) / 180);
      context.drawImage(video, -video.videoWidth / 2, -video.videoHeight / 2);
      context.restore();
      drawGuide();
    }
    requestAnimationFrame(drawPreview);
  }

  // The raw frame goes to the server as it comes: the browser already asks for a
  // portrait stream, so a live frame is upright and must not be turned again.
  function rawFrame() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.95));
  }

  function wait(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // The label and the seconds left are painted by drawGuide from `until`, so the
  // subject watches the countdown rather than a blank frame being sent.
  async function countDown(label, seconds) {
    cue = { label, until: performance.now() + seconds * 1000 };
    await wait(seconds * 1000);
    cue = null;
  }

  const POSE_SECONDS = 5; // hold this pose while the countdown runs
  const MOVE_SECONDS = 5; // to get into the next one
  const RETAKE_SECONDS = 3; // a single view, when a tailor asks for it

  function viewName(item) {
    return item.querySelector(".view-name").textContent;
  }

  // Front, then the two sides, in the order section 3 lists them. Every pose gets
  // a countdown to hold still through; between poses there is longer to turn.
  async function captureAll() {
    for (const [index, item] of [...page.querySelectorAll("li[data-view]")].entries()) {
      const name = viewName(item);
      if (index) await countDown(`next: ${name}`, MOVE_SECONDS);
      await countDown(`${name} — hold still`, POSE_SECONDS);
      flashUntil = performance.now() + FLASH_MS;
      await sendView(item.dataset.view, await rawFrame(), isUpright());
    }
  }

  function row(view) {
    return page.querySelector(`li[data-view="${view}"]`);
  }

  function refreshButtons() {
    const live = Boolean(video.srcObject);
    for (const item of page.querySelectorAll("li[data-view]")) {
      const view = item.dataset.view;
      item.querySelector('[data-action="capture"]').disabled =
        !sessionId || !live || busy || passed.has(view);
      item.querySelector("input[type=file]").disabled = !sessionId;
    }
    el("start-capture").disabled = !sessionId || !live || busy;
    el("measure").disabled = !(passed.has("front") && (passed.has("side_left") || passed.has("side_right")));
  }

  async function sendView(view, blob, upright) {
    const item = row(view);
    const output = item.querySelector("output");
    const data = new FormData();
    data.append("image", blob, `${view}.jpg`);
    // A live capture arrives from a portrait-mounted camera and has to be turned;
    // an upload or a stored sample is a photograph and must not be.
    data.append("upright", String(Boolean(upright)));
    const body = await post(`/api/sessions/${sessionId}/views/${view}`, data, output);
    if (!body) {
      item.querySelector('[data-action="retake"]').hidden = false;
      return;
    }
    if (body.ok) {
      passed.add(view);
      output.textContent = "✓ accepted";
      output.className = "ok";
      item.querySelector('[data-action="retake"]').hidden = false;
    } else {
      passed.delete(view);
      output.textContent = body.reasons.join(" · ");
      output.className = "bad";
      item.querySelector('[data-action="retake"]').hidden = false;
    }
    refreshButtons();
  }

  // A sample only appears when the file is actually there, so the page needs no
  // list to keep in step with whatever is in web/samples/.
  async function offerSamples() {
    for (const item of page.querySelectorAll("li[data-view]")) {
      const response = await fetch(`/samples/${item.dataset.view}.jpg`, { method: "HEAD" });
      item.querySelector('[data-action="sample"]').hidden = !response.ok;
    }
  }

  async function useSample(view) {
    const blob = await (await fetch(`/samples/${view}.jpg`)).blob();
    await sendView(view, blob, true);
  }

  el("start-session").addEventListener("click", async () => {
    const body = await post(
      "/api/sessions",
      form({
        subject_code: el("subject-id").value.trim(),
        sex: el("sex").value,
        consent: el("consent").checked,
        height_cm: el("height-cm").value,
      }),
      el("session-out"),
    );
    if (!body) return;
    sessionId = body.id;
    el("session-out").textContent = `session ${sessionId}`;
    refreshButtons();
  });

  // A portrait stream, so the camera is not mounted on its side. A camera that
  // cannot deliver one hands back landscape, which isUpright() catches.
  el("start-camera").addEventListener("click", async () => {
    try {
      video.srcObject = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1080 }, height: { ideal: 1920 } },
      });
      refreshButtons();
    } catch (error) {
      el("station-status").textContent = `no camera: ${error.message} — upload images instead`;
    }
  });

  el("start-capture").addEventListener("click", async () => {
    busy = true;
    refreshButtons();
    try {
      await captureAll();
    } finally {
      busy = false;
      refreshButtons();
    }
  });

  for (const item of page.querySelectorAll("li[data-view]")) {
    const view = item.dataset.view;
    item.querySelector('[data-action="capture"]').addEventListener("click", async () => {
      // Lock every capture while this one runs, or a double click queues two frames.
      busy = true;
      refreshButtons();
      item.querySelector("output").textContent = "get ready…";
      item.querySelector("output").className = "";
      try {
        await countDown(`${viewName(item)} — hold still`, RETAKE_SECONDS);
        flashUntil = performance.now() + FLASH_MS;
        await sendView(view, await rawFrame(), isUpright());
      } finally {
        busy = false;
        refreshButtons();
      }
    });
    item.querySelector('[data-action="sample"]').addEventListener("click", () => useSample(view));
    item.querySelector("input[type=file]").addEventListener("change", async (event) => {
      if (event.target.files[0]) await sendView(view, event.target.files[0], true);
    });
    item.querySelector('[data-action="retake"]').addEventListener("click", () => {
      passed.delete(view);
      item.querySelector("output").textContent = "";
      item.querySelector("output").className = "";
      item.querySelector('[data-action="retake"]').hidden = true;
      refreshButtons();
    });
  }

  el("measure").addEventListener("click", async () => {
    const body = await post(`/api/sessions/${sessionId}/measure`, new FormData(), el("measure-out"));
    if (!body) return;
    const warnings = body.warnings.length ? ` — ${body.warnings.join(" · ")}` : "";
    el("measure-out").textContent =
      `${Object.keys(body.measurements).length} measurements${warnings}`;
    el("download-body").href = `/api/sessions/${sessionId}/export`;
    el("download-report").href = `/api/sessions/${sessionId}/report`;
    el("downloads").hidden = false;
  });

  refreshButtons();
  offerSamples();
  drawPreview();
}

initCapture();
initReview();

function initReview() {
  const page = el("review");
  if (!page) return;

  const sessionId = decodeURIComponent(window.location.pathname.split("/").pop());
  const schema = { deg: ["shoulder_incl", "arm_pose_angle", "hip_inclination"] };
  const canvases = new Map();
  let landmarks = [];
  let dragging = null;

  el("session-name").textContent = sessionId;
  el("download-body").href = `/api/sessions/${sessionId}/export`;
  el("download-report").href = `/api/sessions/${sessionId}/report`;

  const isRow = (mark) => mark.name.endsWith("_row");

  function unit(name) {
    return schema.deg.includes(name) ? "deg" : "cm";
  }

  function draw(view) {
    const entry = canvases.get(view);
    if (!entry || !entry.image.complete || !entry.image.naturalWidth) return;
    const { canvas, image } = entry;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0);

    for (const mark of landmarks.filter((item) => item.view === view)) {
      const colour = mark.auto ? "rgba(255, 200, 0, 0.95)" : "rgba(255, 0, 255, 0.95)";
      context.strokeStyle = colour;
      context.fillStyle = colour;
      context.lineWidth = 3;
      context.font = "22px sans-serif";
      if (isRow(mark)) {
        context.beginPath();
        context.moveTo(0, mark.y);
        context.lineTo(canvas.width, mark.y);
        context.stroke();
        context.fillText(mark.name, 12, mark.y - 10);
      } else {
        context.beginPath();
        context.arc(mark.x, mark.y, 9, 0, Math.PI * 2);
        context.stroke();
        context.fillText(mark.name, mark.x + 14, mark.y - 8);
      }
    }
  }

  function pointIn(canvas, event) {
    const box = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - box.left) / box.width) * canvas.width,
      y: ((event.clientY - box.top) / box.height) * canvas.height,
    };
  }

  function nearest(view, at) {
    let best = null;
    for (const mark of landmarks.filter((item) => item.view === view)) {
      const distance = isRow(mark) ? Math.abs(mark.y - at.y) : Math.hypot(mark.x - at.x, mark.y - at.y);
      if (distance < 40 && (!best || distance < best.distance)) best = { mark, distance };
    }
    return best ? best.mark : null;
  }

  function addView(view) {
    const figure = document.createElement("figure");
    const caption = document.createElement("figcaption");
    caption.textContent = view;
    const canvas = document.createElement("canvas");
    const image = new Image();
    image.src = `/api/sessions/${sessionId}/overlay/${view}_gate.png`;
    image.addEventListener("load", () => draw(view));
    figure.append(canvas, caption);
    el("views").append(figure);
    canvases.set(view, { canvas, image });

    canvas.addEventListener("pointerdown", (event) => {
      const mark = nearest(view, pointIn(canvas, event));
      if (!mark) return;
      dragging = { view, name: mark.name };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!dragging || dragging.view !== view) return;
      const at = pointIn(canvas, event);
      landmarks = landmarks.map((mark) =>
        mark.name === dragging.name && mark.view === view
          ? { ...mark, x: isRow(mark) ? mark.x : at.x, y: at.y, auto: false }
          : mark,
      );
      draw(view);
    });
    canvas.addEventListener("pointerup", async () => {
      if (!dragging) return;
      dragging = null;
      await send("PUT", `/api/sessions/${sessionId}/landmarks`, landmarks);
    });
  }

  async function send(method, url, payload) {
    const status = el("review-status");
    status.textContent = "recomputing…";
    const response = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
      status.textContent = `${response.status}: ${body.detail ?? "failed"}`;
      return null;
    }
    render(body);
    return body;
  }

  function renderTable(measurements) {
    const body = el("measurements").querySelector("tbody");
    body.replaceChildren();
    for (const [name, entry] of Object.entries(measurements)) {
      const row = document.createElement("tr");
      if (entry.confidence === "estimated" || entry.confidence === "low") {
        row.className = "flagged";
      }

      const value = document.createElement("input");
      value.type = "number";
      value.step = "0.1";
      value.value = entry.value;
      const note = document.createElement("input");
      note.type = "text";
      note.placeholder = "why this value";
      note.value = entry.note ?? "";

      const commit = () =>
        send("PUT", `/api/sessions/${sessionId}/values`, {
          [name]: { value: Number(value.value), note: note.value },
        });
      value.addEventListener("change", commit);
      note.addEventListener("change", commit);

      const cells = [name, value, unit(name), entry.confidence, entry.source, note];
      for (const content of cells) {
        const cell = document.createElement("td");
        if (content instanceof HTMLElement) {
          cell.append(content);
        } else if (content === entry.confidence) {
          const chip = document.createElement("span");
          chip.className = `chip ${entry.confidence}`;
          chip.textContent = entry.confidence;
          cell.append(chip);
        } else {
          cell.textContent = content;
        }
        row.append(cell);
      }
      if (entry.edited) row.classList.add("edited");
      body.append(row);
    }
  }

  function render(payload) {
    landmarks = payload.landmarks;
    renderTable(payload.measurements);
    const warnings = el("warnings");
    warnings.hidden = payload.warnings.length === 0;
    warnings.textContent = payload.warnings.join(" · ");
    for (const view of new Set(landmarks.map((mark) => mark.view))) {
      if (!canvases.has(view)) addView(view);
      draw(view);
    }
    el("review-status").textContent = `${Object.keys(payload.measurements).length} measurements`;
  }

  el("confirm-session").addEventListener("click", async () => {
    const body = await post(
      `/api/sessions/${sessionId}/confirm`,
      new FormData(),
      el("confirm-out"),
    );
    if (body) {
      el("confirm-out").textContent =
        `confirmed · photos deleted after ${body.images_purge_after}`;
    }
  });

  el("delete-session").addEventListener("click", async () => {
    if (!window.confirm(`Delete ${sessionId} and every photo in it?`)) return;
    const response = await fetch(`/api/sessions/${sessionId}`, { method: "DELETE" });
    if (response.ok) window.location.href = "/";
  });

  post(`/api/sessions/${sessionId}/measure`, new FormData(), el("review-status")).then(
    (body) => body && render(body),
  );
}
