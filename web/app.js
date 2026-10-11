// One script for every page; each init() returns early when its page is absent.

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

function initCalibrate() {
  const page = el("calibrate");
  if (!page) return;

  const video = el("preview");
  let frames = [];
  let lastBoard = null;

  const boardParams = () => ({
    pattern_cols: el("cols").value,
    pattern_rows: el("rows").value,
    square_mm: el("square").value,
  });

  // The preview runs at the camera's full resolution, so a grabbed frame is
  // exactly what the station will capture later.
  function snapshot() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  }

  async function show(url, data, output) {
    const body = await post(url, data, output);
    if (body) output.textContent = JSON.stringify(body, null, 2);
    return body;
  }

  function countFrames() {
    el("frame-count").textContent = `${frames.length} frames`;
    el("compute-intrinsics").disabled = frames.length === 0;
  }

  el("start-camera").addEventListener("click", async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1920 }, height: { ideal: 1080 } },
    });
    video.srcObject = stream;
    el("grab").disabled = false;
    el("measure-scale").disabled = false;
  });

  el("grab").addEventListener("click", async () => {
    frames.push(await snapshot());
    countFrames();
  });

  el("upload-frames").addEventListener("change", (event) => {
    frames = Array.from(event.target.files);
    countFrames();
  });

  el("compute-intrinsics").addEventListener("click", async () => {
    const data = form(boardParams());
    frames.forEach((frame, index) => data.append("frames", frame, `frame${index}.png`));
    const body = await show("/api/calibration/checkerboard", data, el("intrinsics-out"));
    if (body) {
      frames = [];
      countFrames();
    }
  });

  // Measuring grabs a fresh frame; saving reuses it, so the operator commits the
  // exact numbers shown.
  async function board(fresh) {
    const uploaded = el("upload-board").files[0];
    if (uploaded) return uploaded;
    if (fresh || !lastBoard) lastBoard = await snapshot();
    return lastBoard;
  }

  async function measure(save) {
    const frame = await board(!save);
    const data = form({ ...boardParams(), bottom_margin_mm: el("margin").value, save });
    data.append("frame", frame, "board.png");
    return show("/api/calibration/scale", data, save ? el("save-out") : el("scale-out"));
  }

  el("upload-board").addEventListener("change", (event) => {
    lastBoard = event.target.files[0];
    el("measure-scale").disabled = false;
  });

  el("measure-scale").addEventListener("click", async () => {
    el("save").disabled = (await measure(false)) === null;
  });

  el("save").addEventListener("click", () => measure(true));
}

function initCapture() {
  const page = el("capture");
  if (!page) return;

  const video = el("source");
  const preview = el("preview");
  const context = preview.getContext("2d");
  const passed = new Set();
  let sessionId = null;
  let calibration = null;
  let countdown = 0;

  fetch("/api/health")
    .then((response) => response.json())
    .then((health) => {
      calibration = health.calibration;
      const parts = [
        health.calibrated ? "calibrated" : "NOT calibrated — run the calibration page",
        health.weights ? "models loaded" : "model weights missing from models/",
      ];
      el("station-status").textContent = parts.join(" · ");
      el("consent-version").textContent = `consent form `;
    })
    .catch((error) => {
      el("station-status").textContent = `backend unreachable: ${error.message}`;
    });

  // The camera is mounted portrait and the server rotates every frame, so the
  // preview is rotated to match. What the operator sees is what gets measured.
  function rotation() {
    return calibration ? ((calibration.rotation % 360) + 360) % 360 : 0;
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

    if (calibration) {
      // floor_row is in calibrated pixels; the camera may run at another size.
      const row = (calibration.floor_row / calibration.image_size[1]) * height;
      context.strokeStyle = "rgba(255, 80, 80, 0.9)";
      context.lineWidth = 3;
      context.beginPath();
      context.moveTo(0, row);
      context.lineTo(width, row);
      context.stroke();
      context.fillStyle = "rgba(255, 80, 80, 0.9)";
      context.font = "20px sans-serif";
      context.fillText("floor mark", 10, row - 10);
    }

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

    if (countdown > 0) {
      context.fillStyle = "rgba(0, 0, 0, 0.55)";
      context.fillRect(0, 0, width, height);
      context.fillStyle = "#fff";
      context.font = `${Math.round(height * 0.2)}px sans-serif`;
      context.textAlign = "center";
      context.fillText(String(countdown), centre, height / 2);
      context.textAlign = "start";
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

  // The raw frame goes to the server unrotated: the server applies the
  // calibrated rotation itself, and rotating twice would ruin the scale.
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

  async function runCountdown(seconds) {
    for (countdown = seconds; countdown > 0; countdown -= 1) await wait(1000);
  }

  function row(view) {
    return page.querySelector(`li[data-view="${view}"]`);
  }

  function refreshButtons() {
    const live = Boolean(video.srcObject);
    for (const item of page.querySelectorAll("li[data-view]")) {
      const view = item.dataset.view;
      item.querySelector('[data-action="capture"]').disabled =
        !sessionId || !live || passed.has(view);
      item.querySelector("input[type=file]").disabled = !sessionId;
    }
    el("measure").disabled = !(passed.has("front") && (passed.has("side_left") || passed.has("side_right")));
  }

  async function sendView(view, blob) {
    const item = row(view);
    const output = item.querySelector("output");
    const data = new FormData();
    data.append("image", blob, `${view}.jpg`);
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

  el("start-session").addEventListener("click", async () => {
    const body = await post(
      "/api/sessions",
      form({
        subject_code: el("subject-id").value.trim(),
        sex: el("sex").value,
        consent: el("consent").checked,
      }),
      el("session-out"),
    );
    if (!body) return;
    sessionId = body.id;
    el("session-out").textContent = `session ${sessionId}`;
    refreshButtons();
  });

  el("start-camera").addEventListener("click", async () => {
    try {
      video.srcObject = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1920 }, height: { ideal: 1080 } },
      });
      refreshButtons();
    } catch (error) {
      el("station-status").textContent = `no camera: ${error.message} — upload images instead`;
    }
  });

  for (const item of page.querySelectorAll("li[data-view]")) {
    const view = item.dataset.view;
    item.querySelector('[data-action="capture"]').addEventListener("click", async () => {
      await runCountdown(3);
      await sendView(view, await rawFrame());
    });
    item.querySelector("input[type=file]").addEventListener("change", async (event) => {
      if (event.target.files[0]) await sendView(view, event.target.files[0]);
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
  drawPreview();
}

initCalibrate();
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
