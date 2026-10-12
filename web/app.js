// One script for every page; each init() returns early when its page is absent.

const FLASH_MS = 150; // the shutter flash, long enough for the subject to see
const HOLD_SECONDS = 3; // brief 16.1: a three-second countdown before each shot
const MOVE_SECONDS = 5; // longer, to turn into the next pose

function el(id) {
  return document.getElementById(id);
}

function form(entries) {
  const data = new FormData();
  for (const [key, value] of Object.entries(entries)) data.append(key, value);
  return data;
}

function setState(output, text, kind) {
  if (!output) return;
  output.textContent = text;
  output.className = `${output.classList.contains("progress") ? "progress " : ""}meta state ${kind}`;
}

async function request(method, url, { body, output, waiting = "working…" } = {}) {
  if (output) setState(output, waiting, "wait");
  let response;
  try {
    response = await fetch(url, body instanceof FormData
      ? { method, body }
      : { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch (error) {
    if (output) setState(output, `backend unreachable: ${error.message}`, "bad");
    return null;
  }
  const payload = response.headers.get("content-type")?.includes("json")
    ? await response.json()
    : {};
  if (!response.ok) {
    if (output) setState(output, `${response.status}: ${payload.detail ?? "failed"}`, "bad");
    return null;
  }
  return payload;
}

const post = (url, body, output) => request("POST", url, { body, output });
const get = (url, output) => request("GET", url, { output, waiting: "loading…" });

function reasonText(payload) {
  return (payload.reasons ?? []).map((reason) => `${reason.code}: ${reason.message}`).join(" · ");
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function sessionIdFromPath() {
  return decodeURIComponent(window.location.pathname.split("/").pop());
}

// --- theme ---------------------------------------------------------------

function initTheme() {
  const button = el("theme");
  if (!button) return;
  const stored = window.localStorage.getItem("theme");
  if (stored) document.documentElement.dataset.theme = stored;
  const sync = () => {
    const dark = document.documentElement.dataset.theme === "dark";
    button.setAttribute("aria-pressed", String(dark));
    button.textContent = dark ? "Light theme" : "Dark theme";
  };
  button.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    window.localStorage.setItem("theme", next);
    sync();
  });
  sync();
}

// --- the stage viewer, shared by the processing and review pages ---------

/* Shows one stage of one view, with the comparison and zoom controls of brief
   8.1. The image is never stretched: only its width is set, so the aspect
   ratio - which is the measurement - is whatever the file says it is. */
function createViewer({ sessionId, onStage }) {
  const frame = el("frame");
  const strip = el("filmstrip");
  let stages = [];
  let manifest = [];
  let view = null;
  let index = 0;
  let zoom = 1;
  let comparing = false;
  let sideBySide = false;
  const overlays = new Set();

  const entry = (viewName, stageId) =>
    manifest.find((item) => item.view === viewName && item.stage_id === stageId);

  const available = () => stages.filter((stage) => entry(view, stage.stage_id));

  const current = () => available()[index] ?? null;

  const url = (stageId) => `/api/sessions/${sessionId}/artifacts/${view}/${stageId}`;

  function renderStrip() {
    if (!strip) return;
    strip.replaceChildren();
    available().forEach((stage, position) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.setAttribute("aria-pressed", String(position === index));
      const number = document.createElement("span");
      number.className = "num";
      number.textContent = stage.stage_id.slice(0, 2);
      const label = document.createElement("span");
      label.textContent = stage.label;
      button.append(number, label);
      button.addEventListener("click", () => show(position));
      item.append(button);
      strip.append(item);
    });
    const missing = stages.filter((stage) => !entry(view, stage.stage_id));
    for (const stage of missing) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.disabled = true;
      button.setAttribute("aria-disabled", "true");
      button.title = "not produced yet";
      const number = document.createElement("span");
      number.className = "num";
      number.textContent = stage.stage_id.slice(0, 2);
      const label = document.createElement("span");
      label.textContent = `${stage.label} — not yet`;
      button.append(number, label);
      item.append(button);
      strip.append(item);
    }
  }

  function renderDetails(stage, record) {
    const title = el("stage-title");
    if (title) title.textContent = `${stage.label} — ${stage.description}`;
    const list = el("stage-details");
    if (!list) return;
    const pairs = [
      ["Stage", stage.stage_id],
      ["View", record.view],
      ["Dimensions", `${record.width} × ${record.height} px`],
      ["Runtime", record.runtime_ms === undefined ? "not timed" : `${record.runtime_ms} ms`],
      ["Algorithm", record.algorithm],
      ["Created", record.created_at],
      ["From", (record.source_files ?? []).join(", ") || "the camera"],
      ...Object.entries(record.parameters ?? {}).map(([key, value]) => [
        key,
        Array.isArray(value) ? value.join(", ") : String(value),
      ]),
    ];
    if (record.note) pairs.push(["Note", record.note]);
    list.replaceChildren();
    for (const [key, value] of pairs) {
      const dt = document.createElement("dt");
      dt.textContent = key;
      const dd = document.createElement("dd");
      dd.textContent = value;
      list.append(dt, dd);
    }
  }

  function picture(stageId, extra = {}) {
    const image = document.createElement("img");
    image.src = url(stageId);
    image.alt = "";
    Object.assign(image.style, extra);
    return image;
  }

  function render() {
    const stage = current();
    frame.replaceChildren();
    if (!stage) {
      const empty = document.createElement("p");
      empty.className = "hint";
      empty.textContent = view
        ? "This view has no stages yet. Capture it, then process the session."
        : "No view selected.";
      frame.append(empty);
      return;
    }
    const record = entry(view, stage.stage_id);
    frame.classList.toggle("side-by-side", sideBySide && comparing);

    const skeleton = document.createElement("p");
    skeleton.className = "skeleton";
    skeleton.textContent = `loading ${stage.label}…`;

    if (sideBySide && comparing) {
      frame.append(picture("00_raw"), picture(stage.stage_id));
    } else {
      const wrapper = document.createElement("div");
      wrapper.className = "overlay-stack";
      wrapper.style.width = zoom === 1 ? "100%" : `${Math.round(record.width * zoom)}px`;
      if (comparing) wrapper.append(picture("00_raw", { position: "absolute", inset: "0" }));
      const main = picture(stage.stage_id, {
        opacity: comparing ? String(Number(el("blend")?.value ?? 100) / 100) : "1",
        position: "relative",
      });
      main.addEventListener("load", () => skeleton.remove());
      main.addEventListener("error", () => {
        skeleton.textContent = `${stage.label} could not be loaded. Earlier stages are still there.`;
        skeleton.className = "skeleton";
      });
      wrapper.append(main);
      for (const stageId of overlays) {
        if (!entry(view, stageId)) continue;
        wrapper.append(
          picture(stageId, {
            position: "absolute",
            inset: "0",
            mixBlendMode: "screen",
            opacity: "0.65",
            pointerEvents: "none",
          }),
        );
      }
      frame.append(wrapper, skeleton);
      if (onStage) onStage(wrapper, record, stage);
    }
    renderDetails(stage, record);
    renderStrip();
  }

  function show(position) {
    const list = available();
    if (!list.length) return;
    index = Math.max(0, Math.min(position, list.length - 1));
    render();
  }

  el("prev")?.addEventListener("click", () => show(index - 1));
  el("next")?.addEventListener("click", () => show(index + 1));
  el("zoom-in")?.addEventListener("click", () => {
    zoom = Math.min(zoom * 1.4, 8);
    render();
  });
  el("zoom-out")?.addEventListener("click", () => {
    zoom = Math.max(zoom / 1.4, 0.2);
    render();
  });
  el("fit")?.addEventListener("click", () => {
    zoom = 1;
    render();
  });
  el("compare")?.addEventListener("click", (event) => {
    comparing = !comparing;
    event.currentTarget.setAttribute("aria-pressed", String(comparing));
    const label = el("blend-label");
    if (label) label.hidden = !comparing || sideBySide;
    render();
  });
  el("side-by-side")?.addEventListener("click", (event) => {
    sideBySide = !sideBySide;
    event.currentTarget.setAttribute("aria-pressed", String(sideBySide));
    const label = el("blend-label");
    if (label) label.hidden = !comparing || sideBySide;
    render();
  });
  el("blend")?.addEventListener("input", render);

  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, select, textarea")) return;
    if (event.key === "[") show(index - 1);
    if (event.key === "]") show(index + 1);
  });

  return {
    load(payload) {
      stages = payload.stages;
      manifest = payload.manifest;
      if (!view) view = payload.views.find((name) => manifest.some((e) => e.view === name));
      render();
    },
    setView(name) {
      view = name;
      index = 0;
      render();
    },
    get view() {
      return view;
    },
    get stageId() {
      return current()?.stage_id ?? null;
    },
    toggleOverlay(stageId, on) {
      if (on) overlays.add(stageId);
      else overlays.delete(stageId);
      render();
    },
    refresh: render,
  };
}

// --- capture -------------------------------------------------------------

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

  get("/api/health").then((health) => {
    if (!health) {
      setState(el("station-status"), "backend unreachable", "bad");
      return;
    }
    rotationDegrees = ((health.camera_rotation % 360) + 360) % 360;
    setState(
      el("station-status"),
      [
        health.calibrated ? "lens corrected" : "lens distortion not corrected",
        health.weights ? "models loaded" : "model weights missing from models/",
      ].join(" · "),
      health.weights ? "ok" : "bad",
    );
    el("consent-version").textContent = `consent text ${health.consent_version}`;
    el("height-hint").textContent = health.height_hint;
    for (const [id, options] of [
      ["scale-mode", health.scale_modes],
      ["profile", health.profiles],
    ]) {
      el(id).replaceChildren(
        ...options.map((name) => {
          const option = document.createElement("option");
          option.value = name;
          option.textContent = name.replace(/_/g, " ");
          return option;
        }),
      );
    }
    el("profile").value = "full";
  });

  // The browser is asked for a portrait stream, so the camera needs no physical
  // rotation and the frame arrives upright. A camera that ignores the request and
  // hands back a landscape stream still falls back to the configured rotation.
  function isUpright() {
    return video.videoHeight > video.videoWidth;
  }

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

  async function countDown(label, seconds) {
    cue = { label, until: performance.now() + seconds * 1000 };
    await wait(seconds * 1000);
    cue = null;
  }

  function viewName(item) {
    return item.querySelector(".view-name").textContent;
  }

  async function captureAll() {
    for (const [index, item] of [...page.querySelectorAll("li[data-view]")].entries()) {
      const name = viewName(item);
      if (index) await countDown(`next: ${name}`, MOVE_SECONDS);
      await countDown(`${name} — hold still`, HOLD_SECONDS);
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
    const ready = passed.has("front") && (passed.has("side_left") || passed.has("side_right"));
    el("go-process").disabled = !ready;
    setState(
      el("capture-progress"),
      `${passed.size} of 3 accepted${ready ? " — enough to process" : " — front and one side needed"}`,
      ready ? "ok" : "wait",
    );
  }

  async function sendView(view, blob, upright) {
    const item = row(view);
    const output = item.querySelector("output");
    const data = new FormData();
    data.append("image", blob, `${view}.jpg`);
    // A live capture arrives from a portrait-mounted camera and has to be turned;
    // an upload or a stored sample is a photograph and must not be.
    data.append("upright", String(Boolean(upright)));
    item.querySelector('[data-action="retake"]').hidden = false;
    const body = await post(`/api/sessions/${sessionId}/views/${view}`, data, output);
    if (!body) return;
    if (body.ok) {
      passed.add(view);
      setState(output, "accepted", "ok");
    } else {
      passed.delete(view);
      setState(output, reasonText(body), "bad");
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
        profile: el("profile").value,
        scale_mode: el("scale-mode").value,
      }),
      el("session-out"),
    );
    if (!body) return;
    sessionId = body.id;
    setState(el("session-out"), `session ${sessionId}`, "ok");
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
      setState(el("station-status"), `no camera: ${error.message} — upload images instead`, "bad");
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
      setState(item.querySelector("output"), "get ready…", "wait");
      try {
        await countDown(`${viewName(item)} — hold still`, HOLD_SECONDS);
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
      item.querySelector('[data-action="retake"]').hidden = true;
      refreshButtons();
    });
  }

  el("go-process").addEventListener("click", () => {
    window.location.href = `/process/${sessionId}`;
  });

  refreshButtons();
  offerSamples();
  drawPreview();
}

// --- processing ----------------------------------------------------------

function initProcess() {
  const page = el("process");
  if (!page) return;

  const sessionId = sessionIdFromPath();
  let cancelled = false;
  el("session-name").textContent = sessionId;
  el("to-review").href = `/review/${sessionId}`;

  const viewer = createViewer({ sessionId });

  async function refresh() {
    const payload = await get(`/api/sessions/${sessionId}/artifacts`);
    if (!payload) return;
    const select = el("view-select");
    const withStages = payload.views.filter((view) =>
      payload.manifest.some((entry) => entry.view === view),
    );
    if (select.options.length !== withStages.length) {
      select.replaceChildren(
        ...withStages.map((view) => {
          const option = document.createElement("option");
          option.value = view;
          option.textContent = view.replace(/_/g, " ");
          return option;
        }),
      );
      if (withStages.length) viewer.setView(withStages[0]);
    }
    viewer.load(payload);

    const status = await get(`/api/sessions/${sessionId}/process/status`);
    if (!status) return;
    const views = Object.entries(status.views);
    const pending = views.filter(([, info]) => info.stages_missing.length);
    setState(
      el("process-status"),
      status.measured
        ? `measured · ${views.length} view(s)`
        : `${views.length} view(s) captured · ${pending.length} not measured yet`,
      status.measured ? "ok" : "wait",
    );
  }

  async function run() {
    cancelled = false;
    const error = el("process-error");
    error.hidden = true;
    setState(el("process-status"), "processing…", "wait");
    const payload = await post(`/api/sessions/${sessionId}/process`, new FormData());
    if (cancelled) return;
    if (!payload) {
      // The stage that failed is the first one missing; everything before it stays.
      const status = await get(`/api/sessions/${sessionId}/process/status`);
      const failed = Object.entries(status?.views ?? {})
        .map(([view, info]) => `${view}: ${info.stages_missing[0] ?? "complete"}`)
        .join(" · ");
      error.hidden = false;
      error.textContent = `Processing failed. First missing stage per view — ${failed}. Earlier stages are still viewable.`;
      setState(el("process-status"), "failed", "bad");
      await refresh();
      return;
    }
    await refresh();
    setState(el("process-status"), "measured — open Review", "ok");
  }

  el("view-select").addEventListener("change", (event) => viewer.setView(event.target.value));
  el("run").addEventListener("click", run);
  el("retry").addEventListener("click", run);
  el("cancel").addEventListener("click", () => {
    cancelled = true;
    setState(el("process-status"), "cancelled — nothing already written was removed", "wait");
  });

  refresh();
}

// --- review --------------------------------------------------------------

const CORE = ["shoulder_length", "arm_length", "chest", "bust", "waist", "hips"];
const GROUPS = {
  Torso: [
    "neck_w", "shoulder_w", "shoulder_incl", "armscye_depth", "bust", "underbust", "waist",
    "back_width", "waist_back_width", "waist_line", "waist_over_bust_line", "bust_line",
    "vert_bust_line", "bust_points", "head_l",
  ],
  Arms: ["arm_length", "arm_pose_angle", "wrist"],
  "Lower body": [
    "hips", "hip_back_width", "hips_line", "leg_circ", "crotch_hip_diff", "hip_inclination",
    "bum_points", "height",
  ],
};
const FILTERS = {
  core: CORE,
  torso: GROUPS.Torso,
  arm: GROUPS.Arms,
  lower: GROUPS["Lower body"],
};

function initReview() {
  const page = el("review");
  if (!page) return;

  const sessionId = sessionIdFromPath();
  let landmarks = [];
  let automatic = {}; // name -> the value before any operator edit, for comparison
  let measurements = {};
  let primaryKeys = [];
  let filter = "all";
  let selected = null;
  let dragging = null;
  const history = [];
  let undone = [];
  const shown = new Set(["landmarks", "names", "rows"]);

  el("session-name").textContent = sessionId;
  el("to-process").href = `/process/${sessionId}`;
  el("download-body").href = `/api/sessions/${sessionId}/export/body`;
  el("download-extended").href = `/api/sessions/${sessionId}/export/extended`;
  el("download-report").href = `/api/sessions/${sessionId}/export/report`;
  el("download-csv").href = `/api/sessions/${sessionId}/export/csv`;

  const isRow = (mark) => mark.name.endsWith("_row");

  const viewer = createViewer({
    sessionId,
    onStage: (wrapper, record) => drawLandmarks(wrapper, record),
  });

  // --- the landmark layer -------------------------------------------------

  function drawLandmarks(wrapper, record) {
    const canvas = document.createElement("canvas");
    canvas.width = record.width;
    canvas.height = record.height;
    Object.assign(canvas.style, {
      position: "absolute",
      inset: "0",
      width: "100%",
      height: "auto",
    });
    wrapper.append(canvas);
    paint(canvas);
    attachDragging(canvas);
  }

  function paint(canvas) {
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!shown.has("landmarks")) return;
    const scale = canvas.width / 900; // keep marks legible at any capture size
    for (const mark of landmarks.filter((item) => item.view === viewer.view)) {
      if (isRow(mark) && !shown.has("rows")) continue;
      const chosen = selected && selected.name === mark.name && selected.view === mark.view;
      const colour = mark.auto ? "rgba(255, 190, 0, 0.95)" : "rgba(235, 60, 220, 0.95)";
      context.strokeStyle = chosen ? "rgba(90, 220, 255, 0.98)" : colour;
      context.fillStyle = context.strokeStyle;
      context.lineWidth = 3 * scale;
      context.font = `${Math.round(20 * scale)}px system-ui, sans-serif`;
      if (isRow(mark)) {
        context.beginPath();
        context.moveTo(0, mark.y);
        context.lineTo(canvas.width, mark.y);
        context.stroke();
      } else {
        context.beginPath();
        context.arc(mark.x, mark.y, 9 * scale, 0, Math.PI * 2);
        context.stroke();
      }
      if (shown.has("names")) {
        const label = shown.has("values") ? `${mark.name} (${Math.round(mark.x)}, ${Math.round(mark.y)})` : mark.name;
        context.fillText(label, isRow(mark) ? 12 * scale : mark.x + 14 * scale, mark.y - 10 * scale);
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

  function nearest(at, canvas) {
    const tolerance = (40 / 900) * canvas.width;
    let best = null;
    for (const mark of landmarks.filter((item) => item.view === viewer.view)) {
      if (isRow(mark) && !shown.has("rows")) continue;
      const distance = isRow(mark)
        ? Math.abs(mark.y - at.y)
        : Math.hypot(mark.x - at.x, mark.y - at.y);
      if (distance < tolerance && (!best || distance < best.distance)) best = { mark, distance };
    }
    return best ? best.mark : null;
  }

  function attachDragging(canvas) {
    canvas.addEventListener("pointerdown", (event) => {
      const mark = nearest(pointIn(canvas, event), canvas);
      if (!mark) return;
      history.push(structuredClone(landmarks));
      undone = [];
      selected = { name: mark.name, view: mark.view };
      el("landmark-select").value = mark.name;
      dragging = { name: mark.name, view: mark.view };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      const at = pointIn(canvas, event);
      moveTo(dragging, at.x, at.y);
      paint(canvas);
    });
    canvas.addEventListener("pointerup", async () => {
      if (!dragging) return;
      dragging = null;
      await saveLandmarks();
    });
  }

  function moveTo(target, x, y) {
    landmarks = landmarks.map((mark) =>
      mark.name === target.name && mark.view === target.view
        ? { ...mark, x: isRow(mark) ? mark.x : x, y, auto: false }
        : mark,
    );
  }

  function nudge(dx, dy) {
    if (!selected) return;
    const mark = landmarks.find(
      (item) => item.name === selected.name && item.view === selected.view,
    );
    if (!mark) return;
    history.push(structuredClone(landmarks));
    undone = [];
    moveTo(selected, mark.x + dx, mark.y + dy);
    viewer.refresh();
    saveLandmarks();
  }

  async function saveLandmarks() {
    const payload = await request("PUT", `/api/sessions/${sessionId}/landmarks`, {
      body: landmarks,
      output: el("review-status"),
      waiting: "recomputing…",
    });
    if (payload) render(payload);
  }

  // --- the numbers --------------------------------------------------------

  function cell(content, className) {
    const td = document.createElement("td");
    if (className) td.className = className;
    if (content instanceof HTMLElement) td.append(content);
    else td.textContent = content;
    return td;
  }

  function chip(text, kind) {
    const span = document.createElement("span");
    span.className = `chip ${kind ?? text}`;
    span.textContent = text;
    return span;
  }

  function shows(name, entry) {
    if (filter === "all") return true;
    if (filter === "warning") return entry.warnings.length > 0;
    if (filter === "missing") return entry.value === null;
    if (filter === "mesh") return entry.source === "mesh_visual_hull";
    return (FILTERS[filter] ?? []).includes(name);
  }

  function renderCore() {
    const host = el("core");
    host.replaceChildren();
    for (const name of CORE) {
      const entry = measurements[name];
      if (!entry) continue;
      const card = document.createElement("div");
      card.className = "core-card";
      const title = document.createElement("h3");
      title.textContent = name.replace(/_/g, " ");
      const big = document.createElement("p");
      big.className = "big";
      big.textContent = entry.value === null ? "—" : `${entry.value} ${entry.unit}`;
      card.append(title, big);

      if (entry.value === null) {
        const why = document.createElement("p");
        why.className = "pair";
        why.textContent = entry.warnings.join(" · ") || entry.status;
        card.append(why);
      }
      for (const side of ["left", "right"]) {
        const pair = measurements[`${name}_${side}`];
        if (!pair) continue;
        const line = document.createElement("p");
        line.className = "pair";
        line.textContent = `${side}: ${pair.value ?? "—"} ${pair.unit}`;
        card.append(line);
      }
      const inputs = entry.inputs ?? {};
      if (inputs.front_breadth_mm !== undefined) {
        const line = document.createElement("p");
        line.className = "pair";
        line.textContent =
          `front breadth ${inputs.front_breadth_mm} mm · side depth ${inputs.side_depth_mm} mm · ` +
          `row ${inputs.row_height_mm} mm above the floor · ${entry.formula}`;
        card.append(line);
      }
      const views = document.createElement("p");
      views.className = "pair";
      views.textContent = `views: ${entry.views.join(", ") || "none"} · ${entry.status}`;
      card.append(views);
      host.append(card);
    }
  }

  function renderGroups() {
    const host = el("groups");
    host.replaceChildren();
    for (const [group, names] of Object.entries(GROUPS)) {
      const details = document.createElement("details");
      details.className = "panel";
      details.open = true;
      const summary = document.createElement("summary");
      summary.textContent = group;
      details.append(summary);
      const table = document.createElement("table");
      const body = document.createElement("tbody");
      for (const name of names) {
        const entry = measurements[name];
        if (!entry || !primaryKeys.includes(name)) continue;
        const row = document.createElement("tr");
        if (entry.value === null) row.className = "missing";
        row.append(
          cell(name.replace(/_/g, " ")),
          cell(entry.value === null ? "—" : String(entry.value), "value"),
          cell(entry.unit),
          cell(chip(entry.status)),
        );
        if (entry.value === null) {
          const why = document.createElement("tr");
          const td = cell(entry.warnings.join(" · ") || `${entry.status}: no value`);
          td.colSpan = 4;
          td.className = "hint";
          why.append(td);
          body.append(row, why);
          continue;
        }
        body.append(row);
      }
      table.append(body);
      details.append(table);
      host.append(details);
    }
  }

  function renderTable() {
    const body = el("measurements").querySelector("tbody");
    body.replaceChildren();
    for (const [name, entry] of Object.entries(measurements)) {
      if (!shows(name, entry)) continue;
      const row = document.createElement("tr");
      if (entry.value === null) row.classList.add("missing");
      else if (entry.confidence === "estimated" || entry.confidence === "low") {
        row.classList.add("flagged");
      }
      if (CORE.includes(name)) row.classList.add("core");

      const field = document.createElement("input");
      field.type = "number";
      field.step = "0.1";
      field.value = entry.value ?? "";
      field.setAttribute("aria-label", `${name} value in ${entry.unit}`);
      field.addEventListener("change", async () => {
        const note = window.prompt(`Why are you setting ${name} to ${field.value} ${entry.unit}?`);
        if (!note || !note.trim()) {
          field.value = entry.value ?? "";
          setState(el("review-status"), "a typed value needs a note: nothing changed", "bad");
          return;
        }
        const payload = await request("PUT", `/api/sessions/${sessionId}/measurements/${name}`, {
          body: { value: Number(field.value), note },
          output: el("review-status"),
          waiting: "recomputing…",
        });
        if (payload) render(payload);
      });

      row.append(
        cell(name.replace(/_/g, " ")),
        cell(automatic[name] === undefined || automatic[name] === null ? "—" : String(automatic[name]), "value"),
        cell(field),
        cell(entry.unit),
        cell(chip(entry.status)),
        cell(chip(entry.confidence)),
        cell(entry.method.replace(/_/g, " ")),
      );
      row.addEventListener("click", () => renderDrawer(name, entry));
      body.append(row);
    }
  }

  function renderExtended() {
    const body = el("extended").querySelector("tbody");
    body.replaceChildren();
    for (const [name, entry] of Object.entries(measurements)) {
      if (primaryKeys.includes(name)) continue;
      const row = document.createElement("tr");
      row.append(
        cell(name.replace(/_/g, " ")),
        cell(entry.value === null ? "—" : String(entry.value), "value"),
        cell(entry.unit),
        cell(chip(entry.status)),
      );
      body.append(row);
    }
  }

  function renderDrawer(name, entry) {
    const host = el("drawer");
    host.replaceChildren();
    const title = document.createElement("h3");
    title.textContent = name.replace(/_/g, " ");
    const list = document.createElement("dl");
    list.className = "pairs";
    const pairs = [
      ["Automatic value", automatic[name] ?? "—"],
      ["Final value", entry.value === null ? "—" : `${entry.value} ${entry.unit}`],
      ["Status", entry.status],
      ["Source", entry.source],
      ["Method", entry.method || "—"],
      ["Formula", entry.formula || "—"],
      ["Views", entry.views.join(", ") || "none"],
      ["Confidence", entry.confidence],
      ["Validated against manual measurement", "no — none collected yet"],
      ["Edited by an operator", entry.edited ? "yes" : "no"],
      ...Object.entries(entry.inputs ?? {}).map(([key, value]) => [key, String(value)]),
    ];
    if (entry.note) pairs.push(["Note", entry.note]);
    if (entry.warnings.length) pairs.push(["Warnings", entry.warnings.join(" · ")]);
    for (const [key, value] of pairs) {
      const dt = document.createElement("dt");
      dt.textContent = key;
      const dd = document.createElement("dd");
      dd.textContent = String(value);
      list.append(dt, dd);
    }
    host.append(title, list);
  }

  function renderViewButtons() {
    const host = el("view-buttons");
    const views = [...new Set(landmarks.map((mark) => mark.view))];
    host.replaceChildren();
    for (const view of views) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = view.replace(/_/g, " ");
      button.setAttribute("aria-pressed", String(view === viewer.view));
      button.addEventListener("click", () => {
        viewer.setView(view);
        renderViewButtons();
        renderLandmarkSelect();
      });
      host.append(button);
    }
  }

  function renderLandmarkSelect() {
    const select = el("landmark-select");
    const previous = select.value;
    select.replaceChildren(
      ...landmarks
        .filter((mark) => mark.view === viewer.view)
        .map((mark) => {
          const option = document.createElement("option");
          option.value = mark.name;
          option.textContent = `${mark.name}${mark.auto ? "" : " (edited)"}`;
          return option;
        }),
    );
    if ([...select.options].some((option) => option.value === previous)) select.value = previous;
    selected = select.value ? { name: select.value, view: viewer.view } : null;
  }

  function render(payload) {
    landmarks = payload.landmarks;
    measurements = payload.measurements;
    primaryKeys = payload.primary_keys;
    for (const [name, entry] of Object.entries(measurements)) {
      if (!entry.edited && automatic[name] === undefined) automatic[name] = entry.value;
    }
    const warnings = el("warnings");
    warnings.hidden = payload.warnings.length === 0;
    warnings.textContent = payload.warnings.join(" · ");

    const gate = el("export-gate");
    gate.hidden = false;
    gate.className = `banner ${payload.export.ok ? "good" : "warn"}`;
    gate.textContent = payload.export.ok
      ? `Pattern-ready export allowed for the ${payload.export.profile} profile.`
      : `Pattern-ready export is blocked for the ${payload.export.profile} profile — ` +
        Object.entries(payload.export.blocked)
          .map(([name, why]) => `${name}: ${why}`)
          .join(" · ");

    renderCore();
    renderGroups();
    renderTable();
    renderExtended();
    renderViewButtons();
    renderLandmarkSelect();
    viewer.refresh();
    setState(
      el("review-status"),
      `${Object.keys(measurements).length} measurements`,
      payload.export.ok ? "ok" : "wait",
    );
  }

  // --- wiring -------------------------------------------------------------

  for (const button of page.querySelectorAll("[data-filter]")) {
    button.addEventListener("click", () => {
      filter = button.dataset.filter;
      for (const other of page.querySelectorAll("[data-filter]")) {
        other.setAttribute("aria-pressed", String(other === button));
      }
      renderTable();
    });
  }

  const OVERLAY_STAGES = {
    mask: "07_mask_clean",
    contour: "09_edges",
    keypoints: "10_keypoints",
  };
  for (const button of page.querySelectorAll("[data-overlay]")) {
    button.addEventListener("click", () => {
      const key = button.dataset.overlay;
      const on = button.getAttribute("aria-pressed") !== "true";
      button.setAttribute("aria-pressed", String(on));
      if (OVERLAY_STAGES[key]) {
        viewer.toggleOverlay(OVERLAY_STAGES[key], on);
        return;
      }
      if (on) shown.add(key);
      else shown.delete(key);
      viewer.refresh();
    });
  }

  el("landmark-select").addEventListener("change", (event) => {
    selected = { name: event.target.value, view: viewer.view };
    viewer.refresh();
  });
  el("nudge-up").addEventListener("click", () => nudge(0, -1));
  el("nudge-down").addEventListener("click", () => nudge(0, 1));
  el("nudge-left").addEventListener("click", () => nudge(-1, 0));
  el("nudge-right").addEventListener("click", () => nudge(1, 0));

  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, textarea")) return;
    const step = event.shiftKey ? 10 : 1;
    const moves = {
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
    };
    if (moves[event.key] && selected) {
      event.preventDefault();
      nudge(...moves[event.key]);
    }
  });

  el("undo").addEventListener("click", async () => {
    if (!history.length) return;
    undone.push(structuredClone(landmarks));
    landmarks = history.pop();
    await saveLandmarks();
  });
  el("redo").addEventListener("click", async () => {
    if (!undone.length) return;
    history.push(structuredClone(landmarks));
    landmarks = undone.pop();
    await saveLandmarks();
  });
  el("reset-landmarks").addEventListener("click", async () => {
    if (!window.confirm("Throw away every landmark edit and typed value?")) return;
    const payload = await post(
      `/api/sessions/${sessionId}/landmarks/reset`,
      new FormData(),
      el("review-status"),
    );
    if (payload) {
      automatic = {};
      render(payload);
    }
  });

  el("confirm-session").addEventListener("click", async () => {
    const body = await post(
      `/api/sessions/${sessionId}/confirm`,
      new FormData(),
      el("confirm-out"),
    );
    if (body) {
      setState(
        el("confirm-out"),
        `confirmed · photographs deleted after ${body.images_purge_after}`,
        "ok",
      );
    }
  });

  el("delete-session").addEventListener("click", async () => {
    if (!window.confirm(`Delete ${sessionId} and every photograph in it?`)) return;
    const response = await fetch(`/api/sessions/${sessionId}`, { method: "DELETE" });
    if (response.ok) window.location.href = "/sessions.html";
  });

  get(`/api/sessions/${sessionId}/artifacts`).then((payload) => payload && viewer.load(payload));
  post(`/api/sessions/${sessionId}/process`, new FormData(), el("review-status")).then(
    (payload) => payload && render(payload),
  );
}

// --- sessions ------------------------------------------------------------

function initSessions() {
  const page = el("sessions");
  if (!page) return;

  async function refresh() {
    const status = el("status-filter").value;
    const rows = await get(`/api/sessions${status ? `?status=${status}` : ""}`);
    if (!rows) return;
    const list = el("session-list");
    list.replaceChildren();
    for (const row of rows) {
      const item = document.createElement("li");
      const left = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = row.subject_code;
      const meta = document.createElement("p");
      meta.className = "meta";
      meta.textContent = `${row.id} · ${row.status} · ${row.height_cm} cm · created ${row.created_at}`;
      left.append(title, meta);

      const right = document.createElement("div");
      right.className = "row";
      for (const [label, href] of [
        ["Processing", `/process/${row.id}`],
        ["Review", `/review/${row.id}`],
      ]) {
        const link = document.createElement("a");
        link.href = href;
        link.textContent = label;
        right.append(link);
      }
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "danger";
      remove.textContent = "Delete";
      remove.addEventListener("click", async () => {
        if (!window.confirm(`Delete ${row.subject_code} and every photograph in it?`)) return;
        await fetch(`/api/sessions/${row.id}`, { method: "DELETE" });
        refresh();
      });
      right.append(remove);
      item.append(left, right);
      list.append(item);
    }
    setState(el("sessions-status"), `${rows.length} session(s)`, "ok");
  }

  el("status-filter").addEventListener("change", refresh);
  refresh();
}

initTheme();
initCapture();
initProcess();
initReview();
initSessions();
