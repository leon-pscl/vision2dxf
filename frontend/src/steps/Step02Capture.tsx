import { useRef, useState } from "react";
import { readFileAsDataUrl } from "../api";
import type { CaptureResult } from "../types";

/**
 * Draw a stand-in body photo on a canvas, so the whole pipeline can be tried
 * without uploading a real photo. Landmarks and masks are positional, so any
 * upright figure works.
 */
function samplePhoto(view: "front" | "side"): string {
  const w = 420;
  const h = 700;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const g = c.getContext("2d");
  if (!g) return "";

  // studio backdrop
  const bg = g.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, "#f3f4f7");
  bg.addColorStop(1, "#e6e8ee");
  g.fillStyle = bg;
  g.fillRect(0, 0, w, h);

  // body: head, torso, arms, legs. Slightly narrower in side view.
  const cx = w / 2;
  const k = view === "side" ? 0.55 : 1;
  const skin = "#d8b49a";
  g.fillStyle = view === "side" ? "#7d8698" : "#5f6a7d";

  // legs
  g.beginPath();
  g.roundRect(cx - 62 * k, 400, 48 * k, 270, 18);
  g.roundRect(cx + 14 * k, 400, 48 * k, 270, 18);
  g.fill();

  // torso
  g.beginPath();
  g.roundRect(cx - 78 * k, 190, 156 * k, 220, 34);
  g.fill();

  // arms
  g.beginPath();
  g.roundRect(cx - 96 * k, 200, 22 * k, 200, 11);
  g.roundRect(cx + 74 * k, 200, 22 * k, 200, 11);
  g.fill();

  // head
  g.fillStyle = skin;
  g.beginPath();
  g.arc(cx, 120, 46 * (view === "side" ? 0.82 : 1), 0, Math.PI * 2);
  g.fill();

  // neck
  g.fillRect(cx - 16, 150, 32, 44);

  // floor line, so the figure reads as standing
  g.fillStyle = "rgba(0,0,0,0.06)";
  g.fillRect(0, 670, w, 30);

  return c.toDataURL("image/png").split(",")[1];
}

interface Props {
  run: <T>(step: number, payload?: Record<string, unknown>, backend?: string) => Promise<T>;
  busy: boolean;
  result: CaptureResult | null;
  images: { front: string; side: string };
  setImages: (v: { front: string; side: string }) => void;
}

const CHECKS: { key: keyof CaptureResult; label: string }[] = [
  { key: "distance_ok", label: "Distance" },
  { key: "pose_ok", label: "A-pose" },
  { key: "full_body_visible", label: "Full body" },
  { key: "clothing_ok", label: "Clothing" },
];

export function Step02Capture({ run, busy, result, images, setImages }: Props) {
  const [height, setHeight] = useState("178");
  const [weight, setWeight] = useState("");
  const webcamRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [webcamOn, setWebcamOn] = useState(false);

  const pick = async (which: "front" | "side", file: Blob) => {
    setImages({ ...images, [which]: await readFileAsDataUrl(file) });
  };

  const toggleWebcam = async () => {
    if (webcamOn) {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      setWebcamOn(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (webcamRef.current) webcamRef.current.srcObject = stream;
      setWebcamOn(true);
    } catch {
      setWebcamOn(false);
    }
  };

  const shoot = (which: "front" | "side") => {
    const v = webcamRef.current;
    if (!v || !v.videoWidth) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")?.drawImage(v, 0, 0);
    c.toBlob(
      async (blob) => {
        if (blob) setImages({ ...images, [which]: await readFileAsDataUrl(blob) });
      },
      "image/jpeg",
      0.92,
    );
  };

  const useSample = () => {
    setImages({ front: samplePhoto("front"), side: samplePhoto("side") });
  };

  const analyse = async () => {
    if (!images.front || !images.side || !height) return;
    await run(2, {
      front_image: images.front,
      side_image: images.side,
      height_cm: Number(height),
      weight_kg: weight ? Number(weight) : null,
    });
  };

  const ready = Boolean(images.front && images.side && height);

  return (
    <div className="content">
      <div className="pane-side" style={{ float: "left", maxWidth: 300 }}>
        <div className="field">
          <span className="label">Height (cm)</span>
          <input value={height} onChange={(e) => setHeight(e.target.value)} inputMode="decimal" />
        </div>
        <div className="field">
          <span className="label">Weight (kg, optional)</span>
          <input value={weight} onChange={(e) => setWeight(e.target.value)} inputMode="decimal" />
        </div>

        <hr className="rule" />
        <p className="label" style={{ margin: "12px 0 6px" }}>
          Capture checks
        </p>
        {result ? (
          <>
            <ul className="checklist">
              {CHECKS.map(({ key, label }) => {
                const ok = Boolean(result[key]);
                return (
                  <li key={key}>
                    <span className={`check ${ok ? "pass" : "fail"}`}>{ok ? "pass" : "fail"}</span>
                    <span>{label}</span>
                  </li>
                );
              })}
            </ul>
            {result.guidance.length > 0 && (
              <ul className="guidance">
                {result.guidance.map((g) => (
                  <li key={g}>{g}</li>
                ))}
              </ul>
            )}
          </>
        ) : (
          <p className="notice">Run the check to see distance, pose, framing and clothing.</p>
        )}
      </div>

      <div style={{ marginLeft: 336 }}>
        <div className="capture-grid">
          {(["front", "side"] as const).map((which) => (
            <div key={which}>
              <p className="label" style={{ marginBottom: 6 }}>
                {which} view
              </p>
              <div className="drop">
                {images[which] ? (
                  <img src={images[which]} alt={`${which} view`} />
                ) : (
                  <p>Drop a file or choose one</p>
                )}
              </div>
              <div className="btn-row" style={{ marginTop: 8 }}>
                <label className="btn btn-ghost btn-sm" style={{ cursor: "pointer" }}>
                  Upload
                  <input
                    type="file"
                    accept="image/*"
                    hidden
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) pick(which, f);
                    }}
                  />
                </label>
                {webcamOn && (
                  <button className="btn btn-ghost btn-sm" onClick={() => shoot(which)}>
                    Capture
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="btn-row" style={{ marginTop: 14 }}>
          <button className="btn" disabled={busy} onClick={useSample}>
            Use sample photos
          </button>
          <button className="btn btn-ghost btn-sm" onClick={toggleWebcam}>
            {webcamOn ? "Stop webcam" : "Use webcam"}
          </button>
          <button className="btn" disabled={!ready || busy} onClick={analyse}>
            {busy ? "Checking…" : "Run capture check"}
          </button>
        </div>

        {webcamOn && (
          <video
            ref={webcamRef}
            autoPlay
            playsInline
            muted
            style={{ marginTop: 12, maxWidth: "100%", border: "1px solid var(--line)" }}
          />
        )}

        <p className="notice">Images stay in memory. Nothing is written to disk.</p>
      </div>
    </div>
  );
}
