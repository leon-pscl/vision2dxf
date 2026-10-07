import { useMemo, useState } from "react";
import { Step01Garment } from "./steps/Step01Garment";
import { Step02Capture } from "./steps/Step02Capture";
import { Step03Segmentation } from "./steps/Step03Segmentation";
import { Step04Calibration } from "./steps/Step04Calibration";
import { Step05Landmarks } from "./steps/Step05Landmarks";
import { Step06Measurement } from "./steps/Step06Measurement";
import { Step07Validation } from "./steps/Step07Validation";
import { Step08Draft } from "./steps/Step08Draft";
import { Step09Production } from "./steps/Step09Production";
import { Step10Drape } from "./steps/Step10Drape";
import { isLocked, STEPS } from "./steps";
import { useSession } from "./useSession";
import type {
  CalibrationResult,
  CaptureResult,
  DraftResult,
  DrapeResult,
  GarmentSpec,
  GarmentType,
  LandmarkResult,
  MeasurementResult,
  ProductionResult,
  SegmentationResult,
  ValidationResult,
} from "./types";

export default function App() {
  const sess = useSession();
  const [step, setStep] = useState(1);
  const [garment, setGarment] = useState<GarmentType>("polo_shirt");
  const [images, setImages] = useState({ front: "", side: "" });
  const [s3view, setS3view] = useState<"sewing" | "view3d">("sewing");
  const [s1tab, setS1tab] = useState<"design" | "body">("design");

  const r = sess.results;
  const confirmed = sess.state?.landmarks_confirmed ?? false;
  const busy = sess.busy > 0;

  const typed = useMemo(
    () => ({
      spec: (r["1"] as unknown as GarmentSpec) ?? null,
      capture: (r["2"] as unknown as CaptureResult) ?? null,
      seg: (r["3"] as unknown as SegmentationResult) ?? null,
      calib: (r["4"] as unknown as CalibrationResult) ?? null,
      lm: (r["5"] as unknown as LandmarkResult) ?? null,
      meas: (r["6"] as unknown as MeasurementResult) ?? null,
      val: (r["7"] as unknown as ValidationResult) ?? null,
      draft: (r["8"] as unknown as DraftResult) ?? null,
      prod: (r["9"] as unknown as ProductionResult) ?? null,
      drape: (r["10"] as unknown as DrapeResult) ?? null,
    }),
    [r],
  );

  const stepView = () => {
    switch (step) {
      case 1:
        return (
          <Step01Garment
            run={sess.run}
            busy={busy}
            spec={typed.spec}
            current={garment}
            onPick={setGarment}
          />
        );
      case 2:
        return (
          <Step02Capture
            run={sess.run}
            busy={busy}
            result={typed.capture}
            images={images}
            setImages={setImages}
          />
        );
      case 3:
        return <Step03Segmentation run={sess.run} busy={busy} result={typed.seg} front={images.front} />;
      case 4:
        return <Step04Calibration run={sess.run} busy={busy} result={typed.calib} />;
      case 5:
        return (
          <Step05Landmarks
            run={sess.run}
            busy={busy}
            result={typed.lm}
            confirmed={confirmed}
            front={images.front}
            onUpdated={() => void sess.reload()}
          />
        );
      case 6:
        return <Step06Measurement run={sess.run} busy={busy} result={typed.meas} />;
      case 7:
        return (
          <Step07Validation
            run={sess.run}
            busy={busy}
            result={typed.val}
            sid={sess.sid}
            state={sess.state}
          />
        );
      case 8:
        return <Step08Draft run={sess.run} busy={busy} result={typed.draft} />;
      case 9:
        return <Step09Production run={sess.run} busy={busy} result={typed.prod} />;
      case 10:
        return <Step10Drape run={sess.run} busy={busy} result={typed.drape} />;
      default:
        return null;
    }
  };

  return (
    <div className="app">
      {sess.state?.any_mock && (
        <div className="mock-banner" role="status">
          Mock data – not real measurements
        </div>
      )}
      {sess.error && (
        <div className="error-banner" role="alert">
          {sess.error} <button className="btn btn-sm" onClick={sess.clearError}>Dismiss</button>
        </div>
      )}

      <div className="workspace">
        <nav className="stepper" aria-label="Pipeline steps">
          {STEPS.map((s) => {
            const st = sess.status(s.n);
            const locked = isLocked(s.n, confirmed, r);
            return (
              <button
                key={s.n}
                className={`step-item ${step === s.n ? "current" : ""}`}
                onClick={() => setStep(s.n)}
                disabled={locked}
                title={locked ? "Confirm the landmarks in step 5 first" : undefined}
                aria-current={step === s.n ? "step" : undefined}
              >
                <span className="step-num">{String(s.n).padStart(2, "0")}</span>
                <span className="step-name">{s.title}</span>
                {locked ? (
                  <span className="badge">locked</span>
                ) : st !== "pending" ? (
                  <span className={`badge ${st}`}>{st}</span>
                ) : null}
              </button>
            );
          })}
        </nav>

        <main className="main">
          <div className="headbar">
            <div className="tabs" role="tablist">
              <button
                role="tab"
                aria-selected={s1tab === "design"}
                className={`tab ${s1tab === "design" ? "active" : ""}`}
                onClick={() => setS1tab("design")}
              >
                Design parameters
              </button>
              <button
                role="tab"
                aria-selected={s1tab === "body"}
                className={`tab ${s1tab === "body" ? "active" : ""}`}
                onClick={() => setS1tab("body")}
              >
                Body parameters
              </button>
            </div>

            <div className="tabs" role="tablist">
              <button
                role="tab"
                aria-selected={s3view === "sewing"}
                className={`tab ${s3view === "sewing" ? "active" : ""}`}
                onClick={() => setS3view("sewing")}
              >
                Sewing pattern
              </button>
              <button
                role="tab"
                aria-selected={s3view === "view3d"}
                className={`tab ${s3view === "view3d" ? "active" : ""}`}
                onClick={() => setS3view("view3d")}
                title="Phase 3. The 3D view is not built yet."
              >
                3D view
              </button>
            </div>
          </div>

          {s1tab === "body" ? (
            <div className="content">
              <p className="label" style={{ marginBottom: 8 }}>
                Body parameters
              </p>
              <table style={{ maxWidth: 420 }}>
                <thead>
                  <tr>
                    <th>Measurement</th>
                    <th>Value (cm)</th>
                    <th>Interval (cm)</th>
                  </tr>
                </thead>
                <tbody>
                  {typed.meas
                    ? Object.entries(typed.meas.measurements).map(([k, m]) => (
                        <tr key={k}>
                          <td>{k.replace(/_/g, " ")}</td>
                          <td className="num">{m.value_cm.toFixed(1)}</td>
                          <td className="num">
                            {m.lower_cm === null ? "—" : `${m.lower_cm.toFixed(1)} – ${m.upper_cm?.toFixed(1)}`}
                          </td>
                        </tr>
                      ))
                    : (
                        <tr>
                          <td colSpan={3} style={{ color: "var(--muted)" }}>
                            Run step 6 to see body parameters.
                          </td>
                        </tr>
                      )}
                </tbody>
              </table>
            </div>
          ) : s3view === "view3d" ? (
            <div className="content">
              <p className="notice">
                The 3D view is not built in this phase. Switch back to Sewing pattern.
              </p>
            </div>
          ) : (
            stepView()
          )}

          <div className="footer-actions">
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => setStep((s) => Math.max(1, s - 1))}
              disabled={step === 1}
            >
              Back
            </button>
            <button
              className="btn btn-sm"
              onClick={() => setStep((s) => Math.min(10, s + 1))}
              disabled={step === 10 || isLocked(step + 1, confirmed, r)}
            >
              Next
            </button>
          </div>
        </main>
      </div>
    </div>
  );
}
