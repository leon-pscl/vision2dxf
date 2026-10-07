import { useState } from "react";
import { SilhouetteCanvas } from "../components/SilhouetteCanvas";
import { GARMENTS, type GarmentSpec, type GarmentType } from "../types";

interface Props {
  run: <T>(step: number, payload?: Record<string, unknown>, backend?: string) => Promise<T>;
  busy: boolean;
  spec: GarmentSpec | null;
  current: GarmentType;
  onPick: (g: GarmentType) => void;
}

export function Step01Garment({ run, busy, spec, current, onPick }: Props) {
  const [showSilhouette, setShowSilhouette] = useState(true);

  const choose = async (g: GarmentType) => {
    onPick(g);
    await run(1, { garment_type: g });
  };

  return (
    <div className="pane">
      <div className="pane-side">
        <div className="field">
          <span className="label">Garment</span>
          {GARMENTS.map((g) => (
            <button
              key={g.type}
              className={`btn ${g.type === current ? "" : "btn-ghost"}`}
              disabled={busy}
              onClick={() => choose(g.type)}
              style={{ justifyContent: "flex-start", textAlign: "left", marginBottom: 6 }}
            >
              <span>
                {g.label}
                <br />
                <small style={{ opacity: 0.7, textTransform: "none", letterSpacing: 0 }}>
                  {g.note}
                </small>
              </span>
            </button>
          ))}
        </div>

        {spec && (
          <>
            <hr className="rule" />
            <p className="label" style={{ margin: "12px 0 6px" }}>
              Required measurements
            </p>
            <ul className="checklist">
              {spec.required_measurements.map((m) => (
                <li key={m}>
                  <span style={{ flex: 1 }}>{m.replace(/_/g, " ")}</span>
                  {spec.ease_cm[m] !== undefined && (
                    <span style={{ color: "var(--muted)" }}>+{spec.ease_cm[m]} cm ease</span>
                  )}
                </li>
              ))}
            </ul>

            <p className="label" style={{ margin: "16px 0 6px" }}>
              Design rules
            </p>
            <ul className="checklist">
              {Object.entries(spec.design_rules).map(([k, v]) => (
                <li key={k}>
                  <span style={{ flex: 1, color: "var(--muted)" }}>{k.replace(/_/g, " ")}</span>
                  <span style={{ textAlign: "right" }}>{String(v)}</span>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>

      <div className="pane-main">
        <div className="switch-row">
          <span className="label">Body silhouette</span>
          <button
            className={`switch ${showSilhouette ? "" : "off"}`}
            onClick={() => setShowSilhouette((v) => !v)}
            aria-pressed={showSilhouette}
            aria-label="Toggle body silhouette"
          />
        </div>
        <div className="canvas">
          {showSilhouette && <SilhouetteCanvas garment={current} />}
        </div>
      </div>
    </div>
  );
}


