import { useState } from "react";
import type { MeasurementBackend, MeasurementResult } from "../types";

interface Props {
  run: <T>(step: number, payload?: Record<string, unknown>, backend?: string) => Promise<T>;
  busy: boolean;
  result: MeasurementResult | null;
}

const BACKENDS: { key: MeasurementBackend; label: string }[] = [
  { key: "regression", label: "Regression" },
  { key: "body_model", label: "Body model" },
];

/** Interval drawn as a bar, with the value as a tick in the middle. */
function Interval({ lo, hi, value }: { lo: number | null; hi: number | null; value: number }) {
  if (lo === null || hi === null) {
    return (
      <div className="interval" title="no interval, backend is uncalibrated">
        <div className="interval-fill" style={{ left: 0, right: 0, background: "var(--line)" }} />
      </div>
    );
  }
  const pad = Math.max((hi - lo) * 0.25, 0.5);
  const a = lo - pad;
  const b = hi + pad;
  const pct = (v: number) => `${((v - a) / (b - a)) * 100}%`;

  return (
    <div className="interval" title={`${lo.toFixed(1)} to ${hi.toFixed(1)} cm`}>
      <div
        className="interval-fill"
        style={{ left: pct(lo), width: `${((hi - lo) / (b - a)) * 100}%` }}
      />
      <div className="interval-mid" style={{ left: pct(value) }} />
    </div>
  );
}

export function Step06Measurement({ run, busy, result }: Props) {
  const [backend, setBackend] = useState<MeasurementBackend>("regression");

  const go = async (b: MeasurementBackend) => {
    setBackend(b);
    await run(6, {}, b);
  };

  const rows = result ? Object.entries(result.measurements) : [];

  return (
    <div className="content">
      <div className="stat-row" style={{ marginBottom: 16 }}>
        <div className="toggle-group" role="group" aria-label="Measurement backend">
          {BACKENDS.map((b) => (
            <button
              key={b.key}
              className={backend === b.key ? "on" : ""}
              disabled={busy}
              onClick={() => go(b.key)}
            >
              {b.label}
            </button>
          ))}
        </div>
        <div className="stat">
          <span className="label-muted">Backend</span>
          <b>{result?.backend ?? backend}</b>
        </div>
        <div className="stat">
          <span className="label-muted">Measurements</span>
          <b>{rows.length}</b>
        </div>
        <div className="btn-row" style={{ marginLeft: "auto" }}>
          <button className="btn" disabled={busy} onClick={() => go(backend)}>
            {busy ? "Measuring…" : "Measure"}
          </button>
        </div>
      </div>

      {rows.length === 0 ? (
        <p className="notice">Run the measurement to see values and intervals.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Measurement</th>
              <th>Value (cm)</th>
              <th>Interval (cm)</th>
              <th style={{ width: "38%" }}>Range</th>
              <th>Method</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, m]) => (
              <tr key={name}>
                <td>{name.replace(/_/g, " ")}</td>
                <td className="num">{m.value_cm.toFixed(1)}</td>
                <td className="num">
                  {m.lower_cm === null || m.upper_cm === null
                    ? "—"
                    : `${m.lower_cm.toFixed(1)} – ${m.upper_cm.toFixed(1)}`}
                </td>
                <td>
                  <Interval lo={m.lower_cm} hi={m.upper_cm} value={m.value_cm} />
                </td>
                <td style={{ color: "var(--muted)", fontSize: 11 }}>{m.method}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {result && result.warnings.length > 0 && (
        <ul className="guidance" style={{ marginTop: 12 }}>
          {result.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
