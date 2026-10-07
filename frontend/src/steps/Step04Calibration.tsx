import type { CalibrationResult } from "../types";

interface Props {
  run: <T>(step: number, payload?: Record<string, unknown>, backend?: string) => Promise<T>;
  busy: boolean;
  result: CalibrationResult | null;
}

export function Step04Calibration({ run, busy, result }: Props) {
  return (
    <div className="content">
      <div className="stat-row" style={{ marginBottom: 16 }}>
        <div className="stat">
          <span className="label-muted">Method</span>
          <b>{result?.method ?? "—"}</b>
        </div>
        <div className="stat">
          <span className="label-muted">Relative uncertainty</span>
          <b>{result ? `${(result.relative_uncertainty * 100).toFixed(2)}%` : "—"}</b>
        </div>
        <div className="btn-row" style={{ marginLeft: "auto" }}>
          <button className="btn" disabled={busy} onClick={() => run(4)}>
            {busy ? "Calibrating…" : "Calibrate"}
          </button>
        </div>
      </div>

      <p className="label" style={{ marginBottom: 6 }}>
        Pixels per cm, by view
      </p>
      {result ? (
        <table style={{ maxWidth: 420 }}>
          <thead>
            <tr>
              <th>View</th>
              <th>px per cm</th>
              <th>1 cm equals</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(result.px_per_cm).map(([view, v]) => (
              <tr key={view}>
                <td>{view}</td>
                <td className="num">{v.toFixed(3)}</td>
                <td className="num" style={{ color: "var(--muted)" }}>
                  {v.toFixed(1)} px
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="notice">Run the calibration to convert pixels to centimetres.</p>
      )}
    </div>
  );
}
