import { useState } from "react";
import { PatternGrid } from "../components/PatternGrid";
import type { DraftResult, DraftRoute } from "../types";

interface Props {
  run: <T>(step: number, payload?: Record<string, unknown>, backend?: string) => Promise<T>;
  busy: boolean;
  result: DraftResult | null;
}

const ROUTES: { key: DraftRoute; label: string }[] = [
  { key: "block", label: "Block" },
  { key: "garmentcode", label: "GarmentCode" },
];

export function Step08Draft({ run, busy, result }: Props) {
  const [route, setRoute] = useState<DraftRoute>("block");

  const draft = async (r: DraftRoute) => {
    setRoute(r);
    await run(8, { route: r });
  };

  return (
    <div className="content">
      <div className="stat-row" style={{ marginBottom: 14 }}>
        <div className="toggle-group" role="group" aria-label="Drafting route">
          {ROUTES.map((r) => (
            <button
              key={r.key}
              className={route === r.key ? "on" : ""}
              disabled={busy}
              onClick={() => draft(r.key)}
            >
              {r.label}
            </button>
          ))}
        </div>
        <div className="stat">
          <span className="label-muted">Route</span>
          <b>{result?.route ?? route}</b>
        </div>
        <div className="stat">
          <span className="label-muted">Pieces</span>
          <b>{result?.pieces.length ?? 0}</b>
        </div>
        <div className="btn-row" style={{ marginLeft: "auto" }}>
          <button className="btn" disabled={busy} onClick={() => draft(route)}>
            {busy ? "Drafting…" : "Draft"}
          </button>
        </div>
      </div>

      {result ? <PatternGrid pieces={result.pieces} /> : <p className="notice">Run the drafting to see the pattern pieces.</p>}
    </div>
  );
}
