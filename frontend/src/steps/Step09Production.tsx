import { PatternGrid } from "../components/PatternGrid";
import type { ProductionResult } from "../types";

interface Props {
  run: <T>(step: number, payload?: Record<string, unknown>, backend?: string) => Promise<T>;
  busy: boolean;
  result: ProductionResult | null;
}

export function Step09Production({ run, busy, result }: Props) {
  return (
    <div className="content">
      <div className="stat-row" style={{ marginBottom: 14 }}>
        <div className="stat">
          <span className="label-muted">Pieces</span>
          <b>{result?.pieces.length ?? 0}</b>
        </div>
        <div className="stat">
          <span className="label-muted">Export</span>
          <b>{result?.export_formats.join(", ") ?? "—"}</b>
        </div>
        <div className="btn-row" style={{ marginLeft: "auto" }}>
          <button className="btn" disabled={busy} onClick={() => run(9)}>
            {busy ? "Building…" : "Add production details"}
          </button>
        </div>
      </div>

      {result ? (
        <>
          <PatternGrid pieces={result.pieces} showNotches />
          <p className="notice">
            Dashed outline is the seam allowance. Blue ticks are notches, one pair per matched seam.
          </p>
        </>
      ) : (
        <p className="notice">Run the step to add seam allowance and notches.</p>
      )}
    </div>
  );
}
