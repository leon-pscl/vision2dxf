import { useState } from "react";
import type { DrapeResult } from "../types";

interface Props {
  run: <T>(step: number, payload?: Record<string, unknown>, backend?: string) => Promise<T>;
  busy: boolean;
  result: DrapeResult | null;
}

export function Step10Drape({ run, busy, result }: Props) {
  const [ticked, setTicked] = useState<Record<number, boolean>>({});

  return (
    <div className="content">
      <div className="pane">
        <div className="pane-main">
          {result ? (
            <img
              className="preview-img"
              src={`data:image/png;base64,${result.placeholder_image}`}
              alt="drape placeholder"
            />
          ) : (
            <p className="notice">Run the step to get the drape placeholder and the toile checklist.</p>
          )}
        </div>

        <div className="pane-side" style={{ flex: 1 }}>
          <div className="btn-row" style={{ marginBottom: 12 }}>
            <button className="btn" disabled={busy} onClick={() => run(10)}>
              {busy ? "Running…" : "Prepare toile"}
            </button>
          </div>

          {result && (
            <>
              <p className="label" style={{ marginBottom: 6 }}>
                Toile checklist
              </p>
              <ul className="toile-list">
                {result.toile_checklist.map((c, i) => (
                  <li key={c}>
                    <input
                      id={`toile-${i}`}
                      type="checkbox"
                      checked={Boolean(ticked[i])}
                      onChange={(e) => setTicked((p) => ({ ...p, [i]: e.target.checked }))}
                    />
                    <label htmlFor={`toile-${i}`}>{c}</label>
                  </li>
                ))}
              </ul>

              <p className="notice">
                {Object.values(ticked).filter(Boolean).length} of {result.toile_checklist.length}{" "}
                checked.
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
