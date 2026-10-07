import { exportYamlUrl } from "../api";
import type { SessionState, ValidationResult } from "../types";

interface Props {
  run: <T>(step: number, payload?: Record<string, unknown>, backend?: string) => Promise<T>;
  busy: boolean;
  result: ValidationResult | null;
  sid: string | null;
  state: SessionState | null;
}

export function Step07Validation({ run, busy, result, sid, state }: Props) {
  const ready = result !== null;

  return (
    <div className="content">
      <div className="btn-row" style={{ marginBottom: 14 }}>
        <button className="btn" disabled={busy} onClick={() => run(7)}>
          {busy ? "Validating…" : "Validate"}
        </button>
        <a
          className={`btn ${ready ? "" : "btn-ghost"}`}
          href={sid ? exportYamlUrl(sid) : "#"}
          aria-disabled={!ready}
          onClick={(e) => {
            if (!ready) e.preventDefault();
          }}
          style={ready ? undefined : { opacity: 0.4, pointerEvents: "none" }}
        >
          Download YAML
        </a>
      </div>

      {!ready ? (
        <p className="notice">Run the validation to see flags and the YAML payload.</p>
      ) : (
        <>
          <p className="label" style={{ marginBottom: 6 }}>
            Flags
          </p>
          <ul className="flags">
            {result.flags.map((f, i) => (
              <li key={`${f.rule}-${f.measurement}-${i}`}>
                <span className={`sev ${f.severity}`}>{f.severity}</span>
                <span style={{ minWidth: 90 }}>{f.measurement.replace(/_/g, " ")}</span>
                <span style={{ color: "var(--muted)" }}>{f.rule}</span>
                <span style={{ flex: 1 }}>{f.detail}</span>
              </li>
            ))}
          </ul>

          <p className="label" style={{ margin: "16px 0 6px" }}>
            YAML preview
          </p>
          <pre className="yaml">{result.yaml}</pre>

          {state?.any_mock && (
            <p className="notice">This payload contains mock measurements.</p>
          )}
        </>
      )}
    </div>
  );
}
