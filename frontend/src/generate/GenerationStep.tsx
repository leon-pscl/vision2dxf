import { Badge, Callout } from '../components/ui'
import { formatMm } from '../pattern-viewer/viewer-math'
import type { TransformResponse } from '../types'

export function GenerationStep({ result }: { result: TransformResponse }) {
  const { trace, diagnostics } = result
  const solvers = Object.values(diagnostics.solver)
  const worstResidual = Math.max(...solvers.map((s) => s.maxResidualMm))
  const rounds = diagnostics.seamBalanceRounds
  const settled = rounds.length === 0 || rounds[rounds.length - 1].settled

  return (
    <div className="stack">
      <header className="row spread">
        <h2>Transformation trace</h2>
        <Badge tone={settled ? 'ok' : 'warn'}>
          {solvers.length} panel{solvers.length === 1 ? '' : 's'} solved
        </Badge>
      </header>

      <p className="muted">
        Method <code>{trace.method}</code> · base block{' '}
        <code>{trace.base_block.id}</code> v{trace.base_block.version}
        {trace.base_block.reviewed ? '' : ' (not reviewed)'} · configuration v
        {trace.transformation_config.version} · influence map v
        {trace.transformation_config.influence_map_version} · software{' '}
        {trace.software_version} · {new Date(trace.generated_at).toLocaleString()}
      </p>
      <p className="muted">
        Source template <code>{trace.base_block.source_file}</code>, pinned at sha256{' '}
        {trace.base_block.source_sha256.slice(0, 12)}… · units {trace.units.source} →{' '}
        {trace.units.pattern}
      </p>

      <div className="stat-row">
        {solvers.map((solver) => (
          <div key={solver.panel} className="stat">
            <strong>{solver.panel}</strong>
            <span>{solver.landmarkCount} driver landmarks</span>
            <span>{solver.converged ? 'converged' : 'did not converge'}</span>
            <span>residual {solver.maxResidualMm.toExponential(2)} mm</span>
            <span className="muted small">condition {solver.conditionNumber.toFixed(0)}</span>
          </div>
        ))}
        <div className="stat">
          <strong>Seam balancing</strong>
          <span>{rounds.length} round{rounds.length === 1 ? '' : 's'}</span>
          <span>{settled ? 'settled' : 'stopped at the limit'}</span>
        </div>
        <div className="stat">
          <strong>Curve reconstruction</strong>
          <span>{diagnostics.curveRefit.refits.length} edges rebuilt</span>
          <span>{diagnostics.curveRefit.issues.length} issues</span>
        </div>
      </div>

      {!solvers.every((s) => s.converged) ? (
        <Callout tone="error" title="Solver did not converge">
          The deformation is not usable. The last iterate is never returned as a pattern.
        </Callout>
      ) : null}

      {diagnostics.warnings.length > 0 ? (
        <Callout tone="warn" title="Reported, not silently applied">
          <ul className="tight">
            {diagnostics.warnings.map((w, i) => (
              <li key={i}>
                <code>{w.field ?? w.code}</code>: {w.message}
              </li>
            ))}
          </ul>
        </Callout>
      ) : null}

      <details>
        <summary>Objective and how each term is enforced</summary>
        <table className="rules">
          <thead>
            <tr>
              <th scope="col">Term</th>
              <th scope="col">Weight</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(trace.transformation_config.objective)
              .filter(([key]) => key.startsWith('w_'))
              .map(([key, value]) => (
                <tr key={key}>
                  <th scope="row"><code>{key}</code></th>
                  <td className="nowrap">{String(value)}</td>
                </tr>
              ))}
          </tbody>
        </table>
        <table className="rules">
          <thead>
            <tr>
              <th scope="col">Term</th>
              <th scope="col">Realised by</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(
              (trace.transformation_config.objective.realisation ?? {}) as Record<string, string>,
            ).map(([term, how]) => (
              <tr key={term}>
                <th scope="row"><code>{term}</code></th>
                <td className="interpret">{how}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>

      <details>
        <summary>Field interpretations used ({Object.keys(trace.field_interpretations).length})</summary>
        <table className="rules">
          <thead>
            <tr>
              <th scope="col">Source field</th>
              <th scope="col">Canonical variable</th>
              <th scope="col">Status</th>
              <th scope="col">Confirmed by</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(trace.field_interpretations).map(([field, i]) => (
              <tr key={field}>
                <th scope="row"><code>{field}</code></th>
                <td><code>{i.meaning}</code></td>
                <td>{i.status}</td>
                <td>{i.confirmed_by}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>

      <details>
        <summary>Differences applied ({diagnostics.deltas.filter((d) => d.deltaValue).length})</summary>
        <div className="table-scroll">
          <table className="rules">
            <thead>
              <tr>
                <th scope="col">Measurement</th>
                <th scope="col">Difference</th>
                <th scope="col">Region moved</th>
              </tr>
            </thead>
            <tbody>
              {diagnostics.deltas
                .filter((d) => Math.abs(d.deltaValue) > 1e-9)
                .map((d) => (
                  <tr key={d.name}>
                    <th scope="row"><code>{d.name}</code></th>
                    <td className="nowrap">
                      {d.deltaValue > 0 ? '+' : ''}{d.deltaValue.toFixed(1)} {d.unit}
                    </td>
                    <td className="interpret">
                      {d.mapped ? d.region : <span className="muted">not mapped · {d.region}</span>}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </details>

      <details>
        <summary>Rebuilt edges ({diagnostics.curveRefit.refits.length})</summary>
        <div className="table-scroll">
          <table className="rules">
            <thead>
              <tr>
                <th scope="col">Edge</th>
                <th scope="col">Role</th>
                <th scope="col">Method</th>
                <th scope="col">Points</th>
                <th scope="col">Length</th>
                <th scope="col">Change</th>
              </tr>
            </thead>
            <tbody>
              {diagnostics.curveRefit.refits.map((refit) => (
                <tr key={refit.edgeId}>
                  <th scope="row"><code>{refit.edgeId.slice(0, 8)}</code></th>
                  <td><code>{refit.role}</code></td>
                  <td>{refit.method}</td>
                  <td className="nowrap">{refit.points}</td>
                  <td className="nowrap">{formatMm(refit.lengthMm)}</td>
                  <td className="nowrap">
                    {refit.lengthDeltaMm > 0 ? '+' : ''}{refit.lengthDeltaMm.toFixed(2)} mm
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      <p className="muted">
        Worst landmark residual {worstResidual.toExponential(2)} mm against a tolerance of{' '}
        <code>{String(trace.transformation_config.solver.landmark_residual_tol_mm)}</code> mm.
      </p>
    </div>
  )
}
