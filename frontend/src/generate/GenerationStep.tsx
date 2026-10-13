import { Badge, Callout } from '../components/ui'
import { formatMm } from '../pattern-viewer/viewer-math'
import type { GenerationResponse } from '../types'

export function GenerationStep({ result }: { result: GenerationResponse }) {
  const diagnostics = result.diagnostics
  const residuals = Object.values(diagnostics.crossSectionResiduals)
  const worstResidual = Math.max(0, ...residuals)

  return (
    <div className="stack">
      <header className="row spread">
        <div>
          <h2>Surface reconstruction and flattening</h2>
          <p className="muted">
            {result.trace.method} · configuration v{diagnostics.configVersion} · software v
            {result.trace.software_version} · generated {new Date(result.trace.generated_at).toLocaleString()}
          </p>
        </div>
        <Badge tone={result.validation.status === 'pass' ? 'ok' : 'error'}>
          {result.validation.status === 'pass' ? 'VALIDATED' : 'CHECKS FAILED'}
        </Badge>
      </header>

      <div className="stat-row">
        <div className="stat">
          <strong>{diagnostics.panels.length} flattened panels</strong>
          <span>{diagnostics.method}</span>
          <span>{diagnostics.panels.every((panel) => panel.solver.converged) ? 'All solvers converged' : 'A solver did not converge'}</span>
        </div>
        <div className="stat">
          <strong>Cross-section residual</strong>
          <span>Maximum {formatMm(worstResidual, 4)}</span>
          <span>Limit {formatMm(result.crossSections?.residualToleranceMm ?? 0.5, 3)}</span>
        </div>
        <div className="stat">
          <strong>Flattening quality</strong>
          <span>{diagnostics.maxFlippedTriangles} flipped triangles</span>
          <span>Mean angular error ≤ {diagnostics.maxMeanAngularDistortionDeg.toFixed(3)}°</span>
        </div>
        <div className="stat">
          <strong>Geometry validation</strong>
          <span>{result.validation.summary.passed}/{result.validation.summary.total} checks passed</span>
          <span>{result.validation.summary.criticalFailures} critical failures</span>
        </div>
      </div>

      <h3>Pipeline progress</h3>
      <ol className="tight">
        {result.progress.map((stage) => (
          <li key={stage.stage}>
            <code>{stage.stage}</code> — {stage.status}, {stage.elapsedMs.toFixed(1)} ms
          </li>
        ))}
      </ol>

      <h3>Panel flattening diagnostics</h3>
      <div className="table-scroll">
        <table className="rules">
          <thead>
            <tr>
              <th scope="col">Panel</th>
              <th scope="col">Converged</th>
              <th scope="col">Iterations</th>
              <th scope="col">Mean angular</th>
              <th scope="col">Max edge-length</th>
              <th scope="col">Boundary crossings</th>
            </tr>
          </thead>
          <tbody>
            {diagnostics.panels.map((panel) => (
              <tr key={panel.panel}>
                <th scope="row">{panel.panel}</th>
                <td>{panel.solver.converged ? 'Yes' : 'No'}</td>
                <td>{panel.solver.iterations}</td>
                <td>{Number(panel.distortion.meanAngularDistortionDeg ?? 0).toFixed(3)}°</td>
                <td>{Number(panel.distortion.maxEdgeLengthDistortion ?? 0).toFixed(5)}</td>
                <td>{Number(panel.distortion.boundarySelfIntersections ?? 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {result.assumptions.length > 0 ? (
        <>
          <h3>Recorded geometric assumptions</h3>
          <ul className="tight">
            {result.assumptions.map((assumption) => (
              <li key={assumption.id}>
                <strong>{assumption.name}</strong>: <code>{String(assumption.value)}{assumption.unit ? ` ${assumption.unit}` : ''}</code>
                {' — '}{assumption.reason}
                {assumption.sourceFields.length > 0 ? (
                  <span className="muted"> Source fields: {assumption.sourceFields.join(', ')}.</span>
                ) : null}
              </li>
            ))}
          </ul>
        </>
      ) : null}

      {result.trace.warnings.length > 0 ? (
        <Callout tone="warn" title="Measurement warnings">
          <ul className="tight">
            {result.trace.warnings.map((warning, index) => (
              <li key={`${warning.code}-${index}`}><code>{warning.field}</code>: {warning.message}</li>
            ))}
          </ul>
        </Callout>
      ) : null}
    </div>
  )
}
