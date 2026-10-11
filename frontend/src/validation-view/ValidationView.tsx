import { Badge, Callout } from '../components/ui'
import { formatMm } from '../pattern-viewer/viewer-math'
import type { ValidationReport } from '../types'

export function ValidationView({ report }: { report: ValidationReport }) {
  const failed = report.checks.filter((c) => c.status === 'fail')
  const passed = report.checks.filter((c) => c.status === 'pass')

  return (
    <div className="stack">
      <header className="row spread">
        <h2>Geometry validation</h2>
        <Badge tone={report.status === 'pass' ? 'ok' : 'error'}>
          {report.status === 'pass' ? 'PASS' : 'BLOCKED'}
        </Badge>
      </header>

      <Callout tone={report.status === 'pass' ? 'ok' : 'error'}>
        {report.summary.passed}/{report.summary.total} checks passed
        {report.summary.criticalFailures > 0
          ? ` - export is blocked by ${report.summary.criticalFailures} critical failure(s).`
          : ' - export is enabled.'}
      </Callout>

      <h3>Tolerances</h3>
      <ul className="tight mono">
        {Object.entries(report.tolerances).map(([k, v]) => (
          <li key={k}>
            {k.replace(/_/g, ' ')}: <strong>{v}</strong>
          </li>
        ))}
      </ul>

      {failed.length > 0 ? (
        <>
          <h3>Failures</h3>
          {failed.map((c) => (
            <Callout key={c.id} tone="error" title={c.id}>
              <p>{c.message}</p>
              {c.suggestedAction ? <p className="muted">Suggested: {c.suggestedAction}</p> : null}
              {c.affectedIds.length > 0 ? <p className="mono tight">{c.affectedIds.join(', ')}</p> : null}
            </Callout>
          ))}
        </>
      ) : null}

      <h3>Metrics</h3>
      <ul className="tight mono">
        {Object.entries(report.metrics).map(([k, v]) => (
          <li key={k}>
            {k}: <strong>{typeof v === 'number' ? formatMm(v, 3) : JSON.stringify(v)}</strong>
          </li>
        ))}
      </ul>

      <details open={failed.length === 0}>
        <summary>All checks ({passed.length} passed)</summary>
        <table className="rules">
          <thead>
            <tr>
              <th scope="col">Check</th>
              <th scope="col">Status</th>
              <th scope="col">Severity</th>
              <th scope="col">Message</th>
            </tr>
          </thead>
          <tbody>
            {report.checks.map((c) => (
              <tr key={c.id}>
                <th scope="row">
                  <code>{c.id}</code>
                </th>
                <td>{c.status === 'pass' ? 'PASS' : 'FAIL'}</td>
                <td>{c.severity}</td>
                <td className="interpret">{c.message}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  )
}
