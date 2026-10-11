import { useMemo, useState } from 'react'
import { Badge, Callout } from '../components/ui'
import { formatMm } from '../pattern-viewer/viewer-math'
import type { GenerateResponse } from '../types'

export function GenerationStep({ result }: { result: GenerateResponse }) {
  const [filter, setFilter] = useState('')
  const rules = useMemo(() => {
    const needle = filter.trim().toLowerCase()
    return needle
      ? result.trace.rules.filter((r) => r.ruleId.toLowerCase().includes(needle) || r.outputName.toLowerCase().includes(needle))
      : result.trace.rules
  }, [result, filter])

  const panels = result.pattern as { panels: { id: string; label: string; seqEdges: unknown[] }[] }

  return (
    <div className="stack">
      <header className="row spread">
        <h2>Generation trace</h2>
        <Badge tone="ok">{result.trace.rules.length} rules evaluated</Badge>
      </header>
      <p className="muted">
        Method <code>{result.trace.method}</code> · registry v{result.trace.ruleRegistryVersion} · geometry{' '}
        {result.trace.units.geometry} · source <code>{result.trace.sourceFile}</code>
      </p>

      <div className="stat-row">
        {result.preview.panels.map((panel) => (
          <div key={panel.id} className="stat">
            <strong>{panel.label}</strong>
            <span>{panel.edges.length} edges</span>
            <span>{panel.darts.length ? `${panel.darts.length} dart, ${formatMm(panel.darts[0].intakeMm)} intake` : 'no dart'}</span>
          </div>
        ))}
        <div className="stat">
          <strong>Stitches</strong>
          <span>{(panels.panels?.length ?? 0)} panels paired</span>
        </div>
      </div>

      {result.trace.assumptions.length > 0 ? (
        <Callout tone="warn" title="Assumptions and clamps">
          <ul className="tight">
            {result.trace.assumptions.map((a, i) => (
              <li key={i}>
                <code>{a.field ?? a.code}</code>: {a.message}
              </li>
            ))}
          </ul>
        </Callout>
      ) : null}

      <label className="field">
        <span className="field-label">Filter rules</span>
        <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="front_bust_half_width, lm.front…" />
      </label>

      <div className="table-scroll">
        <table className="rules">
          <thead>
            <tr>
              <th scope="col">Rule</th>
              <th scope="col">Output</th>
              <th scope="col">Value</th>
              <th scope="col">Equation</th>
              <th scope="col">Source</th>
            </tr>
          </thead>
          <tbody>
            {rules.map((rule) => (
              <tr key={rule.ruleId}>
                <th scope="row">
                  <code>{rule.ruleId}</code>
                </th>
                <td>{rule.outputName}</td>
                <td className="nowrap">{formatMm(rule.value, 3)}</td>
                <td className="interpret">{rule.explanation}</td>
                <td className="interpret">{rule.sourceNote}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
