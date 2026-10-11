import { useMemo, useState } from 'react'
import { Badge, Button, Callout } from '../components/ui'
import type { MeasurementRow, Review } from '../types'

export function MeasurementReview({ review }: { review: Review }) {
  const [query, setQuery] = useState('')
  const [edited, setEdited] = useState<Record<string, number>>({})

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return review.rows.filter(
      (row) =>
        !needle ||
        row.label.toLowerCase().includes(needle) ||
        row.name.includes(needle) ||
        row.landmark.toLowerCase().includes(needle),
    )
  }, [review, query])

  const overrides = Object.keys(edited).length

  return (
    <div className="stack">
      <header className="row spread">
        <h2>Confirm measurements</h2>
        <Badge tone="ok">{review.rows.length} values imported</Badge>
      </header>
      <p className="muted">
        Source: <code>{review.filename}</code> · unit <strong>{review.unit}</strong>. Geometry is drafted in
        millimetres; original centimetre values are kept separately and never overwritten.
      </p>

      {review.warnings.length > 0 ? (
        <Callout tone="warn" title={`${review.warnings.length} value warning(s) - nothing was changed`}>
          <ul className="tight">
            {review.warnings.map((w, i) => (
              <li key={i}>
                <code>{w.field ?? w.code}</code>: {w.message}
              </li>
            ))}
          </ul>
        </Callout>
      ) : null}

      <label className="field">
        <span className="field-label">Filter</span>
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="bust, waist, neck…" />
      </label>

      <div className="table-scroll">
        <table className="measurements">
          <thead>
            <tr>
              <th scope="col">Field</th>
              <th scope="col">Value ({review.unit})</th>
              <th scope="col">Drafted (mm)</th>
              <th scope="col">Origin</th>
              <th scope="col">Interpretation</th>
              <th scope="col">Range</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <Row key={row.name} row={row} value={edited[row.name]} onChange={(v) => setEdited((p) => ({ ...p, [row.name]: v }))} />
            ))}
          </tbody>
        </table>
      </div>

      {overrides > 0 ? (
        <Callout tone="warn" title={`${overrides} local override(s)`}>
          Edited values are only used for this session and are reported as assumptions in the manifest.
          <div className="row">
            <Button variant="ghost" onClick={() => setEdited({})}>Reset overrides</Button>
          </div>
        </Callout>
      ) : null}

      <details>
        <summary>Measurement dictionary checks</summary>
        <ul className="tight">
          {review.dictionary.checks.map((c) => (
            <li key={c.id}>
              <code>{c.id}</code>: {c.description}
            </li>
          ))}
        </ul>
      </details>
    </div>
  )
}

function Row({
  row,
  value,
  onChange,
}: {
  row: MeasurementRow
  value: number | undefined
  onChange: (value: number) => void
}) {
  const shown = value ?? row.value
  const outOfRange = row.min !== null && row.max !== null && (shown < row.min || shown > row.max)
  return (
    <tr className={outOfRange ? 'row-warn' : undefined}>
      <th scope="row">
        {row.label}
        <small>
          <code>{row.name}</code>
        </small>
      </th>
      <td>
        <input
          type="number"
          step="0.1"
          aria-label={`${row.label} value`}
          value={shown}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      </td>
      <td>{(shown * 10).toFixed(1)}</td>
      <td>
        <code>{row.origin}</code>
        <small>{row.direction}</small>
      </td>
      <td className="interpret">{row.interpretation}</td>
      <td className="nowrap">
        {row.min ?? '–'} … {row.max ?? '–'}
        {outOfRange ? <Badge tone="warn">outside</Badge> : null}
      </td>
    </tr>
  )
}
