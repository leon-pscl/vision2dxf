import { useMemo, useState } from 'react'
import { Badge, Button, Callout } from '../components/ui'
import type { Review } from '../types'

export function MeasurementReview({
  review,
  confirmations,
  onCorrections,
}: {
  review: Review
  confirmations: Record<string, string>
  onCorrections: (corrections: Record<string, number>) => void
}) {
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
  const commit = () => {
    if (overrides === 0) return
    onCorrections({ ...edited })
    setEdited({})
  }

  return (
    <div className="stack">
      <header className="row spread">
        <h2>Review measurements</h2>
        <Badge tone="ok">{review.rows.length} values imported</Badge>
      </header>
      <p className="muted">
        Source: <code>{review.filename}</code> · unit <strong>{review.unit}</strong>. These are the values as
        the file reports them. Meaning is decided on the previous screen, never here.
      </p>

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
              <th scope="col">Confirmed meaning</th>
              <th scope="col">Normalized (mm)</th>
              <th scope="col">Original capture</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const confirmed = Boolean(confirmations[row.name])
              return (
                <tr key={row.name} className={confirmed ? undefined : 'row-warn'}>
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
                      value={edited[row.name] ?? row.value}
                      onChange={(e) => setEdited((p) => ({ ...p, [row.name]: Number(e.target.value) }))}
                    />
                  </td>
                  <td className="interpret">
                    {confirmed ? 'Interpretation confirmed' : 'Not confirmed yet'}
                    <small>{row.landmark}</small>
                  </td>
                  <td>{(edited[row.name] ?? row.valueMm).toFixed(1)}</td>
                  <td className="interpret">{row.interpretation}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {overrides > 0 ? (
        <Callout tone="warn" title={`${overrides} corrected value(s) not yet applied`}>
          Corrections change this subject&apos;s numbers only. They are recorded in the generation manifest.
          <div className="row">
            <Button variant="primary" onClick={commit}>
              Apply corrections
            </Button>
            <Button variant="ghost" onClick={() => setEdited({})}>
              Discard
            </Button>
          </div>
        </Callout>
      ) : null}
    </div>
  )
}
