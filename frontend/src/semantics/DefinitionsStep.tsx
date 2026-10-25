import { useCallback, useMemo, useState } from 'react'
import { api, saveJson, type FieldResolution, type GenerationPayload, type SemanticsResponse } from '../api'
import { Badge, Button, Callout } from '../components/ui'
import { canGenerate } from './gate'

const STATUS_TONE: Record<string, 'ok' | 'warn' | 'error' | 'neutral'> = {
  CONFIRMED: 'ok',
  DERIVED: 'ok',
  ASSUMED: 'warn',
  OPTIONAL: 'neutral',
  MISSING: 'error',
  AMBIGUOUS: 'error',
  INVALID: 'error',
}

const STATUS_HINT: Record<string, string> = {
  CONFIRMED: 'Pinned by the schema. May enter drafting equations.',
  DERIVED: 'Computed by a recorded equation.',
  ASSUMED: 'Operator confirmed an unpinned definition.',
  OPTIONAL: 'Not required by the selected garment.',
  MISSING: 'Required by the selected garment and absent.',
  AMBIGUOUS: 'Definition not confirmed. Generation is blocked.',
  INVALID: 'Value rejected by the schema. Generation is blocked.',
}

export function DefinitionsStep({
  input,
  onChange,
  onBack,
  onReady,
}: {
  input: GenerationPayload
  onChange: (input: GenerationPayload) => void
  onBack: () => void
  onReady: (response: SemanticsResponse) => void
}) {
  const [state, setState] = useState<SemanticsResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [openField, setOpenField] = useState<string | null>(null)
  const [onlyUnresolved, setOnlyUnresolved] = useState(false)

  const load = useCallback(
    async (next: GenerationPayload) => {
      setBusy(true)
      setError(null)
      try {
        const response = await api.semantics(next)
        setState(response)
        if (canGenerate(response.source.fields)) onReady(response)
        return response
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Semantics resolution failed')
        return null
      } finally {
        setBusy(false)
      }
    },
    [onReady],
  )

  const apply = (next: GenerationPayload) => {
    onChange(next)
    void load(next)
  }

  const rows = useMemo(() => {
    if (!state) return []
    return state.source.fields.filter((f) =>
      onlyUnresolved ? ['AMBIGUOUS', 'MISSING', 'INVALID'].includes(f.status) : true,
    )
  }, [state, onlyUnresolved])

  const open = state ? canGenerate(state.source.fields) : false
  const ambiguous = state?.source?.fields?.filter((f) => f.status === 'AMBIGUOUS') ?? []

  const confirmAll = () => {
    const confirmations: Record<string, string> = { ...input.confirmations }
    for (const f of ambiguous) {
      const def = state?.registry.fields.find((d) => d.name === f.name)
      if (def && !confirmations[f.name]) confirmations[f.name] = def.candidates[0].id
    }
    apply({ ...input, confirmations })
  }

  return (
    <div className="stack">
      <header className="row spread">
        <h2>Measurement definitions</h2>
        <div className="row">
          {state ? (
            <Badge tone={open ? 'ok' : 'error'}>
              {open ? 'READY TO GENERATE' : `${state.source.blocking.length} BLOCKING`}
            </Badge>
          ) : null}
          <Badge tone="neutral">schema v{state?.registry.schemaVersion ?? '—'}</Badge>
        </div>
      </header>

      <p className="muted">
        Field meanings come from the schema configuration, not from their names. An operator may confirm one
        of the offered interpretations or correct a numeric value; meaning itself is changed by editing{' '}
        <code>config/measurement-semantics.yaml</code>.
      </p>

      {!state ? (
        <Button variant="primary" disabled={busy} onClick={() => void load(input)}>
          {busy ? 'Resolving…' : 'Resolve field definitions'}
        </Button>
      ) : null}
      {error ? <Callout tone="error">{error}</Callout> : null}

      {state ? (
        <>
          <Callout tone={open ? 'ok' : 'warn'} title="Coordinate convention">
            Vertical origin <strong>{state.registry.coordinateSystem.verticalOrigin}</strong>, direction{' '}
            <strong>{state.registry.coordinateSystem.verticalDirection}</strong>; source unit{' '}
            <strong>{state.registry.coordinateSystem.unit}</strong>, pattern unit{' '}
            <strong>{state.registry.coordinateSystem.patternUnit}</strong>. Pattern frame y = 0 at{' '}
            <strong>{state.registry.coordinateSystem.patternFrame.y_zero_landmark}</strong>.
          </Callout>

          {state.source.blocking.length > 0 ? (
            <Callout tone="error" title="Generation is blocked">
              <ul className="tight">
                {state.source.blocking.map((b, i) => (
                  <li key={i}>
                    <code>{b.field ?? b.code}</code> [{b.code}]: {b.message}
                    {b.suggestedAction ? <em className="muted"> — {b.suggestedAction}</em> : null}
                  </li>
                ))}
              </ul>
            </Callout>
          ) : null}

          {state.source.assumptions.length > 0 ? (
            <Callout tone="warn" title={`${state.source.assumptions.length} assumption(s) recorded`}>
              <ul className="tight">
                {state.source.assumptions.map((a, i) => (
                  <li key={i}>
                    <code>{a.field}</code>: {a.message}
                  </li>
                ))}
              </ul>
            </Callout>
          ) : null}

          <div className="row">
            <Button variant="primary" disabled={busy || ambiguous.length === 0} onClick={confirmAll}>
              Confirm all offered definitions ({ambiguous.length})
            </Button>
            <Button onClick={() => void load(input)} disabled={busy}>
              Re-resolve
            </Button>
            <Button
              onClick={() => {
                saveJson('measurement-report.json', state.source)
              }}
            >
              Download validation report
            </Button>
            <Button variant="ghost" onClick={onBack}>
              Return to measurement capture
            </Button>
          </div>

          <label className="toggle">
            <input
              type="checkbox"
              checked={onlyUnresolved}
              onChange={(e) => setOnlyUnresolved(e.target.checked)}
            />
            <span>Show only unresolved fields</span>
          </label>

          <div className="table-scroll">
            <table className="definitions">
              <thead>
                <tr>
                  <th scope="col">Field</th>
                  <th scope="col">Value</th>
                  <th scope="col">Unit</th>
                  <th scope="col">Interpreted meaning</th>
                  <th scope="col">Status</th>
                  <th scope="col">Actions</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((f) => (
                  <DefinitionRow
                    key={f.name}
                    field={f}
                    open={openField === f.name}
                    onToggle={() => setOpenField(openField === f.name ? null : f.name)}
                    onConfirm={(candidateId) =>
                      apply({ ...input, confirmations: { ...input.confirmations, [f.name]: candidateId } })
                    }
                    onCorrect={(value) =>
                      apply({ ...input, corrections: { ...input.corrections, [f.name]: value } })
                    }
                    candidates={state.registry.fields.find((d) => d.name === f.name)?.candidates ?? []}
                  />
                ))}
              </tbody>
            </table>
          </div>

          {state.draftingMeasurements ? (
            <details>
              <summary>Drafting input adapter ({Object.keys(state.draftingMeasurements).length} variables)</summary>
              <table className="rules">
                <thead>
                  <tr>
                    <th scope="col">Drafting variable</th>
                    <th scope="col">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(state.draftingMeasurements).map(([k, v]) => (
                    <tr key={k}>
                      <th scope="row">
                        <code>{k}</code>
                      </th>
                      <td>{v === null ? '—' : `${v} mm`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted">
                These are the only measurement names a drafting equation can see. Source field names never
                reach the drafting engine.
              </p>
            </details>
          ) : null}

          <details>
            <summary>Schema consistency rules</summary>
            <ul className="tight">
              {state.registry.consistencyRules.map((r) => (
                <li key={r.id}>
                  <Badge tone={r.severity === 'blocking' ? 'error' : 'warn'}>{r.severity}</Badge>{' '}
                  <code>{r.id}</code>: {r.description}
                </li>
              ))}
            </ul>
          </details>
        </>
      ) : null}
    </div>
  )
}

function DefinitionRow({
  field,
  candidates,
  open,
  onToggle,
  onConfirm,
  onCorrect,
}: {
  field: FieldResolution
  candidates: { id: string; meaning: string; representation: string | null; scale: number; startLandmark: string | null; endLandmark: string | null; origin: string | null; draftingVariable: string }[]
  open: boolean
  onToggle: () => void
  onConfirm: (candidateId: string) => void
  onCorrect: (value: number) => void
}) {
  const [draft, setDraft] = useState<string>(String(field.sourceValue ?? ''))
  return (
    <>
      <tr className={field.status === 'INVALID' ? 'row-error' : undefined}>
        <th scope="row">
          {field.label}
          <small>
            <code>{field.name}</code>
          </small>
        </th>
        <td>
          <input
            type="number"
            step="0.1"
            aria-label={`${field.label} value`}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={() => {
              const value = Number(draft)
              if (!Number.isNaN(value) && value !== field.sourceValue) onCorrect(value)
            }}
          />
          {field.corrected ? <small>from {field.originalValue}</small> : null}
        </td>
        <td>{field.unitSource}</td>
        <td className="interpret">
          {field.status === 'AMBIGUOUS' ? 'Definition not confirmed' : field.meaning}
          <small>
            {field.draftingVariable ? <code>{field.draftingVariable}</code> : null}
            {field.normalizedMm !== null && field.status !== 'AMBIGUOUS' ? (
              <> → {field.normalizedMm} mm</>
            ) : null}
          </small>
        </td>
        <td>
          <Badge tone={STATUS_TONE[field.status]}>{field.status}</Badge>
          <small className="muted">{field.confirmedBy ? `by ${field.confirmedBy}` : ''}</small>
        </td>
        <td>
          <div className="row">
            <Button variant="ghost" onClick={onToggle}>
              {open ? 'Hide' : 'View'}
            </Button>
            {field.status === 'AMBIGUOUS' && candidates.length > 0 ? (
              <Button variant="primary" onClick={() => onConfirm(candidates[0].id)}>
                Confirm
              </Button>
            ) : null}
          </div>
        </td>
      </tr>
      {open ? (
        <tr className="definition-detail">
          <td colSpan={6}>
            <p>
              <strong>Schema definition.</strong> {field.message ?? ''}
            </p>
            <p className="muted">{STATUS_HINT[field.status]}</p>
            <ul className="tight">
              {candidates.map((c) => (
                <li key={c.id}>
                  <label className="toggle">
                    <input
                      type="radio"
                      name={`cand-${field.name}`}
                      checked={field.candidateId === c.id}
                      onChange={() => onConfirm(c.id)}
                      disabled={field.status === 'INVALID'}
                    />
                    <span>
                      <strong>{c.id}</strong> — {c.meaning}{' '}
                      {c.representation ? <Badge tone="neutral">{c.representation}</Badge> : null}{' '}
                      {c.origin ? <code>origin: {c.origin}</code> : null}{' '}
                      {c.startLandmark && c.endLandmark ? (
                        <code>
                          {c.startLandmark} → {c.endLandmark}
                        </code>
                      ) : null}{' '}
                      {c.scale !== 1 ? <code>scale ×{c.scale}</code> : null}
                    </span>
                  </label>
                </li>
              ))}
            </ul>
            <p className="muted">
              Conversion applied: <code>{field.conversion}</code>
            </p>
          </td>
        </tr>
      ) : null}
    </>
  )
}
