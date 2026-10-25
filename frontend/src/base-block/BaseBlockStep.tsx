import { useEffect, useState } from 'react'
import { api, RequestError, type GenerationPayload } from '../api'
import { Badge, Button, Callout, Spinner, Toggle } from '../components/ui'
import { formatMm } from '../pattern-viewer/viewer-math'
import type { BaseBlockCatalog, BaseBlockDetail, DeltaResponse, Settings } from '../types'

export function BaseBlockStep({
  input,
  settings,
  onChange,
  onReady,
}: {
  input: GenerationPayload
  settings: Settings
  onChange: (settings: Settings) => void
  onReady: (ready: boolean) => void
}) {
  const [catalog, setCatalog] = useState<BaseBlockCatalog | null>(null)
  const [detail, setDetail] = useState<BaseBlockDetail | null>(null)
  const [deltas, setDeltas] = useState<DeltaResponse | null>(null)
  const [adHoc, setAdHoc] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const selected = settings.base_block_id ?? catalog?.default ?? null

  useEffect(() => {
    api.baseBlocks().then(setCatalog).catch(() => setError('The base-block catalog could not be loaded.'))
  }, [])

  // The block and the differences against it are one view: a block you cannot transform
  // onto this body is not a block you should be able to walk past.
  useEffect(() => {
    if (!selected) return
    let current = true
    setBusy(true)
    setError(null)
    setDetail(null)
    setDeltas(null)
    Promise.all([
      api.baseBlock(selected),
      api.deltas(input, { ...settings, base_block_id: selected }).catch((e) => {
        if (e instanceof RequestError && e.code === 'semantics_unresolved') return null
        throw e
      }),
    ])
      .then(([block, difference]) => {
        if (!current) return
        setDetail(block)
        setDeltas(difference)
      })
      .catch((e) => {
        if (!current) return
        setError(e instanceof RequestError ? e.message : 'The base block could not be loaded.')
      })
      .finally(() => current && setBusy(false))
    return () => {
      current = false
    }
  }, [selected, input, settings.ease_preset, settings.confirm_envelope])

  const envelope = deltas?.envelope
  const blocked =
    envelope?.status === 'stop' || (envelope?.requiresConfirmation === true && !settings.confirm_envelope)

  useEffect(() => {
    onReady(Boolean(detail && detail.validationStatus === 'pass' && deltas && !blocked))
  }, [detail, deltas, blocked, onReady])

  const select = (id: string) => onChange({ ...settings, base_block_id: id })

  return (
    <div className="stack">
      <h2>Base block and differences</h2>
      <p className="muted">
        A base block is a reviewed pattern file plus the semantic layer derived from it when it is
        loaded. Pick the block to transform; the differences below are measured against the body that
        block is declared to fit, not against the block's own garment measurements.
      </p>

      {error ? <Callout tone="error" title="Request failed">{error}</Callout> : null}

      <div className="block-grid">
        {catalog?.blocks?.map((block) => (
          <button
            key={block.id}
            type="button"
            className={`block-card${block.id === selected ? ' selected' : ''}`}
            aria-pressed={block.id === selected}
            onClick={() => select(block.id)}
          >
            <div className="row">
              <strong>{block.label}</strong>
              {block.validationStatus ? (
                <Badge tone={block.validationStatus === 'pass' ? 'ok' : 'error'}>
                  {block.validationStatus}
                </Badge>
              ) : null}
              {block.reviewed ? null : <Badge tone="warn">not reviewed</Badge>}
            </div>
            <dl className="block-facts">
              <div>
                <dt>Panels</dt>
                <dd>{block.panelCount ?? '—'}</dd>
              </div>
              <div>
                <dt>Seams</dt>
                <dd>{block.stitchPairs ?? '—'}</dd>
              </div>
              <div>
                <dt>Darts</dt>
                <dd>{block.hasDarts ? 'yes' : 'none'}</dd>
              </div>
              <div>
                <dt>Bust</dt>
                <dd>
                  {block.referenceMeasurementsMm?.bust_circumference
                    ? formatMm(block.referenceMeasurementsMm.bust_circumference, 0)
                    : '—'}
                </dd>
              </div>
            </dl>
            <small className="muted">{block.error ?? block.note}</small>
          </button>
        ))}
      </div>

      {catalog?.templateSelection?.supported ? (
        <fieldset>
          <legend>Load any template in the library</legend>
          <p className="muted small">{catalog.templateSelection.note}</p>
          <div className="row">
            <input
              aria-label="Template file name"
              placeholder={catalog.templateSelection.idFormat}
              value={adHoc}
              onChange={(e) => setAdHoc(e.target.value)}
            />
            <Button
              onClick={() => select(adHoc.startsWith('template:') ? adHoc : `template:${adHoc}`)}
              disabled={adHoc.trim().length === 0}
            >
              Load template
            </Button>
          </div>
        </fieldset>
      ) : null}

      {busy ? <Spinner label="Loading the block, deriving its semantics, measuring the differences…" /> : null}

      {detail ? (
        <>
          {detail.reviewed ? null : (
            <Callout tone="warn" title="Not reviewed">
              This template has not been reviewed as a base block. Its derived semantics have not been
              checked by a person, and it carries the same validation report as any other block.
            </Callout>
          )}
          {detail?.notes?.map((note) => (
            <Callout key={note.code} tone="warn" title={note.code.replace(/_/g, ' ')}>
              {note.message}
            </Callout>
          ))}

          <div className="stat-row">
            <div className="stat">
              <span>Source file</span>
              <strong className="small">{detail.sourceFile}</strong>
            </div>
            <div className="stat">
              <span>Pinned sha256</span>
              <strong className="small">{detail.sourceSha256.slice(0, 12)}…</strong>
            </div>
            <div className="stat">
              <span>Declared reference body</span>
              <strong className="small">{detail.referenceProfile.sourceFile}</strong>
            </div>
            <div className="stat">
              <span>Declared ease</span>
              <strong className="small">{detail.referenceEasePreset}</strong>
            </div>
          </div>

          <div className="preview-frame" dangerouslySetInnerHTML={{ __html: detail.previewSvg }} />
        </>
      ) : null}

      {deltas ? (
        <>
          <div className="table-scroll">
            <table className="edge-table">
              <caption>
                Differences from the body this block is declared to fit. A difference the influence
                map cannot act on is reported here rather than absorbed somewhere it does not belong.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Measurement</th>
                  <th scope="col">Block fits</th>
                  <th scope="col">This body</th>
                  <th scope="col">Difference</th>
                  <th scope="col">Moves geometry</th>
                </tr>
              </thead>
              <tbody>
                {deltas?.deltas?.map((delta) => (
                  <tr key={delta.name} className={!delta.mapped && delta.deltaValue !== 0 ? 'row-warn' : undefined}>
                    <th scope="row">{delta.name.replace(/_/g, ' ')}</th>
                    <td className="nowrap">{delta.baseValue.toFixed(1)} {delta.unit}</td>
                    <td className="nowrap">{delta.targetValue.toFixed(1)} {delta.unit}</td>
                    <td className="nowrap">
                      {delta.deltaValue >= 0 ? '+' : ''}
                      {delta.deltaValue.toFixed(1)} {delta.unit}
                    </td>
                    <td>
                      {delta.mapped ? (
                        <Badge tone="ok">mapped</Badge>
                      ) : (
                        <Badge tone={delta.deltaValue === 0 ? 'neutral' : 'warn'}>unmapped</Badge>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {envelope && envelope.status !== 'ok' ? (
            <Callout
              tone={envelope.status === 'stop' ? 'error' : 'warn'}
              title={
                envelope.status === 'stop'
                  ? 'Outside the transformation envelope'
                  : 'Past the tested range'
              }
            >
              <ul className="stack-tight">
                {envelope?.entries
                  ?.filter((entry) => entry.status !== 'ok')
                  .map((entry) => (
                    <li key={entry.name}>{entry.message}</li>
                  ))}
              </ul>
              {envelope.status === 'stop' ? (
                <p>
                  This is too far past the tested range to be confirmed. No reliable extrapolation is
                  claimed here; pick a block closer to this body.
                </p>
              ) : (
                <Toggle
                  label="Proceed anyway — the result is an extrapolation, and the confirmation is recorded in the manifest"
                  checked={settings.confirm_envelope}
                  onChange={(on) => onChange({ ...settings, confirm_envelope: on })}
                />
              )}
            </Callout>
          ) : null}
        </>
      ) : null}
    </div>
  )
}
