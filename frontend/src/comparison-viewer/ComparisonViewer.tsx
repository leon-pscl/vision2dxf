import { useEffect, useMemo, useRef, useState } from 'react'
import { Badge, Button, Toggle } from '../components/ui'
import {
  COMPARE_MODE_LABELS,
  LAYER_LABELS,
  formatMm,
  visibleLayers,
  wipeInset,
  type CompareMode,
  type LayerName,
} from '../pattern-viewer/viewer-math'
import type { Diagnostics, Preview } from '../types'

/** Layers that read on top of either block and stay on whichever mode is picked. */
const OVERLAYS: LayerName[] = ['displacement', 'heatmap']

const MODES: CompareMode[] = ['overlay', 'base', 'target', 'slider']

export function ComparisonViewer({
  preview,
  diagnostics,
}: {
  preview: Preview
  diagnostics: Diagnostics
}) {
  const hostRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)
  const [mode, setMode] = useState<CompareMode>('overlay')
  const [overlays, setOverlays] = useState<LayerName[]>([])
  const [wipe, setWipe] = useState(0.5)

  const available = useMemo(
    () => (preview.comparisonLayers.length ? preview.comparisonLayers : preview.layers) as LayerName[],
    [preview],
  )

  // One server-rendered SVG holds both blocks in one coordinate frame, so comparing
  // them is a matter of hiding groups rather than fetching a second render.
  useEffect(() => {
    const node = hostRef.current
    if (!node) return
    node.innerHTML = preview.svg
    svgRef.current = node.querySelector('svg')
  }, [preview])

  useEffect(() => {
    const shown = visibleLayers(mode, available, overlays)
    svgRef.current?.querySelectorAll<SVGGElement>('[data-layer]').forEach((group) => {
      const layer = group.dataset.layer as LayerName
      group.style.display = shown.includes(layer) ? '' : 'none'
      // The wipe reveals the transformed block over the base one, so it clips
      // everything except the base and whatever overlay is pinned on top.
      group.style.clipPath =
        mode === 'slider' && layer !== 'base' && !overlays.includes(layer) ? wipeInset(wipe) : ''
    })
  }, [mode, overlays, wipe, available, preview])

  const dimensions = Object.entries(diagnostics.requestedDimensions)
  const moved = [...preview.displacements].sort((a, b) => b.displacementMm - a.displacementMm)

  return (
    <section className="viewer">
      <h2>Compare the base block and the transformed pattern</h2>
      <p className="muted">
        Both blocks are drawn from one server-side render in one coordinate frame, so every
        difference you can see is a difference in the geometry and not in how it was laid out.
      </p>

      <div className="viewer-toolbar">
        <div className="toolbar-group" role="group" aria-label="Comparison mode">
          {MODES.map((m) => (
            <Button key={m} variant={mode === m ? 'primary' : 'default'} onClick={() => setMode(m)}>
              {COMPARE_MODE_LABELS[m]}
            </Button>
          ))}
        </div>
        <div className="toolbar-group wrap" role="group" aria-label="Overlays">
          {OVERLAYS.filter((layer) => available.includes(layer)).map((layer) => (
            <Toggle
              key={layer}
              label={LAYER_LABELS[layer]}
              checked={overlays.includes(layer)}
              onChange={(on) =>
                setOverlays((prev) => (on ? [...prev, layer] : prev.filter((l) => l !== layer)))
              }
            />
          ))}
        </div>
      </div>

      {mode === 'slider' ? (
        <label className="field slider-field">
          <span className="field-label">
            Wipe <em>{Math.round(wipe * 100)}% transformed</em>
          </span>
          <input
            type="range"
            min={0}
            max={100}
            value={Math.round(wipe * 100)}
            onChange={(e) => setWipe(Number(e.target.value) / 100)}
          />
        </label>
      ) : null}

      <div className="viewer-canvas" ref={hostRef} />

      <div className="stat-row">
        <div className="stat">
          <span>Solver</span>
          <strong>{diagnostics.method}</strong>
        </div>
        <div className="stat">
          <span>Largest landmark move</span>
          <strong>{moved.length ? formatMm(moved[0].displacementMm) : '—'}</strong>
        </div>
        <div className="stat">
          <span>Seam balancing</span>
          <strong>
            {diagnostics.seamBalanceRounds.length} round
            {diagnostics.seamBalanceRounds.length === 1 ? '' : 's'}
            {diagnostics.seamBalanceRounds.at(-1)?.settled === false ? ' (unsettled)' : ''}
          </strong>
        </div>
        <div className="stat">
          <span>Influence map</span>
          <strong>{diagnostics.influenceMapVersion}</strong>
        </div>
      </div>

      <div className="table-scroll">
        <table className="edge-table">
          <caption>Achieved against requested dimensions</caption>
          <thead>
            <tr>
              <th scope="col">Dimension</th>
              <th scope="col">Base</th>
              <th scope="col">Requested</th>
              <th scope="col">Achieved</th>
              <th scope="col">Difference</th>
            </tr>
          </thead>
          <tbody>
            {dimensions.map(([name, requested]) => {
              const achieved = diagnostics.achievedDimensions[name]
              const unit = requested.unit === 'deg' ? '°' : ' mm'
              const gap = achieved === undefined ? null : achieved - requested.requestedMm
              return (
                <tr key={name}>
                  <th scope="row">{name.replace(/_/g, ' ')}</th>
                  <td className="nowrap">{requested.baseMm.toFixed(1)}{unit}</td>
                  <td className="nowrap">{requested.requestedMm.toFixed(1)}{unit}</td>
                  <td className="nowrap">{achieved === undefined ? '—' : achieved.toFixed(1) + unit}</td>
                  <td className="nowrap">
                    {gap === null ? (
                      '—'
                    ) : (
                      <Badge tone={Math.abs(gap) <= 1 ? 'ok' : Math.abs(gap) <= 3 ? 'warn' : 'error'}>
                        {gap >= 0 ? '+' : ''}
                        {gap.toFixed(1)}
                        {unit}
                      </Badge>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="table-scroll">
        <table className="edge-table">
          <caption>
            Seam mismatch before and after balancing. What is corrected is the gap the
            transformation opened, not the gap the block arrived with.
          </caption>
          <thead>
            <tr>
              <th scope="col">Seam</th>
              <th scope="col">On the base block</th>
              <th scope="col">Before balancing</th>
              <th scope="col">After balancing</th>
              <th scope="col">Tolerance</th>
            </tr>
          </thead>
          <tbody>
            {diagnostics.seamMismatchBeforeBalancing.map((seam, i) => {
              const after = diagnostics.seamMismatchAfterMm.find(
                (a) => a.edges.join() === seam.edges.join(),
              )
              return (
                <tr key={seam.edges.join('-') + i} className={after && !after.balanced ? 'row-warn' : undefined}>
                  <th scope="row">{seam.role.replace(/_/g, ' ')}</th>
                  <td className="nowrap">{formatMm(seam.baseMismatchMm, 2)}</td>
                  <td className="nowrap">{formatMm(seam.mismatchMm, 2)}</td>
                  <td className="nowrap">{after ? formatMm(after.mismatchMm, 2) : '—'}</td>
                  <td className="nowrap">{formatMm(seam.toleranceMm, 2)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="table-scroll">
        <table className="edge-table">
          <caption>Landmark displacement, and the measurement difference that caused it</caption>
          <thead>
            <tr>
              <th scope="col">Panel</th>
              <th scope="col">Landmark</th>
              <th scope="col">Moved</th>
              <th scope="col">Attributed to</th>
            </tr>
          </thead>
          <tbody>
            {moved.map((target) => (
              <tr key={target.key}>
                <td>{target.panel}</td>
                <td>
                  <code>
                    {target.role}
                    {target.side ? ` (${target.side})` : ''}
                  </code>
                </td>
                <td className="nowrap">{formatMm(target.displacementMm, 2)}</td>
                <td>
                  {target.ops.length === 0 ? (
                    <span className="muted">anchor — nothing moved it</span>
                  ) : (
                    target.ops.map((op, i) => (
                      <small className="op" key={`${target.key}-${i}`}>
                        <code>{op.measurement}</code> {op.mode}
                        {op.direction ? ` ${op.direction}` : ''}
                        {op.pivot ? ` about ${op.pivot}` : ''}
                        {op.appliedMm !== undefined ? ` ${op.appliedMm.toFixed(2)} mm` : ''}
                        {op.appliedDeg !== undefined ? ` ${op.appliedDeg.toFixed(2)}°` : ''}
                      </small>
                    ))
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
