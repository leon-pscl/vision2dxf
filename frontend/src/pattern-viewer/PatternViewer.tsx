import { useCallback, useEffect, useRef, useState } from 'react'
import { Button, Toggle } from '../components/ui'
import {
  LAYER_LABELS,
  MAX_SCALE,
  MIN_SCALE,
  formatMm,
  niceRulerMm,
  type LayerName,
  type ViewBox,
} from './viewer-math'
import type { Preview } from '../types'

const DEFAULT_LAYERS: LayerName[] = [
  'construction',
  'landmarks',
  'controls',
  'sewing',
  'cutting',
  'darts',
  'notches',
  'labels',
  'rulers',
]

export function PatternViewer({ preview }: { preview: Preview }) {
  const hostRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)
  const [base, setBase] = useState<ViewBox>({ x: 0, y: 0, w: 1, h: 1 })
  const [view, setView] = useState<ViewBox>({ x: 0, y: 0, w: 1, h: 1 })
  const [layers, setLayers] = useState<LayerName[]>(DEFAULT_LAYERS)
  const [rulerMode, setRulerMode] = useState(false)
  const [marks, setMarks] = useState<[number, number][]>([])
  const drag = useRef<{ x: number; y: number } | null>(null)
  const [widthPx, setWidthPx] = useState(0)

  // Mount the server-rendered SVG once per preview.
  useEffect(() => {
    const node = hostRef.current
    if (!node) return
    node.innerHTML = preview.svg
    const svg = node.querySelector('svg')
    if (!svg) return
    svgRef.current = svg as SVGSVGElement
    const vb = svg.viewBox.baseVal
    const next: ViewBox = { x: vb.x, y: vb.y, w: vb.width, h: vb.height }
    setBase(next)
    setView(next)
    setMarks([])
  }, [preview])

  useEffect(() => {
    setLayers(DEFAULT_LAYERS.filter((layer) => preview.layers.includes(layer)))
  }, [preview])

  // Layer visibility is toggled straight on the SVG groups: one render, no round trip.
  useEffect(() => {
    svgRef.current?.querySelectorAll<SVGGElement>('[data-layer]').forEach((group) => {
      group.style.display = layers.includes(group.dataset.layer as LayerName) ? '' : 'none'
    })
  }, [layers, preview])

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    const observer = new ResizeObserver(() => setWidthPx(host.clientWidth))
    observer.observe(host)
    setWidthPx(host.clientWidth)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return
    svg.setAttribute('viewBox', `${view.x} ${view.y} ${view.w} ${view.h}`)
  }, [view])

  const toUser = useCallback((event: { clientX: number; clientY: number }): [number, number] => {
    const svg = svgRef.current
    const ctm = svg?.getScreenCTM()
    if (!ctm) return [0, 0]
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse())
    return [point.x, point.y]
  }, [])

  const zoom = (factor: number) => {
    const cx = view.x + view.w / 2
    const cy = view.y + view.h / 2
    const w = Math.min(base.w / MIN_SCALE, Math.max(base.w / MAX_SCALE, view.w / factor))
    const k = w / view.w
    setView({ x: cx - (cx - view.x) * k, y: cy - (cy - view.y) * k, w, h: view.h * k })
  }

  const onWheel = (event: React.WheelEvent) => {
    const factor = event.deltaY < 0 ? 1 / 1.12 : 1.12
    const [px, py] = toUser(event)
    const w = Math.min(base.w / MIN_SCALE, Math.max(base.w / MAX_SCALE, view.w * factor))
    const k = w / view.w
    setView({ x: px - (px - view.x) * k, y: py - (py - view.y) * k, w, h: view.h * k })
  }

  const rulerMm = niceRulerMm(widthPx / Math.max(1e-9, view.w))
  const measured = marks.length === 2 ? Math.hypot(marks[1][0] - marks[0][0], marks[1][1] - marks[0][1]) : null

  return (
    <section className="viewer">
      <div className="viewer-toolbar">
        <div className="toolbar-group" role="group" aria-label="Zoom">
          <Button onClick={() => zoom(1 / 1.25)} title="Zoom out">−</Button>
          <span className="toolbar-readout">{Math.round((base.w / view.w) * 100)}%</span>
          <Button onClick={() => zoom(1.25)} title="Zoom in">+</Button>
          <Button onClick={() => setView(base)} title="Fit to view">Fit</Button>
        </div>
        <div className="toolbar-group" role="group" aria-label="Layers">
          {preview.layers.filter((layer): layer is LayerName => layer in LAYER_LABELS).map((layer) => (
            <Toggle
              key={layer}
              label={LAYER_LABELS[layer]}
              checked={layers.includes(layer)}
              onChange={(on) =>
                setLayers((prev) => (on ? [...prev, layer] : prev.filter((l) => l !== layer)))
              }
            />
          ))}
        </div>
        <div className="toolbar-group">
          <Toggle
            label="Ruler"
            checked={rulerMode}
            onChange={(on) => {
              setRulerMode(on)
              setMarks([])
            }}
          />
          <Button variant="ghost" onClick={() => setMarks([])} disabled={marks.length === 0}>
            Clear ruler
          </Button>
          <span className="toolbar-readout">scale {rulerMm} mm</span>
        </div>
      </div>

      <div
        className="viewer-canvas"
        ref={hostRef}
        onPointerDown={(event) => {
          if (event.button !== 0) return
          drag.current = { x: event.clientX, y: event.clientY }
        }}
        onPointerMove={(event) => {
          if (!drag.current || !widthPx) return
          const dx = (event.clientX - drag.current.x) * (view.w / widthPx)
          const dy = (event.clientY - drag.current.y) * (view.h / (widthPx * (view.h / view.w)))
          drag.current = { x: event.clientX, y: event.clientY }
          setView((v) => ({ ...v, x: v.x - dx, y: v.y - dy }))
        }}
        onPointerUp={(event) => {
          const moved = drag.current && (Math.abs(event.clientX - drag.current.x) > 3 || Math.abs(event.clientY - drag.current.y) > 3)
          drag.current = null
          if (moved || !rulerMode) return
          setMarks((prev) => (prev.length >= 2 ? [toUser(event)] : [...prev, toUser(event)]))
        }}
        onPointerLeave={() => { drag.current = null }}
        onWheel={onWheel}
        onDoubleClick={() => setView(base)}
      >
        {/* Overlay shares the viewBox, so the ruler follows zoom and pan for free. */}
        {marks.length > 0 ? (
          <svg className="ruler-overlay" viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`} aria-hidden="true">
            {marks.length === 2 ? (
              <line
                x1={marks[0][0]}
                y1={marks[0][1]}
                x2={marks[1][0]}
                y2={marks[1][1]}
                stroke="var(--ruler, #a16207)"
                strokeWidth={view.w / 400}
                strokeDasharray={`${view.w / 80} ${view.w / 80}`}
              />
            ) : null}
            {marks.map((m, i) => (
              <circle
                key={i}
                cx={m[0]}
                cy={m[1]}
                r={view.w / 200}
                fill="var(--ruler, #a16207)"
              />
            ))}
          </svg>
        ) : null}
        <p className="ruler-readout" role="status" aria-live="polite">
          {measured !== null
            ? `Measured ${formatMm(measured)}`
            : rulerMode
              ? 'Click two points to measure'
              : 'Drag to pan, scroll to zoom, double-click to fit'}
        </p>
      </div>

      <table className="edge-table">
        <caption>Panel edges</caption>
        <thead>
          <tr>
            <th scope="col">Panel</th>
            <th scope="col">Edge</th>
            <th scope="col">Role</th>
            <th scope="col">Length</th>
          </tr>
        </thead>
        <tbody>
          {preview.panels.flatMap((panel) =>
            panel.edges.map((edge) => (
              <tr key={edge.id}>
                <td>{panel.label}</td>
                <td>{edge.label}</td>
                <td>
                  <code>{edge.role}</code>
                </td>
                <td>{formatMm(edge.lengthMm)}</td>
              </tr>
            )),
          )}
        </tbody>
      </table>
    </section>
  )
}
