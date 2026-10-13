import { useMemo, useState } from 'react'
import { Badge, Callout } from '../components/ui'
import { SurfaceMeshViewer } from './SurfaceMeshViewer'
import type { CrossSections, GenerationResponse, SurfaceMesh, SurfaceSeamPath } from '../types'

const SECTION_COLORS = ['#1f5fd0', '#a83f76', '#187c66', '#ba6a16', '#6941c6', '#687582']

function sectionPath(section: CrossSections['sections'][number], size: number): string {
  const points: string[] = []
  const exponent = section.exponent
  for (let i = 0; i <= 96; i += 1) {
    const angle = i / 96 * Math.PI * 2
    const x = section.halfWidthMm * Math.sign(Math.cos(angle)) * Math.abs(Math.cos(angle)) ** (2 / exponent)
    const y = section.halfDepthMm * Math.sign(Math.sin(angle)) * Math.abs(Math.sin(angle)) ** (2 / exponent)
    points.push(`${(size / 2 + x).toFixed(2)},${(size / 2 - y).toFixed(2)}`)
  }
  return `M ${points.join(' L ')} Z`
}

function makePanelMesh(panels: NonNullable<GenerationResponse['surfaceStages']>['panels']): SurfaceMesh {
  const vertices: number[][] = []
  const faces: number[][] = []
  for (const panel of panels) {
    const offset = vertices.length
    vertices.push(...panel.vertices)
    faces.push(...panel.faces.map((face) => face.map((index) => index + offset)))
  }
  return { label: 'segmented panels', columns: 0, rows: 0, vertices, faces, levelRows: {}, rings: [] }
}

export function SurfaceReview({
  result,
  view,
}: {
  result: GenerationResponse
  view: 'sections' | 'surfaces'
}) {
  const data = result.surfaceStages
  const sections = result.crossSections
  const [selected, setSelected] = useState<'proxy' | 'garment' | 'panels'>('proxy')
  const panelsMesh = useMemo(() => data ? makePanelMesh(data.panels) : null, [data])

  if (!data || !sections) {
    return (
      <Callout tone="error" title="Intermediate geometry missing">
        Regenerate with intermediate surface output enabled.
      </Callout>
    )
  }

  const mesh: SurfaceMesh = selected === 'proxy'
    ? data.proxy
    : selected === 'garment'
      ? data.garment
      : panelsMesh ?? data.garment
  const seamPaths: SurfaceSeamPath[] = selected === 'proxy' ? [] : data.seamPaths

  return (
    <div className="stack">
      <h2>Reconstructed surfaces and seam network</h2>
      <p className="muted">
        This is a measurement-constrained proxy torso, not an exact scan reconstruction.
        Rotate the mesh to inspect its shape and the fixed seam paths.
      </p>

      {view === 'sections' ? <fieldset>
        <legend>Numerically solved cross-sections</legend>
        <div className="cross-section-view">
          <svg viewBox="0 0 420 420" role="img" aria-label="Overlay of reconstructed torso cross-sections">
            <rect x="0" y="0" width="420" height="420" fill="var(--panel)" />
            <line x1="210" y1="24" x2="210" y2="396" stroke="var(--line)" strokeDasharray="4 5" />
            {sections.sections.map((section, index) => (
              <path
                key={section.name}
                d={sectionPath(section, 420)}
                fill="none"
                stroke={SECTION_COLORS[index % SECTION_COLORS.length]}
                strokeWidth="2"
              />
            ))}
          </svg>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th scope="col">Level</th>
                  <th scope="col">Target</th>
                  <th scope="col">Achieved</th>
                  <th scope="col">Residual</th>
                  <th scope="col">Solved for</th>
                </tr>
              </thead>
              <tbody>
                {sections.sections.map((section, index) => (
                  <tr key={section.name}>
                    <th scope="row">
                      <span className="legend-swatch" style={{ backgroundColor: SECTION_COLORS[index % SECTION_COLORS.length] }} />
                      {section.name}
                    </th>
                    <td>{section.targetPerimeterMm === null ? '—' : `${section.targetPerimeterMm.toFixed(1)} mm`}</td>
                    <td>{section.achievedPerimeterMm.toFixed(1)} mm</td>
                    <td>{section.residualMm.toFixed(4)} mm</td>
                    <td>{section.solvedFor}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <p className="muted small">
          Cross-section shape overlay is normalized for comparison; the adjacent table reports
          dimensions in millimetres. Maximum perimeter residual {sections.maxResidualMm.toFixed(5)} mm
          (configured limit {sections.residualToleranceMm} mm).
        </p>
      </fieldset> : null}

      {view === 'surfaces' ? <fieldset>
        <legend>Surface stages</legend>
        <div className="row" role="group" aria-label="Select surface stage">
          {([
            ['proxy', 'Proxy torso'],
            ['garment', 'Ease-adjusted garment + seams'],
            ['panels', 'Segmented panels'],
          ] as const).map(([value, label]) => (
            <button
              key={value}
              type="button"
              className={`btn ${selected === value ? 'btn-primary' : ''}`}
              aria-pressed={selected === value}
              onClick={() => setSelected(value)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="row">
          <Badge tone="neutral">{mesh.label}</Badge>
          {selected !== 'proxy' ? <Badge tone="warn">{data.seamPaths.length} semantic seam paths</Badge> : null}
          {selected === 'panels' ? <Badge tone="neutral">{data.panels.length} panels</Badge> : null}
        </div>
        <SurfaceMeshViewer mesh={mesh} seams={seamPaths} />
      </fieldset> : null}

      {view === 'surfaces' ? <fieldset>
        <legend>Garment ease achieved</legend>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th scope="col">Level</th>
                <th scope="col">Body circumference</th>
                <th scope="col">Garment circumference</th>
                <th scope="col">Requested ease</th>
                <th scope="col">Residual</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.trace.garment_surface?.levels ?? {}).map(([name, value]) => {
                const level = value as Record<string, number>
                return (
                  <tr key={name}>
                    <th scope="row">{name}</th>
                    <td>{Number(level.bodyCircumferenceMm ?? 0).toFixed(1)} mm</td>
                    <td>{Number(level.achievedCircumferenceMm ?? 0).toFixed(1)} mm</td>
                    <td>{Number(level.requestedEaseMm ?? 0).toFixed(1)} mm</td>
                    <td>{Number(level.easeResidualMm ?? 0).toFixed(5)} mm</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </fieldset> : null}
    </div>
  )
}
