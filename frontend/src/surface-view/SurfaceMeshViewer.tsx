import { useMemo, useRef, useState } from 'react'
import type { SurfaceMesh, SurfaceSeamPath } from '../types'

const VIEW_WIDTH = 720
const VIEW_HEIGHT = 440

type Projected = { x: number; y: number; depth: number }

function rotate(point: number[], yaw: number, pitch: number): Projected {
  const [x, y, z] = point
  const yawX = Math.cos(yaw) * x + Math.sin(yaw) * z
  const yawZ = -Math.sin(yaw) * x + Math.cos(yaw) * z
  return {
    x: yawX,
    y: Math.cos(pitch) * y - Math.sin(pitch) * yawZ,
    depth: Math.sin(pitch) * y + Math.cos(pitch) * yawZ,
  }
}

export function SurfaceMeshViewer({
  mesh,
  seams = [],
}: {
  mesh: Pick<SurfaceMesh, 'vertices' | 'faces'>
  seams?: SurfaceSeamPath[]
}) {
  const [yaw, setYaw] = useState(-0.58)
  const [pitch, setPitch] = useState(0.12)
  const drag = useRef<{ x: number; y: number } | null>(null)

  const scene = useMemo(() => {
    const rotated = mesh.vertices.map((point) => rotate(point, yaw, pitch))
    const xs = rotated.map((point) => point.x)
    const ys = rotated.map((point) => point.y)
    const spanX = Math.max(...xs) - Math.min(...xs) || 1
    const spanY = Math.max(...ys) - Math.min(...ys) || 1
    const scale = Math.min((VIEW_WIDTH - 72) / spanX, (VIEW_HEIGHT - 72) / spanY)
    const minX = (Math.min(...xs) + Math.max(...xs)) / 2
    const minY = (Math.min(...ys) + Math.max(...ys)) / 2
    const projected = rotated.map((point) => ({
      x: VIEW_WIDTH / 2 + (point.x - minX) * scale,
      y: VIEW_HEIGHT / 2 - (point.y - minY) * scale,
      depth: point.depth,
    }))
    const faces = mesh.faces
      .map((face, index) => ({ face, index }))
      .filter(({ face }) => face.length >= 3 && face.every((i) => projected[i] !== undefined))
      .map(({ face, index }) => {
        const a = rotated[face[0]]
        const b = rotated[face[1]]
        const c = rotated[face[2]]
        const ab = [b.x - a.x, b.y - a.y, b.depth - a.depth]
        const ac = [c.x - a.x, c.y - a.y, c.depth - a.depth]
        const normalZ = ab[0] * ac[1] - ab[1] * ac[0]
        const shade = Math.max(0, Math.min(1, Math.abs(normalZ) / (Math.hypot(...ab) * Math.hypot(...ac) || 1)))
        return {
          index,
          points: face.map((i) => `${projected[i].x.toFixed(1)},${projected[i].y.toFixed(1)}`).join(' '),
          depth: face.reduce((sum, i) => sum + projected[i].depth, 0) / face.length,
          shade,
        }
      })
      .sort((a, b) => a.depth - b.depth)
    const paths = seams.map((seam) => ({
      ...seam,
      points: seam.points.map((point) => rotate(point, yaw, pitch)).map((point) => ({
        x: VIEW_WIDTH / 2 + (point.x - minX) * scale,
        y: VIEW_HEIGHT / 2 - (point.y - minY) * scale,
      })),
    }))
    return { faces, paths }
  }, [mesh, pitch, seams, yaw])

  return (
    <div className="mesh-viewer">
      <svg
        className="mesh-scene"
        viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
        role="img"
        aria-label={`${mesh.vertices.length} vertex surface mesh with ${mesh.faces.length} triangles. Drag to rotate.`}
        onPointerDown={(event) => {
          if (event.button !== 0) return
          drag.current = { x: event.clientX, y: event.clientY }
          event.currentTarget.setPointerCapture(event.pointerId)
        }}
        onPointerMove={(event) => {
          if (!drag.current) return
          const dx = event.clientX - drag.current.x
          const dy = event.clientY - drag.current.y
          drag.current = { x: event.clientX, y: event.clientY }
          setYaw((value) => value + dx * 0.008)
          setPitch((value) => Math.max(-1.35, Math.min(1.35, value + dy * 0.008)))
        }}
        onPointerUp={() => { drag.current = null }}
        onPointerCancel={() => { drag.current = null }}
      >
        <title>Interactive surface mesh. Drag to rotate; use the sliders below for keyboard control.</title>
        <rect width={VIEW_WIDTH} height={VIEW_HEIGHT} fill="var(--panel)" />
        {scene.faces.map((face) => (
          <polygon
            key={face.index}
            points={face.points}
            fill={`hsl(208 46% ${Math.round(34 + face.shade * 24)}%)`}
            stroke="var(--mesh-edge, rgba(10, 30, 55, 0.28))"
            strokeWidth="0.65"
            vectorEffect="non-scaling-stroke"
          />
        ))}
        {scene.paths.map((path, index) => (
          <polyline
            key={`${path.panel}-${path.role}-${index}`}
            points={path.points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(' ')}
            fill="none"
            stroke={path.role.includes('center') ? 'var(--accent)' : '#d97706'}
            strokeWidth="2.4"
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>
      <div className="mesh-controls">
        <label className="field">
          <span className="field-label">Horizontal rotation</span>
          <input
            type="range"
            min={-180}
            max={180}
            step={1}
            value={Math.round(yaw * 180 / Math.PI)}
            onChange={(event) => setYaw(Number(event.target.value) * Math.PI / 180)}
          />
        </label>
        <label className="field">
          <span className="field-label">Vertical rotation</span>
          <input
            type="range"
            min={-75}
            max={75}
            step={1}
            value={Math.round(pitch * 180 / Math.PI)}
            onChange={(event) => setPitch(Number(event.target.value) * Math.PI / 180)}
          />
        </label>
        <span className="muted small">{mesh.vertices.length} vertices · {mesh.faces.length} triangles</span>
      </div>
      <p className="muted small">Drag the model or use the keyboard-operable rotation controls.</p>
    </div>
  )
}
