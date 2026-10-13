import { useMemo } from 'react'
import { Callout } from '../components/ui'
import type { GenerationResponse } from '../types'

const WIDTH = 900
const HEIGHT = 560

export function DistortionHeatMap({ result }: { result: GenerationResponse }) {
  const heatMap = useMemo(() => {
    const triangles = result.heatMaps.flatMap((panel) =>
      panel.triangles.map((triangle, index) => ({
        id: `${panel.panel}-${index}`,
        panel: panel.label,
        value: panel.values[index],
        points: triangle.map(([x, y]) => [x + panel.center[0], y + panel.center[1]] as [number, number]),
      })),
    )
    if (triangles.length === 0) return null

    const points = triangles.flatMap((triangle) => triangle.points)
    const minX = Math.min(...points.map(([x]) => x))
    const maxX = Math.max(...points.map(([x]) => x))
    const minY = Math.min(...points.map(([, y]) => y))
    const maxY = Math.max(...points.map(([, y]) => y))
    const spanX = maxX - minX || 1
    const spanY = maxY - minY || 1
    const scale = Math.min((WIDTH - 60) / spanX, (HEIGHT - 60) / spanY)
    const midX = (minX + maxX) / 2
    const midY = (minY + maxY) / 2
    const maxValue = Math.max(1e-9, ...triangles.map(({ value }) => value))

    return {
      triangles: triangles.map((triangle) => ({
        ...triangle,
        valueRatio: triangle.value / maxValue,
        points: triangle.points.map(([x, y]) => [
          WIDTH / 2 + (x - midX) * scale,
          HEIGHT / 2 - (y - midY) * scale,
        ] as [number, number]),
      })),
      maxValue,
    }
  }, [result.heatMaps])

  if (!heatMap) {
    return <Callout tone="warn" title="Distortion heat map unavailable">This generation did not include triangle distortion data.</Callout>
  }

  return (
    <section className="stack">
      <h2>Flattening distortion heat map</h2>
      <p className="muted">
        Triangle color encodes angular distortion from the local surface metric.
        The scale runs from lower distortion (blue) to higher distortion (red).
      </p>
      <svg
        className="heat-map-scene"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`Flattened panel distortion heat map, maximum angular distortion ${heatMap.maxValue.toFixed(3)} degrees`}
      >
        <rect width={WIDTH} height={HEIGHT} fill="var(--panel)" />
        {heatMap.triangles.map((triangle) => (
          <polygon
            key={triangle.id}
            points={triangle.points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ')}
            fill={`hsl(${Math.round(210 * (1 - triangle.valueRatio))} 78% 54%)`}
            stroke="rgba(20, 24, 32, 0.35)"
            strokeWidth="0.55"
            vectorEffect="non-scaling-stroke"
            aria-label={`${triangle.panel}: ${triangle.value.toFixed(3)} degrees`}
          />
        ))}
      </svg>
      <div className="heat-map-legend" aria-hidden="true">
        <span>Lower</span><span className="heat-map-gradient" /><span>Higher</span>
      </div>
      <p className="muted small">
        Maximum angular distortion shown: {heatMap.maxValue.toFixed(4)}°. Individual values are
        included in the flattening diagnostics download.
      </p>
    </section>
  )
}
