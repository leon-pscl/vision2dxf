/** Pure helpers for the pattern viewer. Kept free of React so they are unit-testable. */

export type LayerName =
  | 'construction'
  | 'landmarks'
  | 'controls'
  | 'sewing'
  | 'cutting'
  | 'darts'
  | 'notches'
  | 'labels'
  | 'rulers'

export const LAYER_LABELS: Record<LayerName, string> = {
  construction: 'Construction lines',
  landmarks: 'Named landmarks',
  controls: 'Bezier control points',
  sewing: 'Sewing lines',
  cutting: 'Cutting lines',
  darts: 'Darts',
  notches: 'Notches',
  labels: 'Edge labels',
  rulers: 'Rulers',
}

export type ViewBox = { x: number; y: number; w: number; h: number }

export const MIN_SCALE = 0.2
export const MAX_SCALE = 40

export function clampScale(scale: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale))
}

/** Zoom around a fixed point, so the pixel under the cursor stays put. */
export function zoomAt(
  current: { scale: number; x: number; y: number },
  factor: number,
  px: number,
  py: number,
): { scale: number; x: number; y: number } {
  const scale = clampScale(current.scale * factor)
  const k = scale / current.scale
  return { scale, x: px - (px - current.x) * k, y: py - (py - current.y) * k }
}

export function pan(current: { scale: number; x: number; y: number }, dx: number, dy: number) {
  return { ...current, x: current.x + dx, y: current.y + dy }
}

export function formatMm(value: number, digits = 1): string {
  return `${value.toFixed(digits)} mm`
}

export function distanceMm(a: [number, number], b: [number, number], scale: number): number {
  return Math.hypot(b[0] - a[0], b[1] - a[1]) * scale
}

/** Scale bar length in mm that lands on a round number for the current zoom. */
export function niceRulerMm(pxPerMm: number, maxPx = 140): number {
  const target = maxPx / Math.max(pxPerMm, 1e-9)
  const steps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
  return steps.find((s) => s >= target) ?? 1000
}
