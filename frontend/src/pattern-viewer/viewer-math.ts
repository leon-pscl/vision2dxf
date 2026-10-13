/** Pure helpers for the pattern viewer. Kept free of React so they are unit-testable. */

export type LayerName =
  | 'base'
  | 'displacement'
  | 'heatmap'
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
  base: 'Base block',
  displacement: 'Displacement vectors',
  heatmap: 'Deformation heat map',
  construction: 'Construction lines',
  landmarks: 'Named landmarks',
  controls: 'Control points',
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

/* ---- comparison viewer --------------------------------------------------- */

export type CompareMode = 'overlay' | 'base' | 'target' | 'slider'

export const COMPARE_MODE_LABELS: Record<CompareMode, string> = {
  overlay: 'Overlay',
  base: 'Base only',
  target: 'Transformed only',
  slider: 'Slider',
}

/** The comparison layers that stand for the base block rather than the result. */
export const BASE_LAYERS: LayerName[] = ['base']

/**
 * Which layers a comparison mode shows. `base` is the untransformed block and every
 * other layer draws the result, so a mode is just a choice of which of the two to keep.
 * The slider keeps both and wipes between them instead, which is a clip, not a layer.
 */
export function visibleLayers(mode: CompareMode, available: LayerName[], extras: LayerName[]): LayerName[] {
  const isBase = (layer: LayerName) => BASE_LAYERS.includes(layer)
  return available.filter((layer) => {
    if (extras.includes(layer)) return true
    if (mode === 'base') return isBase(layer)
    if (mode === 'target') return !isBase(layer)
    return true
  })
}

/** Clip inset for the slider wipe: the result is revealed from the left. */
export function wipeInset(fraction: number): string {
  const pct = Math.min(100, Math.max(0, fraction * 100))
  return `inset(0 ${(100 - pct).toFixed(2)}% 0 0)`
}
