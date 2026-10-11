import { describe, expect, it } from 'vitest'
import { firstIncompleteStep, STEPS, type StepId } from '../steps'
import { clampScale, distanceMm, formatMm, niceRulerMm, pan, zoomAt } from './viewer-math'

describe('viewer transforms', () => {
  it('keeps the point under the cursor fixed while zooming', () => {
    const cursor: [number, number] = [200, 120]
    const zoomed = zoomAt({ scale: 1, x: 0, y: 0 }, 2, cursor[0], cursor[1])
    const before = distanceMm(cursor, [cursor[0] + 100, cursor[1]], 1)
    const after = distanceMm(cursor, [cursor[0] + 100 * zoomed.scale, cursor[1]], 1)
    expect(before * 2).toBeCloseTo(after, 6)
    expect(zoomed.scale).toBe(2)
  })

  it('clamps the scale to the usable range', () => {
    expect(clampScale(0.01)).toBe(0.2)
    expect(clampScale(1000)).toBe(40)
    expect(zoomAt({ scale: 1, x: 0, y: 0 }, 1000, 0, 0).scale).toBe(40)
    expect(zoomAt({ scale: 1, x: 0, y: 0 }, 0.00001, 0, 0).scale).toBe(0.2)
  })

  it('pans by a delta', () => {
    expect(pan({ scale: 1, x: 0, y: 0 }, 12, -8)).toEqual({ scale: 1, x: 12, y: -8 })
  })
})

describe('measurement formatting', () => {
  it('formats millimetres', () => {
    expect(formatMm(274.53)).toBe('274.5 mm')
    expect(formatMm(20, 0)).toBe('20 mm')
  })

  it('measures a distance in millimetres', () => {
    expect(distanceMm([0, 0], [30, 40], 1)).toBe(50)
    expect(distanceMm([0, 0], [30, 40], 0.5)).toBe(25)
  })

  it('picks a round ruler length', () => {
    expect(niceRulerMm(1)).toBe(200) // target 140 mm, rounded up to the next 1-2-5 step
    expect(niceRulerMm(10)).toBe(20)
    expect(niceRulerMm(0.01)).toBe(1000) // beyond the table: fall back to a metre
  })
})

describe('workflow steps', () => {
  it('stops at the first incomplete step', () => {
    const ready = Object.fromEntries(STEPS.map((s) => [s.id, true])) as Record<StepId, boolean>
    expect(firstIncompleteStep(ready)).toBe(STEPS.length - 1)
    ready.review = false
    expect(firstIncompleteStep(ready)).toBe(STEPS.findIndex((s) => s.id === 'review'))
  })
})
