import { describe, expect, it } from 'vitest'
import { canGenerate, unresolved } from './gate'
import type { FieldResolution } from '../api'

const field = (over: Partial<FieldResolution>): FieldResolution => ({
  name: 'bust',
  label: 'Bust',
  category: 'circumference',
  quantityType: 'circumference',
  status: 'CONFIRMED',
  required: true,
  registryStatus: 'confirmed',
  sourceValue: 85.8,
  originalValue: 85.8,
  normalizedMm: 858,
  unitSource: 'cm',
  unitPattern: 'mm',
  conversion: 'cm -> mm (x10)',
  meaning: 'full girth',
  representation: 'full',
  startLandmark: null,
  endLandmark: null,
  origin: null,
  draftingVariable: 'bust_circumference',
  candidateId: 'schema_definition',
  confirmedBy: 'schema',
  definitionVersion: '1',
  schemaVersion: '1',
  corrected: false,
  message: null,
  suggestedAction: null,
  ...over,
})

describe('generation gate', () => {
  it('is open when every required field is confirmed', () => {
    expect(canGenerate([field({}), field({ name: 'waist' })])).toBe(true)
  })

  it('is blocked by an ambiguous required field', () => {
    const fields = [
      field({}),
      field({ name: 'shoulder_w', status: 'AMBIGUOUS', candidateId: null, confirmedBy: null }),
    ]
    expect(canGenerate(fields)).toBe(false)
    expect(unresolved(fields).map((f) => f.name)).toEqual(['shoulder_w'])
  })

  it('is blocked by a missing or invalid required field', () => {
    expect(canGenerate([field({ status: 'MISSING' })])).toBe(false)
    expect(canGenerate([field({ status: 'INVALID' })])).toBe(false)
  })

  it('ignores optional fields', () => {
    expect(canGenerate([field({}), field({ name: 'wrist', required: false, status: 'AMBIGUOUS' })])).toBe(true)
  })

  it('treats an operator confirmation as assumed, not confirmed', () => {
    const confirmed = field({ status: 'ASSUMED', confirmedBy: 'operator', candidateId: 'half' })
    expect(canGenerate([confirmed])).toBe(true)
    expect(unresolved([confirmed])).toHaveLength(0)
  })
})
