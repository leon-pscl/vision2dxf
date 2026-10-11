import type { FieldResolution } from '../api'

/**
 * The gate logic the Measurement Definitions screen relies on. Mirrors the server's
 * contract: generation is blocked while any required field is MISSING, AMBIGUOUS or
 * INVALID, and the Generate control stays disabled until that is false.
 */
export const BLOCKING = ['MISSING', 'AMBIGUOUS', 'INVALID'] as const

export function canGenerate(fields: FieldResolution[]): boolean {
  return fields.every((f) => !f.required || !BLOCKING.includes(f.status as (typeof BLOCKING)[number]))
}

export function unresolved(fields: FieldResolution[]): FieldResolution[] {
  return fields.filter((f) => BLOCKING.includes(f.status as (typeof BLOCKING)[number]))
}
