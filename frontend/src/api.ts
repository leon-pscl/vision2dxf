import type {
  ApiError,
  AppConfig,
  GenerationResponse,
  Preview,
  Review,
  Settings,
  ValidationReport,
} from './types'

export class RequestError extends Error {
  code: string
  details: ApiError['details']
  constructor(error: ApiError) {
    super(error.message)
    this.code = error.code
    this.details = error.details ?? []
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = await response.json().catch(() => null)
  if (!response.ok) {
    const error = (data as { error?: ApiError })?.error
    throw new RequestError(error ?? { code: 'http_error', message: `Request failed (${response.status})`, details: [] })
  }
  return data as T
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path)
  const data = await response.json().catch(() => null)
  if (!response.ok) {
    const error = (data as { error?: ApiError })?.error
    throw new RequestError(error ?? {
      code: 'http_error',
      message: `Request failed (${response.status})`,
      details: [],
    })
  }
  return data as T
}

/** Everything a request needs to reproduce a generation, exactly as the server saw it. */
export type GenerationPayload = {
  yamlText: string
  filename: string
  confirmations: Record<string, string>
  corrections: Record<string, number>
}

function payload(input: GenerationPayload, settings?: Settings) {
  return settings ? { ...input, settings } : input
}

export const api = {
  config: () => get<AppConfig>('/api/config'),

  validateMeasurements: (filename: string, content: string) =>
    post<Review>('/api/measurements/validate', { filename, content }),

  semantics: (input: GenerationPayload) =>
    post<SemanticsResponse>('/api/measurements/semantics', payload(input)),

  report: (input: GenerationPayload) =>
    post<{ can_generate: boolean; blocking: unknown[] }>('/api/measurements/report', payload(input)),

  generate: (input: GenerationPayload, settings: Settings) =>
    post<GenerationResponse>('/api/patterns/flatten', {
      ...payload(input, settings),
      include_mesh: true,
      include_surfaces: true,
    }),

  validatePattern: (pattern: unknown) =>
    post<{ ok: boolean; issues: { check: string; message: string }[] }>('/api/patterns/validate', { pattern }),

  async download(path: string, filename: string, input: GenerationPayload, settings: Settings): Promise<void> {
    const response = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload(input, settings)),
    })
    if (!response.ok) {
      const data = (await response.json().catch(() => null)) as { error?: ApiError } | null
      throw new RequestError(data?.error ?? { code: 'export_failed', message: 'Export failed', details: [] })
    }
    const blob = await response.blob()
    saveBlob(blob, filename)
  },
}

export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

export function saveJson(name: string, data: unknown): void {
  saveBlob(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }), name)
}

export type Candidate = {
  id: string
  meaning: string
  representation: string | null
  startLandmark: string | null
  endLandmark: string | null
  origin: string | null
  referenceAxis: string | null
  signConvention: string | null
  axis: string | null
  draftingVariable: string
  scale: number
}

export type FieldResolution = {
  name: string
  label: string
  category: string
  quantityType: string
  status: 'CONFIRMED' | 'DERIVED' | 'ASSUMED' | 'OPTIONAL' | 'MISSING' | 'AMBIGUOUS' | 'INVALID'
  required: boolean
  registryStatus: string
  sourceValue: number | null
  originalValue: number | null
  normalizedMm: number | null
  unitSource: string
  unitPattern: string
  conversion: string
  meaning: string
  representation: string | null
  startLandmark: string | null
  endLandmark: string | null
  origin: string | null
  draftingVariable: string | null
  candidateId: string | null
  confirmedBy: string | null
  definitionVersion: string
  schemaVersion: string
  corrected: boolean
  message: string | null
  suggestedAction: string | null
}

export type Blocking = { field: string | null; code: string; message: string; suggestedAction: string; severity: string }

export type SemanticsResponse = {
  filename: string
  source: {
    schemaVersion: string
    definitionVersion: string
    canGenerate: boolean
    fields: FieldResolution[]
    derivedMeasurements: Record<string, { value: number | null; equation: string; dependencies?: string[] }>
    assumptions: { field: string; kind: string; message: string }[]
    warnings: { field: string; code: string; message: string }[]
    blocking: Blocking[]
  }
  registry: {
    schemaVersion: string
    definitionVersion: string
    coordinateSystem: {
      verticalOrigin: string
      verticalDirection: string
      unit: string
      patternUnit: string
      patternFrame: { y_zero_landmark: string }
      angles: { default_unit: string }
    }
    consistencyRules: { id: string; severity: string; description: string }[]
    fields: {
      name: string
      label: string
      definition: string
      candidates: Candidate[]
      required: boolean
      registryStatus: string
    }[]
    statuses: string[]
    blockingStatuses: string[]
  }
  draftingMeasurements: Record<string, number | null> | null
}

export type { AppConfig, GenerationResponse, Preview, Review, Settings, ValidationReport }
