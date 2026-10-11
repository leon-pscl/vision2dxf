import type { ApiError, AppConfig, GenerateResponse, Review, Settings, ValidationReport } from './types'

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

export const api = {
  config: () => fetch('/api/config').then((r) => r.json() as Promise<AppConfig>),

  validateMeasurements: (filename: string, content: string) =>
    post<Review>('/api/measurements/validate', { filename, content }),

  generate: (yamlText: string, filename: string, settings: Settings) =>
    post<GenerateResponse>('/api/patterns/generate', { yamlText, filename, settings }),

  validatePattern: (pattern: unknown) =>
    post<{ ok: boolean; issues: { check: string; message: string }[] }>('/api/patterns/validate', { pattern }),

  async download(path: string, filename: string, payload: unknown): Promise<void> {
    const response = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!response.ok) {
      const data = (await response.json().catch(() => null)) as { error?: ApiError } | null
      throw new RequestError(data?.error ?? { code: 'export_failed', message: 'Export failed', details: [] })
    }
    const blob = await response.blob()
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    link.click()
    URL.revokeObjectURL(url)
  },
}

export function reportSummary(report: ValidationReport): string {
  return `${report.summary.passed}/${report.summary.total} checks passed`
}
