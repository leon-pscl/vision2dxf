import { useState } from 'react'
import { Button, Callout } from '../components/ui'
import { api, RequestError, saveJson, type GenerationPayload } from '../api'
import type { GenerateResponse, Settings } from '../types'

const FILES = [
  { path: '/api/patterns/export/bundle', name: 'pattern-bundle.zip', label: 'Everything (zip)', note: 'all files' },
  { path: '/api/patterns/export/pdf', name: 'pattern-a0.pdf', label: 'pattern-a0.pdf', note: 'full scale, A0' },
  { path: '/api/patterns/export/dxf', name: 'pattern.dxf', label: 'pattern.dxf', note: 'full scale, mm' },
  { path: '/api/patterns/export/svg', name: 'pattern-preview.svg', label: 'pattern-preview.svg', note: 'same geometry as the exports' },
]

export function ExportView({
  result,
  input,
  settings,
}: {
  result: GenerateResponse
  input: GenerationPayload
  settings: Settings
}) {
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const blocked = result.validation.status !== 'pass'

  const download = async (path: string, name: string) => {
    setBusy(name)
    setError(null)
    try {
      await api.download(path, name, input, settings)
    } catch (e) {
      setError(e instanceof RequestError ? e.message : 'Export failed')
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="stack">
      <h2>Export</h2>
      {blocked ? (
        <Callout tone="error" title="Export blocked">
          Geometry validation failed. Fix the failing checks before producing production files.
        </Callout>
      ) : null}

      <div className="export-grid">
        {FILES.map((file) => (
          <div key={file.name} className="export-card">
            <div>
              <strong>{file.label}</strong>
              <small>{file.note}</small>
            </div>
            <Button variant="primary" disabled={busy !== null} onClick={() => void download(file.path, file.name)}>
              {busy === file.name ? 'Preparing…' : 'Download'}
            </Button>
          </div>
        ))}
        <div className="export-card">
          <div>
            <strong>pattern.json</strong>
            <small>GarmageSet-like primary pattern</small>
          </div>
          <Button disabled={busy !== null} onClick={() => saveJson('pattern.json', result.pattern)}>Download</Button>
        </div>
        <div className="export-card">
          <div>
            <strong>generation-manifest.json</strong>
            <small>method, settings, assumptions, rule traces</small>
          </div>
          <Button disabled={busy !== null} onClick={() => saveJson('generation-manifest.json', result.trace)}>Download</Button>
        </div>
        <div className="export-card">
          <div>
            <strong>validation-report.json</strong>
            <small>checks, metrics, tolerances</small>
          </div>
          <Button disabled={busy !== null} onClick={() => saveJson('validation-report.json', result.validation)}>Download</Button>
        </div>
      </div>

      {error ? <Callout tone="error">{error}</Callout> : null}

      <Callout tone="info" title="Fixture-compatible defaults">
        <p>
          The panel-local geometry and the <code>center</code> placement follow the reference file. Values the
          fixture cannot explain are preserved verbatim and documented in the manifest:{' '}
          <code>{JSON.stringify(result.trace.fixtureDefaults)}</code>
        </p>
      </Callout>
    </div>
  )
}
