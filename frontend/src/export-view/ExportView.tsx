import { useState } from 'react'
import { Button, Callout } from '../components/ui'
import { api, RequestError, saveJson, type GenerationPayload } from '../api'
import type { Settings, TransformResponse } from '../types'

const FILES = [
  { path: '/api/patterns/export/bundle', name: 'pattern-bundle.zip', label: 'Everything (zip)', note: 'every file below, plus the base block and its derived maps' },
  { path: '/api/patterns/export/pdf', name: 'pattern-a0.pdf', label: 'pattern-a0.pdf', note: 'full scale, A0' },
  { path: '/api/patterns/export/dxf', name: 'pattern.dxf', label: 'pattern.dxf', note: 'full scale, mm' },
  { path: '/api/patterns/export/svg', name: 'base-vs-target-preview.svg', label: 'base-vs-target-preview.svg', note: 'the comparison, same geometry as the exports' },
]

export function ExportView({
  result,
  input,
  settings,
}: {
  result: TransformResponse
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
            <small>base block, its version and hash, the configuration and every difference</small>
          </div>
          <Button disabled={busy !== null} onClick={() => saveJson('generation-manifest.json', result.trace)}>Download</Button>
        </div>
        <div className="export-card">
          <div>
            <strong>transformation-diagnostics.json</strong>
            <small>solver, landmark targets, seam balancing, achieved dimensions</small>
          </div>
          <Button disabled={busy !== null} onClick={() => saveJson('transformation-diagnostics.json', result.diagnostics)}>Download</Button>
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

      <Callout tone="info" title="What identifies this pattern">
        <p>
          Base block <code>{result.trace.base_block.id}</code> v{result.trace.base_block.version},
          derived from template <code>{result.trace.base_block.source_file}</code> pinned at sha256{' '}
          {result.trace.base_block.source_sha256.slice(0, 16)}…, transformed with configuration v
          {result.trace.transformation_config.version} and influence map v
          {result.trace.transformation_config.influence_map_version}. All of it is recorded in the
          manifest.
        </p>
      </Callout>

      <Callout tone="info" title="Fixture-compatible defaults">
        <p>
          Panel-local geometry and the <code>center</code> placement follow the reference file. Values
          the fixture cannot explain are preserved verbatim and documented in the manifest:{' '}
          <code>{JSON.stringify(result.trace.fixture_defaults)}</code>
        </p>
      </Callout>
    </div>
  )
}
