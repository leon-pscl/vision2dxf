import { useRef, useState } from 'react'
import { Button, Callout } from '../components/ui'
import { api, RequestError } from '../api'
import type { Review } from '../types'

const SAMPLE_HINT = 'Paste or drop a body_measurements.yaml file. Only .yaml and .yml are accepted.'

export function UploadStep({
  onUploaded,
  yaml,
  filename,
  setYaml,
  setFilename,
}: {
  onUploaded: (review: Review) => void
  yaml: string
  filename: string
  setYaml: (text: string) => void
  setFilename: (name: string) => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const load = async (text: string, name: string) => {
    setBusy(true)
    setError(null)
    try {
      onUploaded(await api.validateMeasurements(name, text))
    } catch (e) {
      setError(e instanceof RequestError ? e.message : 'Upload failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="stack">
      <h2>Upload body measurements</h2>
      <p className="muted">{SAMPLE_HINT}</p>

      <div
        className="dropzone"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault()
          const file = e.dataTransfer.files[0]
          if (file) file.text().then((text) => { setYaml(text); setFilename(file.name); void load(text, file.name) })
        }}
      >
        <input
          ref={inputRef}
          id="yaml-file"
          type="file"
          accept=".yaml,.yml"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (!file) return
            void file.text().then((text) => { setYaml(text); setFilename(file.name); void load(text, file.name) })
          }}
        />
        <label htmlFor="yaml-file">Choose a .yaml file or drop one here</label>
      </div>

      <label className="field">
        <span className="field-label">YAML source</span>
        <textarea
          rows={16}
          spellCheck={false}
          value={yaml}
          onChange={(e) => setYaml(e.target.value)}
          placeholder={'body:\n  bust: 85.8\n  ...'}
        />
      </label>

      <div className="row">
        <Button
          variant="primary"
          disabled={!yaml.trim() || busy}
          onClick={() => void load(yaml, filename.endsWith('.yaml') || filename.endsWith('.yml') ? filename : 'body_measurements.yaml')}
        >
          {busy ? 'Validating…' : 'Validate measurements'}
        </Button>
        <Button variant="ghost" onClick={() => inputRef.current?.click()}>
          Browse…
        </Button>
      </div>

      {error ? <Callout tone="error" title="Import failed">{error}</Callout> : null}
    </div>
  )
}
