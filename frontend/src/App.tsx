import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, RequestError } from './api'
import { Stepper } from './components/Stepper'
import { Button, Callout, Spinner } from './components/ui'
import { UploadStep } from './upload/UploadStep'
import { MeasurementReview } from './measurement-review/MeasurementReview'
import { ConfigureStep } from './configure/ConfigureStep'
import { GenerationStep } from './generate/GenerationStep'
import { PatternViewer } from './pattern-viewer/PatternViewer'
import { ValidationView } from './validation-view/ValidationView'
import { ExportView } from './export-view/ExportView'
import { STEPS, firstIncompleteStep, type StepId } from './steps'
import type { AppConfig, GenerateResponse, Review, Settings } from './types'

const DEFAULT_SETTINGS: Settings = {
  garment_type: 'basic_bodice',
  ease_preset: 'fitted',
  bust_ease_mm: null,
  waist_ease_mm: null,
  hip_ease_mm: null,
  seam_allowances: {},
  cutting_join: null,
  paper_size: null,
  paper_orientation: null,
  shoulder_slope_deg: null,
  armscye_depth_mm: null,
  bust_apex_y_mm: null,
  dart_intake_mm: null,
  unit: 'cm',
}

export default function App() {
  const [step, setStep] = useState(0)
  const [yaml, setYaml] = useState('')
  const [filename, setFilename] = useState('body_measurements.yaml')
  const [review, setReview] = useState<Review | null>(null)
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS)
  const [result, setResult] = useState<GenerateResponse | null>(null)
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [theme, setTheme] = useState<'light' | 'dark'>(() =>
    (localStorage.getItem('theme') as 'light' | 'dark') ?? 'light',
  )

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('theme', theme)
  }, [theme])

  useEffect(() => {
    api.config().then(setConfig).catch(() => undefined)
  }, [])

  const generate = useCallback(async () => {
    setBusy(true)
    setError(null)
    try {
      setResult(await api.generate(yaml, filename, settings))
      setStep(STEPS.findIndex((s) => s.id === 'inspect'))
    } catch (e) {
      setError(e instanceof RequestError ? `${e.message} ${e.details.map((d) => d.field).join(', ')}` : 'Generation failed')
    } finally {
      setBusy(false)
    }
  }, [yaml, filename, settings])

  const ready = useMemo(
    () =>
      ({
        upload: yaml.trim().length > 0,
        review: Boolean(review),
        configure: Boolean(review),
        generate: Boolean(review),
        inspect: Boolean(result),
        validation: Boolean(result),
        export: Boolean(result),
      }) as Record<StepId, boolean>,
    [yaml, review, result],
  )

  const furthest = firstIncompleteStep(ready)

  return (
    <div className="app">
      <header className="app-bar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <strong>Formula drafting</strong>
            <small>anthropometric bodice from documented equations</small>
          </div>
        </div>
        <div className="row">
          {result ? (
            <span className={`status ${result.validation.status}`}>
              {result.validation.status === 'pass' ? 'validated' : 'validation failed'}
            </span>
          ) : null}
          <Button variant="ghost" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
            {theme === 'dark' ? 'Light theme' : 'Dark theme'}
          </Button>
        </div>
      </header>

      <Stepper current={step} reachable={(i) => i <= furthest || i === step} onSelect={setStep} />

      <main>
        {error ? (
          <Callout tone="error" title="Request failed">
            {error}
          </Callout>
        ) : null}

        {step === 0 ? (
          <UploadStep
            yaml={yaml}
            filename={filename}
            setYaml={setYaml}
            setFilename={setFilename}
            onUploaded={(r) => {
              setReview(r)
              setStep(1)
            }}
          />
        ) : null}

        {step === 1 && review ? (
          <MeasurementReview review={review} />
        ) : (
          step === 1 ? <p>Import a YAML file first.</p> : null
        )}

        {step === 2 && review ? (
          <ConfigureStep
            settings={settings}
            onChange={setSettings}
            config={config}
            assumptions={result?.trace.assumptions ?? []}
          />
        ) : null}

        {step === 3 ? (
          <div className="stack">
            <h2>Generate the pattern</h2>
            <p className="muted">
              Drafting runs the rule registry end to end and validates the geometry before anything is shown.
            </p>
            <Button variant="primary" onClick={() => void generate()} disabled={busy}>
              {busy ? 'Drafting…' : result ? 'Regenerate pattern' : 'Generate pattern'}
            </Button>
            {busy ? <Spinner label="Drafting panels, running geometry checks…" /> : null}
            {result ? <GenerationStep result={result} /> : null}
          </div>
        ) : null}

        {step === 4 && result ? <PatternViewer preview={result.preview} /> : null}
        {step === 5 && result ? <ValidationView report={result.validation} /> : null}
        {step === 6 && result ? (
          <ExportView result={result} yaml={yaml} filename={filename} settings={settings} />
        ) : null}
      </main>

      <footer className="app-foot">
        <div className="row">
          <Button disabled={step === 0} onClick={() => setStep((s) => Math.max(0, s - 1))}>
            Back
          </Button>
          <Button
            variant="primary"
            disabled={step >= STEPS.length - 1 || !ready[STEPS[step + 1]?.id ?? 'export']}
            onClick={() => setStep((s) => Math.min(STEPS.length - 1, s + 1))}
          >
            Next
          </Button>
        </div>
        <small className="muted">
          Geometry in millimetres · tolerance for endpoint continuity 0.1 mm · no master pattern is loaded
        </small>
      </footer>
    </div>
  )
}
