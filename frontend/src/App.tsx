import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, RequestError, type GenerationPayload } from './api'
import { Stepper } from './components/Stepper'
import { Button, Callout, Spinner } from './components/ui'
import { UploadStep } from './upload/UploadStep'
import { DefinitionsStep } from './semantics/DefinitionsStep'
import { MeasurementReview } from './measurement-review/MeasurementReview'
import { ConfigureStep } from './configure/ConfigureStep'
import { GenerationStep } from './generate/GenerationStep'
import { PatternViewer } from './pattern-viewer/PatternViewer'
import { ValidationView } from './validation-view/ValidationView'
import { ExportView } from './export-view/ExportView'
import { STEPS, firstIncompleteStep, stepIndex, type StepId } from './steps'
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

const EMPTY_PAYLOAD: GenerationPayload = {
  yamlText: '',
  filename: 'body_measurements.yaml',
  confirmations: {},
  corrections: {},
}

export default function App() {
  const [step, setStep] = useState(0)
  const [input, setInput] = useState<GenerationPayload>(EMPTY_PAYLOAD)
  const [review, setReview] = useState<Review | null>(null)
  const [semanticsReady, setSemanticsReady] = useState(false)
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
      setResult(await api.generate(input, settings))
      setStep(stepIndex('inspect'))
    } catch (e) {
      setError(e instanceof RequestError ? e.message : 'Generation failed')
      setStep(stepIndex('definitions'))
    } finally {
      setBusy(false)
    }
  }, [input, settings])

  const ready = useMemo(
    () =>
      ({
        upload: input.yamlText.trim().length > 0,
        definitions: semanticsReady,
        review: semanticsReady,
        configure: semanticsReady,
        generate: semanticsReady,
        inspect: Boolean(result),
        validation: Boolean(result),
        export: Boolean(result),
      }) as Record<StepId, boolean>,
    [input, semanticsReady, result],
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
            yaml={input.yamlText}
            filename={input.filename}
            setYaml={(yamlText) => setInput({ ...EMPTY_PAYLOAD, yamlText, filename: input.filename })}
            setFilename={(filename) => setInput({ ...EMPTY_PAYLOAD, filename, yamlText: input.yamlText })}
            onUploaded={(r) => {
              setReview(r)
              setSemanticsReady(false)
              setStep(stepIndex('definitions'))
            }}
          />
        ) : null}

        {step === 1 ? (
          <DefinitionsStep
            input={input}
            onChange={(next) => {
              setInput(next)
              setResult(null)
              setSemanticsReady(false)
            }}
            onBack={() => setStep(0)}
            onReady={() => setSemanticsReady(true)}
          />
        ) : null}

        {step === 2 && review ? (
          <MeasurementReview
            review={review}
            confirmations={input.confirmations}
            onCorrections={(corrections) => setInput({ ...input, corrections })}
          />
        ) : (
          step === 2 ? <p>Import a YAML file first.</p> : null
        )}

        {step === 3 ? (
          <ConfigureStep
            settings={settings}
            onChange={setSettings}
            config={config}
            assumptions={result?.trace.assumptions ?? []}
          />
        ) : null}

        {step === 4 ? (
          <div className="stack">
            <h2>Generate the pattern</h2>
            <p className="muted">
              Drafting runs the rule registry end to end and validates the geometry before anything is shown.
              Confirmations from the definitions screen travel with the request.
            </p>
            <Button variant="primary" onClick={() => void generate()} disabled={busy || !semanticsReady}>
              {busy ? 'Drafting…' : result ? 'Regenerate pattern' : 'Generate pattern'}
            </Button>
            {!semanticsReady ? (
              <Callout tone="warn">
                Every required field needs a confirmed definition before generation is allowed.
              </Callout>
            ) : null}
            {busy ? <Spinner label="Drafting panels, running geometry checks…" /> : null}
            {result ? <GenerationStep result={result} /> : null}
          </div>
        ) : null}

        {step === 5 && result ? <PatternViewer preview={result.preview} /> : null}
        {step === 6 && result ? <ValidationView report={result.validation} /> : null}
        {step === 7 && result ? (
          <ExportView result={result} input={input} settings={settings} />
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
          Geometry in millimetres · endpoint tolerance 0.1 mm · no master pattern is loaded
        </small>
      </footer>
    </div>
  )
}
