import { useCallback, useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { ErrorBoundary } from './components/ErrorBoundary'
import { api, RequestError, type GenerationPayload } from './api'
import { Stepper } from './components/Stepper'
import { Button, Callout, Spinner } from './components/ui'
import { UploadStep } from './upload/UploadStep'
import { DefinitionsStep } from './semantics/DefinitionsStep'
import { MeasurementReview } from './measurement-review/MeasurementReview'
import { BaseBlockStep } from './base-block/BaseBlockStep'
import { ConfigureStep } from './configure/ConfigureStep'
import { GenerationStep } from './generate/GenerationStep'
import { ComparisonViewer } from './comparison-viewer/ComparisonViewer'
import { ValidationView } from './validation-view/ValidationView'
import { ExportView } from './export-view/ExportView'
import { STEPS, firstIncompleteStep, stepIndex, type StepId } from './steps'
import type { AppConfig, Review, Settings, TransformResponse } from './types'

const DEFAULT_SETTINGS: Settings = {
  garment_type: 'basic_bodice',
  base_block_id: null,
  ease_preset: 'fitted',
  bust_ease_mm: null,
  waist_ease_mm: null,
  hip_ease_mm: null,
  seam_allowances: {},
  cutting_join: null,
  paper_size: null,
  paper_orientation: null,
  confirm_envelope: false,
  unit: 'cm',
}

const EMPTY_PAYLOAD: GenerationPayload = {
  yamlText: '',
  filename: 'body_measurements.yaml',
  confirmations: {},
  corrections: {},
}

export default function App() {
  const location = useLocation()
  const navigate = useNavigate()
  const currentId: StepId = (STEPS.some((s) => s.id === location.pathname.slice(1))
    ? location.pathname.slice(1)
    : 'upload') as StepId
  const step = stepIndex(currentId)
  const go = (id: StepId) => navigate('/' + id)
  const [input, setInput] = useState<GenerationPayload>(EMPTY_PAYLOAD)
  const [review, setReview] = useState<Review | null>(null)
  const [semanticsReady, setSemanticsReady] = useState(false)
  const [blockReady, setBlockReady] = useState(false)
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS)
  const [result, setResult] = useState<TransformResponse | null>(null)
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
      setResult(await api.transform(input, settings))
      go('compare')
    } catch (e) {
      setError(e instanceof RequestError ? e.message : 'Transformation failed')
      go('base')
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
        base: semanticsReady,
        configure: semanticsReady && blockReady,
        generate: semanticsReady && blockReady,
        compare: Boolean(result),
        validation: Boolean(result),
        export: Boolean(result),
      }) as Record<StepId, boolean>,
    [input, semanticsReady, blockReady, result],
  )

  const furthest = firstIncompleteStep(ready)
  const invalidate = () => {
    setResult(null)
  }

  return (
    <div className="app">
      <header className="app-bar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <strong>Base-block transformation</strong>
            <small>a validated pattern block, deformed to measured body data</small>
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

      <Stepper current={step} reachable={(i) => i <= furthest || i === step} onSelect={(i) => go(STEPS[i].id)} />

      <main>
        {error ? (
          <Callout tone="error" title="Request failed">
            {error}
          </Callout>
        ) : null}

        {step === stepIndex('upload') ? (
          <UploadStep
            yaml={input.yamlText}
            filename={input.filename}
            setYaml={(yamlText) => setInput((prev) => ({ ...EMPTY_PAYLOAD, yamlText, filename: prev.filename }))}
            setFilename={(filename) => setInput((prev) => ({ ...EMPTY_PAYLOAD, filename, yamlText: prev.yamlText }))}
            onUploaded={(r) => {
              setReview(r)
              setSemanticsReady(false)
              setBlockReady(false)
              invalidate()
              go('definitions')
            }}
          />
        ) : null}

        {step === stepIndex('definitions') ? (
          <DefinitionsStep
            input={input}
            onChange={(next) => {
              setInput(next)
              invalidate()
              setSemanticsReady(false)
            }}
            onBack={() => go('upload')}
            onReady={() => setSemanticsReady(true)}
          />
        ) : null}

        {step === stepIndex('review') ? (
          review ? (
            <MeasurementReview
              review={review}
              confirmations={input.confirmations}
              onCorrections={(corrections) => {
                setInput({ ...input, corrections })
                invalidate()
              }}
            />
          ) : (
            <p>Import a YAML file first.</p>
          )
        ) : null}

        {step === stepIndex('base') ? (
          <ErrorBoundary>
          <BaseBlockStep
            input={input}
            settings={settings}
            onChange={(next) => {
              setSettings(next)
              invalidate()
            }}
            onReady={setBlockReady}
          />
          </ErrorBoundary>
        ) : null}

        {step === stepIndex('configure') ? (
          <ConfigureStep
            settings={settings}
            onChange={(next) => {
              setSettings(next)
              invalidate()
            }}
            config={config}
            assumptions={result?.trace.assumptions ?? []}
          />
        ) : null}

        {step === stepIndex('generate') ? (
          <div className="stack">
            <h2>Transform the base block</h2>
            <p className="muted">
              The block is loaded, validated, and deformed onto the target landmarks with a thin-plate
              spline. Nothing is redrafted from a formula: panel count, boundary order, edge roles,
              identifiers, stitches and grain direction all survive the transformation.
            </p>
            <Button variant="primary" onClick={() => void generate()} disabled={busy || !ready.generate}>
              {busy ? 'Transforming…' : result ? 'Transform again' : 'Transform pattern'}
            </Button>
            {!semanticsReady ? (
              <Callout tone="warn">
                Every required field needs a confirmed definition before a transformation is allowed.
              </Callout>
            ) : null}
            {semanticsReady && !blockReady ? (
              <Callout tone="warn">
                Pick a base block whose validation passes, and confirm the transformation limits if the
                target sits outside the tested range.
              </Callout>
            ) : null}
            {busy ? <Spinner label="Solving the deformation, balancing seams, running geometry checks…" /> : null}
            {result ? <GenerationStep result={result} /> : null}
          </div>
        ) : null}

        {step === stepIndex('compare') && result ? (
          <ErrorBoundary>
          <ComparisonViewer preview={result.preview} diagnostics={result.diagnostics} />
          </ErrorBoundary>
        ) : null}
        {step === stepIndex('validation') && result ? <ValidationView report={result.validation} /> : null}
        {step === stepIndex('export') && result ? (
          <ExportView result={result} input={input} settings={settings} />
        ) : null}
      </main>

      <footer className="app-foot">
        <div className="row">
          <Button disabled={step === 0} onClick={() => step > 0 && go(STEPS[step - 1].id)}>
            Back
          </Button>
          <Button
            variant="primary"
            disabled={step >= STEPS.length - 1 || !ready[STEPS[step + 1]?.id ?? 'export']}
            onClick={() => step < STEPS.length - 1 && go(STEPS[step + 1].id)}
          >
            Next
          </Button>
        </div>
        <small className="muted">
          Geometry in millimetres · endpoint tolerance 0.1 mm · the base block is deformed, never redrafted
        </small>
      </footer>
    </div>
  )
}
