import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, RequestError, type GenerationPayload } from './api'
import { Stepper } from './components/Stepper'
import { Button, Callout, Spinner } from './components/ui'
import { UploadStep } from './upload/UploadStep'
import { DefinitionsStep } from './semantics/DefinitionsStep'
import { MeasurementReview } from './measurement-review/MeasurementReview'
import { ConfigureStep } from './configure/ConfigureStep'
import { GenerationStep } from './generate/GenerationStep'
import { DistortionHeatMap } from './flattening-view/DistortionHeatMap'
import { PatternViewer } from './pattern-viewer/PatternViewer'
import { SurfaceReview } from './surface-view/SurfaceReview'
import { ValidationView } from './validation-view/ValidationView'
import { ExportView } from './export-view/ExportView'
import { STEPS, firstIncompleteStep, stepIndex, type StepId } from './steps'
import type { AppConfig, GenerationResponse, Review, Settings } from './types'

const DEFAULT_SETTINGS: Settings = {
  garment_type: 'basic_torso',
  section_model: 'superellipse',
  section_exponent: null,
  cross_section_resolution: null,
  longitudinal_segments: null,
  ease_preset: 'fitted',
  bust_ease_mm: null,
  waist_ease_mm: null,
  hip_ease_mm: null,
  armhole_ease_mm: null,
  seam_preset: 'basic_torso_v1',
  front_neck_drop_mm: null,
  back_neck_drop_mm: null,
  flattening_method: 'lscm_arap',
  w_distortion: null,
  w_boundary: null,
  w_seam: null,
  w_grain: null,
  w_flip: null,
  seam_allowances: {},
  cutting_join: null,
  paper_size: null,
  paper_orientation: null,
  unit: 'cm',
  confirm_units: false,
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
  const [result, setResult] = useState<GenerationResponse | null>(null)
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
    api.config()
      .then(setConfig)
      .catch((e: unknown) => {
        setError(e instanceof RequestError ? e.message : 'Could not load reconstruction configuration.')
      })
  }, [])

  const invalidate = () => setResult(null)

  const generate = useCallback(async () => {
    setBusy(true)
    setError(null)
    try {
      const generated = await api.generate(input, settings)
      setResult(generated)
      setStep(stepIndex('cross-sections'))
    } catch (e) {
      setError(e instanceof RequestError
        ? `${e.message}${e.details.length ? `: ${e.details.map((d) => d.message).join('; ')}` : ''}`
        : 'Surface reconstruction failed.')
    } finally {
      setBusy(false)
    }
  }, [input, settings])

  const ready = useMemo(
    () =>
      ({
        upload: input.yamlText.trim().length > 0 && Boolean(review),
        definitions: semanticsReady,
        review: semanticsReady && Boolean(review),
        configure: semanticsReady,
        generate: semanticsReady && settings.confirm_units,
        'cross-sections': Boolean(result?.crossSections),
        surface: Boolean(result?.surfaceStages),
        flattening: Boolean(result),
        pattern: Boolean(result),
        validation: Boolean(result),
        export: Boolean(result),
      }) as Record<StepId, boolean>,
    [input.yamlText, review, result, semanticsReady, settings.confirm_units],
  )

  const furthest = firstIncompleteStep(ready)

  return (
    <div className="app">
      <header className="app-bar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <strong>Anthropometric surface reconstruction</strong>
            <small>Measured proxy torso · seam segmentation · computational flattening</small>
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
        {error ? <Callout tone="error" title="Request failed">{error}</Callout> : null}

        {step === stepIndex('upload') ? (
          <UploadStep
            yaml={input.yamlText}
            filename={input.filename}
            setYaml={(yamlText) => {
              setInput((prev) => ({ ...EMPTY_PAYLOAD, yamlText, filename: prev.filename }))
              setReview(null)
              setSemanticsReady(false)
              invalidate()
            }}
            setFilename={(filename) => {
              setInput((prev) => ({ ...EMPTY_PAYLOAD, filename, yamlText: prev.yamlText }))
              setReview(null)
              setSemanticsReady(false)
              invalidate()
            }}
            onUploaded={(nextReview) => {
              setReview(nextReview)
              setSemanticsReady(false)
              invalidate()
              setStep(stepIndex('definitions'))
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
            onBack={() => setStep(stepIndex('upload'))}
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
          ) : <p>Upload a measurement file first.</p>
        ) : null}

        {step === stepIndex('configure') ? (
          <ConfigureStep
            settings={settings}
            config={config}
            onChange={(next) => {
              setSettings(next)
              invalidate()
            }}
          />
        ) : null}

        {step === stepIndex('generate') ? (
          <div className="stack">
            <h2>Generate from the reconstructed surface</h2>
            <p className="muted">
              The pipeline solves anatomical cross-sections, lofts and eases a proxy torso,
              segments it along the fixed seam network, then flattens each triangulated panel.
              It does not use a two-dimensional master block or drafting-formula sequence.
            </p>
            {!semanticsReady ? (
              <Callout tone="warn">Resolve and confirm every required measurement definition before generating.</Callout>
            ) : null}
            {!settings.confirm_units ? (
              <Callout tone="warn">Confirm the measurement units in Configure before generating.</Callout>
            ) : null}
            <Button variant="primary" onClick={() => void generate()} disabled={busy || !ready.generate}>
              {busy ? 'Reconstructing and flattening…' : result ? 'Regenerate surface pattern' : 'Generate pattern'}
            </Button>
            {busy ? <Spinner label="Solving sections, reconstructing surfaces, segmenting seams, flattening panels, and validating geometry…" /> : null}
            {result ? <GenerationStep result={result} /> : null}
          </div>
        ) : null}

        {step === stepIndex('cross-sections') && result ? (
          <SurfaceReview result={result} view="sections" />
        ) : null}

        {step === stepIndex('surface') && result ? (
          <SurfaceReview result={result} view="surfaces" />
        ) : null}

        {step === stepIndex('flattening') && result ? (
          <div className="stack">
            <GenerationStep result={result} />
            <DistortionHeatMap result={result} />
          </div>
        ) : null}

        {step === stepIndex('pattern') && result ? (
          <div className="stack">
            <h2>Flattened two-dimensional panels</h2>
            <p className="muted">
              Pattern geometry is reconstructed from the triangulated garment surface.
              Drag to pan, scroll to zoom, and use the ruler to inspect lengths in millimetres.
            </p>
            <PatternViewer preview={result.preview} />
          </div>
        ) : null}

        {step === stepIndex('validation') && result ? (
          <ValidationView report={result.validation} />
        ) : null}

        {step === stepIndex('export') && result ? (
          <ExportView result={result} input={input} settings={settings} />
        ) : null}
      </main>

      <footer className="app-foot">
        <Button variant="ghost" disabled={step === 0} onClick={() => setStep((current) => Math.max(0, current - 1))}>
          Previous
        </Button>
        <Button
          variant="primary"
          disabled={step >= STEPS.length - 1 || !ready[STEPS[step + 1]?.id ?? 'export']}
          onClick={() => setStep((current) => Math.min(STEPS.length - 1, current + 1))}
        >
          Next
        </Button>
        <small className="muted">
          Coordinates in millimetres · proxy geometry is assumption-recorded · 2D panels come from surface flattening
        </small>
      </footer>
    </div>
  )
}
