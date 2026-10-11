export type Warning = {
  code: string
  field: string | null
  message: string
  original_value?: unknown
  action?: string
}

export type MeasurementRow = {
  name: string
  label: string
  value: number
  unit: string
  valueMm: number
  kind: string
  origin: string
  direction: string
  landmark: string
  interpretation: string
  min: number | null
  max: number | null
  usedBy: string[]
}

export type Review = {
  ok: boolean
  filename: string
  unit: string
  rows: MeasurementRow[]
  warnings: Warning[]
  dictionary: { unit: string; checks: { id: string; description: string }[]; fieldCount: number }
}

export type CheckResult = {
  id: string
  status: 'pass' | 'fail'
  severity: 'critical' | 'warning'
  message: string
  affectedIds: string[]
  metrics: Record<string, unknown>
  suggestedAction: string
}

export type ValidationReport = {
  status: 'pass' | 'fail'
  passed: boolean
  summary: { total: number; passed: number; failed: number; criticalFailures: number }
  metrics: Record<string, unknown>
  warnings: unknown[]
  checks: CheckResult[]
  tolerances: Record<string, number>
}

export type RuleTrace = {
  ruleId: string
  name: string
  outputName: string
  group: string
  inputs: Record<string, number>
  value: number
  explanation: string
  sourceNote: string
}

export type Manifest = {
  method: string
  ruleRegistryVersion: string
  sourceFile: string
  units: { input: string; geometry: string }
  settings: Settings
  ease: { preset: string; bustMm: number; waistMm: number; hipMm: number }
  assumptions: Warning[]
  tolerances: Record<string, number>
  fixtureDefaults: Record<string, unknown>
  rules: RuleTrace[]
  validation: ValidationReport['summary']
}

export type Preview = {
  svg: string
  extent: [number, number, number, number]
  layers: string[]
  panels: {
    id: string
    label: string
    center: number[]
    edges: { id: string; label: string; role: string; lengthMm: number }[]
    darts: { id: string; intakeMm: number }[]
  }[]
}

export type GenerateResponse = {
  pattern: unknown
  trace: Manifest
  validation: ValidationReport
  preview: Preview
  settings: Settings
}

export type Settings = {
  garment_type: string
  ease_preset: string
  bust_ease_mm: number | null
  waist_ease_mm: number | null
  hip_ease_mm: number | null
  seam_allowances: Record<string, number>
  cutting_join: string | null
  paper_size: string | null
  paper_orientation: string | null
  shoulder_slope_deg: number | null
  armscye_depth_mm: number | null
  bust_apex_y_mm: number | null
  dart_intake_mm: number | null
  unit: string
}

export type AppConfig = {
  easePresets: Record<string, { bust_mm: number; waist_mm: number; hip_mm: number }>
  defaultEasePreset: string
  seamAllowances: Record<string, number>
  paper: { default_size: string; default_orientation: string; sizes_mm: Record<string, number[]> }
  tolerances: Record<string, number>
  viewLayers: string[]
  fixtureDefaults: Record<string, unknown>
}

export type ApiError = {
  code: string
  message: string
  details: { field: string; message: string; code?: string }[]
}
