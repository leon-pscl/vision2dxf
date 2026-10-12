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

/* ---- base blocks --------------------------------------------------------- */

export type BaseBlockRow = {
  id: string
  label: string
  source: string
  version: string
  reviewed: boolean
  note: string
  validationStatus?: 'pass' | 'fail'
  panelCount?: number
  referenceMeasurementsMm?: Record<string, number>
  hasDarts?: boolean
  stitchPairs?: number
  error?: string
}

export type BaseBlockCatalog = {
  catalogVersion: string
  default: string
  reference: { profile?: string; ease_preset?: string }
  blocks: BaseBlockRow[]
  templateSelection: { supported: boolean; idFormat: string; note: string }
}

export type Landmark = {
  id: string
  panel: string
  side: string
  role: string
  positionMm: number[]
  lockedX: boolean
  lockedY: boolean
  symmetryGroup: string | null
}

export type ReferenceProfile = {
  sourceFile: string
  confirmations: Record<string, string>
  measurementsMm: Record<string, number>
  coordinateOrigin: string
  definitionVersion: string
}

export type BaseBlockDetail = {
  id: string
  version: string
  label: string
  sourceFile: string
  sourceSha256: string
  reviewed: boolean
  sourceLabels: Record<string, string>
  referenceProfile: ReferenceProfile
  referenceEasePreset: string
  referenceMeasurementsMm: Record<string, number>
  garmentDimensionsMm: Record<string, number>
  measurementDefinitions: Record<string, string>
  notes: { field: string; code: string; message: string }[]
  panels: {
    id: string
    key: string
    label: string
    center: number[]
    frame: { axisX: number; neckAtMaxY: boolean; symmetric: boolean; mirrorErrorMm: number }
    levels: Record<string, number>
    edgeRoles: string[]
    grainline: number[][] | null
  }[]
  semanticMap: {
    sides: string[]
    centreLandmarks: string[]
    pairedLandmarks: string[]
    conditionalLandmarks: Record<string, string>
    pairedEdgeRoles: string[]
    seamRoles: string[]
  }
  maps: {
    landmarks: { panels: Record<string, Record<string, Landmark>> }
    edges: { panels: Record<string, { roles: string[]; ids: string[]; labels: string[]; lengthsMm: number[] }> }
    seams: { pairs: { edges: string[]; roles: string[] }[] }
    darts: { panels: Record<string, { edgeId: string; lengthMm: number }[]> }
    grainlines: { panels: Record<string, { fromMm: number[]; toMm: number[] } | null> }
  }
  previewSvg: string
  validation: ValidationReport
  validationStatus: 'pass' | 'fail'
  provenance: string
  influenceMapVersion: string
}

/* ---- differences and the envelope ---------------------------------------- */

export type Delta = {
  name: string
  baseValue: number
  targetValue: number
  deltaValue: number
  pct: number
  unit: string
  region: string
  mapped: boolean
}

export type EnvelopeEntry = {
  name: string
  absoluteDelta: number
  percentageDelta: number
  unit: string
  region: string
  limit: number
  limitPct: number | null
  status: 'ok' | 'warn' | 'stop'
  message: string
}

export type Envelope = {
  status: 'ok' | 'warn' | 'stop'
  confirmed: boolean
  requiresConfirmation: boolean
  solverResidualMm: number | null
  entries: EnvelopeEntry[]
  blocking: { field: string; code: string; message: string; suggestedAction: string }[]
}

export type DeltaResponse = {
  baseBlock: { id: string; label: string; version: string; reviewed: boolean }
  referenceProfile: ReferenceProfile
  referenceEasePreset: string
  baseMeasurementsMm: Record<string, number>
  garmentDimensionsMm: Record<string, number>
  measurementDefinitions: Record<string, string>
  targetEaseMm: { bust_mm: number; waist_mm: number; hip_mm: number; preset: string }
  deltas: Delta[]
  envelope: Envelope
  influenceMap: {
    version: string
    measurements: Record<string, { region: string; ops: { mode: string; roles: string[]; panels: string[]; weight: number }[] }>
  }
}

/* ---- transformation ------------------------------------------------------ */

export type LandmarkTarget = {
  panel: string
  side: string
  role: string
  key: string
  baseMm: number[]
  targetMm: number[]
  displacementMm: number
  ops: {
    measurement: string
    mode: string
    direction?: string
    pivot?: string
    weight?: number
    appliedMm?: number
    appliedDeg?: number
    angleDeg?: number
  }[]
}

export type SolverDiagnostics = {
  panel: string
  method: string
  landmarkCount: number
  iterations: number
  converged: boolean
  lambda: number
  maxResidualMm: number
  meanResidualMm: number
  conditionNumber: number
  residualsMm: Record<string, number>
  constraintViolations: unknown[]
  mergedLandmarks: string[][]
}

export type SeamMismatch = {
  role: string
  edges: string[]
  joins: { panel: string; side: string }[]
  baseLengthsMm: number[]
  lengthsMm: number[]
  baseMismatchMm: number
  mismatchMm: number
  toleranceMm: number
}

export type Diagnostics = {
  method: string
  solver: Record<string, SolverDiagnostics>
  influenceMapVersion: string
  transformationConfigVersion: string
  baseBlock: { id: string; version: string }
  deltas: Delta[]
  envelope: Envelope
  landmarkTargets: LandmarkTarget[]
  seamMismatchBeforeBalancing: SeamMismatch[]
  seamBalanceRounds: { iteration: number; corrections: unknown[]; settled: boolean }[]
  seamMismatchAfterMm: {
    edges: string[]
    lengthsMm: number[]
    mismatchMm: number
    toleranceMm: number
    balanced: boolean
  }[]
  curveRefit: {
    refits: {
      edgeId: string
      role: string
      method: string
      points: number
      lengthMm: number
      baseLengthMm: number
      lengthDeltaMm: number
      maxTurnDeg: number
    }[]
    issues: { check: string; edge?: string; message: string }[]
    snappedLandmarks: unknown[]
  }
  achievedDimensions: Record<string, number>
  requestedDimensions: Record<string, {
    baseMm: number
    requestedMm: number
    contributions: Record<string, number>
    unit: string
  }>
  baseDimensions: Record<string, number>
  warnings: Warning[]
}

export type Manifest = {
  method: string
  measurement_schema_version: string
  definition_version: string
  software_version: string
  generated_at: string
  source_file: string
  units: { source: string; pattern: string }
  coordinate_convention: Record<string, string>
  base_block: {
    id: string
    version: string
    label: string
    source_file: string
    source_sha256: string
    reviewed: boolean
    source_panel_labels: Record<string, string>
    reference_measurements_mm: Record<string, number>
    measurement_definitions: Record<string, string>
    notes: { field: string; code: string; message: string }[]
  }
  transformation_config: {
    version: string
    influence_map_version: string
    objective: Record<string, unknown>
    solver: Record<string, unknown>
    curve_refit: Record<string, unknown>
    seam_balance: Record<string, unknown>
    envelope: Record<string, unknown>
  }
  field_interpretations: Record<string, {
    meaning: string
    status: string
    confirmed_by: string
    candidate: string | null
  }>
  field_records: unknown[]
  derived_measurements: Record<string, { value: number | null; equation: string }>
  drafting_measurements_mm: Record<string, number | null>
  assumptions: Warning[]
  warnings: Warning[]
  settings: Settings
  ease: { preset: string; bustMm: number; waistMm: number; hipMm: number }
  tolerances: Record<string, number>
  fixture_defaults: Record<string, unknown>
  deltas: Delta[]
  envelope: Envelope
  validation: ValidationReport['summary']
}

export type PanelSummary = {
  id: string
  key: string
  label: string
  center: number[]
  edges: { id: string; label: string; role: string; lengthMm: number }[]
  darts: { id: string; intakeMm: number }[]
  landmarks: Record<string, number[]>
}

export type Preview = {
  svg: string
  extent: [number, number, number, number]
  layers: string[]
  comparisonLayers: string[]
  panels: PanelSummary[]
  basePanels: PanelSummary[]
  displacements: LandmarkTarget[]
}

export type TransformResponse = {
  pattern: unknown
  trace: Manifest
  diagnostics: Diagnostics
  validation: ValidationReport
  preview: Preview
  settings: Settings
}

export type Settings = {
  garment_type: string
  /** A catalog id, or `template:<file>.json` to load any template in the library. */
  base_block_id: string | null
  ease_preset: string
  bust_ease_mm: number | null
  waist_ease_mm: number | null
  hip_ease_mm: number | null
  seam_allowances: Record<string, number>
  cutting_join: string | null
  paper_size: string | null
  paper_orientation: string | null
  /** Proceed with a target outside the tested transformation envelope. */
  confirm_envelope: boolean
  unit: string
}

export type AppConfig = {
  easePresets: Record<string, { bust_mm: number; waist_mm: number; hip_mm: number }>
  defaultEasePreset: string
  seamAllowances: Record<string, number>
  paper: { default_size: string; default_orientation: string; sizes_mm: Record<string, number[]> }
  tolerances: Record<string, number>
  viewLayers: string[]
  comparisonLayers: string[]
  fixtureDefaults: Record<string, unknown>
  transformation: {
    version: string
    method: string
    objective: Record<string, unknown>
    solver: Record<string, number | string>
    curveRefit: Record<string, number>
    seamBalance: Record<string, unknown>
    envelope: {
      default: { max_abs_mm: number; max_pct: number }
      stop_factor: number
      measurements: Record<string, Record<string, number>>
    }
    influenceMapVersion: string
  }
}

export type ApiError = {
  code: string
  message: string
  details: { field: string; message: string; code?: string }[]
}
