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

export type Settings = {
  garment_type: string
  section_model: string
  section_exponent: number | null
  cross_section_resolution: number | null
  longitudinal_segments: number | null
  ease_preset: string
  bust_ease_mm: number | null
  waist_ease_mm: number | null
  hip_ease_mm: number | null
  armhole_ease_mm: number | null
  seam_preset: string
  front_neck_drop_mm: number | null
  back_neck_drop_mm: number | null
  flattening_method: string
  w_distortion: number | null
  w_boundary: number | null
  w_seam: number | null
  w_grain: number | null
  w_flip: number | null
  seam_allowances: Record<string, number>
  cutting_join: string | null
  paper_size: string | null
  paper_orientation: string | null
  unit: string
  confirm_units: boolean
}

export type AppConfig = {
  version: string
  crossSection: { model: string; exponent: number; exponent_range: number[] }
  mesh: {
    cross_section_resolution: number
    longitudinal_segments: number
    panel_rows: number
  }
  garment: {
    ease_presets: Record<string, { bust_mm: number; waist_mm: number; hip_mm: number; armhole_mm: number }>
    default_ease_preset: string
  }
  seams: { preset: string; front_neck_drop_mm: number; back_neck_drop_mm: number }
  flattening: { method: string; objective: Record<string, number>; arap: Record<string, number> }
  easePresets: Record<string, { bust_mm: number; waist_mm: number; hip_mm: number; armhole_mm: number }>
  defaultEasePreset: string
  seamAllowances: Record<string, number>
  paper: {
    default_size: string
    default_orientation: string
    sizes_mm: Record<string, number[]>
  }
  tolerances: Record<string, number>
  viewLayers: string[]
  fixtureDefaults: Record<string, unknown>
  debugArtifacts: string[]
  unmappedFields: Record<string, string>
  measurementSchema: {
    schemaVersion: string
    definitionVersion: string
    coordinateSystem: Record<string, unknown>
  }
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
  displacements: unknown[]
}

export type SurfaceMesh = {
  label: string
  columns: number
  rows: number
  vertices: number[][]
  faces: number[][]
  levelRows: Record<string, number>
  rings: unknown[]
}

export type SurfaceSeamPath = {
  panel: string
  role: string
  points: number[][]
}

export type SegmentedSurfacePanel = {
  key: string
  label: string
  vertices: number[][]
  faces: number[][]
  boundary: number[]
  semanticBoundaries: Record<string, number[]>
}

export type SurfaceStages = {
  proxy: SurfaceMesh
  garment: SurfaceMesh
  seamNetwork: Record<string, unknown>
  seamPaths: SurfaceSeamPath[]
  surfaceSeamLengths: Record<string, unknown>[]
  panels: SegmentedSurfacePanel[]
}

export type CrossSections = {
  levels: Record<string, unknown>[]
  sections: {
    name: string
    yMm: number
    halfWidthMm: number
    halfDepthMm: number
    widthMm: number
    depthMm: number
    exponent: number
    targetPerimeterMm: number | null
    achievedPerimeterMm: number
    residualMm: number
    solvedFor: string
    iterations: number
    sourceFields: string[]
  }[]
  model: string
  residualToleranceMm: number
  maxResidualMm: number
}

export type SurfaceAssumption = {
  id: string
  name: string
  value: number | string
  unit: string | null
  reason: string
  sourceFields: string[]
  configurable: boolean
}

export type FlatteningPanelDiagnostics = {
  panel: string
  mesh: Record<string, number>
  solver: { method: string; converged: boolean; iterations: number; [key: string]: unknown }
  distortion: Record<string, number>
  boundaryFits: Record<string, unknown>[]
}

export type FlatteningDiagnostics = {
  method: string
  weights: Record<string, number>
  configVersion: string
  panels: FlatteningPanelDiagnostics[]
  crossSectionResiduals: Record<string, number>
  garmentEaseResiduals: Record<string, number>
  surfaceSeamLengths: Record<string, unknown>[]
  seamBalance: Record<string, unknown>
  smoothing: Record<string, unknown>
  maxFlippedTriangles: number
  maxMeanAngularDistortionDeg: number
}

export type ProgressStage = { stage: string; status: string; elapsedMs: number }

export type GenerationManifest = {
  method: string
  software_version: string
  generated_at: string
  source_file: string
  units: { source: string; pattern: string }
  measurement_schema_version: string
  definition_version: string
  settings: Settings
  assumptions: SurfaceAssumption[]
  warnings: { field: string; code: string; message: string }[]
  anatomical_levels: Record<string, unknown>[]
  cross_sections: CrossSections['sections']
  garment_surface: {
    levels: Record<string, {
      bodyCircumferenceMm?: number
      achievedCircumferenceMm?: number
      requestedEaseMm?: number
      easeResidualMm?: number
    }>
  }
  [key: string]: unknown
}

export type GenerationResponse = {
  pattern: unknown
  trace: GenerationManifest
  diagnostics: FlatteningDiagnostics
  validation: ValidationReport
  preview: Preview
  heatMaps: {
    panel: string
    label: string
    center: number[]
    triangles: number[][][]
    values: number[]
    max: number
  }[]
  settings: Settings
  assumptions: SurfaceAssumption[]
  progress: ProgressStage[]
  flattenedMesh?: {
    panels: {
      panel: string
      vertices2d: number[][]
      vertices3d: number[][]
      faces: number[][]
      boundary: number[]
      semanticBoundaries: Record<string, number[]>
    }[]
  }
  crossSections?: CrossSections
  surfaceStages?: SurfaceStages
}

export type ApiError = {
  code: string
  message: string
  details: { field?: string | null; message: string; code?: string; [key: string]: unknown }[]
}
