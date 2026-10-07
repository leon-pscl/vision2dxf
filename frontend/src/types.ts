// Hand-mirrored from backend/contracts.py. Keep the two in step.

export type GarmentType =
  | "polo_shirt"
  | "long_sleeve_polo"
  | "short_sleeve_polo"
  | "slacks";

export type MeasurementBackend = "regression" | "body_model";
export type LandmarksSource = "model" | "heuristic" | "user";
export type DraftRoute = "block" | "garmentcode";
export type StepStatus = "pending" | "done" | "warning";

export interface GarmentSpec {
  is_mock: boolean;
  warnings: string[];
  garment_type: GarmentType;
  required_measurements: string[];
  ease_cm: Record<string, number>;
  design_rules: Record<string, unknown>;
}

export interface CaptureResult {
  is_mock: boolean;
  warnings: string[];
  distance_ok: boolean;
  pose_ok: boolean;
  full_body_visible: boolean;
  clothing_ok: boolean;
  guidance: string[];
}

export interface SegmentationResult {
  is_mock: boolean;
  warnings: string[];
  mask_png: string;
  boundary_quality: number;
  model_name: string;
}

export interface CalibrationResult {
  is_mock: boolean;
  warnings: string[];
  method: "height" | "reference_object";
  px_per_cm: Record<string, number>;
  relative_uncertainty: number;
}

export interface LandmarkPoint {
  x: number;
  y: number;
  confidence: number;
  source: LandmarksSource;
}

export interface LandmarkResult {
  is_mock: boolean;
  warnings: string[];
  points: Record<string, LandmarkPoint>;
  confirmed_by_user: boolean;
}

export interface Measurement {
  value_cm: number;
  lower_cm: number | null;
  upper_cm: number | null;
  method: string;
}

export interface MeasurementResult {
  is_mock: boolean;
  warnings: string[];
  backend: MeasurementBackend | null;
  measurements: Record<string, Measurement>;
}

export interface ValidationFlag {
  measurement: string;
  rule: string;
  severity: "info" | "warning" | "error";
  detail: string;
}

export interface ValidationResult {
  is_mock: boolean;
  warnings: string[];
  flags: ValidationFlag[];
  yaml: string;
}

export interface PatternPiece {
  name: string;
  svg: string;
  seam_allowance_cm: number | null;
  notches: { piece: string; x_cm: number; y_cm: number }[];
}

export interface DraftResult {
  is_mock: boolean;
  warnings: string[];
  route: DraftRoute;
  pieces: PatternPiece[];
}

export interface ProductionResult {
  is_mock: boolean;
  warnings: string[];
  pieces: PatternPiece[];
  export_formats: string[];
}

export interface DrapeResult {
  is_mock: boolean;
  warnings: string[];
  placeholder_image: string;
  toile_checklist: string[];
}

export interface SessionState {
  session_id: string;
  created_at: string;
  results: Record<string, Record<string, unknown>>;
  status: Record<string, { status: StepStatus; is_mock: boolean }>;
  any_mock: boolean;
  landmarks_confirmed: boolean;
}

export interface GarmentChoice {
  type: GarmentType;
  label: string;
  note: string;
}

export const GARMENTS: GarmentChoice[] = [
  {
    type: "polo_shirt",
    label: "Polo shirt",
    note: "Two-piece collar, placket, short turned cuff",
  },
  {
    type: "long_sleeve_polo",
    label: "Long sleeve polo",
    note: "Two-piece collar, placket, long turned cuff",
  },
  {
    type: "short_sleeve_polo",
    label: "Short sleeve polo",
    note: "Flat rib collar, pullover, no placket",
  },
  {
    type: "slacks",
    label: "Slacks",
    note: "Straight leg, waistband, J-stitch fly",
  },
];
