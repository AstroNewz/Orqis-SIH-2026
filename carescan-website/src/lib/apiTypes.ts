/**
 * Wire types for the Orqis FastAPI backend.
 *
 * These mirror `backend/schemas/result.py`, `backend/schemas/screening.py` and
 * `backend/schemas/clinic.py`. Field names are camelCase because the backend
 * serialises with camelCase aliases; the one exception is `quantumVisual`, whose
 * inner object deliberately keeps snake_case (it reuses the QuantumVisualSummary
 * contract rather than duplicating a projection that could drift).
 *
 * Nothing here is inferred or guessed. If a field is optional below it is because
 * the backend can genuinely omit it -- most often because no trained artifact is
 * loaded, or because the value is live-only and absent when a result is re-read.
 */

// ---------------------------------------------------------------- Screening

/** Risk band exactly as the backend stores and returns it. */
export type RiskBand = 'LOW RISK' | 'MODERATE RISK' | 'HIGH RISK';

/**
 * How the quantum circuit was actually executed. `mock` means the labelled
 * development stub ran and the numbers carry no clinical meaning at all.
 */
export type ExecutionMode =
  | 'ideal_simulation'
  | 'noisy_simulation'
  | 'ibm_hardware'
  | 'mock';

/** EXPERIMENTAL E2 quantum-visual telemetry (DEC-035). Never the verdict. */
export interface QuantumVisualSummary {
  model_version?: string | null;
  n_qubits?: number | null;
  execution_mode?: string | null;
  backend_name?: string | null;
  circuit_depth?: number | null;
  raw_score?: number | null;
  probability?: number | null;
  execution_time_ms?: number | null;
  [key: string]: unknown;
}

/** `GET /api/results/{id}` and `POST /api/screening/analyze`. */
export interface AssessmentResult {
  id: string;
  assessmentId: string;
  riskLevel: string;
  details: string;

  classicalProbability?: number | null;
  quantumProbability?: number | null;
  finalProbability?: number | null;
  threshold?: number | null;
  classification?: string | null;
  modelVersion?: string | null;
  quantumQubits?: number | null;
  /** Null under exact statevector simulation, where nothing is sampled. */
  quantumShots?: number | null;
  executionTimeMs?: number | null;
  isMock: boolean;
  createdAt?: string | null;

  // Provenance
  inferenceId?: string | null;
  /** Quantum measurement before calibration. Not a probability. */
  rawScore?: number | null;
  probabilityUncalibrated?: number | null;
  highRiskThreshold?: number | null;
  calibrationMethod?: string | null;
  bandsSource?: string | null;
  executionMode?: ExecutionMode | string | null;
  backendName?: string | null;
  circuitDepth?: number | null;
  featureMode?: 'image_only' | 'clinical_only' | 'multimodal' | string | null;
  quantumTimeMs?: number | null;

  // Displayed (headline) verdict -- DEC-034.
  // When `primaryCalibrated` is false, `primaryProbability` is a RANKING SCORE and
  // must never be rendered as a percentage. See `isRankingScore()` in ./verdict.
  primaryModel?: string | null;
  primaryRiskLevel?: string | null;
  primaryProbability?: number | null;
  primaryThreshold?: number | null;
  primaryCalibrated?: boolean | null;

  /** Live-only: null when the result is re-read from history. */
  quantumVisual?: QuantumVisualSummary | null;

  /** Required patient-facing wording. The backend attaches it to every result. */
  disclaimer: string;
}

/** `POST /api/screening/upload`. */
export interface UploadResponse {
  status: string;
  image_path: string;
  patient_id: string;
  size_bytes: string;
}

/**
 * `POST /api/screening/analyze` payload.
 *
 * The three risk factors are tri-state on purpose: `false` means the patient was
 * asked and said no, `null`/omitted means it was not collected. The clinical
 * encoder keeps those apart, so never collapse one into the other.
 */
export interface AnalyzeRequest {
  patient_id?: string | null;
  image_path: string;
  scan_type?: string;
  smoking_history?: boolean | null;
  alcohol_consumption?: boolean | null;
  betel_quid?: boolean | null;
  age?: number | null;
  sex?: string | null;
  features?: number[] | null;
  is_mock?: boolean;
}

/** `GET /api/screening/{id}`. */
export interface Assessment {
  id: string;
  imagePath: string;
  timestamp: string;
  type: string;
}

/** `GET /api/patients/{id}/history`. */
export interface HistoryEntry {
  assessment: Assessment;
  result: AssessmentResult | null;
}

/** `POST /api/localize`. Advisory overlay only -- never gates a screening. */
export interface LocalizationResult {
  status: string;
  localized: boolean;
  confidence: number;
  box_normalised: number[] | null;
  roi_box_pixels: number[] | null;
  source_width: number;
  source_height: number;
  roi_source: string | null;
  reasons: string[];
  localizer_version: string;
}

// ------------------------------------------------------------------- System

/** `GET /health`. */
export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
  /** False when no trained artifacts are loaded. The process is still healthy. */
  model_ready: boolean;
  model_version: string | null;
  /** Effective mode. Differs from `_configured` when IBM hardware was unreachable. */
  quantum_execution_mode: string;
  quantum_execution_mode_configured: string;
  quantum_backend: string;
  quantum_qubits: number;
}

/** `GET /api/model/info`. Shape varies with readiness; read defensively. */
/**
 * `GET /api/model/info`.
 *
 * The live backend groups most of this under `quantum`, `calibration` and
 * `pipeline`; only `ready`, `model_version` and the descriptor dimension sit at
 * the top level. The flat variants below are kept because a backend that has not
 * finished loading reports a shallower object. Read it through
 * {@link summariseModelInfo} rather than reaching for either shape directly.
 */
export interface ModelInfo {
  ready?: boolean;
  model_version?: string | null;
  quantum?: {
    n_qubits?: number | null;
    circuit_depth?: number | null;
    execution_mode?: string | null;
    backend_name?: string | null;
    shots?: number | null;
    is_exact?: boolean | null;
    /** True when a requested hardware backend was unavailable and Aer ran instead. */
    fell_back?: boolean | null;
    fallback_reason?: string | null;
  } | null;
  calibration?: {
    method?: string | null;
    /** False on this deployment, and load-bearing: the bands are not validated. */
    thresholds_are_clinically_validated?: boolean | null;
  } | null;
  pipeline?: {
    feature_mode?: string | null;
    n_qubits?: number | null;
    clinical_encoder?: { n_features?: number | null } | null;
    fusion?: { n_clinical_features?: number | null } | null;
  } | null;
  expected_descriptor_dimension?: number | null;
  /**
   * A BOOLEAN — whether the model expects clinical features at all, not how many
   * there are. The count lives at `pipeline.clinical_encoder.n_features`. Reading
   * this as a number renders a blank cell, because React prints `true` as nothing.
   */
  clinical_features_expected?: boolean | null;
  /** Which classical model sets the displayed band (DEC-034). */
  classical_reference?: string | null;
  /** Flat fallbacks. Present only on a not-yet-ready response. */
  n_qubits?: number | null;
  circuit_depth?: number | null;
  calibration_method?: string | null;
  execution_mode?: string | null;
  backend_name?: string | null;
  feature_mode?: string | null;
  [key: string]: unknown;
}

// ------------------------------------------------------------ Clinic portal

export interface ClinicUser {
  id: string;
  email: string;
  clinicId: string;
  fullName: string | null;
  role: string;
}

export interface LoginResponse {
  accessToken: string;
  tokenType: string;
  expiresInMinutes: number;
  user: ClinicUser;
}

export interface WorklistItem {
  screeningId: string;
  patientId: string;
  scanType: string;
  status: string;
  createdAt: string;
  /** The *displayed* band (DEC-034): primary model's band when present. */
  riskLevel: string | null;
  primaryModel: string | null;
  primaryCalibrated: boolean | null;
  /** Calibrated quantum probability. Secondary readout, not the verdict. */
  finalProbability: number | null;
  threshold: number | null;
  modelVersion: string | null;
  isMock: boolean;
  hasResult: boolean;
}

export interface WorklistResponse {
  items: WorklistItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface ClinicPatientSummary {
  patientId: string;
  screeningCount: number;
  lastScreeningAt: string | null;
  latestRiskLevel: string | null;
  createdAt: string;
  isActive: boolean;
}

export interface ClinicPatientsResponse {
  items: ClinicPatientSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface ClinicStats {
  clinicId: string;
  totalScreenings: number;
  totalPatients: number;
  completedScreenings: number;
  pendingScreenings: number;
  /** Results flagged is_mock. Development stubs, not clinical throughput. */
  mockResults: number;
  /** Keyed by the band string as stored, e.g. `"HIGH RISK"`. */
  bandCounts: Record<string, number>;
}

/** FHIR R4 bundle from `GET /api/screening/{id}/fhir`. Rendered, not parsed. */
export type FhirBundle = Record<string, unknown>;
