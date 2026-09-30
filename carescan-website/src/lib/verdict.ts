/**
 * Presentation rules for a screening verdict (DEC-034).
 *
 * Why this file exists
 * --------------------
 * The backend returns two different scores and it is dangerously easy to render
 * the wrong one as the headline:
 *
 *   - `primaryRiskLevel` / `primaryProbability` come from the strongest *validated*
 *     model on this dataset, which is currently the classical baseline. That model
 *     sets the band the clinician reads. But when `primaryCalibrated` is false its
 *     score is a RANKING, not a probability -- "0.82" does not mean "82% chance".
 *     Printing it with a percent sign would be the single most misleading thing
 *     the portal could do.
 *
 *   - `finalProbability` is the calibrated quantum probability. It is a real
 *     probability and it is the value that goes into FHIR, but on this dataset the
 *     quantum model has not beaten the classical baseline, so it is reported as a
 *     clearly-labelled *secondary* readout and never as the verdict.
 *
 * Every component that shows a score imports from here rather than formatting a
 * number itself.
 */

import type { AssessmentResult, ModelInfo, WorklistItem } from './apiTypes';

export type BandTone = 'low' | 'moderate' | 'high' | 'unknown';

/**
 * Verbatim wording required whenever the quantum readout is shown next to the
 * headline. Kept as a constant so it cannot drift between screens.
 */
export const QUANTUM_SECONDARY_NOTE =
  'The current dataset has not demonstrated a classical bottleneck that justifies ' +
  'quantum processing. The quantum model is reported alongside the verdict for ' +
  'transparency, not as the basis for it.';

/** Shown in place of a percentage for an uncalibrated primary score. */
export const RANKING_NOT_PROBABILITY =
  'Ranking score — not a probability. This model orders cases by suspicion; the ' +
  'number is not a percentage chance of cancer.';

/**
 * Why `details` is not printed under the headline band.
 *
 * The backend builds `details` in the same breath as the *quantum* band, from the
 * quantum probability (`quantum_ml/calibration.py`). The band on screen is the
 * primary model's (DEC-034). When the two models disagree — which on this dataset
 * they routinely do — putting `details` under the headline produces a flat
 * contradiction: a LOW RISK heading above the sentence "risk estimate exceeds the
 * experimental screening threshold. Professional clinical evaluation is
 * recommended."
 *
 * So `details` is rendered verbatim, but beside the readout it actually describes.
 */
/**
 * Attribution shown beside `details`. Deliberately position-neutral — it is
 * rendered under the narrative on the result screen and in the demo, and a
 * "below"/"above" in here would rot the moment either layout moved.
 */
export const DETAILS_BELONG_TO_QUANTUM =
  'This narrative was written by the quantum model for its own band, not for the ' +
  'headline verdict.';

/** Points a reader to where the per-capture narrative went. */
export const DETAILS_MOVED_NOTE =
  'The band above is set by the validated classical baseline. The quantum model’s ' +
  'own narrative for this capture is shown with the secondary readout below.';

/**
 * The band to display, per DEC-034: the primary model's band when the result has
 * one, otherwise the quantum band.
 */
export function displayedBand(
  result: Pick<AssessmentResult, 'primaryRiskLevel' | 'riskLevel'> | WorklistItem,
): string | null {
  if ('primaryRiskLevel' in result && result.primaryRiskLevel) {
    return result.primaryRiskLevel;
  }
  return result.riskLevel ?? null;
}

/** Map any band spelling onto a visual tone. Unrecognised bands stay neutral. */
export function bandTone(band: string | null | undefined): BandTone {
  if (!band) return 'unknown';
  const upper = band.toUpperCase();
  if (upper.includes('HIGH')) return 'high';
  if (upper.includes('MODERATE') || upper.includes('MEDIUM')) return 'moderate';
  if (upper.includes('LOW')) return 'low';
  return 'unknown';
}

/**
 * True when the headline score must NOT be rendered as a percentage.
 *
 * Deliberately conservative: an absent `primaryCalibrated` is treated as
 * uncalibrated, because suppressing a percentage that could have been shown is a
 * harmless loss of detail, while showing one that should have been suppressed is
 * a clinical misstatement.
 */
export function isRankingScore(
  result: Pick<AssessmentResult, 'primaryCalibrated' | 'primaryProbability'> | WorklistItem,
): boolean {
  return result.primaryCalibrated !== true;
}

/** Format a probability in [0,1] as a percentage. Returns null for absent values. */
export function asPercent(value: number | null | undefined, digits = 1): string | null {
  if (value == null || !Number.isFinite(value)) return null;
  return `${(value * 100).toFixed(digits)}%`;
}

/** Format a ranking score as a bare number, with no unit that implies probability. */
export function asRanking(value: number | null | undefined, digits = 3): string | null {
  if (value == null || !Number.isFinite(value)) return null;
  return value.toFixed(digits);
}

/** Format a signed expectation value such as the Pauli-Z readout. */
export function asExpectation(value: number | null | undefined, digits = 3): string | null {
  if (value == null || !Number.isFinite(value)) return null;
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}

/**
 * Human label for an execution mode. Never invent hardware: `ideal_simulation`
 * says "simulator", because claiming a Heron run that did not happen is a
 * fabricated provenance claim.
 */
export function executionModeLabel(mode: string | null | undefined): string {
  switch (mode) {
    case 'ideal_simulation':
      return 'Ideal statevector simulation (Qiskit Aer)';
    case 'noisy_simulation':
      return 'Noisy simulation (Qiskit Aer noise model)';
    case 'ibm_hardware':
      return 'IBM Quantum hardware';
    case 'mock':
      return 'Mock stub — no inference was run';
    default:
      return mode ? String(mode) : 'Unknown';
  }
}

/** Tailwind classes per tone, matching the site's existing badge palette. */
export const TONE_CLASSES: Record<
  BandTone,
  { text: string; bg: string; border: string; bar: string; dot: string }
> = {
  low: {
    text: 'text-emerald-800 dark:text-emerald-300',
    bg: 'bg-emerald-50 dark:bg-emerald-950/50',
    border: 'border-emerald-200 dark:border-emerald-800',
    bar: 'bg-emerald-500',
    dot: 'bg-emerald-500',
  },
  moderate: {
    text: 'text-amber-800 dark:text-amber-300',
    bg: 'bg-amber-50 dark:bg-amber-950/50',
    border: 'border-amber-200 dark:border-amber-800',
    bar: 'bg-amber-500',
    dot: 'bg-amber-500',
  },
  high: {
    text: 'text-rose-800 dark:text-rose-300',
    bg: 'bg-rose-50 dark:bg-rose-950/50',
    border: 'border-rose-200 dark:border-rose-800',
    bar: 'bg-rose-500',
    dot: 'bg-rose-500',
  },
  unknown: {
    text: 'text-slate-700 dark:text-slate-300',
    bg: 'bg-slate-50 dark:bg-slate-900',
    border: 'border-slate-200 dark:border-slate-800',
    bar: 'bg-slate-400',
    dot: 'bg-slate-400',
  },
};

/** Badge variant names used by the site's existing `<Badge>` component. */
export function badgeVariantForTone(
  tone: BandTone,
): 'success' | 'warning' | 'danger' | 'neutral' {
  switch (tone) {
    case 'low':
      return 'success';
    case 'moderate':
      return 'warning';
    case 'high':
      return 'danger';
    default:
      return 'neutral';
  }
}

/** Round a millisecond duration for display. `548.3462999691255` is not a fact. */
export function asMillis(value: number | null | undefined): string | null {
  if (value == null || !Number.isFinite(value)) return null;
  return `${value < 10 ? value.toFixed(2) : value.toFixed(1)} ms`;
}

/**
 * One factual line describing how the *primary* model scored a capture.
 *
 * Used where a compact row needs a subtitle under the band. It restates the
 * numbers already labelled on the result screen and asserts nothing beyond them,
 * which is the point: the alternative is `details`, which narrates the quantum
 * band and therefore contradicts the band shown beside it.
 */
export function primaryScoreLine(
  result: Pick<
    AssessmentResult,
    'primaryModel' | 'primaryProbability' | 'primaryThreshold' | 'primaryCalibrated'
  >,
): string | null {
  const score = isRankingScore(result)
    ? asRanking(result.primaryProbability)
    : asPercent(result.primaryProbability);
  if (score == null) return null;

  const unit = isRankingScore(result) ? 'ranking score' : 'calibrated probability';
  const threshold = asRanking(result.primaryThreshold);
  const model = result.primaryModel ? `${result.primaryModel} · ` : '';

  return threshold
    ? `${model}${unit} ${score}, decision threshold ${threshold}`
    : `${model}${unit} ${score}`;
}

/**
 * Flatten `GET /api/model/info` into the fields the System panel shows.
 *
 * The ready response nests almost everything (`quantum.circuit_depth`,
 * `calibration.method`, `pipeline.feature_mode`); a backend still loading reports
 * a shallower object with the same names at the top level. Reading only the flat
 * names — which is what this panel used to do — renders six populated fields as
 * em dashes and makes a healthy backend look broken.
 */
export function summariseModelInfo(model: ModelInfo): {
  ready: boolean | undefined;
  modelVersion: string | null;
  qubits: number | null;
  circuitDepth: number | null;
  calibrationMethod: string | null;
  executionMode: string | null;
  backendName: string | null;
  featureMode: string | null;
  descriptorDimension: number | null;
  clinicalFeatureCount: number | null;
  classicalReference: string | null;
  thresholdsValidated: boolean | null;
  fellBack: boolean | null;
  fallbackReason: string | null;
} {
  const first = <T>(...values: (T | null | undefined)[]): T | null =>
    values.find((value) => value != null) ?? null;

  return {
    ready: model.ready,
    modelVersion: first(model.model_version),
    qubits: first(model.quantum?.n_qubits, model.pipeline?.n_qubits, model.n_qubits),
    circuitDepth: first(model.quantum?.circuit_depth, model.circuit_depth),
    calibrationMethod: first(model.calibration?.method, model.calibration_method),
    executionMode: first(model.quantum?.execution_mode, model.execution_mode),
    backendName: first(model.quantum?.backend_name, model.backend_name),
    featureMode: first(model.pipeline?.feature_mode, model.feature_mode),
    descriptorDimension: first(model.expected_descriptor_dimension),
    // `clinical_features_expected` is a boolean flag, not a count. The count is
    // the encoder's own feature width.
    clinicalFeatureCount: first(
      model.pipeline?.clinical_encoder?.n_features,
      model.pipeline?.fusion?.n_clinical_features,
    ),
    classicalReference: first(model.classical_reference),
    thresholdsValidated: first(model.calibration?.thresholds_are_clinically_validated),
    fellBack: first(model.quantum?.fell_back),
    fallbackReason: first(model.quantum?.fallback_reason),
  };
}
