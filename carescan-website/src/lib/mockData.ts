import { TestCasePreset } from '@/types';

/**
 * Written clinical presentations used by the triage sandbox.
 *
 * These are *descriptions*, not photographs, and they carry no scores. Earlier
 * revisions shipped pre-baked `expectedQuantumExpectation` / `expectedFinalProb`
 * values and a `calculateSimulatedRisk()` helper that turned them into a verdict
 * in the browser. That produced clinical-looking numbers no model had ever
 * computed, so both are gone: the sandbox submits these cases to the backend as
 * labelled test data and renders whatever the server returns.
 */
export const TEST_CASE_PRESETS: TestCasePreset[] = [
  {
    id: 'case-benign',
    name: 'Healthy Buccal Mucosa',
    category: 'Normal / Benign',
    clinicalDescription:
      'Uniform pink mucosal surface with smooth vascularization and no demonstrable ulceration, keratosis, or architectural disruption.',
    lifestyle: {
      smoking: false,
      alcohol: false,
      betelQuid: false,
    },
    clinicalSeverity: 'Low',
  },
  {
    id: 'case-moderate',
    name: 'Oral Lichenoid Keratosis',
    category: 'Low-Grade Dysplasia / Reactive',
    clinicalDescription:
      'Reticular white striae on right lateral buccal mucosa with localized mild erythema. Patient reports mild episodic spicy food sensitivity.',
    lifestyle: {
      smoking: true,
      alcohol: false,
      betelQuid: false,
    },
    clinicalSeverity: 'Moderate',
  },
  {
    id: 'case-high',
    name: 'Homogeneous Leukoplakia with Ulceration',
    category: 'High-Risk Premalignant Lesion',
    clinicalDescription:
      'Non-scrapable thick, elevated white plaque on left lateral tongue border with central focal erythematous ulceration (>1.2 cm). Longstanding betel quid exposure.',
    lifestyle: {
      smoking: true,
      alcohol: true,
      betelQuid: true,
    },
    clinicalSeverity: 'High',
  },
];
