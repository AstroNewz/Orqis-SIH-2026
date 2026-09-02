# CareScan — Issue Tracker

This document tracks genuine discovered defects, blockers, and verification items for the CareScan Patient Mobile Application.

**Rules**:
- Only record actual discovered issues — do not invent bugs.
- Unknown requirements are recorded as verification items, not fake defects.
- Status: `OPEN` | `IN PROGRESS` | `RESOLVED` | `CLOSED` | `WONT FIX`
- Severity: `CRITICAL` | `HIGH` | `MEDIUM` | `LOW` | `VERIFICATION`

---

## Issue Template

```
### ISS-XXX: [Title]

| Field | Value |
|---|---|
| **ID** | ISS-XXX |
| **Status** | OPEN |
| **Severity** | MEDIUM |
| **Title** | [Title] |
| **Description** | [What is the issue] |
| **Expected Behavior** | [What should happen] |
| **Actual Behavior** | [What actually happens] |
| **Reproduction** | [Steps to reproduce] |
| **Impact** | [Effect on users/project] |
| **Resolution** | [How it was fixed, or N/A] |
| **Validation** | [How to verify it's fixed] |
```

---

## Open Issues

*(No open critical or blocking issues at this stage)*

---

## Resolved Issues

### ISS-004: API Contract Undefined

| Field | Value |
|---|---|
| **ID** | ISS-004 |
| **Status** | RESOLVED |
| **Severity** | VERIFICATION |
| **Title** | Backend API contract defined and implemented |
| **Description** | The API endpoints, authentication method, request/response format for assessment submission and history retrieval were pending implementation. |
| **Expected Behavior** | FastAPI endpoints implemented matching Group 1 Flutter models and HL7 FHIR R4 standard. |
| **Actual Behavior** | Fully implemented in `backend/routes/screening_routes.py` and validated via `tests/test_api.py` and `tests/test_e2e.py`. |
| **Reproduction** | POST `/api/screening/analyze`, GET `/api/screening/{id}`, GET `/api/results/{id}`, GET `/api/patients/{id}/history`, GET `/api/screening/{id}/fhir`. |
| **Impact** | Seamless end-to-end integration between Flutter client and FastAPI backend. |
| **Resolution** | Pydantic v2 schemas configured with AliasChoices matching Group 1 Flutter JSON keys (`assessmentId`, `riskLevel`, `details`). |
| **Validation** | All 13 pytest unit and integration tests pass. |


## Resolved Issues

### ISS-001: Flutter Project Not Yet Initialized

| Field | Value |
|---|---|
| **ID** | ISS-001 |
| **Status** | RESOLVED |
| **Severity** | VERIFICATION |
| **Title** | Flutter project not yet initialized in workspace |
| **Description** | The workspace does not contain a Flutter project (`pubspec.yaml`, `lib/`, etc.). |
| **Expected Behavior** | Workspace contains a valid Flutter project. |
| **Actual Behavior** | Project initialized and verified. |
| **Resolution** | Project initialized with complete Clean Architecture and design tokens. |
| **Validation** | `flutter analyze` passes. All tests pass. |

### ISS-002: Stitch Design Token Values Not Yet Extracted

| Field | Value |
|---|---|
| **ID** | ISS-002 |
| **Status** | RESOLVED |
| **Severity** | VERIFICATION |
| **Title** | Exact Stitch design token values require inspection |
| **Description** | Extracted color, typography, spacing, and shape tokens from the current Stitch designs. |
| **Expected Behavior** | Design tokens in code match the current approved Stitch design. |
| **Actual Behavior** | Tokens codified in `AppColors`, `AppTypography`, `AppSpacing`, `AppShapes`, and `AppTheme`. |
| **Resolution** | Full Stitch project inspected and verified across all 7 screens. |
| **Validation** | Design system tests and UI visual checks pass. |

### ISS-003: Navigation Pattern Unconfirmed

| Field | Value |
|---|---|
| **ID** | ISS-003 |
| **Status** | RESOLVED |
| **Severity** | VERIFICATION |
| **Title** | Primary navigation pattern requires Stitch confirmation |
| **Description** | Whether the app uses bottom navigation, drawer, or another pattern. |
| **Expected Behavior** | Navigation pattern is confirmed and documented. |
| **Actual Behavior** | Confirmed bottom navigation bar with central FAB. |
| **Resolution** | Implemented using `go_router` and `StatefulShellRoute.indexedStack`. |
| **Validation** | Navigation tests pass. |

### ISS-005: State Management Approach Not Finalized

| Field | Value |
|---|---|
| **ID** | ISS-005 |
| **Status** | RESOLVED |
| **Severity** | VERIFICATION |
| **Title** | State management library decision pending |
| **Description** | The specific state management approach needs to be selected. |
| **Expected Behavior** | Clean state management pattern with error/loading/success handling. |
| **Actual Behavior** | State management implemented via typed `AsyncState` + `Result` + `Repository` pattern. |
| **Resolution** | Standard reactive state pattern implemented without external bloated dependencies. |
| **Validation** | T-DATA-04 complete, all screen state transitions tested. |

### ISS-006: Camera Scan Design Missing

| Field | Value |
|---|---|
| **ID** | ISS-006 |
| **Status** | RESOLVED |
| **Severity** | HIGH |
| **Title** | Camera Scan UI missing from Stitch |
| **Description** | Camera Scan screen presence verified from Stitch project 4290608824728960355. |
| **Expected Behavior** | Stitch contains HTML design for Camera Scan interface. |
| **Actual Behavior** | Confirmed present and implemented faithfully. |
| **Resolution** | Re-inspected Stitch project and implemented T-SCREEN-02. |
| **Validation** | CameraScanScreen matches Stitch specs with hardware preview, animated scan line, and permission handling. |

### ISS-007: Risk Band Boundaries Were Unreachable

| Field | Value |
|---|---|
| **ID** | ISS-007 |
| **Status** | RESOLVED |
| **Severity** | CRITICAL |
| **Title** | Configured screening and high-risk thresholds could never be crossed |
| **Description** | `SCREENING_THRESHOLD = 0.50` and `HIGH_RISK_THRESHOLD = 0.70` were chosen before the calibrator existed. The trained model's highest calibrated probability on the held-out test partition is 0.4127, so neither boundary was reachable. |
| **Expected Behavior** | A screening tool flags cases above its threshold and can report a HIGH risk band. |
| **Actual Behavior** | At 0.50 the screen flagged nothing at all on test; validation sensitivity was 0.048 (1 of 21 positives). The HIGH band was unreachable code the API could never return. |
| **Resolution** | Bands are now derived per model from out-of-fold calibrated probabilities on train+validation and persisted in `calibration.json`; see DEC-018. For `v1-handcrafted`: screening 0.0407, HIGH 0.1887. Config values retained as fallbacks only. |
| **Validation** | `backend.evaluation.evaluate` asserts reachability against the observed test score range: `screening_threshold_reachable_observed` and `high_risk_threshold_reachable_observed` both now `true`. A degenerate-band guard fires if the HIGH boundary is ever not above the screening threshold. |

### ISS-008: Quantum Model Underperforms Classical Baselines

| Field | Value |
|---|---|
| **ID** | ISS-008 |
| **Status** | OPEN — reported as measured, not a defect to hide |
| **Severity** | HIGH |
| **Title** | The VQC ranks last of four models on the held-out test set |
| **Description** | On the held-out test partition the calibrated VQC scores PR-AUC 0.2180 / ROC-AUC 0.7671, against random forest 0.5750 / 0.9411, gradient boosting 0.5625 / 0.9327 and logistic regression 0.5125 / 0.9256. It also degraded from validation (PR-AUC 0.4636 → 0.2180), indicating a generalisation gap rather than only a capacity limit. |
| **Expected Behavior** | PART 10 requires the quantum contribution to be *evaluated*, not assumed. There is no requirement that it win. |
| **Actual Behavior** | It loses, consistently, on both validation and test. |
| **Analysis** | Plausible contributors, none yet isolated: amplitude encoding into 8 qubits compresses 181 features into 256 amplitudes whose relative magnitudes carry the signal, and L2 normalisation discards overall feature scale that the tree ensembles use freely; the 2-layer ansatz has only 32 parameters against a 600-tree forest; SPSA at maxiter 200 may be underconverged. The PART 16 ablations (qubit count 8/10/12/16, circuit depth, SPSA vs COBYLA) exist to test these and have not all been run. |
| **Constraint on interpretation** | 18 test positives. The ordering is not statistically meaningful and must not be quoted as a general result about quantum machine learning. The evaluator emits this caveat automatically whenever `n_positive < 20`. |

### ISS-009: Calibration Mapping Is Steep Enough to Be Shot-Noise Sensitive

| Field | Value |
|---|---|
| **ID** | ISS-009 |
| **Status** | OPEN |
| **Severity** | MEDIUM |
| **Title** | Platt slope of −35.94 amplifies small raw-score changes |
| **Description** | The fitted Platt mapping has `a = −35.94`, concentrating almost the entire probability transition into a raw-score window of roughly [0.55, 0.62]. A raw-score shift of 0.01 near the transition moves the reported probability by tens of percentage points. |
| **Expected Behavior** | Reported probabilities should be stable against the sampling noise of the execution backend. |
| **Actual Behavior** | Under `ideal_simulation` the raw score is exact, so this is currently latent. Under `noisy_simulation` or real hardware, shot noise at 1024 shots gives the expectation value a standard error near 0.03 — larger than the transition window. |
| **Impact** | The calibrator is fitted per execution mode (`scored_in_mode`), and `backend.evaluation.evaluate` warns when the evaluation mode differs from the fitting mode, so the mismatch cannot pass silently. But a mode-matched calibrator does not remove the underlying variance: the *same image* may receive materially different probabilities across two hardware runs. |
| **Proposed Resolution** | Report a probability interval rather than a point estimate when `is_exact` is false, derived from the shot-noise standard error propagated through the calibrator. Not yet implemented. |

### ISS-010: Clinical Feature Channel Is Constant Within Every Patient

| Field | Value |
|---|---|
| **ID** | ISS-010 |
| **Status** | OPEN — documented limitation |
| **Severity** | MEDIUM |
| **Title** | Clinical-only mode cannot discriminate within a patient, and 44 training patients carry both labels |
| **Description** | 0 of 230 training patients have a non-constant clinical vector across their images, which is expected — habits and demographics are patient-level attributes. Separately, 44 of 230 training patients have images labelled both positive and negative. |
| **Impact** | For those 44 patients the clinical-only configuration is being asked to produce two different answers from one identical input vector, which bounds achievable clinical-only performance by construction. Measured: clinical-only PR-AUC 0.2052 (LR) / 0.2047 (GB) on validation, against multimodal 0.5873 / 0.5986. |
| **Why not "fixed"** | The label is per-image (a lesion is present in this photograph) while the clinical vector is per-patient. That is a property of the dataset, not a bug. It is recorded because it explains the clinical-only ceiling and because the ablation numbers would otherwise look like an encoder defect. |
| **Confirmed non-issue** | A leakage audit verified the clinical encoder reads only habit and demographic columns (Smoking/Alcohol/Arecanut/Chewing families, `Habit_history`, `Type_of_habit`, `Form_of_tobacco`, `Age`, `Sex`) and no diagnosis field. The high ROC-AUC is not target leakage. |

---

## Summary

| Severity | Open | Resolved |
|---|---|---|
| CRITICAL | 0 | 1 |
| HIGH | 1 | 1 |
| MEDIUM | 2 | 0 |
| LOW | 0 | 0 |
| VERIFICATION | 0 | 5 |
| **Total** | **3** | **7** |

Open items ISS-008, ISS-009 and ISS-010 are **measured findings recorded honestly**, not unresolved defects. ISS-008 in particular is the answer to the question PART 10 asks, and closing it by tuning against the held-out set would be the actual error.


