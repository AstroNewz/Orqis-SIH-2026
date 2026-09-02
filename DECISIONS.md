# CareScan — Decision Log

This document records durable architectural, product, and engineering decisions for the CareScan Patient Mobile Application.

**Rules**:
- Decisions are immutable once `ACCEPTED` (they can be superseded by a new decision, not edited).
- Status: `PROPOSED` | `ACCEPTED` | `SUPERSEDED` | `REJECTED`
- Include rationale and alternatives to provide future context.

---

## DEC-001: New Independent Project

| Field | Value |
|---|---|
| **ID** | DEC-001 |
| **Date** | 2026-08-29 |
| **Status** | ACCEPTED |
| **Context** | The CareScan Patient Mobile Application needs a clean workspace to establish the project without inheriting technical debt, design decisions, or implementation patterns from any prior project. |
| **Decision** | CareScan starts as a new, independent project in `/Users/pratyakshranjan/workinggggg`. No code, documentation, configuration, or design decisions are migrated or referenced from any prior project (including QuOra). |
| **Rationale** | A clean start ensures the project is built on current requirements and the current Stitch design, not outdated assumptions. |
| **Consequences** | All architecture, code, and configuration must be created from scratch. No shortcuts from prior work. |
| **Alternatives** | Fork/migrate from prior project — rejected due to risk of inheriting outdated design and technical debt. |

---

## DEC-002: Workspace Isolation

| Field | Value |
|---|---|
| **ID** | DEC-002 |
| **Date** | 2026-08-29 |
| **Status** | ACCEPTED |
| **Context** | The workspace `/Users/pratyakshranjan/workinggggg` must be the sole project workspace. Other directories (e.g., `/Users/pratyakshranjan/QuOra`) are out of scope. |
| **Decision** | All project operations are confined to the `workinggggg` workspace. No inspection, comparison, or reuse of files from other project directories. |
| **Rationale** | Prevents accidental contamination from prior project code or decisions. |
| **Consequences** | Agents must not access, reference, or copy from other workspaces. |
| **Alternatives** | Shared workspace — rejected to maintain project isolation. |

---

## DEC-003: Previous QuOra Project Out of Scope

| Field | Value |
|---|---|
| **ID** | DEC-003 |
| **Date** | 2026-08-29 |
| **Status** | ACCEPTED |
| **Context** | A prior project (QuOra) exists elsewhere. It must not influence CareScan. |
| **Decision** | QuOra is treated as nonexistent for this project. No migration, reference, comparison, or reuse of any kind. |
| **Rationale** | CareScan must be built on its own requirements and current design, not on legacy implementation. |
| **Consequences** | Potentially longer initial setup, but cleaner foundation. |
| **Alternatives** | Selective migration — rejected per explicit project requirement. |

---

## DEC-004: Current Stitch Design as Visual Source of Truth

| Field | Value |
|---|---|
| **ID** | DEC-004 |
| **Date** | 2026-08-29 |
| **Status** | ACCEPTED |
| **Context** | The app's UI must faithfully implement a visual design. Multiple versions of design may exist (old screenshots, prior implementation, current Stitch). |
| **Decision** | The current approved Google Stitch project ("CareScan Mobile Patient Portal") is the authoritative visual source of truth. Prior screenshots or implementations do not override it. |
| **Rationale** | Single authoritative source prevents design drift and conflicting implementations. |
| **Consequences** | When Stitch cannot be inspected, values must be marked `VERIFY WITH STITCH` rather than guessed. UI may require revision when Stitch is inspected. |
| **Alternatives** | Use prior implementation as reference — rejected because it may not reflect current design. |

---

## DEC-005: Flutter / Dart Implementation Technology

| Field | Value |
|---|---|
| **ID** | DEC-005 |
| **Date** | 2026-08-29 |
| **Status** | ACCEPTED |
| **Context** | The application needs a cross-platform mobile framework targeting Android and iOS. |
| **Decision** | Use Flutter with Dart for the CareScan patient application. |
| **Rationale** | Flutter provides cross-platform support, strong widget system, good performance, and active ecosystem. Explicitly specified as the project technology. |
| **Consequences** | All application code is Dart. Platform-specific code uses Flutter's platform channel mechanism when needed. |
| **Alternatives** | React Native, native Android+iOS — not considered; Flutter specified by project requirements. |

---

## DEC-006: Human-Created AAS Skills as External Expertise

| Field | Value |
|---|---|
| **ID** | DEC-006 |
| **Date** | 2026-08-29 |
| **Status** | ACCEPTED |
| **Context** | The project uses human-created Agentic Awesome Skills (AAS) located in `.agent/skills/` as engineering and design guidance. |
| **Decision** | AAS skills remain the external expertise layer, provided and managed separately by the user. AI agents must not create, modify, copy, or recreate SKILL.md files. |
| **Rationale** | Skills are human-curated expertise. AI-generated replacements would dilute their value and create confusion. |
| **Consequences** | Agents use skills as-is for guidance. Project documentation (PROJECT.md, DESIGN.md, etc.) is separate from skills. |
| **Alternatives** | AI-generated skills — rejected per explicit project rule. |

---

## DEC-007: Project Documentation Separate from Skills

| Field | Value |
|---|---|
| **ID** | DEC-007 |
| **Date** | 2026-08-29 |
| **Status** | ACCEPTED |
| **Context** | The project needs both engineering guidance (skills) and project-specific documentation (requirements, architecture, design, tasks, issues, decisions). |
| **Decision** | Project documentation files (PROJECT.md, REQUIREMENTS.md, ARCHITECTURE.md, etc.) are maintained at the project root, separate from `.agent/skills/`. They serve different purposes. |
| **Rationale** | Skills are reusable engineering expertise. Project docs are project-specific control files. Mixing them creates confusion. |
| **Consequences** | Clear separation of concerns. Skills inform engineering. Project docs control the specific CareScan implementation. |
| **Alternatives** | Embed project docs in skills — rejected for clarity. |

---

## DEC-008: Atomic Task Execution

| Field | Value |
|---|---|
| **ID** | DEC-008 |
| **Date** | 2026-08-29 |
| **Status** | ACCEPTED |
| **Context** | Large monolithic implementation steps are error-prone and hard to validate. |
| **Decision** | Implementation uses atomic tasks — small, independently understandable, testable, and validatable units of work. |
| **Rationale** | Atomic tasks reduce risk, simplify validation, and make progress trackable. |
| **Consequences** | More tasks in TASKS.md, but each is safer and easier to verify. |
| **Alternatives** | Large feature-level tasks — rejected because they're harder to validate incrementally. |

---

## DEC-009: No Invented Clinical Functionality

| Field | Value |
|---|---|
| **ID** | DEC-009 |
| **Date** | 2026-08-29 |
| **Status** | ACCEPTED |
| **Context** | CareScan handles health-related image assessment. It is critical that no clinical behavior is fabricated. |
| **Decision** | Clinical, ML, diagnostic, or medical functionality must not be invented. If backend/ML behavior is unavailable, use a clean abstraction (UI → Service Interface → Mock). Mock data must be clearly identifiable as test data. |
| **Rationale** | Invented clinical functionality could mislead users or create liability. |
| **Consequences** | UI development proceeds with mock services. Real clinical behavior comes only from confirmed backend/ML integration. |
| **Alternatives** | Simulate realistic-looking results — rejected due to safety and trust concerns. |

---

## Decision Template

```
## DEC-XXX: [Title]

| Field | Value |
|---|---|
| **ID** | DEC-XXX |
| **Date** | YYYY-MM-DD |
| **Status** | PROPOSED |
| **Context** | [Why this decision is needed] |
| **Decision** | [What was decided] |
| **Rationale** | [Why this option was chosen] |
| **Consequences** | [What follows from this decision] |
| **Alternatives** | [What else was considered and why it was rejected] |
```
## DEC-010: Centralized Routing with go_router

| Field | Value |
|---|---|
| **ID** | DEC-010 |
| **Date** | 2026-08-29 |
| **Status** | ACCEPTED |
| **Context** | The app requires centralized declarative routing to handle navigation between the Home Dashboard, Blogs, Camera Scan, History, Profile, and other assessment screens. |
| **Decision** | Use `go_router` for centralized routing. Define routes for all confirmed top-level destinations in `app_router.dart`. |
| **Rationale** | `go_router` provides robust declarative routing as specified by `ARCHITECTURE.md`, enabling clean deep linking and simplified navigation logic compared to Navigator 2.0. |
| **Consequences** | `go_router` becomes a core dependency. All navigation must use `context.go()` or `context.push()` instead of raw `Navigator` methods. |
| **Alternatives** | Custom Navigator 2.0 (rejected as overly complex), standard Navigator 1.0 (rejected per architecture's declarative requirement). |

---

## DEC-011: Unit Test File Co-location Layout

| Field | Value |
|---|---|
| **ID** | DEC-011 |
| **Date** | 2026-08-29 |
| **Status** | ACCEPTED |
| **Context** | T-TEST-01 prescribed `test/unit/` as the validation path. All required unit tests for models, error types, and result pattern were written adjacent to their sources (`test/core/errors/`, `test/features/assessment/models/`, `test/features/settings/models/`). |
| **Decision** | Retain tests co-located under `test/` mirroring the `lib/` structure. Do not create a flat `test/unit/` directory. |
| **Rationale** | Co-location is idiomatic in Dart/Flutter projects and was the layout established when T-ARCH-02 and T-DATA-01 tests were written. Moving tests to `test/unit/` would be pure reorganisation with no coverage gain, violating the minimal-change rule. All tests pass under `flutter test`. |
| **Consequences** | The TASKS.md validation command `flutter test test/unit/` is not literally honoured; `flutter test` (full suite) is used instead. |
| **Alternatives** | Move all unit tests to `test/unit/` — rejected as speculative reorganisation with no functional benefit. |

---

## DEC-012: FastAPI & Qiskit Aer Stack for Group 3 Backend and QML Engine

| Field | Value |
|---|---|
| **ID** | DEC-012 |
| **Date** | 2026-08-30 |
| **Status** | ACCEPTED |
| **Context** | Group 3 requires an asynchronous API gateway and a local-first, hardware-aware Quantum Machine Learning pipeline implementing amplitude encoding and a Variational Quantum Classifier (VQC). |
| **Decision** | Use FastAPI + Pydantic v2 + SQLAlchemy for backend services and Qiskit + Qiskit Aer (`AerSimulator`) for quantum circuit execution. Real IBM Quantum credentials remain strictly optional and are never required for local dev or tests. |
| **Rationale** | Qiskit and Qiskit Aer provide exact compliance with the research methodology in `QuOra.pdf` (parameterized $R_z(\psi)R_y(\phi)$ rotations, hardware coupling topology, and Pauli-$Z$ measurement). Decoupling the execution layer ensures simulation runs with zero cloud credentials. |
| **Consequences** | Python backend and QML algorithms execute locally and can be containerized via Docker Compose. |
| **Alternatives** | PennyLane (rejected to maintain direct alignment with Qiskit NISQ hardware transpilation); Custom matrix simulator (rejected per project constraint). |

---

## DEC-013: Relational Data Model and HL7 FHIR R4 Interoperability

| Field | Value |
|---|---|
| **ID** | DEC-013 |
| **Date** | 2026-08-30 |
| **Status** | ACCEPTED |
| **Context** | The backend must persist screening events with data minimization and provide interoperable clinical records matching Group 1's Flutter client contract and healthcare data exchange standards. |
| **Decision** | Implement SQLite/PostgreSQL tables using UUID v4 pseudonymous keys (`patients`, `screenings`, `screening_results`, `audit_logs`). Pydantic models expose camelCase aliases matching Group 1 Flutter models (`assessmentId`, `riskLevel`, `details`) and export HL7 FHIR R4 `Observation` and `RiskAssessment` resources. |
| **Rationale** | Ensures zero friction with Group 1's Flutter models while meeting clinical compliance, auditability, and interoperability requirements. |
| **Consequences** | Seamless integration between frontend and backend with full FHIR compliance. |
| **Alternatives** | Raw unstructured JSON blobs — rejected due to lack of type safety and query performance. |

---

## DEC-014: Standard Dart HTTP Client and Remote Integration Layer

| Field | Value |
|---|---|
| **ID** | DEC-014 |
| **Date** | 2026-08-30 |
| **Status** | ACCEPTED |
| **Context** | Flutter mobile client requires communication with FastAPI backend endpoints (`/api/screening/analyze` and `/api/patients/{id}/history`) across Android emulator, iOS simulator, and physical iPhone with robust error translation. |
| **Decision** | Use standard library `dart:io` `HttpClient` in `AssessmentRemoteDataSourceImpl` with `ApiConstants` dynamic base URL resolution (`10.0.2.2:8000` on Android emulator, `localhost:8000` on iOS simulator, configurable `--dart-define=BACKEND_BASE_URL` or `setBaseUrl` for physical devices). Translate socket/timeout exceptions to `NetworkFailure` and non-2xx status to `ServerFailure` within `ApiAssessmentRepository`. |
| **Rationale** | Avoids introducing heavy external networking packages when standard Dart libraries fulfill all timeout, JSON serialization, and streaming requirements cleanly. Preserves domain `Result<T>` pattern and allows `MockAssessmentRepository` to remain intact for tests. |
| **Consequences** | Zero dependency bloat in `pubspec.yaml`; clean separation between data source and domain repository. |
| **Alternatives** | Dio / Http package (rejected to avoid unnecessary dependency churn). |

---

## DEC-015: Patient-Level Dataset Splitting with a Persisted Manifest

| Field | Value |
|---|---|
| **ID** | DEC-015 |
| **Date** | 2026-09-01 |
| **Status** | ACCEPTED |
| **Context** | The SMART-OM dataset contains multiple images per patient (2436 images across 326 patients). Random image-level splitting puts different photographs of the same lesion on both sides of the split, which inflates every metric and is the single most common way a medical imaging result turns out to be meaningless. |
| **Decision** | Split by patient, never by image. The split is deterministic (seed 42), stratified by patient-level class where metadata allows, and persisted as `backend/artifacts/dataset/split_manifest.json`. `SplitManifest.assert_no_patient_leakage` runs when the manifest is written **and again on every `load_partitions` call**, not only in the test suite. |
| **Rationale** | Verifying on read as well as write means a manifest copied between machines or regenerated by hand cannot quietly reintroduce leakage. The check is cheap and the failure mode it prevents is silent. |
| **Consequences** | Resulting partitions: train 1692 images / 230 patients / 104 positive, validation 363 / 48 / 21, test 381 / 48 / 18. Test prevalence is 4.72%. With only 18 test positives, differences between models are not statistically meaningful — recorded as a limitation, not worked around. |
| **Alternatives** | Random image-level split (rejected: leakage). Leave-one-patient-out CV (rejected: 326 fits per experiment is not affordable for a VQC whose training is hundreds of circuit evaluations). |

---

## DEC-016: No-Pickle Artifact Policy

| Field | Value |
|---|---|
| **ID** | DEC-016 |
| **Date** | 2026-09-01 |
| **Status** | ACCEPTED |
| **Context** | Fitted scikit-learn estimators are conventionally persisted with `pickle`. Loading a pickle executes arbitrary code in the loading process, and pickles break across library versions — both unacceptable for an artifact that a backend loads at startup and that must stay reproducible. |
| **Decision** | No artifact in this repository is a pickle. Logistic regression persists as JSON coefficients plus intercept and is re-applied as a dot product by `LogisticBaseline`, which does not import scikit-learn. Isotonic calibration persists as its `(x_thresholds_, y_thresholds_)` knots and is re-applied with `numpy.interp`. Tree ensembles persist as their `BaselineRecord` and are **refit from the record**, which is exact. |
| **Rationale** | Refit-from-record doubles as a test: if the record does not fully determine the model, the refit differs and the discrepancy is visible. Verified — all three baselines refit bit-identically (`max|original - refit| = 0.000e+00`). |
| **Consequences** | `RandomForestClassifier` is constructed with `n_jobs=None` (serial) rather than `n_jobs=-1`. With thread parallelism the per-tree probability summation order depends on thread scheduling, which changed the last bit of every predicted probability (observed: 1.11e-16). Slower fits were traded for an auditable reproducibility claim. Serving a tree ensemble requires the training partition to be present; only logistic regression is loadable standalone. |
| **Alternatives** | Pickle / joblib (rejected: code execution on load, version fragility). ONNX (rejected: a new dependency and a new format to validate, for three baseline models). |

---

## DEC-017: Calibration Method Selected by Out-of-Fold CV with a Paired-SE Tie Rule

| Field | Value |
|---|---|
| **ID** | DEC-017 |
| **Date** | 2026-09-01 |
| **Status** | ACCEPTED |
| **Context** | The VQC's raw output `(1 - <Z_0>)/2` is bounded and monotone but is not a probability: trained scores span roughly [0.38, 0.59] against a 6% base rate. Reporting that as "48% risk" would be wrong by nearly an order of magnitude. Both Platt scaling and isotonic regression are defensible, and picking by in-sample fit always favours isotonic, which has far more freedom to overfit. |
| **Decision** | Fit on train+validation only — never test — and select between Platt and isotonic by **out-of-fold** stratified k-fold CV inside that fitting set. Selection uses a paired standard-error rule rather than `min(brier)`: the more flexible model must beat the two-parameter one by more than 1 SE of the paired difference. |
| **Rationale** | With 125 positives in the fitting set, a raw Brier comparison is well inside noise. Measured: `Brier(isotonic) − Brier(Platt) = +0.000916` against a paired SE of `0.000518`, so Platt wins the tie on parsimony rather than on a coin flip. |
| **Consequences** | Platt selected. Out-of-fold Brier 0.05101, log-loss 0.18970, ECE 0.00706 (isotonic: 0.05192 / 0.21552 / 0.01124). Out-of-fold mean predicted 0.06084 against observed 0.06083. The fitted mapping is steep (`a = −35.94`), which is itself a robustness concern under shot noise — see ISSUES.md. |
| **Limitation** | CV folds are stratified by label but **not grouped by patient**, so two images of one patient can land in different folds and the out-of-fold calibration metrics are slightly optimistic. Grouping was rejected because 125 positives across 5 patient-grouped folds gives fold sizes too small for a stable isotonic fit. The held-out test evaluation is unaffected, since patients never cross partitions. |
| **Alternatives** | Fit calibration on test (rejected: forbidden, and would invalidate the only clean estimate). Always Platt (rejected: PART 14 asks for the choice to be made on data and documented). |

---

## DEC-018: Risk Band Boundaries Are Per-Model and Derived from Out-of-Fold Probabilities

| Field | Value |
|---|---|
| **ID** | DEC-018 |
| **Date** | 2026-09-02 |
| **Status** | ACCEPTED |
| **Context** | `SCREENING_THRESHOLD = 0.50` and `HIGH_RISK_THRESHOLD = 0.70` were round-number defaults. Once the calibrator was fitted, both turned out to be **unreachable**: the highest calibrated probability the trained model emits on the test partition is 0.4127. At 0.50 the screen flagged nothing whatsoever (sensitivity 0.048 on validation — 1 true positive out of 21), and the HIGH band was dead code the API could never return. This is not a calibration failure; it is what being calibrated at 6% prevalence means, combined with band boundaries chosen before the score distribution was known. |
| **Decision** | Band boundaries are a property of a specific trained model, not a global constant. `backend.training.calibrate` derives both from **out-of-fold** calibrated probabilities on train+validation and persists them in `calibration.json` under `resolved_bands`. The screening threshold is derived for `SCREENING_TARGET_SENSITIVITY` (0.85); the HIGH boundary is derived for `HIGH_RISK_TARGET_SPECIFICITY` (0.95), because the upper band answers "who first", not "who at all". `ProbabilityCalibrator.from_calibration_payload` reads them, falling back field-by-field to the configured values and reporting which was used via `bands_source`. |
| **Rationale** | Deriving them out-of-fold rather than in-sample avoids a threshold that is too high because in-sample positives score better than field ones. Deriving them on train+validation rather than test keeps the held-out estimate clean. Persisting them with the model means they travel with the version they were derived for. |
| **Consequences** | For `v1-handcrafted`: screening 0.0407 (out-of-fold sensitivity 0.856, specificity 0.600, precision 0.122), HIGH 0.1887 (specificity 0.950, sensitivity 0.328). Both verified reachable on test. `SCREENING_THRESHOLD` / `HIGH_RISK_THRESHOLD` in config are retained **as fallbacks only**, for an uncalibrated or artifact-less model where 0.50 is the only defensible arbitrary choice. A degenerate-band guard fires if the derived HIGH boundary is not above the screening threshold. |
| **Clinical status** | Both are **experimental engineering operating points, not clinically validated cut-offs**, and are labelled as such in the artifact, the API response, and the UI. Specificity 0.600 at the screening threshold means roughly 40% of healthy people are flagged — poor, and reported rather than hidden. |
| **Alternatives** | Keep 0.50/0.70 (rejected: measurably flags nothing). Hardcode 0.0407 into config (rejected: it is specific to this model's calibrated distribution and would be wrong for any other version). |

---

## DEC-019: The Held-Out Test Set Is Scored Once, and the Quantum Result Is Reported As Measured

| Field | Value |
|---|---|
| **ID** | DEC-019 |
| **Date** | 2026-09-02 |
| **Status** | ACCEPTED |
| **Context** | PART 10 requires evaluating the quantum contribution rather than assuming it, and PART 15 restricts the test partition to final evaluation. Both are disciplines that code cannot fully enforce. |
| **Decision** | `backend.evaluation.evaluate` is the only module that reads the `test` partition; every training stage returns `test_partition_used: False` and this one returns `True`. Comparison fairness is enforced structurally: all models are scored on the same persisted pipeline's matrix (loaded, never refitted), and each model's operating threshold is chosen on **validation** for the target sensitivity and then applied unchanged to test. Ranking is by PR-AUC. A prior report is treated as a finding — the script warns, increments `times_scored`, and keeps the earlier headline numbers in the new report so drift is visible rather than overwritten. |
| **Rationale** | A threshold chosen on test inflates whichever model it was chosen for. Ranking on PR-AUC rather than Brier avoids comparing calibrated quantum probabilities against baseline probabilities distorted by `class_weight="balanced"` — PR-AUC is rank-based and unaffected. |
| **Consequences (measured, held-out test: 381 images / 48 patients / 18 positive)** | Random forest PR-AUC 0.5750 / ROC-AUC 0.9411; gradient boosting 0.5625 / 0.9327; logistic regression 0.5125 / 0.9256; **calibrated VQC 0.2180 / 0.7671 — rank 4 of 4**. On this dataset the variational circuit does **not** outperform a classical baseline, and the repository says so in the artifact, the summary, and the final report. The VQC also degraded from validation (PR-AUC 0.4636 → 0.2180), a genuine generalisation gap. |
| **Honest caveat** | 18 test positives means the ordering is not statistically meaningful and must not be quoted as a result about quantum machine learning in general. The evaluator emits this caveat itself whenever `n_positive < 20`, in either direction. |
| **Alternatives** | Report only the quantum model (rejected: PART 10 requires the comparison). Tune the VQC until it wins (rejected: that is tuning on test, and the honest negative result is the deliverable). |



