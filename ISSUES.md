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

### ISS-SIH-01: A trained oral-presence validator is absent

OPEN / HIGH. The existing localizer predicts lesion boxes and cannot distinguish an oral image from an unrelated object. The new client measures quality and applies an explicitly unvalidated red-tissue plausibility heuristic before upload. It blocks many unsuitable inputs but can accept red objects and reject valid oral images. A separately evaluated oral/non-oral classifier with diverse negative images is required for reliable wrong-object rejection. Do not describe the heuristic as mouth detection or clinical validation.

*(No open issue currently blocks the E4 research track. ISS-SIH-01 is a client-side product gap, and ISS-008/009/010 are measured findings.)*

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

### ISS-011: Degeneracy Guard Read a Clipped Standard Deviation and Could Never Fire

| Field | Value |
|---|---|
| **ID** | ISS-011 |
| **Status** | RESOLVED |
| **Severity** | HIGH |
| **Title** | `e4_quantum.fit_map`'s degeneracy guard was unfireable by construction |
| **Description** | The guard that refuses to score a map whose every output column is constant compared `np.max(out_std)` against `DEGENERATE_MAP_FLOOR = 1e-9`, where `out_std` came from `_standardiser`. That helper deliberately returns `np.where(std > _EPS, std, 1.0)` so that dividing by a constant column's std stays finite. A fully degenerate map therefore reported a maximum std of exactly **1.0**, never `<= 1e-9`, and the guard could not fire for any input. |
| **How it was found** | A real-data test of the map-arm path asserted that `fit_map("zz", X, reps=1)` raises. It printed `GUARD FAILED: reps=1 was accepted`. The assertion existed because the `reps=1` product-state defect had been proven separately and the guard was written to enforce it. |
| **Impact if shipped** | `reps=1` produces a provably information-free product state: measured max per-column TRAIN std 6.8e-17, all 36 Z-observables identically zero. A head fitted on it can only learn an intercept, so the arm would have reported a chance-level AUROC as if it were a model. `QUANTUM_REPS_GRID = (2,)` meant no scored cell reached it, so **no recorded number was affected** — the defect was in the second line of defence, not the first. |
| **Resolution** | `fit_map` now measures the raw per-column spread of the encoded TRAIN matrix and guards on that, keeping `_standardiser`'s clipped divisor for scaling only. Verified firing at 7.031e-17 on an 8-dimensional fixture; `reps=2` and both classical controls are unaffected. |
| **Regression cover** | `tests/test_e4_quantum.py::TestDegeneracyGuard` — including `test_the_refusal_quotes_a_spread_at_or_below_the_floor`, which parses the number out of the exception message and asserts it is `< 1.0`. Asserting only that *something* raises would not have caught this class of defect, and would not catch its return. |

### ISS-012: `e4_deep_baseline` Has No Test Module

| Field | Value |
|---|---|
| **ID** | ISS-012 |
| **Status** | RESOLVED |
| **Severity** | MEDIUM |
| **Title** | The module that produced `cnn@resnet_small` = 0.940476 is uncovered |
| **Description** | The house rule is one `tests/test_*.py` per module. `backend/evaluation/e4_deep_baseline.py` has none, and no other test module imports it. Its siblings `e4_classical_baseline` and (now) `e4_quantum` are both covered. |
| **Why it matters** | That module produced `cnn@resnet_small` 0.9404758944556537 and, through it, the fused bar `fusion@cnn+gbm` 0.9462926980022166 that every quantum arm is measured against. It also exports `split_inner`, `train_model`, `predict_logits` and `_logit`, which `e4_quantum.recover_members` depends on entirely — so a defect there would propagate into the quantum rung's members rather than staying local. |
| **What does *not* close it** | `recover_members` asserts the re-run reproduces the recorded scores to 5e-7 (inner) and 1e-9 (fold 9). That proves **determinism**, not correctness: a systematic defect would reproduce exactly and pass. The partition honesty of that module is currently attested only by its own inline guards. |
| **Resolution** | `tests/test_e4_deep_baseline.py` — **30 tests green**, organised around the four failure modes that would be silent: inner-split leakage (`split_inner`'s fold-8 boundary, the patient-span refusal, and both degenerate-TRAIN refusals), amplitude normalisation (conditioning a record inside two different batches must be bit-identical; doubling the input must double the output), score/label misalignment (the memmap batcher sorts rows for sequential access — a wrong inverse permutation would scramble predictions against labels and raise nothing, so shuffled row order must permute the answers identically, and batch size must not change them), and drift from the record. No training is exercised: one real epoch costs 66 s, so the trained path stays covered by its reproduction assertion in `e4_quantum`. |
| **Corroboration found** | `count_parameters(build_model("resnet_small"))` returns **126,649**, matching the figure quoted in DEC-046, TASKS.md and the phase doc — the recorded architecture is the one that still builds. |

---

### ISS-013: Two Client Tests Asserted a UI That Three Tasks Ago Stopped Existing

| Field | Value |
|---|---|
| **ID** | ISS-013 |
| **Status** | RESOLVED |
| **Severity** | MEDIUM |
| **Title** | `accessibility_test.dart` and `assessment_flow_test.dart` were red in the working tree against the shipped UI |
| **Description** | `flutter test` reported **2 failures** with no relation to the work in flight: `Accessibility & Semantics Verification HomeScreen accessibility checks` (`Found 0 widgets with icon "IconData(U+0E57F)"`, i.e. `Icons.settings`) and `Assessment Flow Integration Full navigation from Home to Preview, Analyzing and Result` (`Found 0 widgets with text "Good morning, Alex"`). Both pumped `const MyApp()` and asserted strings and icons from the pre-T-SIH HomeScreen: `Icons.settings`, `'Take a Photo'`, `'Good morning, Alex'`, `'Start a New Assessment'`, `'Assessment History'`, `'Personal Information'`, `'Notifications'`. |
| **Two independent causes** | (1) **The session gate.** T-SIH-03 added `AppRouter.router`'s redirect, which sends *every* route to `/welcome` while `appSession.identity == null`. `pumpWidget(const MyApp())` therefore lands on prototype onboarding, not the shell, so no HomeScreen assertion could ever match. (2) **The redesign.** T-SIH-04 moved HomeScreen onto l10n: `Icons.settings` → `Icons.person_outline` with tooltip 'Profile', `'Take a Photo'`/`'Start a New Assessment'` → `l.startScreening` = 'Start Screening', and the hardcoded 'Alex' persona was deliberately removed — `widget_test.dart` now explicitly asserts `find.text('Alex Johnson')` `findsNothing`. The two failing files were simply never updated with the screens they cover. |
| **Why it matters** | A suite with two permanently-red tests stops being a signal. Every subsequent run has to be read as "119 minus the two we know about", which is exactly the state in which a *third*, real regression goes unnoticed — and this repository's standing instruction is that there are to be no bugs. It also made the two features they cover (screen-reader labelling of the primary action, and shell branch navigation) effectively uncovered while appearing covered. |
| **Provenance established before repair** | The failures surfaced during the T-PLAT-06 run, so the first question was whether T-PLAT-06 caused them. The additive `ApiConstants.tracksPath`/`trackDetailPath` block was temporarily removed and both tests re-run: **identical failures**. Pre-existing in the working tree, then, and not a regression from the tracks work — established by experiment rather than by reading the diff and concluding it looked unrelated. |
| **Resolution** | Both repaired to the **current intended** UI, not weakened to pass. Each enters via `appSession.continueAsGuest()` with `addTearDown(appSession.logout)` so the global session does not leak between tests. `accessibility_test.dart` asserts `find.byTooltip('Profile')` (the app bar action is icon-only, so its tooltip is what a screen reader announces) and `'Start Screening'`. `assessment_flow_test.dart` asserts all five bottom-bar destinations **before any branch is visited** — a visited branch contributes colliding titles of its own, since `SettingsScreen`'s app bar also reads 'Profile' — and then asserts the shell's `navigationShell.currentIndex` for each tab change. |
| **The subtlety that made text assertions insufficient** | `StatefulShellRoute.indexedStack` keeps a visited branch mounted and **size-maintaining** rather than offstage, so `find.text` (which skips only offstage widgets) still matches text from a branch that is no longer showing. `expect(find.text('Start Screening'), findsOneWidget)` after switching to History would therefore have passed whether or not navigation worked. Reading `currentIndex` off `ScaffoldWithNav` is what actually establishes which tab is selected. |
| **Proved live rather than assumed** | Per the ISS-011 discipline, a test that passes on the first run is not yet evidence. The branch-index expectation was mutated from `2` to `1`; the suite went red at the mutated line, then green again on revert. The assertion fires. |
| **Found alongside** | `flutter analyze` had one long-standing info-level `unnecessary_import` (`dart:typed_data` in `lib/core/image/image_quality_service.dart`, already provided by `package:flutter/foundation.dart`). Removed, so the analyzer is now clean at zero issues — a clean analyzer is a usable one, and a single permanently-tolerated info has the same corrosive effect on attention as a permanently-red test. |
| **Verification** | `flutter test` **119 passed, 0 failed**; `flutter analyze` **No issues found**; backend track contract re-checked with `tests/test_tracks.py tests/test_track_registry.py` — **56 passed**. No production widget, route, string or API changed. |

---

### ISS-014: The Radius Token File Does Not Describe the Radius the App Uses, and `AppShapes` Is Referenced Nowhere

| Field | Value |
|---|---|
| **ID** | ISS-014 |
| **Status** | **RESOLVED (2026-09-27, T-UI-04)** -- with one part deliberately left open, see *Still unresolved* below |
| **Severity** | MEDIUM |
| **Title** | `AppShapes.radiusMd` is 8; the theme and every screen use 12, reached as a literal |
| **Description** | `grep -rn "AppShapes" lib/ test/` returns exactly **one** line -- the class declaration in `lib/core/theme/app_shapes.dart:6`. Nothing reads it. Meanwhile `app_theme.dart` writes `BorderRadius.circular(12)` three times as a literal, and `ScreeningRecord`, `tracks_screen.dart`, `InfoNote` and (as of T-UI-02) the Result screen all use 12 the same way. So the file that is supposed to be the single source of corner geometry states a value (`radiusMd = 8`) that nothing in the app has ever rendered, while the real scale exists only as a repeated magic number. |
| **Why it matters** | DESIGN.md §2.4 requires radii to come from tokens, and the analyzer cannot enforce a token that nothing imports. Anyone adding a screen has two contradictory sources of truth -- the token file, which is wrong, and the surrounding code, which is right but undocumented -- and the token file is the one they will be told to trust. This is also how the Result screen accumulated three different radii (24, 20, 12) on a single scroll view without anything flagging it. |
| **Discovered during** | T-UI-02 (DEC-050). Standardising the Result screen on one radius required deciding *which* radius, which surfaced that the token file could not answer the question. |
| **Actual scale, measured** | Larger than the issue recorded. `BorderRadius.circular(12)` appeared **23 times across 11 files**, not three. `circular(999)` appeared twice. |
| **Resolution** | `radiusMd` set to **12** and `radiusFull` kept at 999; both are now read by `AppTheme` and by all 11 widget files, and all 25 literals are gone. `radiusSm` (4) and `radiusLg` (16) were **deleted**: nothing used them, their values were unverified, and an unused token that contradicts the app is precisely the defect being closed here. Visually inert by construction -- 12 replaced 12. |
| **Correction to this issue's own text** | The *Found alongside* row understated the input defect. A focused field was not "not distinguished beyond Material's default" -- it was not distinguished **at all**. Because `border` and `enabledBorder` were set explicitly and `focusedBorder` was not, Flutter resolved the focused state back to `border`, which carried the same `outlineVariant` side. Material's default focus styling was suppressed, not merely unenhanced. Confirmed empirically: the new tests were run against the unfixed theme and 10 of 11 failed, reporting `focusedBorder`/`errorBorder` as `null` in both light and dark. |
| **Also fixed** | `focusedBorder` (primary, 2 px), `errorBorder` (error, 1 px) and `focusedErrorBorder` (error, 2 px) added. The error borders were the same defect on the same property -- an invalid field also resolved to the idle outline -- and `WelcomeScreen`'s name field is a real validated input where the rejection was previously carried by helper text alone. Thicker as well as coloured, because DESIGN.md forbids carrying information by colour alone. |
| **Still unresolved -- VERIFY WITH STITCH** | Whether **12 is the designed radius** is still open, and the token file says so in a comment. This change makes the token describe what the app renders; it does not establish that what the app renders is right. The web-portal plan records upstream card radius as **24** and input radius as **12**, so the local 12-everywhere convention may itself be a drift. Settling that needs the Stitch file and is not an agent decision. Tracked forward as ISS-016. |
| **Verification** | `flutter analyze` clean; `flutter test` **193 passed** (was 181): 11 new theme tests plus 1 new `WelcomeScreen` test. Tests were written and run **before** the fix and failed as expected (10/11). Mutation test after the fix: removing `errorBorder` from the theme killed 5 tests, including the `WelcomeScreen` one. `grep` confirms 0 remaining `circular(12)`/`circular(999)` literals in `lib/` and 12 files reading `AppShapes`. |

---

### ISS-015: The String Table Ships Two Editorial Categories With No Body Text Anywhere, and No Requirement Describing Them

| Field | Value |
|---|---|
| **ID** | ISS-015 |
| **Status** | OPEN -- **REQUIRES CLARIFICATION** |
| **Severity** | MEDIUM |
| **Title** | `categoryAwareness` and `categorySigns` are translated category labels for articles that do not exist |
| **Description** | `app_en.arb` / `app_hi.arb` carry a complete editorial vocabulary: `featured`, `allArticles`, `relatedArticle`, `readingTime`, `author`, `articleNotFound`, `published`, `editorialSubtitle`, `educationDisclaimer`, and three category labels -- `categoryCapture` ("Capture guide"), `categoryAwareness` ("Understanding oral health") and `categorySigns` ("Signs & symptoms"). Only `categoryCapture` has body text behind it, and it is complete in both locales (`guideLighting`/`guidePosition`/`guideDistance`/`guideSteady` plus their `*Body` pairs). For the other two there is **no body text in the ARB, no content file, no model, and no entry in REQUIREMENTS.md** describing what they should contain. `grep` for signs/symptom/awareness body strings returns the category labels and nothing else. |
| **Why it matters** | Those two labels name patient-facing oral-cancer education: what to look for in your own mouth, and when a change matters. AGENTS.md §4.3 forbids this agent from inventing clinical functionality and §7 forbids generating medical recommendations, so the content cannot be authored here -- and it should not be, because unreviewed symptom guidance in a screening app is exactly the kind of copy that needs a clinician's name against it before a patient reads it. The labels existing in both languages implies the section was planned; nothing records what it was planned to say. |
| **Discovered during** | T-UI-03. Replacing the `/blogs` placeholder required knowing what the editorial section was supposed to contain. |
| **Not fixed here, deliberately** | T-UI-03 built the section around the one article whose copy already exists, and the screen does not advertise a category it has nothing to put in -- an empty "Signs & symptoms" tab is a worse promise than no tab. The editorial shell is complete and content-driven: adding an entry to `educationArticles()` in `lib/features/education/models/education_article.dart` lights up the featured/all-articles split and "Read next" with no screen changes, and the `>1 article` paths are already covered by tests using an injected second article. |
| **What is needed to close** | Reviewed body copy for the two categories, from the project owner or a clinician, added to `app_en.arb` and `app_hi.arb`. Also a publication date per article: `published` ("Published") is a translated label the article screen currently does not render, because the repo holds no date for the capture guide and fabricating provenance on a clinical document is not an acceptable default. |
| **Verification** | None -- open by nature. The T-UI-03 work around it is green (`flutter analyze` clean, `flutter test` 181 passed, 16 new). |

---

### ISS-016: Eleven One-Off Corner Radii Survive Tokenisation, and Nothing Says Which Are Intentional

| Field | Value |
|---|---|
| **ID** | ISS-016 |
| **Status** | OPEN -- VERIFY WITH STITCH |
| **Severity** | LOW |
| **Title** | After T-UI-04, `lib/` still hand-writes radii of 6, 8, 10, 16, 18, 20 and 24 |
| **Description** | T-UI-04 tokenised the two values with an unambiguous majority -- 12 (23 sites) and 999 (2 sites). It left the rest alone: `circular(20)` ×3 (`localized_capture_view.dart`, the ROI overlay), `circular(16)` ×3 (`settings_screen.dart` section cards and its bottom-sheet top), `circular(8)` ×2 (ROI corner marks, a badge in `screening_record.dart`), and one each of 24 (`image_preview_screen.dart` sheet), 18, 10 (`CareScanBrand` mark) and 6. |
| **Why it was not fixed** | Normalising these would **change pixels**, unlike the 12→token substitution, which changed none. Each is a distinct value in a distinct context and may well be deliberate -- an ROI overlay and a settings card have no reason to share geometry with a list tile. Choosing which survive is a design decision, and Stitch was not accessible in this session (AGENTS.md §4.4). Guessing would be worse than the current state, which is at least honest about being unreviewed. |
| **Why it matters anyway** | `AppShapes` now has two tokens and the app has nine radii. A contributor reading the token file will reasonably conclude the app uses two, and the next new surface will pick one of the eleven undocumented values by copying whatever screen happened to be open. That is the same mechanism that produced ISS-014. |
| **What is needed to close** | The Stitch radius scale. Then either promote the intentional values to `AppShapes` (`settings_screen.dart`'s three 16s are the strongest candidate -- one value, one purpose, three sites) or fold the accidental ones into `radiusMd`, with a DECISIONS.md entry for any surface whose geometry changes. |
| **Verification** | None -- open by nature. The T-UI-04 work that surfaced it is green (`flutter analyze` clean, `flutter test` 193 passed). Counts above are from `grep -rho "circular([0-9.]*)" lib/ | sort | uniq -c`. |

---

## Summary

| Severity | Open | Resolved |
|---|---|---|
| CRITICAL | 0 | 1 |
| HIGH | 2 | 2 |
| MEDIUM | 3 | 3 |
| LOW | 1 | 0 |
| VERIFICATION | 0 | 5 |
| **Total** | **6** | **11** |

Open items ISS-008, ISS-009 and ISS-010 are **measured findings recorded honestly**, not unresolved defects. ISS-008 in particular is the answer to the question PART 10 asks, and closing it by tuning against the held-out set would be the actual error. ISS-SIH-01 is the one open item that is a genuine gap: it is a client-side product limitation (no trained oral/non-oral validator), not a research-track defect, and it had been omitted from this tally until 2026-09-21.


