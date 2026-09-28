# ORQIS — FINAL VERIFIED FINDINGS

**Problem statement:** SIH26139 — Hybrid Quantum Machine Learning Platform for Early Disease Detection
**Repository audited:** `C:\Users\ISHAN SHUKLA\Downloads\Orqis-main\Orqis-main`
**Audit date:** 27 September 2026
**Evidence basis:** machine-generated evaluation artifacts, tested source code, and repository decision logs. The PPT was **not** used as a source. Every number below is traceable to a named file.

**Environment of record (from artifact `environment` blocks):** Python 3.12.10, numpy 1.26.4, Windows-11-10.0.26200-SP0, seed 42 throughout.

---

## A. PROBLEM AND MOTIVATION

Oral cancer is diagnosed late, and late diagnosis is the dominant driver of poor outcomes. The engineering problem this repository addresses is narrow and honest: **triage of smartphone-captured oral photographs into risk bands, patient-disjointly validated, with the quantum contribution measured rather than asserted.**

The repository does **not** contain, and this report does not claim, any evidence about survival, mortality, treatment invasiveness, or clinical deployment.

**Scope note on the second modality.** The repository contains a second, fully separate track on 12-lead ECG (PTB-XL). It exists to answer one research question the oral dataset could not answer for lack of scale: *does the quantum feature map contribute anything a same-shape classical map does not, when given a large patient-disjoint cohort?* ECG results are **never** mixed with oral-cancer results anywhere in this document.

---

## B. PRODUCT BUILT

| Component | State |
|---|---|
| Flutter patient/clinic mobile client | Implemented, 193 tests pass, `flutter analyze` clean |
| FastAPI backend, 20 HTTP endpoints | Implemented, 1,038 tests collected |
| Oral-lesion screening track | Implemented and serving |
| 12-lead ECG track | Registered, **no model attached — gated** |
| Frozen lesion localizer | Implemented, SHA256-pinned, evaluated on the held-out test partition |
| Two-stage image quality gate (client + backend) | Implemented |
| FHIR R4 export | Implemented (`GET /api/screening/{id}/fhir`) |
| Clinic-staff JWT authentication | Implemented (bcrypt + HS256) |
| Patient authentication | **Does not exist** |

---

## C. CURRENT ARCHITECTURE

```
Flutter client
  → client-side quality gate (exposure / focus / clipping / red-chromatic heuristic)
  → POST /api/screening/upload  (multipart)
  → backend quality gate (luminance, Laplacian focus, clipping, resolution,
                          compression proxy, ROI adequacy, ROI truncation)
  → frozen localizer (MobileNetV3-Small box regressor, fallback to whole image)
  → features → TRAIN-fitted reduction
  → classical reference (headlines the verdict)  +  quantum head (secondary, separate)
  → TRAIN-fitted calibration → banded verdict → SQLAlchemy persistence → FHIR R4
```

**Platform layer.** `backend/ml/track.py`, `backend/ml/tracks/{oral_lesion,ecg}.py`, `backend/routes/track_routes.py`. Per `docs/PLATFORM_ARCHITECTURE.md`, this layer is **additive**: nothing in it is imported by the existing screening path, and `tests/test_tracks.py::test_the_existing_screening_surface_is_untouched` fails if that stops being true.

**Endpoints (20, enumerated from source):** `/health`, `/`; `POST /api/screening/upload`, `POST /api/screening/analyze`, `POST /api/localize`, `GET /api/model/info`, `GET /api/screening/{id}`, `GET /api/screening/{id}/image`, `GET /api/results/{id}`, `GET /api/patients/{id}/history`, `GET /api/screening/{id}/fhir`; `POST /api/auth/login`, `GET /api/auth/me`; `GET /api/clinics/{id}/screenings`, `/patients`, `/stats`; `GET /api/tracks`, `GET /api/tracks/{id}`, `POST /api/tracks/{id}/analyze`, `POST /api/tracks/{id}/analyze-image`.

---

## D. DATASETS

### D.1 Oral cohort — three distinct numbers, never interchangeable

| Stage | Images | Patients | Source |
|---|---|---|---|
| Source corpus (all annotation trees) | 7,731 | — | `inspection_report.json` |
| Canonical corpus (de-duplicated) | 2,469 | 329 | `inspection_report.json` |
| QC-filtered split cohort | **2,436** | **328** | `split_manifest.json` |
| → TRAIN | 1,692 | 230 | 104 positive |
| → VALIDATION | 363 | 50 | 21 positive |
| → TEST (held out) | 381 | 48 | 18 positive |

**ACHIEVEMENT** — Reproducible, patient-disjoint, QC-audited oral cohort with an enumerable exclusion ledger
**EVIDENCE** — 33 images excluded with named reasons: excessive_blur 24, insufficient_resolution 3, excessive_illumination 2, insufficient_illumination 1, excessive_illumination+excessive_blur 1, excessive_blur+insufficient_resolution 1, insufficient_resolution+incomplete_lesion_visibility 1
**METRIC** — 2,469 canonical → 2,436 retained; 328 patients; 143 positives; `split_manifest_sha256 = fe45df6d890048756b709e9ddb5144903465930219d44ae7910dbaf3dd1ce629`
**SOURCE FILE** — `backend/artifacts/dataset/split_manifest.json`, `backend/artifacts/dataset/inspection_report.json`
**STATUS** — VERIFIED

Class distribution (canonical): normal 2,145 · opmd 125 · oral_cancer 20 · variation_from_normal 179. Binary: 2,324 negative / 145 positive. Capture mode R 1,981 / W 420 / unspecified 68. `n_unreadable = 0`. 27 cross-class pixel-identical duplicate groups were resolved by dropping the lower-severity copy. Clinical coverage: 302/329 patients with demographics; risk factors on 304 patients (smoking 38, alcohol 18, betel quid 48 present, 0 unknown).

**The primary experimental condition is smaller still.** `A_lesion_polygon` — the only condition that isolates the annotator lesion box — is **215 TRAIN / 52 VALIDATION images**, 103/21 positives, 112/25 patients, `patient_overlap = 0`.

### D.2 ECG cohort (secondary track)

| Stage | Records | Patients |
|---|---|---|
| Full PTB-XL | 21,799 | 18,869 |
| Working set (folds 1–9) | **19,601** | **16,965** |
| → TRAIN (folds 1–8) | 17,084 labelled | 14,823 |
| → VALIDATION (fold 9) | 2,146 | 1,917 |
| → TEST (fold 10) | 2,198 | 1,904 — **never used** |

**ACHIEVEMENT** — Byte-level verified ECG signal corpus
**EVIDENCE** — 19,601 of 19,601 stems complete; 39,202 of 39,202 files checksum-OK; 0 mismatch, 0 unknown, 0 corrupt; 482,260,870 bytes; `fully_verified: true`; `patient_leakage_free: true`
**METRIC** — 100% signal integrity; features matrix (19,601 × 97) = 1,901,297 cells, `failures: {}`
**SOURCE FILE** — `backend/artifacts/dataset/ptbxl/signal_verification_report.json`, `audit_report.json`, `features_ecg-v1.npz`
**STATUS** — VERIFIED

---

## E. PREPROCESSING

Oral: 224×224 target, 0.8 centre-crop fraction, `carescan-preproc-1`. CIELAB is used rather than HSV because hue is circular and saturation unstable at low value (`backend/ml/preprocessing.py`).

Feature blocks (oral, `e1_cache_rows = 2436`): handcrafted LAB descriptor 163-d · HOG 1,764-d · texture (LBP+GLCM) 156-d · multiscale 456-d · MobileNetV3-Small embedding 576-d. The winning candidate **C7** = `hstack([mobilenet 576, C5 619])` = **1,195-d**, z-scored with **TRAIN mean/scale only**.

ECG: 97 hand-engineered `ecg-v1` features (HRV, QRS/QT/QTc/JT, per-lead Q/R/S/ST60/ST-slope/T-amplitude across all 12 leads, QRS/T axes, relative band powers, spectral entropy, dominant frequency, flatline/saturation fractions, baseline wander, HF noise, RR regularity). Extraction 227.76 s.

**Reductions are fitted on TRAIN only** (`train_patient_ids` length 15,023): pca08 explains 0.600332 of variance · pca16 **0.774452** · pca32 0.927450 · select16 selects 16 supervised features with `dropped_features: []`.

> **Correction to a circulating claim.** "97 retained, 16 discarded, variance ≈ 0.7745" is garbled. All 97 features are retained (`dropped_features: []`); `select16` *selects* 16; and 0.774452 is **pca16**'s explained variance.

---

## F. LOCALIZATION

**What it is:** MobileNetV3-Small regressing a normalised lesion bounding box (DEC-020). Checkpoint SHA256-pinned `27d6036e458f1d660c3378bc9f19aa1f5b6b5cd7f4b6d9c8238774ab79c31dee` (4,434,955 bytes). `min_confidence = 0.3` selected on validation by F1 of accepted vs IoU≥0.25; epoch 17 selected on validation mean IoU. Falls back to the whole image when confidence fails.

**ACHIEVEMENT** — Frozen, threshold-pinned lesion localizer evaluated on the held-out oral test partition
**EVIDENCE** — 381 test images / 48 patients, threshold unchanged from validation: 374 localized, 7 fallback, 0 rejected; all 7 failures `low_confidence`
**METRIC** — localization_rate **0.9816**, fallback_rate 0.0184. IoU **measured on `lesion_annotated_subset_only`, n = 47**: mean 0.5279, median 0.5535, IoU≥0.25 = 0.9574, ≥0.5 = 0.5106, ≥0.75 = 0.1489
**SOURCE FILE** — `backend/artifacts/models/localizer/carescan-localizer-1/test_evaluation_report.json`
**STATUS** — VERIFIED

### What localization does NOT prove

- **It is not an oral-vs-non-oral classifier.** It regresses a box; it does not decide whether the photograph is of a mouth.
- **0.9816 is an acceptance rate, not an accuracy.** Of 381 test images, 332 are unannotated and carry the artifact's own disclaimer: `verified_against_ground_truth: false`, `unverified_localization: true` — *"'localized' means only that the confidence cleared the frozen threshold. It is an acceptance rate, not an accuracy: none of these boxes has been verified to contain a lesion."*
- Confidence is a weak proxy for correctness on the annotated subset: confidence–IoU correlation **0.1884** (versus 0.608 on the development set).
- Per-class IoU (small n): opmd 0.5131 (16 images), oral_cancer 0.6487 (**2 images**), variation_from_normal 0.5252 (31 images).

---

## G. CLASSICAL ML — ORAL

### G.1 Primary research result (validation, `A_lesion_polygon`)

**ACHIEVEMENT** — Best classical candidate on the leak-free primary condition, with the increment gated by a patient-blocked permutation null
**EVIDENCE** — C7 (1,195-d), logistic regression, **C = 0.01** selected on TRAIN out-of-fold average precision via StratifiedGroupKFold grouped by patient
**METRIC** — **validation PR-AUC 0.913038**; ROC-AUC 0.933948; accuracy 0.846154; sensitivity 0.904762; specificity 0.806452; precision 0.76; F1 0.826087; balanced accuracy 0.855607; Brier 0.11062; log-loss 0.361332; ECE 0.119337; confusion TP 19 / FP 6 / TN 25 / FN 2. On **52 validation images (25 patients), 21 positive.** TRAIN CV PR-AUC 0.7749237055117247
**SOURCE FILE** — `reports_e1_failure_map.json` → `/conditions/A_lesion_polygon/candidates/C7/arms/logistic/validation_metrics/pr_auc`
**STATUS** — VERIFIED

> **0.913038 is a VALIDATION PR-AUC on 52 images. It is not an accuracy, not a test-set number, and not measured on the 2,436-image cohort.**

**Statistical gating of the increment over the C1 reference (random forest, 0.884779):**

| Quantity | Value |
|---|---|
| Increment | 0.062496 |
| **Column-permutation null (gating)** | empirical p = **0.004975**, 0/200 at-or-above, null mean −0.012024, null max 0.048885, z = 2.9305, `gates_the_claim: true` |
| Label-permutation null (reported, not gating) | p = 0.144279, `gates_the_claim: false`, with an explicit low-power note |
| BH-adjusted p across 7 candidates | 0.017413 |
| `survives_gating_null` | true |

**E1 verdict: "A", margin over C1 0.028259, `motivates_quantum: false`.**

### G.2 Held-out oral TEST partition — the only oral test numbers in the repository

`backend/artifacts/models/v1-handcrafted/evaluation.json`, `test_partition_used: true`, `times_scored: 3`. 381 images, 18 positives, prevalence 0.047244. Thresholds selected on validation at target sensitivity 0.85.

| Model | Test PR-AUC | Test ROC-AUC | Sens | Spec | Sens@Spec0.90 |
|---|---|---|---|---|---|
| **random_forest** | **0.575009** | 0.941077 | 0.7778 | 0.9311 | 0.8333 |
| gradient_boosting | 0.562539 | 0.932660 | 0.8333 | 0.9174 | **0.8889** |
| logistic_regression | 0.512530 | 0.925620 | 0.7222 | 0.9146 | 0.7222 |
| quantum_vqc_calibrated | **0.218038** | 0.767065 | 0.6667 | 0.6860 | 0.3333 |

**The repository's own verdict, quoted verbatim:** *"The quantum model ranks 4 of 4 on the held-out test set: PR-AUC 0.2180 against random_forest at 0.5750. On this dataset the variational circuit does not outperform a classical baseline. The test partition contains only 18 positive images, so the ordering between models is not statistically meaningful and should not be quoted as a result about quantum machine learning in general."* (`statistically_underpowered: true`, `quantum_outperforms_classical: false`)

**Note the prevalence gap.** The primary-condition validation figure (0.913038) sits at prevalence 0.404; the test figures sit at prevalence 0.047. **PR-AUC is prevalence-dependent — these two numbers must never be compared.**

---

## H. QUANTUM ML — the full evolution, every arm, every result

Seven quantum families were built and measured. **All produced null or negative results against matched classical controls.** This is reported as the finding, not hidden.

### H.1 16-qubit raw-pixel amplitude-encoded VQC (E0)

Amplitude encoding of 256×256 grayscale (65,536 amplitudes), observable Z on qubit 0, score (1−⟨Z₀⟩)/2, class-weighted BCE, SPSA, 32 parameters.

Depth sweep on `A_lesion_polygon` (validation PR-AUC): **L1 0.526376** · L3 0.356754 · L2 0.329401. Final train loss 0.69071195 against a ln(2) ≈ 0.6931 starting point — **the objective barely moved.**

Test partition (`reports_final_oracle.json`, `reports_final_predicted.json`):

| Condition | n | Positives | Quantum PR-AUC | Best classical | Margin |
|---|---|---|---|---|---|
| A_all (oracle) | 381 | 18 | 0.139935 | 0.575009 | **−0.435074** |
| A_lesion_polygon (oracle) | 49 | 18 | 0.544288 | 0.709452 | **−0.165164** |
| B_localized (predicted) | 374 | 16 | 0.043516 | 0.592556 | **−0.549040** |
| B_and_C_all (predicted) | 381 | 18 | 0.052155 | 0.575009 | **−0.522854** |

**ACHIEVEMENT** — Honest hardware-feasibility accounting for amplitude encoding, measured rather than asserted
**EVIDENCE** — CX cost of arbitrary state preparation measured at 6/8/10/12 qubits (57 / 247 / 1,013 / 4,083 CX) and extrapolated, rather than claimed at 16
**METRIC** — extrapolated **62,940 CX** for one 16-qubit state preparation against a Shende–Bullock–Markov upper bound of 131,038, versus **16 CX** for the ansatz itself — **state preparation dominates by ≈3,934×**. Repository conclusion: *"against current hardware coherence times this circuit is not runnable."* Training used Aer `set_statevector`, which the artifact states *"is not state preparation on hardware and implies nothing about hardware feasibility."*
**SOURCE FILE** — `backend/artifacts/models/pixel_vqc/v1-pixel-vqc-1/oracle_A_lesion_polygon_pixel_vqc.json` → `hardware_feasibility`
**STATUS** — VERIFIED (negative / infeasible)

E0 multi-observable readout diagnostic (`reports_e0_plateau.json`): widening from 1 to 16, 24, and 136 observables **did not help** — every variant fell below its own null p95 (`exceeds_null_p95: false` throughout), deltas −0.128 to −0.147.

E0 classical controls (`reports_ctl_oracle_primary.json`): quantum 0.545337 vs random forest 0.790313 (margin −0.244976); stack-minus-RF increment **−0.001731**, increment p = 0.442786, quantum–RF Spearman 0.056177, `complementarity_claimed: false`.

### H.2 Fixed 8-qubit angle-encoded visual reservoir (E2)

Self-declared role, verbatim from the artifact: *"EXPERIMENTAL quantum visual demonstrator (SIH fallback). Secondary signal only; the classical baseline remains the primary clinical decision-maker (DEC-033/DEC-034). **No quantum-advantage claim is made or authorised.**"*

**ACHIEVEMENT** — A quantum visual representation that clears its own patient-blocked null and runs inside the latency budget
**EVIDENCE** — 16-d readout, logistic head (17 trainable parameters, C = 0.001), reported *only* beside its equal-dimension RFF control and raw PCA-8 control on identical rows with an identical head
**METRIC** — quantum validation PR-AUC **0.878649** (`survives_gating_null: true`, p = 0.004975, z = 5.7589, null p95 0.58748) · RFF-16 control 0.871222 (also survives) · PCA-8 0.775453. Non-degenerate: effective rank 16/16, 0 constant observables. Latency 1.468 ms/image sustained mean, 1.957 ms p95, 2.366 ms cold, against a 500 ms budget. Entanglement witness: mean |C_ij| per ring edge 0.077–0.268, max 0.67188126. **KEEP**
**SOURCE FILE** — `reports_e2_quantum_visual.json`
**STATUS** — VERIFIED — but note it **does not beat its own equal-dimension classical control** (0.878649 vs 0.871222), and E3 showed its increment over C7 does not survive gating (below).

### H.3 E3 — four quantum families against the frozen C7 anchor (oral)

`C7_REFERENCE_PR_AUC = 0.913038` is hard-coded at `backend/evaluation/e3_hybrid_fusion.py:113` with an assertion that refuses to run if the anchor does not reproduce.

| Family | Validation PR-AUC | vs C7 (0.913038) | Gating verdict |
|---|---|---|---|
| E2 quantum visual, 16-d | 0.868335 | increment +0.017116 (stack 0.930154) | `survives_gating_null: **false**` |
| Rich ZZ + re-uploading, 36-d | **0.390688** | increment −0.003502 | fails |
| Quantum kernel / QSVM | 0.772148 | (RBF control 0.803026) | fails both |
| Trainable shallow VQC | 0.770660 | — | fails |
| Trainable VQC, **no entangler** | **0.778527** | — | — |
| Classical MLP control | 0.768450 | — | — |
| RFF-36 control | 0.726207 | +0.007149 | fails |

Decisive detail: **the trainable VQC beat the classical MLP control (0.770660 vs 0.768450) but lost to its own entangler-free ablation (0.778527).** The entanglement was not doing the work. `barren_plateau: false`, `selected_vqc_depth: 1`. Flags: `defensible_quantum_advantage_over_c7: false`, `quantum_specific_advantage: false`, `quantum_kernel_beats_c7: false`, `quantum_kernel_beats_rbf_control: false`.

### H.4 E4 — the ZZ feature map in the PTB-XL arena (ECG, NOT oral)

**This section is entirely ECG.** `cache_version: ptbxl-ecg-1`, `feature_set_version: ecg-v1`, arm input "8-dim ecg-fit-1 PCA scores". Metric is **ROC-AUC on fold 9**, not PR-AUC.

**The same-shape contract.** Every map is R⁸ → R³⁶ with **zero trainable parameters**, an identical logistic head, an identical C grid, and an identical 4-cell search budget. The quantum arm therefore competes against dimension- and budget-matched classical controls: a degree-2 polynomial using the *identical* index set, and RFF-36.

Havlicek ZZ map: 8 qubits, reps = 2, full entanglement, circuit depth 67, 112 two-qubit gates, 256-dim state, 36 observables (8 singletons + 28 pairs), 0 trainable parameters.

**ACHIEVEMENT** — Circuit correctness proven to numerical precision against two independent references
**EVIDENCE** — Fast `|ψ|² @ diagᵀ` readout checked against Qiskit **Aer** `save_expectation_value`, and the hand-built ZZ circuit checked against the canonical **Qiskit ZZFeatureMap** up to global phase; tolerance 1e-10; 8 samples × 36 observables
**METRIC** — **reps = 2 (the scored arm): readout-vs-Aer 1.1102230246251565e-15, circuit-vs-ZZFeatureMap 9.992007221626409e-16.** (reps = 1: 1.8041124150158794e-16 and 6.661338147750939e-16.) All passed
**SOURCE FILE** — `backend/artifacts/reports/e4_quantum.json` → `map_validation`
**STATUS** — VERIFIED

**ACHIEVEMENT** — An entanglement witness used to *structurally exclude* a dead configuration before any arm was scored
**EVIDENCE** — Train mean/max |⟨Z_iZ_j⟩ − ⟨Z_i⟩⟨Z_j⟩| over 28 pairs, 256 samples, floor 0.001. At reps = 1 the post-Hadamard Havlicek block is diagonal ⇒ |ψ|² uniform ⇒ all 36 Z-observables identically zero. A `fit_map` runtime guard raises on a constant-output map
**METRIC** — reps = 1: mean **0.0**, max 0.0, `is_entangling: false` (product state). reps = 2: mean **0.07320875**, max 0.71480617, `is_entangling: true`
**SOURCE FILE** — `backend/artifacts/reports/e4_quantum.json` → `entangling_witness`; `docs/PHASE_E4_DATA_EXPANSION.md` §7.4
**STATUS** — VERIFIED. *Note: the prose table in `docs/PHASE_E4_DATA_EXPANSION.md:823` records 0.0701; the final artifact records 0.07320875. Per the evidence hierarchy the artifact governs and the doc row is stale.*

**Single-arm results (fold-9 validation ROC-AUC):**

| Arm | Selected | Inner | Validation ROC-AUC | 95% bootstrap CI | Sens@Spec0.9 |
|---|---|---|---|---|---|
| `q@zz` | C=0.01, reps=2 | 0.745375 | **0.760505** | [0.740144, 0.779565] | 0.483 |
| `c@poly2` | C=10.0 | 0.879400 | **0.884961** | — | 0.7175 |
| `c@rff36` | C=0.1 | 0.895046 | **0.897873** | — | 0.7443 |

**Paired comparison, `single_arm_zz_vs_poly2`:** observed **−0.124457**, mean −0.124638, CI **[−0.145524, −0.105109]**, `excludes_zero: **true**`.

> **The quantum map loses 0.124 ROC-AUC to a same-shape classical polynomial, and the interval excludes zero. This is a statistically significant defeat, stated plainly.**

**Fusion arms and the pre-registered headline:**

| Fusion | Validation ROC-AUC |
|---|---|
| `fusion@cnn+gbm` | 0.9462926980022166 |
| `fusion@cnn+gbm+zz` | 0.9463291085282333 |
| `fusion@cnn+gbm+poly2` | 0.9456879280456961 |
| `fusion@cnn+gbm+rff36` | 0.9456621742590015 |

**ACHIEVEMENT** — A pre-registered, patient-clustered, paired-bootstrap NULL result on the headline quantum question
**EVIDENCE** — `HEADLINE_quantum_fusion_vs_same_shape_classical_fusion`, 2,000 patient-clustered bootstrap draws with shared resamples, verdict conditions frozen before the first fit (`frozen_before_first_fit: true`)
**METRIC** — observed **+0.000641**, CI **[−0.000486, +0.001747]**, `excludes_zero: **false**`. `quantum_beats_same_shape_classical_fusion: false`; `clears_absolute_bar: true`
**SOURCE FILE** — `backend/artifacts/reports/e4_quantum.json` → `verdict`
**STATUS** — VERIFIED NULL

Repository conclusion, verbatim: *"NULL: the paired interval spans zero, so the quantum map contributed nothing a same-shape classical map did not. Per Section 7.10 this is the conclusion regardless of how either arm scores against the absolute bar."*

Total E4 quantum runtime 7,127.9 s; seed 42; 2,000 bootstrap draws; `quantum_version: v1-e4-quantum-1`; `test_partition_used: false`.

### H.5 The research value of these nulls

These are not failures of execution; they are **measurements**, and they are the scientifically defensible core of this submission:

1. **The classical bottleneck was measured before any quantum work was authorised.** E1's repository-level `bottleneck_summary`: `families_motivating_quantum: []`, `classical_bottleneck_measured: false`, conclusion *"No candidate produced verdict B on the primary condition. Under §3 and §6 this is a NULL result and NO QUANTUM IMPLEMENTATION is justified."*
2. **Same-shape controls localise the cause.** Because every map is R⁸→R³⁶ with zero trainable parameters and an identical head, the −0.124 loss is attributable to the *map*, not to dimensionality or tuning budget.
3. **Ablation isolates entanglement.** The trainable VQC losing to its own entangler-free variant is direct evidence that entanglement was not the active ingredient.
4. **Amplitude encoding is quantified as infeasible, not dismissed.** 62,940 CX vs 16 CX for the ansatz.
5. **A dead configuration was excluded by proof, not by scoring.** The reps=1 exclusion is analytic and pre-scoring — it cannot be metric shopping.

---

## I. STATISTICAL VALIDATION

**ACHIEVEMENT** — Leakage-resistant, pre-registered statistical protocol applied uniformly across five experiment phases
**EVIDENCE** — (a) StratifiedGroupKFold grouped by patient with asserted `patient_overlap = 0`; (b) reductions, scalers and calibrators fitted on TRAIN only; (c) `test_partition_used: false` asserted in E1–E4 payloads and PTB-XL fold 10 never read; (d) candidate grids frozen before the first fit; (e) **two** permutation-null families, with the *column*-permutation null (candidate block shuffled in whole patient blocks, labels and reference column untouched) designated as the gating test and the weaker label-permutation null reported beside it with an explicit power note; (f) 2,000-draw patient-clustered bootstrap, paired with shared resamples, `excludes_zero` as the decision rule; (g) BH correction across candidate families
**METRIC** — E1 gating null p = 0.004975 (0/200), BH-adjusted 0.017413; E2 gating null p = 0.004975 (z = 5.7589); E4 headline CI [−0.000486, +0.001747]
**SOURCE FILE** — `reports_e1_failure_map.json`, `reports_e2_quantum_visual.json`, `backend/artifacts/reports/e4_quantum.json`, `docs/PHASE_E4_DATA_EXPANSION.md` §7.4/§7.10
**STATUS** — VERIFIED

### The ECG label-permutation null — reported precisely

`backend/artifacts/reports/e4_permutation_null.json`: 200 patient-blocked label permutations, **model refitted each draw**. Observed ROC-AUC **0.940234** (arm `gbm@f97`); null mean **0.507865**, sd 0.023552, p97.5 0.554956, max 0.567394; **0/200 at-or-above**; **p = 0.004975**; 867.3 s. Cohort 19,601 records / 16,965 patients / 97 dims. Bootstrap [0.929748, 0.949353], SE 0.004988.

> **Attribution:** this result belongs to **PTB-XL ECG NORM-vs-ABNORMAL**, not to oral cancer. The value p = 0.004975 appears in *both* the oral E1 column-permutation null and this ECG label-permutation null simply because both had 0 exceedances of 200 draws → (0+1)/(200+1). The triple 0.940 / 0.508 / 0.567 is unambiguously **ECG**.

**`within_patient_labels_consistent = False` — reported, not corrected away.** Quantified in `DECISIONS.md`: 1,631 of 14,823 TRAIN patients (11.00%) have more than one record, and **281 patients (1.90% of all patients, 17.23% of multi-record patients, 688 records = 4.03%) genuinely carry mixed labels.** Patient-blocking therefore shifts permuted record prevalence from 0.5760 to 0.5525 (sd 0.0018, range [0.5478, 0.5585]). It was retained because it yields the **harder, conservative** null. The synthetic-fixture tests in `tests/test_e4_classical_baseline.py` assert `is True` because each fixture patient carries one label; the real cache legitimately reports `False`.

---

## J. BENCHMARK RESULTS

### J.1 ECG classical leaderboard (fold-9 validation ROC-AUC, 13 arms)

| Rank | Arm | ROC-AUC |
|---|---|---|
| 1 | gbm@f97 | **0.9402343416976896** |
| 2 | rbf_svm@f97 | 0.9336244991332519 |
| 3 | rbf_svm@pca32 | 0.9319611597374180 |
| 4 | rbf_svm@pca16 | 0.9274080678621160 |
| 5 | gbm@pca32 | 0.9229047074369832 |
| 6 | gbm@pca16 | 0.9169173960612692 |
| 7 | rbf_svm@pca08 | 0.9150062874761999 |
| 8 | gbm@pca08 | 0.9070492554491460 |
| 9 | logistic@f97 | 0.9002697931171673 |
| 10 | logistic@pca32 | 0.8955905964932224 |
| 11 | logistic@pca16 | 0.8792964420699652 |
| 12 | logistic@pca08 | 0.8698048395805508 |
| 13 | prevalence@f97 | **exactly 0.5** |

The prevalence floor scoring exactly 0.5 is a **score/label alignment check**; its bootstrap is deliberately `measured: false` because a constant score carries no rank information. RBF-SVM arms are Platt-scaled from TRAIN out-of-fold decision values because `SVC(probability=True)` is deprecated in scikit-learn 1.9.

### J.2 ECG deep baseline — and the honest deep-vs-tabular null

`backend/artifacts/reports/e4_deep_baseline.json`, `v1-e4-deep-1`, 14,061.0 s. Signal cache: 19,601 rows, 1,000 samples, 100.0 Hz, mV, 12 leads, 0 failures, "stores decoded physical units only".

| Arm | Validation ROC-AUC | Sens@Spec0.9 |
|---|---|---|
| `cnn@resnet_small` (raw 12-lead, 0.5–40 Hz, dim 12,000, 126,649 params, epoch 17) | 0.9404758944556537 | 0.8393 |
| `gbm@f97` | 0.9402343416976896 | — |
| **`fusion@cnn+gbm`** | **0.9462926980022166** | 0.8409 (threshold 0.544001) |

**ACHIEVEMENT** — Measured, reported null on the deep-vs-tabular question
**EVIDENCE** — `deep_vs_tabular` paired patient-clustered bootstrap
**METRIC** — observed **+0.000242**, CI **[−0.006101, +0.006374]**, `excludes_zero: **false**` — the 1D-CNN is **not** better than the 97-feature GBM. Both members lose to the fusion with intervals excluding zero
**SOURCE FILE** — `backend/artifacts/reports/e4_deep_baseline.json`
**STATUS** — VERIFIED NULL

Architecture selection (inner scores): resnet_small 126,649 params / 0.946739 / 1,659.2 s · resnet_long_kernel 218,809 / 0.946422 / 1,155.9 s · resnet_wide 251,785 / 0.945696 / 10,235.1 s. Failure map at threshold 0.544001: error rate 0.1356, 207 missed positives, 84 false alarms.

**Secondary MI-vs-NORM task:** `used_for_selection: false`, ceiling gbm@f97 0.973261609530756. Not used for any decision.

---

## K. MOBILE APPLICATION

**ACHIEVEMENT** — Flutter client verified clean on this machine during this audit
**EVIDENCE** — `flutter test` and `flutter analyze` executed 27 September 2026, Flutter 3.47.2 stable
**METRIC** — **193 tests, all passed, 0 failures** (32 test files). `flutter analyze`: **"No issues found!"** in 10.1 s
**SOURCE FILE** — `carescan/test/**` (32 `*_test.dart` files)
**STATUS** — VERIFIED BY EXECUTION

> The circulating figure "119 passed" is **stale**. The measured count is **193**.

Implemented: go_router stateful shell navigation, camera capture with lifecycle handling, typed repositories and result models, multipart upload, server-backed history, model provenance display, independent localization view, client-side quality gate, onboarding, localization catalogues, theme/language preferences, education articles, original Flutter vector illustrations (no external imagery).

Design authority: **VERIFY WITH STITCH.** `docs/SIH_PRODUCT_AUDIT.md` records that Stitch was not accessible; the teal/white/cool-grey redesign proceeds on the user's explicit brief. No claim is made that Stitch was inspected.

Platform caveats from the same audit, unresolved: the Android main manifest lacks INTERNET permission and release signing uses debug keys; iOS carries camera/local-network descriptions and a local-network ATS exception. **Device capture and iOS builds require hardware/macOS verification that has not been performed.**

---

## L. BACKEND AND APIS

**ACHIEVEMENT** — Modality-agnostic track platform added without disturbing the serving screening path
**EVIDENCE** — Seven-stage shared contract (`artifacts → input validation → features → representation → classical reference + optional quantum head → calibration → banded verdict`); only the first two stages are modality-specific. `tests/test_tracks.py::test_the_existing_screening_surface_is_untouched` fails if `/api/screening/analyze` changes
**METRIC** — 20 endpoints; 2 tracks registered (oral serving, ECG **gated with no model attached**); 1,038 backend tests passing
**SOURCE FILE** — `backend/ml/track.py`, `backend/ml/tracks/`, `backend/routes/track_routes.py`, `docs/PLATFORM_ARCHITECTURE.md`
**STATUS** — VERIFIED (ECG track: REGISTERED, NOT SERVING)

---

## M. TESTING AND VERIFICATION

| Suite | Measured | Method |
|---|---|---|
| Flutter tests | **193 passed / 0 failed** | `flutter test`, executed this audit |
| `flutter analyze` | **No issues found** (10.1 s) | executed this audit |
| Backend tests | **1,038 passed / 0 failed** in 732.71 s, exit code 0 | `python -m pytest tests -q`, executed this audit |

**Combined: 1,231 automated tests, all passing, on Flutter 3.47.2 stable and Python 3.12.10.**

**STATUS** — VERIFIED BY EXECUTION (both suites, pass/fail confirmed 27 September 2026).

Representative verification design worth citing: the E3 hybrid-fusion module hard-codes `C7_REFERENCE_PR_AUC = 0.913038` and **asserts** the anchor reproduces before proceeding, so a silent preprocessing drift fails the run rather than quietly shifting every comparison.

---

## N. SECURITY AND PRIVACY — verified features only

| Control | State | Evidence |
|---|---|---|
| Password hashing | bcrypt via passlib `CryptContext` | `backend/auth/security.py:10` |
| Token | JWT HS256, 60-minute expiry | `config.py:41-42` |
| Login enumeration resistance | unknown email, wrong password, and deactivated account all return the same 401 with identical wording | `backend/routes/auth_routes.py` |
| Public sign-up | **deliberately absent**; first account seeded from env vars, further accounts an operator task | `auth_routes.py` docstring |
| Default admin password | **none exists** — unset `CLINIC_ADMIN_PASSWORD` seeds no account | `config.py:46-51` |
| Secrets in VCS | none committed; `.env` git-ignored; `.env.example` carries placeholders only | `.env.example` |
| Credentials in logs | neither password nor stored hash is logged | `auth_routes.py` |

**Honest deficiencies, stated plainly:**
- `SECRET_KEY` has an insecure development default (`"insecure-default-key-for-dev-only-replace-in-prod"`). **Not production-safe unless overridden.**
- `allow_origins=["*"]` in `backend/main.py:45`. Permissive CORS.
- Mobile screening routes are **unauthenticated** for the local demo.
- **`/api/auth/login` and `/api/auth/me` authenticate clinic staff only. No patient signup or login exists.** Per `docs/SIH_PRODUCT_AUDIT.md`: *"Clinic access must not masquerade as patient authentication."*

**This is a prototype/local identity layer, not production clinical authentication. No certification (HIPAA, GDPR, ISO, CDSCO, CE, FDA) is held, claimed, or in progress.** Transport is local HTTP for the demo; HTTPS is a deployment requirement, not a verified property of this checkout.

---

## O. FEASIBILITY

**Verified feasible:** the served classical pipeline, the E2 quantum visual stage at **1.468 ms/image sustained** against a 500 ms budget, and simulator-based quantum work at 8 qubits.

**Verified infeasible on current hardware:** 16-qubit amplitude encoding — **62,940 CX** extrapolated for state preparation versus **16 CX** for the ansatz, ≈3,934× dominance, with the repository's own conclusion that the circuit "is not runnable". This is a property of amplitude-encoding 65,536 values, not of the ansatz.

**Not verified:** on-device capture on physical hardware, iOS builds, any cloud or multi-user deployment, any real quantum hardware execution (all quantum results are ideal statevector simulation; `shots: null` where exact).

---

## P. ECONOMIC VIABILITY

**No cost model exists in this repository.** There is no costing module, no infrastructure pricing artifact, and no deployment budget.

**STATUS — NOT MEASURED.** No percentage cost saving and no rupee figure can be stated. What *can* be said, and only this: the served classical pipeline runs on CPU (measured per-image latency in the low milliseconds for the quantum stage; no GPU required at inference), and the quantum components used are simulator-based, so no quantum hardware access cost is incurred. **Any quantified infrastructure saving would have to be measured first.**

---

## Q. IMPACT

Stated strictly as what the artifacts support:

- A patient-disjoint oral screening cohort with a reproducible, hash-pinned split and an enumerable QC ledger.
- A classical triage model whose best validation PR-AUC (0.913038, primary condition, 52 images) is gated by a patient-blocked permutation null at p = 0.004975.
- A held-out test evaluation on 381 images whose own artifact declares the comparison underpowered at 18 positives.
- A rigorous, pre-registered negative result on quantum contribution, on a 19,601-record patient-disjoint ECG cohort — which is a genuine contribution to the QML literature *because* it is properly controlled.
- A platform layer that makes a third condition a registration rather than a rewrite.

**No claim is made about survival, mortality, earlier detection in practice, reduced treatment invasiveness, or specialist substitution. The repository contains no evidence of any kind on those questions.**

---

## R. LIMITATIONS

1. **Tiny positive counts.** Primary condition: 52 validation images (21 positive). Test partition: 18 positives. The evaluation artifact's own note: *"Only 18 positive samples: every metric here has a wide confidence interval and small differences between models are not meaningful."*
2. **Single dataset, single site.** No external or multi-site oral validation. No temporal validation.
3. **`oral_cancer` class is critically small** — 20 images in the canonical corpus, 14 in TRAIN, **2 in the test partition**. Per-class localizer IoU for oral_cancer (0.6487) rests on 2 images.
4. **Localization is largely unverified at test time** — 332 of 381 test images have no lesion annotation; `unverified_localization: true`.
5. **The quality gate does not verify oral content.** The red-chromatic-fraction check (client-side, threshold 0.18, centre 60% of frame) is by its own source comment *"NOT a clinical test or trained anatomy classifier… It can reject usable images and accept red objects."* **No validated wrong-object detector exists in this system.**
6. **Calibration is imperfect** — C7 ECE 0.119337; logistic test ECE 0.14722. Cross-model Brier/log-loss/ECE comparisons are confounded by `class_weight='balanced'` on the baselines, which is why rankings use rank-based PR-AUC.
7. **No quantum hardware execution.** All quantum results are ideal simulation.
8. **ECG track has no model attached** — registered and gated.
9. **Deep model gives no measured gain** over 97 tabular features (CI spans zero).
10. **`within_patient_labels_consistent = False`** on the real ECG cache — 4.03% of TRAIN records carry mixed within-patient labels.
11. **Security defaults are development-grade** — insecure default `SECRET_KEY`, wildcard CORS, unauthenticated mobile routes, debug release signing on Android.
12. **`PROJECT.md` is comprehensively stale** — it describes a *skin-condition* app, declares backend and ML *out of scope*, and its status table claims the Flutter project is not initialised and no screens or tests exist. All four statements are contradicted by the working tree. It must not be cited as current scope.
13. **Stitch design was never inspected** — all legacy design comparison is `VERIFY WITH STITCH`.
14. **No economic model.**

---

## S. FINAL ACHIEVEMENTS

1. **Reproducible, hash-pinned, patient-disjoint oral cohort** — 2,469 canonical → 2,436 split images, 328 patients, 33 exclusions with named reasons. `split_manifest.json`
2. **Byte-verified 19,601-record ECG corpus** — 39,202/39,202 files checksum-OK, `patient_leakage_free: true`. `signal_verification_report.json`
3. **Gated classical result on the primary oral condition** — validation PR-AUC 0.913038, increment over reference gated at p = 0.004975, BH-adjusted 0.017413. `reports_e1_failure_map.json`
4. **Held-out oral test evaluation with a self-critical verdict** — random forest test PR-AUC 0.575009 / ROC-AUC 0.941077; quantum VQC ranks 4/4 at 0.218038; artifact declares the ordering underpowered. `v1-handcrafted/evaluation.json`
5. **Frozen, SHA256-pinned localizer evaluated at test time** — acceptance rate 0.9816, IoU mean 0.5279 on the 47-image annotated subset, with explicit unverified-localization disclosure. `test_evaluation_report.json`
6. **Quantum circuit correctness proven to ~1e-15** against Aer and the canonical Qiskit ZZFeatureMap at tolerance 1e-10. `e4_quantum.json`
7. **A dead quantum configuration excluded by analytic proof before scoring**, enforced by a runtime guard, with an entanglement witness of exactly 0.0 at reps=1 and 0.07320875 at reps=2. `e4_quantum.json`
8. **A pre-registered NULL on the headline quantum question** — +0.000641, CI [−0.000486, +0.001747], spans zero. `e4_quantum.json`
9. **A statistically significant negative result on the quantum map** — −0.124457 versus a same-shape classical polynomial, CI [−0.145524, −0.105109], excludes zero. `e4_quantum.json`
10. **Entanglement isolated by ablation** — the trainable VQC lost to its own entangler-free variant (0.770660 vs 0.778527). `reports_e3_trainable_vqc.json`
11. **Amplitude-encoding infeasibility quantified, not asserted** — 62,940 CX vs 16 CX, measured at 6–12 qubits and extrapolated. `oracle_A_lesion_polygon_pixel_vqc.json`
12. **Deep-vs-tabular null measured and published** — +0.000242, CI [−0.006101, +0.006374]. `e4_deep_baseline.json`
13. **Additive platform layer with a regression test protecting the serving contract.** `docs/PLATFORM_ARCHITECTURE.md`
14. **Verified test suites** — 193 Flutter tests passed, `flutter analyze` clean, **1,038 backend tests passed / 0 failed** (executed during this audit; 1,231 tests total).

---

## T. CLAIMS WE MUST NOT MAKE

- ❌ Any quantum advantage, quantum speed-up, or "quantum precision" — **every** quantum arm produced a null or negative result against matched controls.
- ❌ "Validated" or "clinically validated" anything — no clinical validation study exists; every artifact carries `is_clinically_validated: false`.
- ❌ Any accuracy figure derived from a PR-AUC or ROC-AUC. **0.913038 is a PR-AUC, not 91.3% accuracy.**
- ❌ Comparing the 0.913038 validation figure (prevalence 0.404) with any test figure (prevalence 0.047).
- ❌ Attributing ECG results (0.940234, 0.760505, 0.884961, 0.897873, +0.000641, −0.124457) to oral cancer.
- ❌ Describing localization as an oral-vs-non-oral classifier, or 0.9816 as an accuracy.
- ❌ Claiming the quality gate guarantees oral content or detects wrong objects.
- ❌ Any survival, mortality, "ultra-early detection", "less invasive care", or specialist-substitution claim.
- ❌ Any infrastructure cost percentage or rupee figure.
- ❌ Any certification, regulatory clearance, or deployment claim.
- ❌ Describing the ECG track as serving, or patient authentication as existing.
- ❌ Citing `PROJECT.md` as current scope.
- ❌ Claiming Stitch was inspected.
- ❌ Claiming quantum hardware execution.

---

# WHAT WE ACTUALLY ACHIEVED — one page, confirmed only

We built a **working two-part system and ran a disciplined quantum research programme that returned an honest negative answer.**

**The product.** A Flutter mobile client (**193 tests passing, `flutter analyze` clean**) talking to a FastAPI backend with **20 endpoints and 1,038 tests passing** — **1,231 automated tests, zero failures**, all executed and verified on 27 September 2026. Images pass a two-stage quality gate, a **SHA256-pinned frozen localizer** proposes a lesion box, a classical model produces a **banded risk verdict**, and the result is persisted and exportable as **FHIR R4**. A **modality-agnostic track layer** was added so a third condition is a registration rather than a rewrite, protected by a regression test that fails if the serving contract changes.

**The data work.** From a 7,731-file source tree we derived a de-duplicated 2,469-image canonical corpus and a **QC-filtered, hash-pinned, patient-disjoint cohort of 2,436 images across 328 patients**, with all 33 exclusions individually justified. Separately we verified a **19,601-record, 16,965-patient ECG corpus byte-for-byte — 39,202 of 39,202 files checksum-clean, provably patient-leakage-free.**

**The classical result.** On the leak-free primary oral condition the best candidate reaches **validation PR-AUC 0.913038** (52 images, 21 positive, prevalence 0.404), and its improvement over the reference model **survives a patient-blocked column-permutation null at p = 0.004975**, BH-adjusted to 0.017413. On the **held-out 381-image test partition** the strongest classical model reaches **PR-AUC 0.575009 / ROC-AUC 0.941077** at prevalence 0.047 — and the artifact itself records that with 18 positives the model ordering is not statistically meaningful.

**The quantum result — the honest centre of this work.** We built and measured **seven** quantum families: a 16-qubit amplitude-encoded pixel VQC, a fixed 8-qubit angle-encoded visual reservoir, a richer ZZ map with data re-uploading, a quantum kernel/QSVM, a trainable VQC with and without an entangler, and the Havlicek ZZ map in a large ECG arena. We proved the circuits correct to **~1e-15** against both Qiskit Aer and the canonical ZZFeatureMap. We **excluded a dead configuration by analytic proof before scoring anything**, backed by an entanglement witness reading exactly 0.0.

And then we measured, pre-registered, and published the answer: on the headline question the quantum fusion beat its same-shape classical fusion by **+0.000641 with a 95% CI of [−0.000486, +0.001747] — spanning zero.** As a single arm the quantum map **lost 0.124457 ROC-AUC to a dimension-matched classical polynomial, CI [−0.145524, −0.105109], excluding zero.** The trainable VQC lost to its own entangler-free ablation. On the oral test set the VQC ranked 4th of 4. And amplitude encoding at 16 qubits costs an extrapolated **62,940 CX gates against 16 for the ansatz** — not runnable on current hardware.

**That null is the achievement.** It is trustworthy precisely because of how it was obtained: every map constrained to R⁸→R³⁶ with **zero trainable parameters** and an identical head, grids frozen before the first fit, held-out partitions never touched, permutation nulls chosen for conservatism over convenience, and 2,000-draw patient-clustered paired bootstraps deciding every comparison. A system built to find a quantum advantage would have found one. This one was built to measure whether there was one, and reports that there was not.
