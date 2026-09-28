# PPT CLAIMS THAT MUST BE REMOVED OR REWRITTEN

**Basis:** every replacement below is traceable to a named repository artifact. Nothing here is softened for presentation; nothing is invented to fill a gap. The repository's own audit (`docs/SIH_PRODUCT_AUDIT.md`, 20 September 2026) independently reaches the same conclusions on the product claims, and states plainly: *"No clinical-validation or quantum-advantage claims."*

---

## 1. "Validated Hybrid QML Platform"

**Why it must go.** Three separate falsehoods in four words.
- **"Validated"** — no clinical validation study exists. Every relevant artifact asserts the opposite: `thresholds_are_clinically_validated: false`, `is_clinically_validated: false`, and the evaluation's own clinical disclaimer: *"These are research metrics… They do not establish clinical validity, and no threshold here is a validated clinical cut-off."*
- **"QML Platform"** — the platform spine is real, but the **quantum head is secondary and gated by design** (`docs/PLATFORM_ARCHITECTURE.md` §6; DEC-033/DEC-034). The classical model headlines every verdict. Calling the platform "QML" inverts the actual architecture.
- **"Platform"** — two tracks are registered; **one serves. The ECG track has no model attached.**

**Defensible replacement:**
> "A modality-agnostic screening platform with a served classical oral-lesion track and a registered ECG track, in which quantum components are implemented as an explicitly secondary, separately reported experimental stage. Seven-stage shared pipeline; 1,038 backend and 193 Flutter tests passing."

---

## 2. "Proven & validated"

**Why it must go.** "Proven" is not available for any model claim in this repository, and the strongest evidence points the other way on the quantum arm. What *is* proven, in the strict sense, is narrower and different: **circuit correctness** (agreement with Qiskit Aer and the canonical `ZZFeatureMap` to ~1e-15 at tolerance 1e-10) and the **analytic exclusion of the reps=1 configuration** (entanglement witness exactly 0.0, enforced by a `fit_map` runtime guard).

**Defensible replacement:**
> "Quantum circuit implementations are verified to ~1e-15 against Qiskit Aer and the canonical ZZFeatureMap, and one configuration was excluded by analytic proof before any arm was scored. Model performance is *evaluated*, not proven: the classical increment on the primary oral condition survives a patient-blocked permutation null at p = 0.004975 (BH-adjusted 0.017413)."

---

## 3. "Quantum Precision"

**Why it must go.** This is the single most damaging claim in the deck, because the repository contains a **statistically significant result in the opposite direction.** On the large patient-disjoint ECG cohort, under a same-shape contract (every map R⁸→R³⁶, zero trainable parameters, identical head, identical search budget), the quantum arm **loses**:

- `single_arm_zz_vs_poly2`: **−0.124457 ROC-AUC**, 95% CI **[−0.145524, −0.105109]**, `excludes_zero: true`
- Pre-registered headline: quantum fusion vs same-shape classical fusion **+0.000641**, CI **[−0.000486, +0.001747]** — **spans zero**
- Oral held-out test: the VQC ranks **4 of 4** (PR-AUC 0.218038 vs random forest 0.575009)
- Trainable VQC loses to its own **entangler-free** ablation (0.770660 vs 0.778527)

**Defensible replacement:**
> "We measured the quantum contribution rather than assuming it. Under a same-shape contract the quantum feature map produced a **pre-registered null** on the headline fusion comparison (+0.000641, CI [−0.000486, +0.001747]) and a statistically significant **deficit** as a single arm (−0.124457, CI [−0.145524, −0.105109]). Reporting this is the contribution: the comparison was controlled tightly enough for the null to mean something."

---

## 4. "Better detection performance"

**Why it must go.** Better than *what*, on *which* task, at *which* prevalence? Unqualified, it reads as a quantum-over-classical claim — which the artifacts refute — and it invites the illegitimate comparison of the 0.913038 validation figure (prevalence 0.404, 52 images) with test figures at prevalence 0.047. **PR-AUC is prevalence-dependent; those numbers are not comparable.**

**Defensible replacement:**
> "On the leak-free primary oral condition (`A_lesion_polygon`, 215 train / 52 validation images, 112/25 patients, patient_overlap = 0), the best classical candidate reaches **validation PR-AUC 0.913038** (ROC-AUC 0.933948; sensitivity 0.905, specificity 0.806), and its increment over the reference model survives a patient-blocked column-permutation null at p = 0.004975. On the held-out 381-image test partition (18 positives, prevalence 0.047) the strongest classical model reaches **PR-AUC 0.575009 / ROC-AUC 0.941077**."

---

## 5. "More robust on unseen data"

**Why it must go.** Robustness on unseen data requires external or multi-site validation. There is **one dataset, one site, no temporal split, and no external cohort.** The held-out partition carries **18 positives**, and the artifact itself states: *"every metric here has a wide confidence interval and small differences between models are not meaningful."*

**Defensible replacement:**
> "Evaluation is patient-disjoint by construction — StratifiedGroupKFold grouped by patient with `patient_overlap = 0` asserted, all scalers and reductions fitted on TRAIN only, and the test partition never read during development (`test_partition_used: false` across E1–E4). This guards against **leakage**, not against distribution shift: with a single dataset, a single site and 18 test positives, generalisation to new populations is untested."

---

## 6. "Ultra-Early Detection"

**Why it must go.** No detection-timing evidence of any kind exists in the repository. There is no longitudinal cohort, no time-to-diagnosis measurement, no comparison against a standard-of-care pathway. The system performs **risk banding on a single photograph**, and the `oral_cancer` class contains **20 images** in the canonical corpus and **2** in the test partition.

**Defensible replacement:**
> "The system produces a risk band from a single oral photograph, intended as a triage aid that flags cases for professional examination. No claim is made about detection timing: the repository contains no longitudinal or time-to-diagnosis data."

---

## 7. "Higher Survival Rates"

**Why it must go.** There is no survival data, no outcome follow-up, and no patient in this dataset whose clinical course is recorded. This claim is unfalsifiable from the repository and, on a medical submission, actively misleading. **Delete it — it has no defensible rewrite as an achievement.**

**Nearest defensible statement (context, explicitly not a result):**
> "Late-stage presentation is the established driver of poor oral-cancer outcomes, which is the clinical motivation for triage research. This project measures screening-model performance; it does not measure, and makes no claim about, survival."

---

## 8. "Less Invasive Care"

**Why it must go.** No comparison against any care pathway exists. The system does not replace biopsy, and the evaluation artifact states directly: *"The system does not replace professional clinical assessment or histopathological confirmation."* **Delete.**

---

## 9. "60–70% lower infrastructure cost"

**Why it must go. This is a fabricated number.** There is **no cost model, no pricing artifact, no deployment budget, and no infrastructure measurement anywhere in the repository.** The percentage cannot be sourced, and no rupee figure may be substituted for it.

**Defensible replacement (only what is measured):**
> "The served pipeline runs on CPU with no GPU requirement at inference; the quantum visual stage was measured at **1.468 ms per image sustained mean** (1.957 ms p95, 2.366 ms cold) against a 500 ms budget. All quantum components are simulator-based, so no quantum-hardware access cost is incurred. **No infrastructure cost comparison has been measured, so no cost-saving figure is claimed.**"

---

## 10. Any figure presented as "accuracy" that is not an accuracy

**Why it must go.** **0.913038 is a PR-AUC.** Writing "91.3% accuracy" is a fabrication, not a rounding. For the record, the *actual* accuracy of that arm is **0.846154** — a different number, on 52 images. The same applies to every ROC-AUC in the deck: 0.940234, 0.760505, 0.884961, 0.897873, 0.941077 are ROC-AUCs, not accuracies.

**Rule for the deck:** every number carries **metric name + dataset + split + cohort level**, e.g. "validation PR-AUC 0.913038 (52 images, 25 patients, primary condition)".

---

## 11. Attribution of ECG numbers to oral cancer

**Why it must go.** Several headline figures in circulation belong to **PTB-XL 12-lead ECG**, not oral cancer: **0.940234** (`gbm@f97`), **0.760505** (`q@zz`), **0.884961** (`c@poly2`), **0.897873** (`c@rff36`), **+0.000641**, **−0.124457**, and the permutation triple **0.940 / 0.508 / 0.567**. The artifact identifies them unambiguously: `cache_version: ptbxl-ecg-1`, `feature_set_version: ecg-v1`, arm input "8-dim ecg-fit-1 PCA scores", metric ROC-AUC on fold 9.

**Trap to avoid:** `p = 0.004975` appears in *both* the oral E1 column-permutation null *and* the ECG label-permutation null — because both had 0 exceedances of 200 draws → (0+1)/(200+1). Identical p-value, **different experiments.**

**Defensible replacement:** label every slide either **ORAL (primary)** or **ECG / PTB-XL (secondary research track)**, and never place the two metric families in the same table.

---

## 12. "~3,000 images | 714 patients" as the final cohort

**Why it must go.** Neither number matches any partition. The repository has three distinct, non-interchangeable counts:

| Stage | Images | Patients |
|---|---|---|
| Source annotation trees | 7,731 | — |
| Canonical corpus (de-duplicated) | 2,469 | 329 |
| **QC-filtered split cohort** | **2,436** | **328** |
| Primary experimental condition | **215 train / 52 validation** | 112 / 25 |

**Defensible replacement:**
> "7,731 source annotation entries reduced to a 2,469-image canonical corpus across 329 patients; after quality control the split cohort is **2,436 images across 328 patients** (33 exclusions, each with a named reason), with 143 positives. The primary leak-free experimental condition is a subset: **215 train / 52 validation images.**"

---

## 13. Localization described as oral detection, or 0.9816 as accuracy

**Why it must go.** The localizer is a **bounding-box regressor** (DEC-020). `DECISIONS.md:11` states it *"remains advisory and is never used as an oral-presence classifier or model attention explanation,"* and `SIH_PRODUCT_AUDIT.md` confirms *"The ROI regressor is explicitly not an oral-presence classifier."* And **0.9816 is an acceptance rate, not an accuracy** — the artifact's own words: *"'localized' means only that the confidence cleared the frozen threshold. It is an acceptance rate, not an accuracy: none of these boxes has been verified to contain a lesion."* 332 of 381 test images are unannotated (`unverified_localization: true`).

**Defensible replacement:**
> "A frozen, SHA256-pinned MobileNetV3-Small regressor proposes a lesion bounding box, falling back to the whole image below confidence 0.3. On the held-out test partition **374 of 381 boxes cleared the threshold (acceptance rate 0.9816, 7 low-confidence fallbacks, 0 rejections)**. Geometric accuracy was measurable only on the **47 lesion-annotated** images: mean IoU **0.5279**, IoU≥0.25 in 95.7%, IoU≥0.5 in 51.1%. The remaining 332 boxes are unverified. The localizer does not decide whether an image shows a mouth."

---

## 14. "The quality gate guarantees the image is of the mouth"

**Why it must go.** It does not, and two independent repository sources say so. The **backend gate contains no colour or anatomy check at all** — `backend/ml/image_quality.py` measures illumination, clipping, focus, resolution, compression, ROI adequacy and ROI truncation, and nothing else. The red-tissue check is **client-side Flutter** (`carescan/lib/core/image/image_quality_service.dart`, threshold `tissue < 0.18` over the central 60% of the frame), and its own source comment reads: *"Engineering checks, NOT a clinical test or trained anatomy classifier… Red chromatic fraction is a conservative plausibility heuristic. It can reject usable images and accept red objects."* `SIH_PRODUCT_AUDIT.md`: *"No validated wrong-object detector is present."*

**Defensible replacement:**
> "Two independent layers screen acquisition quality. The client checks exposure, clipping, focus, resolution and a **red-chromatic-fraction plausibility heuristic** (threshold 0.18, central 60% of frame). The backend enforces the authoritative gate — luminance 40–235, resolution-normalised Laplacian focus ≥ 12, clipped fraction ≤ 0.25, short edge ≥ 224 px, compression proxy ≥ 0.05 bytes/px, ROI fraction ≥ 0.02 and ROI truncation — every threshold a named setting in `config.py` with its observed dataset margin in a comment. These are **engineering acceptability limits derived from this dataset, not clinically validated cut-offs**, and **no trained anatomical classifier and no validated wrong-object detector exists in the system.**"

---

## 15. "Secure patient login" / "ABHA connected" / patient identity

**Why it must go.** **No patient authentication exists.** `/api/auth/login` and `/api/auth/me` authenticate **clinic staff only**, and `SIH_PRODUCT_AUDIT.md` states the rule explicitly: *"Clinic access must not masquerade as patient authentication."* The same audit lists the ABHA connection and the "Alex" identity as **fabricated UI placeholders**. Account creation is deliberately not an endpoint; there is no public sign-up.

**Defensible replacement:**
> "A prototype clinic-staff identity layer: bcrypt password hashing, HS256 JWT with 60-minute expiry, enumeration-resistant uniform 401s, no public sign-up, and no default password (an unset `CLINIC_ADMIN_PASSWORD` seeds no account). This is a **local demo identity layer, not production clinical authentication**: the mobile screening routes are unauthenticated, CORS is `allow_origins=["*"]`, `SECRET_KEY` carries an insecure development default, and **no patient-facing authentication exists.**"

---

## 16. Any certification, compliance or regulatory claim

**Why it must go.** No certification is held, claimed in progress, or evidenced: not HIPAA, GDPR, ISO 13485/27001, CDSCO, CE, or FDA. Do not imply one with badge iconography either. What can be stated is the **implemented technical control** — FHIR R4 export via `GET /api/screening/{id}/fhir`, which is interoperability, not compliance.

---

## 17. Implementation-status claims sourced from `PROJECT.md`

**Why it must go. `PROJECT.md` is comprehensively stale and must not be cited as current scope.** It describes a patient-facing app for **"skin conditions"**; it lists *"Backend/API server implementation"* and *"ML model training or deployment"* as **Out of Scope**; and its status table claims the Flutter project is **❌ Not yet** initialised, with no design system, no screens, no tests and no verified build. All of that is contradicted by the working tree (1,231 passing tests, a serving FastAPI backend, trained artifacts). It also names a macOS workspace path that `SIH_PRODUCT_AUDIT.md` records as not identifying this checkout.

**Action:** treat `DECISIONS.md`, `docs/PLATFORM_ARCHITECTURE.md`, `docs/SIH_PRODUCT_AUDIT.md` and the artifacts as authoritative for status. Flag `PROJECT.md` for rewrite in `ISSUES.md`.

---

## 18. "Quantum hardware validated" / IBM backend imagery

**Why it must go.** **Every quantum result in this repository is ideal statevector simulation.** `IBMQ_API_KEY` is blank in `.env.example`. Worse for the claim, the repository itself proves one family **infeasible** on hardware: 16-qubit amplitude encoding needs an extrapolated **62,940 CX** gates for state preparation against **16 CX** for the ansatz — ≈3,934× dominance — and the artifact states *"against current hardware coherence times this circuit is not runnable."* It also warns that Aer's `set_statevector` *"is not state preparation on hardware and implies nothing about hardware feasibility."*

**Defensible replacement:**
> "All quantum results are ideal statevector simulation (Qiskit/Aer), verified against two independent references. Hardware feasibility was assessed rather than assumed: CX cost of arbitrary state preparation was **measured** at 6, 8, 10 and 12 qubits (57 / 247 / 1,013 / 4,083 CX) and extrapolated to **62,940 CX** at 16 qubits, against a Shende–Bullock–Markov bound of 131,038 — so the 16-qubit amplitude-encoded design is **documented as not runnable on current hardware.** The 8-qubit designs are simulator-tractable; none has been executed on quantum hardware."

---

## 19. Stale technical figures circulating in the deck

| Claim | Status | Correct value | Source |
|---|---|---|---|
| "119 Flutter tests passed" | **STALE** | **193 passed / 0 failed** | executed 27 Sep 2026 |
| "97 retained, 16 discarded, variance ≈ 0.7745" | **GARBLED** | All **97 retained** (`dropped_features: []`); `select16` *selects* 16; **0.774452 is pca16's** explained variance | `*_ecg-fit-1.npz` |
| Circuit validation "1.804e-16 / 6.661e-16" | **WRONG ARM** | Those are **reps=1**. The **scored** arm is reps=2: **1.1102230246251565e-15** and **9.992007221626409e-16** | `e4_quantum.json` |
| Entangling witness "≈0.0701" | **STALE DOC** | **0.07320875** (max 0.71480617). The 0.0701 row in `docs/PHASE_E4_DATA_EXPANSION.md:823` is superseded by the artifact | `e4_quantum.json` |
| "1D-CNN outperforms tabular features" | **UNSUPPORTED** | `deep_vs_tabular` **+0.000242**, CI **[−0.006101, +0.006374]**, spans zero — **no measured gain** | `e4_deep_baseline.json` |
| "Stitch design implemented faithfully" | **UNVERIFIABLE** | Stitch was **not accessible**; all legacy design comparison is **VERIFY WITH STITCH** | `SIH_PRODUCT_AUDIT.md` |

---

## 20. Marketing vocabulary to strike throughout

Remove: *revolutionary · groundbreaking · world-changing · game-changing · unprecedented · state-of-the-art* (unqualified) · *guaranteed · proven · clinically validated · eliminates specialist dependency · prevents invasive treatment · reduces cancer mortality · saves lives.*

**Replace with the measured claim and its interval.** The strongest honest line in this submission is not a superlative — it is that the negative result is trustworthy *because* of how tightly the comparison was controlled: every map constrained to R⁸→R³⁶ with zero trainable parameters and an identical head, grids frozen before the first fit, held-out partitions never read, the more conservative of two permutation nulls chosen as the gate, and 2,000-draw patient-clustered paired bootstraps deciding every comparison.

> **A system built to find a quantum advantage would have found one. This one was built to measure whether there was one, and reports that there was not.**
