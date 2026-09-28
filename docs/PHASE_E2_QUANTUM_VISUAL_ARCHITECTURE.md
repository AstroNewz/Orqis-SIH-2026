# Phase E2 — Quantum Visual-Processing Demonstrator: Architecture Proposal

| | |
|---|---|
| **Status** | PROPOSAL (Phase 1). No code written. No artifact, threshold, split, or evaluation touched. |
| **Decides** | Which single quantum architecture E2 implements, and why it is preferable to the alternatives. |
| **Decision record** | Reserves **DEC-035** (max existing = DEC-034). The DEC-035 body is *drafted* in §13 and is only committed to `DECISIONS.md` when Phase 2 begins, not by this proposal. |
| **Governing briefs** | "URGENT SIH QUANTUM TRACK" build directive + its **fallback clause**; the freeze / Mac-handoff constraints; DEC-030/031/032/033/034. |
| **Author's one-line recommendation** | Build **Candidate B**: an 8-qubit, **fixed (zero-trainable-gate)** angle-encoded quantum feature map over a compact ROI descriptor, read out as an interpretable vector of **local 1- and 2-body Z observables**, feeding a small classical head — shipped as an **experimental secondary** signal while the validated classical model remains the clinical decision-maker. |

---

## 0. Relationship to E0/E1 — read this first (the non-contradiction firewall)

This section exists because the honest reading of the prior work is that **no quantum model is authorised for a quantum-*advantage* claim**, and E2 must not quietly reverse that.

- **DEC-032** closed the 16-qubit whole-image raw-amplitude encoding as a *measured* representation failure (F1): 0 of 15 variant×weight settings beat their own patient-blocked null; an RBF-SVM on the identical 65,536-amplitude vectors also landed near chance. That encoding is the project's **designated negative control**. E2 does not rescue it and shares none of its axes.
- **DEC-033** set the rule "**no second quantum model until a classical bottleneck is measured**," and pre-registered the *advantage* path (projected quantum kernel, Q1) behind two gates: an E1.1 **verdict-B** family **and** a favourable Huang geometric-difference pre-screen.
- **DEC-034** ran E1.1 and returned a **NULL result**: `classical_bottleneck_measured: false`, `families_motivating_quantum: []`. The verdict-B gate failed, so Q1 correctly never triggered. **No advantage-motivated quantum model is authorised.**

**Why E2 is nonetheless in-scope, and does not contradict any of the above.** The build directive's own fallback clause authorises exactly one thing when no advantage can be honestly justified (verbatim):

> "implement the smallest quantum visual-processing demonstrator whose purpose is explicitly: 'experimental quantum visual feature extraction' and keep the validated classical model as the clinical risk decision-maker."

E2 is that demonstrator and **only** that. Concretely, E2 is bound by four invariants that keep it consistent with DEC-032/033/034:

1. **No advantage claim.** E2 does not assert, and its KEEP/KILL (§8) does not require, that quantum beats classical. The expected outcome, pre-disclosed here (§8.4), is a **tie-or-loss** to the matched classical control — consistent with E0/E1 and with the QML literature. That is not a failure of the demonstrator; it is the disclosed prior.
2. **Classical stays primary.** The DEC-034 `primary_*` display contract is untouched (§10). The quantum vector is an additive **secondary** field. It never alters `final_probability`, `threshold`, `risk_level`, or the FHIR value.
3. **The classical surrogate is built in, not omitted.** DEC-033 explicitly named the honest control for a fixed quantum feature map / reservoir: *classical random Fourier features at equal output dimension.* E2 ships that control as a first-class, always-reported arm (§6) — the very comparison DEC-033 said such a model must face — so the demonstrator cannot be mistaken for an advantage result.
4. **The test partition stays frozen.** E2 reads TRAIN + VALIDATION only; `test_partition_used: false`, enforced by source inspection exactly as E0/E1 are (§9). DEC-030's single test read remains the only one.

> **One-sentence reconciliation:** E2 is the E1 *fallback demonstrator*, not the E1 *advantage path*; it builds the same class of model DEC-033 rejected-**for-advantage**, but under a different and weaker claim ("a real quantum circuit runs on real ROI-derived features and yields an honest, non-degenerate, better-than-null, interpretable readout"), with DEC-033's own named classical control attached and the classical model still making the clinical decision.

---

## 1. Objective and scope

**Objective (from the directive, verbatim):** *"Design and implement the smallest scientifically honest quantum visual-processing prototype that can run reliably during a demo."*

**In scope:** one quantum architecture that (a) actually executes a quantum circuit, (b) processes information derived from the localized lesion ROI, (c) uses ~4–8 qubits, (d) runs fast enough for an interactive demo, (e) emits an interpretable quantum feature vector, (f) is measured against a matched classical control, (g) never touches the frozen test partition, (h) makes no advantage claim, and (i) is substantially different from the failed 16-qubit whole-image amplitude VQC.

**Out of scope:** any change to the classical risk decision, the calibration, the thresholds, the localizer, or the frozen evaluations; any advantage claim; any test-set read; any commit (until explicitly instructed).

---

## 2. The one selected architecture (summary box)

**Candidate B — fixed angle-encoded quantum feature map with local-observable readout.**

```
frozen localizer (carescan-localizer-1)                     [Phase C, unchanged]
        │  lesion ROI
        ▼
compact ROI descriptor                                      [reuse extract_mobilenet → 576-d]
        │
        ▼
train-fitted PCA → 8 scalars → standardize to angles        [fit on TRAIN only]
        │  θ ∈ [0, π]^8
        ▼
8-qubit FIXED circuit:  Ry(θ) encode  →  NN entangling ring │ ≤2 data-reuploading blocks
        │                (0 trainable gates)                 │ ~16 two-qubit gates total
        ▼
exact statevector (Aer)                                     [reuse quantum_ml backend]
        │
        ▼
QUANTUM FEATURE VECTOR (16 dims):                           [PRE-REGISTERED, fixed set]
   8 × ⟨Z_i⟩   +   8 × ⟨Z_i Z_{i+1 mod 8}⟩                  │ reuse readout observable layer
        │
        ▼
small classical head (logistic / linear-SVM)               [train-only tuned, patient-grouped CV]
        │
        ▼
EXPERIMENTAL SECONDARY signal + telemetry                  [QuantumVisualSummary; classical stays primary]
```

Key figures: **8 qubits · angle (Ry) encoding · ≤2 re-uploading blocks · 0 trainable gates · ~16 two-qubit gates · 256-dim statevector · 16-dim interpretable quantum feature vector · exact statevector simulation.**

Contrast with the negative control (16-qubit amplitude VQC): 8 vs 16 qubits, angle vs amplitude encoding, **0 vs 32 trainable parameters**, ~16 vs ~62,940 two-qubit gates for state loading, local 1-/2-body vs global-only observables, compact ROI-derived input vs whole-image amplitudes. It is different on **every** axis the directive named.

---

## 3. Candidate comparison (A–E) across the 12 criteria

Candidate families, as framed in the directive:

- **A** — ROI patch → compact features → quantum feature map → **quantum kernel** (SVM on a quantum Gram matrix).
- **B** — ROI patch → compact feature vector → **angle-encoded circuit → multiple local observables → quantum feature vector → classical head**. *(selected)*
- **C** — multiple local ROI patches → **shared small circuit (quanvolution)** → aggregate quantum features → classical head.
- **D** — **quantum kernel on selected local texture/spatial features** (a specialisation of A).
- **E** — other (only if strongly justified).

| # | Criterion | A / D (quantum kernel) | **B (selected)** | C (quanvolution) | E (deep/trainable VQC) |
|---|---|---|---|---|---|
| 1 | Qubits | 6–8 | **8** | 4–8 per patch | 8–16 |
| 2 | Encoding | angle (+re-upload) | **angle Ry, ≤2 re-upload blocks** | angle per patch | amplitude or angle |
| 3 | Circuit depth | shallow, ×N² pairs | **shallow, ×N images (1 circuit/image)** | shallow ×(patches×images) | deep |
| 4 | Trainable params | **0** | **0** | 0 (fixed filter) or many | many (32+) |
| 5 | # measurements | full Gram: O(N²) state overlaps | **16 observables × N images** | obs × patches × N | few global obs |
| 6 | Runtime (demo) | O(N²) overlaps — heaviest | **O(N) — lightest, sub-second/image** | O(patches·N) — medium | slow (train loop) |
| 7 | Simulator cost | N² fidelity evals | **N × 256-dim statevectors** | patches·N small SVs | N × large SV + optimiser |
| 8 | Hardware feasibility | needs many-shot overlaps | **best: O(n) load, shallow** | many small circuits | poor (deep/2-qubit heavy) |
| 9 | Classical control | RBF-SVM (kernel-vs-kernel) | **RFF at equal dim (DEC-033's named control)** | random classical filter bank (Bowles 2024) | any classical NN |
| 10 | Expected failure mode | ties classical kernel (Huang gate unmet) | **ties/loses to RFF (disclosed)** | matched random filters tie (Bowles) | barren plateau / √(T/N) blows up |
| 11 | What makes it *genuinely* quantum | quantum Gram geometry | **⟨Z_iZ_j⟩ ≠ ⟨Z_i⟩⟨Z_j⟩: entanglement-borne correlations, verifiable per-image** | entangled patch features | entangled trainable state |
| 12 | Evidence that would justify keeping it | favourable Huang g_CQ + verdict-B (**neither holds** — DEC-034) | **runs in budget, non-degenerate, beats own null; honest telemetry** — a *demonstrator* bar, not an advantage bar | patch features beat matched random filters | √(T/N) bound non-vacuous (**fails at N≈103**) |

---

## 4. Why B — and why not A/D, C, or E

**Why not A / D (quantum kernel = E1's Q1, the advantage path).** A and D *are* the projected quantum kernel that DEC-033 pre-registered as the advantage direction and gated behind (i) an E1.1 verdict-B family and (ii) a favourable Huang geometric-difference pre-screen. **DEC-034 failed gate (i)** (`families_motivating_quantum: []`), so gate (ii) was correctly never run. Building A/D now would be **executing the gated advantage path with its gate unmet** — a direct contradiction of DEC-033/034. It is also the heaviest option for a demo (O(N²) state-overlap Gram matrix) and its honest control (a classical RBF-SVM) is exactly what already tied/beat quantum on the amplitude vectors in DEC-032. **Excluded by frozen decision, not by preference.**

**Why not C (quanvolution / patch filter = Q2).** Q2 is explicitly **deferred** in DEC-033/034 behind a result that did not arrive. Independently, Bowles et al. (arXiv:2403.07059, 2024) found out-of-the-box classical models matched or beat 12 QML models across 160 datasets, and that a **matched random classical filter bank** typically ties a quanvolutional layer — so C's own honest control predicts a tie, at several-fold the demo cost (one circuit per patch per image). Not a good *primary* demonstrator; may be revisited later as telemetry only.

**Why not E (deep / trainable VQC).** Caro et al. (*Nat Commun* 13:4919, 2022) bound generalization error at O(√(T/N)) in trainable gates T; at **N ≈ 103 train positives**, any T of order 100 makes the bound vacuous. A trainable variational model is ruled out *by the data size*, before any hardware or barren-plateau argument. This is also the family the 16-qubit amplitude VQC belongs to (the negative control). **Excluded.**

**Why B.** B is the only family that is simultaneously: (a) **not** the gated advantage path (A/D), (b) **not** the deferred patch filter (C), (c) **not** a trainable model that violates √(T/N) (E), and (d) able to **run reliably and interpretably in an interactive demo**. Its zero trainable gates make it deterministic and √(T/N)-safe by construction; its angle encoding loads in O(n) gates (vs ~62,940 CX for amplitude loading); its local 1-/2-body observables are interpretable per-image telemetry and avoid the global-observable barren-plateau regime (Cerezo et al. 2021); and its 2-body connected correlations `⟨Z_iZ_j⟩ − ⟨Z_i⟩⟨Z_j⟩` give a concrete, *verifiable* handle on "what the entanglement did." Chosen for **reliability + demonstrability + honesty**, not novelty (§11).

---

## 5. Architecture specification (the thing Phase 2 implements)

### 5.1 Input representation
- **Source:** the frozen `carescan-localizer-1` ROI (Phase C, DEC-020) — same ROI the classical pipeline uses. **No re-localization, no localizer change.**
- **Descriptor:** `extract_mobilenet` (`backend/ml/features_image.py:267`, 576-d, `torchvision:IMAGENET1K_V1`, frozen). This is E1.1's C6 family — the strongest single non-fusion representation there (verdict A). *Reuse the E1 embedding cache (`features_e1.npz`, `v1-e1-features-1`) where present; otherwise re-derive with `extract_mobilenet`. No new preprocessing version, no new dataset pass — same constraint honoured by DEC-032/033/034.*
- **Compression:** `PCA → 8 components`, **fitted on TRAIN only**, persisted with the artifact.
- **Angle map:** robust per-component TRAIN-quantile (2nd/98th percentile) affine-map with clipping to `θ ∈ [0, π/2]`. Reproducible from train stats alone; no validation/test statistics enter. **[AMENDED by DEC-036]** the range was originally specced `[0, π]`; that folds the two-block `cos(2θ)` response (`2θ ∈ [0, 2π]` wraps, non-monotonic) and measurably destroyed signal (validation PR-AUC 0.505). A TRAIN-only nested patient-grouped-CV angle sweep (`backend/evaluation/e2_angle_experiment.py`) selected `[0, π/2]` (keeps `2θ ∈ [0, π]` monotonic while the CZ ring still entangles), which recovered validation PR-AUC to 0.879. `ANGLE_MAX = π/2` in `backend/ml/quantum_visual.py`; it is a fixed deterministic scaling constant and introduces **no trainable parameter**.
- *Documented alternative (not primary):* a compact handcrafted ROI descriptor (subset of the 181-feature `features_handcrafted_lab.npz`) → PCA-8. Kept as a fallback if the MobileNet-PCA angles degenerate (§8).

### 5.2 Quantum circuit (fixed; 0 trainable gates)
- **8 qubits**, one per PCA component.
- Per re-uploading block `b ∈ {1, 2}`: **encode** `Ry(θ_i)` on qubit `i`; then a **nearest-neighbour entangling ring** of CZ gates `(0,1),(1,2),…,(6,7),(7,0)`.
- **≤2 re-uploading blocks** (Schuld et al. 2021: encoding fixes the accessible function class; a second block modestly widens it without any trainable parameter).
- **No parameterised/trainable gates anywhere.** `n_parameters = 0`. The circuit is a deterministic function of the input angles.
- **Two-qubit gate count:** 8 CZ × 2 blocks = **16** (vs ~62,940 CX for 16-qubit amplitude loading). `circuit_depth` is read back from the built Qiskit circuit for honest telemetry, not asserted.

### 5.3 Output representation (interpretable quantum feature vector)
- Execute the circuit to an **exact statevector** via the existing Aer backend (`quantum_ml/backends.py`, `statevector_engine`). 8 qubits ⇒ 256 amplitudes.
- **Pre-registered, fixed observable set (never searched):**
  - 8 single-qubit `⟨Z_i⟩` (per-qubit magnetization),
  - 8 nearest-neighbour `⟨Z_i Z_{i+1 mod 8}⟩` (matched-locality two-body correlations).
  - **Total = 16 quantum features** against 103 train positives — a defensible ratio. *We deliberately do **not** use all C(8,2)=28 pairs:* E0/E1's core lesson is that a large observable menu against ~100 positives is a multiplicity trap (DEC-032 caught exactly this at 136 observables). The set is frozen in code as a named constant, mirroring `readout.ObservableSet`, and is **chosen by structure, never by score**.
- **Genuinely-quantum handle (telemetry):** also expose the **connected correlations** `C_ij = ⟨Z_iZ_j⟩ − ⟨Z_i⟩⟨Z_j⟩` for the 8 NN pairs. On a product state these are 0; nonzero values are direct evidence the entangling layer produced correlations a separable encoding could not. This is shown as "entanglement witness" telemetry, **not** as an advantage metric.

### 5.4 Classical head
- A small **logistic regression** (primary) or **linear SVM**, regularization tuned by **patient-grouped CV inside TRAIN only** (`grouped_folds`, `e0_controls.py:144`).
- Emits the experimental secondary probability. Because the quantum stage has 0 trainable gates, all learning lives in this transparent linear head — nothing here can hide a barren plateau or a √(T/N) blow-up.

---

## 6. Matched classical control (built-in, always reported)

DEC-033 named the honest control for a fixed quantum feature map / reservoir: **classical random Fourier features (RFF) at equal output dimension.** E2 ships it as a first-class arm:

- **Primary control — RFF-16:** random Fourier features approximating an RBF kernel, emitting **16** features (equal to the quantum vector), from the **same PCA-8 input**, into the **same head**, with the **same train-only tuning**. RBF bandwidth via `median_heuristic_gamma` (`e0_controls.py:190`) on train distinct-pair distances (the same estimator E0 used for its RBF control).
- **Ablation control — raw PCA-8 → head:** does the quantum map add anything over its own 8-dim input? Answers "is the circuit doing work, or is PCA doing it all?"
- Both controls are scored on the **identical rows** under the identical protocol and reported **beside** the quantum arm in every payload and every telemetry surface. The demonstrator is never shown without its controls.

---

## 7. Experiment design (11 items, defined before implementation)

1. **Input representation** — frozen-localizer ROI → `extract_mobilenet` 576-d → train-fitted PCA-8 → angles in [0, π] (§5.1).
2. **Quantum circuit** — 8-qubit fixed Ry angle encoding + NN CZ ring, ≤2 re-uploading blocks, 0 trainable gates (§5.2).
3. **Output representation** — 16-dim quantum feature vector (8 `⟨Z_i⟩` + 8 NN `⟨Z_iZ_j⟩`), plus connected-correlation telemetry (§5.3).
4. **Classical control** — RFF-16 at equal dimension (primary) + raw-PCA-8 ablation, same head, same tuning (§6).
5. **Train/validation protocol** — fit PCA, angle-scaler, RFF, and head on **TRAIN**; select head/RFF hyperparameters by patient-grouped CV **inside TRAIN**; score once on **VALIDATION**. `build_patient_level_split` (`split.py:178`); reuse the frozen split manifest (`fe45df6d…`) so partitions match the classical baseline exactly.
6. **Primary metric** — **PR-AUC** on validation, primary condition `oracle/A_lesion_polygon`.
7. **Secondary metrics** — ROC-AUC and calibration (Brier score, ECE) on validation.
8. **Patient-level isolation** — `patient_overlap_train_validation = 0`, asserted in the payload (the split guarantees it; the assertion is re-checked, as E0/E1 do).
9. **Null test** — **patient-blocked column-permutation null** (`_patient_blocked_column_shuffle`, `e0_controls.py:559`), ≥200 draws, on the quantum head. *(Label-permutation is reported alongside but does not gate — E0 measured its power ceiling near p≈0.25.)*
10. **KEEP/KILL criterion** — see §8. A *demonstrator* bar (runs / non-degenerate / beats own null), **not** an advantage bar.
11. **Runtime budget** — quantum stage ≤ **500 ms/image** on the demo machine, measured **cold and under sustained load** per DEC-026; no budget claim rests on an unmeasured extrapolation. (8-qubit exact statevector is expected to be single-digit ms/image; the 500 ms ceiling is deliberately generous.)

**Reported condition:** primary is `oracle/A_lesion_polygon`, reported against the **DEC-024 area-only geometry-leak bar (validation ROC-AUC 0.72)**, never against 0.5. `A_all` may be reported for completeness with its area-leak caveat, but is **not** decision-bearing. **TRAIN + VALIDATION ONLY. TEST FROZEN.**

---

## 8. KEEP / KILL and runtime budget (honest, demonstrator-framed)

The demonstrator's claim is *"a real quantum circuit runs on real ROI-derived features and yields an honest, non-degenerate, better-than-null, interpretable readout."* KEEP/KILL tests exactly that — **not** whether quantum beats classical.

### 8.1 KILL (do not ship even as secondary; fall back to telemetry-only or drop) if **any**:
- **K1 Runtime:** quantum stage > 500 ms/image (measured cold + under load, DEC-026). A demonstrator that stalls the demo is worse than none.
- **K2 Degeneracy:** > half the 16 observables fall below the E0 variance floor (`1e-9`) across validation images, or the quantum feature matrix is rank-deficient — i.e. the "readout" is effectively constant and the telemetry would be theatre.
- **K3 Below-null:** the quantum head's validation PR-AUC does **not** exceed its own patient-blocked permutation-null p95. A secondary signal that cannot beat its own null is noise.

### 8.2 KEEP (ship as **experimental secondary**, clearly labelled; classical stays the clinical verdict) if:
- runs within budget (¬K1) **and** produces non-degenerate interpretable readouts (¬K2) **and** clears its own null (¬K3).
- **Independent of whether it beats the matched classical control.**

### 8.3 Always reported (regardless of KEEP/KILL):
- quantum PR-AUC, RFF-16 control PR-AUC, raw-PCA-8 ablation PR-AUC, the null p95, ROC-AUC, calibration, runtime, and the connected-correlation summary — beside each other.

### 8.4 Pre-disclosed expected outcome (stated *now*, before any fit):
Given DEC-032 (amplitude state empty), DEC-034 (no classical bottleneck at this N), and the QML literature (Schreiber 2023: reuploading models have efficient classical surrogates; Bowles 2024: classical ties/beats QML broadly), the **expected** result is that the quantum feature map **ties or loses** to RFF-16. **That is the disclosed prior, not a KILL condition.** The demonstrator's value is that a genuine quantum computation runs on genuine ROI-derived data and reports itself honestly — never a claim of superiority.

---

## 9. Test-partition firewall

- E2 modules read **TRAIN + VALIDATION only**; every payload asserts `test_partition_used: false`, enforced by **source inspection** the same way `pixel_comparison`, `e0_*`, and `e1_failure_map` are.
- The frozen test partition (381 images / 48 patients / 18 positives), DEC-030's single test read, `reports_final_*.json`, `evaluation.json`, and the localizer `test_evaluation_report.json` are **untouched**.
- No new preprocessing version and no new dataset pass. All E0/E1/Phase-D artifacts (both pixel caches, `e0_readout.py`, `e0_controls.py`, `quantum_ml/readout.py`, `state_diagnostics.py`) are preserved unmodified.
- Any final test read (if ever authorised) would require the E2 pipeline to be frozen first and would be a *new, single* read — not part of this proposal.

---

## 10. Telemetry and UI mapping

The frozen `QuantumSummary` (`backend/schemas/inference.py:57`) describes the **amplitude-VQC** provenance and the DEC-034 `primary_*` display contract — **neither is altered**. E2 adds a **sibling** object so the two paths never collide:

**New `QuantumVisualSummary` (additive, experimental):**

| Field | Value for E2 | Directive telemetry item |
|---|---|---|
| `n_qubits` | 8 | "# qubits" |
| `circuit_depth` | measured from built circuit | "circuit depth" |
| `n_reuploading_blocks` | ≤2 | "quantum layers" |
| `n_trainable_parameters` | 0 | (honest: fixed feature map) |
| `two_qubit_gate_count` | ~16 | contrast vs ~62,940 |
| `state_dimension` | 256 | (2^8) |
| `n_quantum_features` | 16 | "X quantum features" |
| `observable_labels` | `["Z0",…,"Z0Z1",…]` | "measured observables" |
| `quantum_feature_vector` | 16 floats | "quantum feature vector" |
| `connected_correlations` | 8 floats | entanglement witness |
| `execution_time_ms` | measured | "execution time" |
| `backend_name` | e.g. `aer_statevector` | "backend/simulator" |
| `is_experimental` | `true` | secondary label |
| `matched_control_prauc` / `quantum_prauc` / `null_p95` | measured | honest comparison |

**Flutter card ("Quantum Visual Analysis", experimental):** `8 qubits · 2 quantum layers · 16 quantum features · Y ms · aer_statevector`, with the quantum feature vector rendered as a bar/heat strip and the connected-correlation values shown as the "entanglement" indicator. The card is **explicitly badged EXPERIMENTAL** and shows the matched-control number beside the quantum number. **No accuracy claim, no advantage claim, no percentage that could be read as a clinical risk.** The DEC-034 primary verdict card is unchanged and remains the headline.

---

## 11. Honest novelty audit

- Angle encoding, data re-uploading, fixed quantum feature maps / quantum reservoirs, and local 1-/2-body Z observables are each **standard**; none is novel, and E2 claims none as such.
- A fixed quantum feature map generally admits an efficient **classical surrogate** (Schreiber et al. 2023 for reuploading; random Fourier features for reservoirs). E2 therefore makes **no advantage claim** and **ships the classical surrogate as its control** (§6).
- "Genuinely quantum" here means the *computation is real* — an entangled 8-qubit state whose 2-body connected correlations are verifiably nonzero and non-factorising — **not** that it is advantageous. The two are kept strictly separate throughout.
- The only defensible contributions are **methodological**: (i) building the demonstrator under an explicit, pre-registered *demonstrator* bar (runs / non-degenerate / beats-own-null) rather than an advantage bar; (ii) attaching DEC-033's own named classical control so the result cannot be oversold; (iii) preserving the classical model as the clinical decision-maker and the test partition as frozen.

---

## 12. Reuse map (Phase 2 implementation surface)

**Reuse unchanged:**
- `backend/ml/features_image.py:267` `extract_mobilenet` (576-d) — ROI descriptor.
- `backend/dataset/split.py:178` `build_patient_level_split`, `SplitManifest`, `fingerprint_records` — partitions + manifest (frozen `fe45df6d…`).
- `backend/evaluation/e0_controls.py`: `grouped_folds` (:144), `median_heuristic_gamma` (:190) for the RFF/RBF bandwidth, `_patient_blocked_column_shuffle` (:559) for the null, `complementarity` (:589) / `run_condition_controls` (:879) as scoring scaffolds.
- `quantum_ml/readout.py`: the **observable/expectation layer** — `z_string_diagonal`, `observable_expectations` (the `p @ diag` GEMM), and the `ObservableSet` frozen-set pattern — is **encoding-agnostic** and reusable to compute `⟨Z_i⟩`/`⟨Z_iZ_j⟩` from any statevector's basis probabilities. Its **validation discipline** is also reused: assert the fast contraction against Aer `save_expectation_value` to 1e-10 before any reported number depends on it.
- `quantum_ml/backends.py` + `statevector_engine` — exact statevector execution.

**Must be newly written (Phase 2):**
- An **8-qubit angle-encoding circuit builder** (Ry encode + NN CZ ring, ≤2 re-uploading blocks). *Do **not** reuse `readout.evolve_states`* — it is specific to the 16-qubit amplitude-encoding ansatz (tensor-product rotations + CNOT-cascade permutation) and does not describe E2's circuit.
- The train-fitted **PCA-8 + angle-scaler** transform (persisted artifact).
- The **RFF-16** control transform.
- The E2 experiment driver (`backend/evaluation/e2_*.py`) mirroring the E0/E1 payload + source-inspection test conventions.
- `QuantumVisualSummary` schema (additive) + the Flutter experimental card.

---

## 13. Draft DEC-035 (committed to `DECISIONS.md` only when Phase 2 begins)

> **DEC-035 — E2 Builds the Fallback Quantum Visual Demonstrator, Not an Advantage Model.**
> **Status:** PROPOSED (spec only; nothing implemented).
> **Context:** DEC-034 returned a NULL failure map — no classical bottleneck, no advantage-motivated quantum model authorised. The SIH quantum-track directive's fallback clause authorises the *smallest honest quantum visual-processing demonstrator* with the classical model kept as the clinical decision-maker.
> **Decision:** Implement **Candidate B** — 8-qubit fixed (0-trainable-gate) angle-encoded quantum feature map over a train-fitted PCA-8 compression of the frozen-localizer ROI's MobileNet descriptor, read out as 16 pre-registered local Z observables, feeding a small classical head — as an **experimental secondary** signal. The advantage path (A/D = Q1 projected quantum kernel) is **not** built: its DEC-033 gates (verdict-B + Huang g_CQ) are unmet. C (quanvolution) and E (trainable VQC) are excluded (deferred / √(T/N)-vacuous).
> **Rationale:** Different from the DEC-032 negative control on every axis; √(T/N)-safe by construction (0 trainable gates); interpretable local observables (Cerezo 2021); O(n) angle loading; ships DEC-033's named classical control (RFF at equal dimension) so it cannot be oversold.
> **Consequences:** `test_partition_used: false` (source-enforced); DEC-030's single test read stands; DEC-034 `primary_*` display contract and all frozen artifacts untouched; no advantage claim; classical remains the clinical decision-maker. KEEP/KILL is a demonstrator bar (runs ≤500 ms/image / non-degenerate / beats own null), reported beside its matched control and null.
> **What it does/does not license:** licenses a real, honestly-bounded quantum demonstrator; does **not** license any advantage claim, any test read, or any change to the clinical decision.

---

## 14. What this proposal does and does not authorize

- **Authorizes (on acceptance):** proceeding to **Phase 2** — implementing *only* Candidate B as specified above.
- **Does not authorize:** any advantage claim; any test-partition read; any change to the localizer, calibration, thresholds, DEC-034 `primary_*` fields, or any frozen artifact; the quantum kernel (A/D), quanvolution (C), or any trainable VQC (E); any commit.
- **Next phases (unchanged from the directive):** Phase 2 implement → Phase 3 small TRAIN/VAL experiment (PR-AUC primary, patient isolation, null, KEEP/KILL, test frozen) → Phase 4 additive experimental secondary field → Phase 5 experimental telemetry card.
