# Phase E0 — Controlled Readout Diagnostics for the 16-Qubit Amplitude VQC

> **Status:** SPECIFICATION. Nothing here is implemented yet. This document defines
> experiments, not fixes. It is written so that a negative result is as publishable as
> a positive one, and so that a positive result cannot be manufactured by tuning.
>
> **Governing decision record:** DEC-031.
> **Predecessor:** Phase D final report; DEC-024 (oracle-ROI geometry confound),
> DEC-026 (depth transfer and two wrong cost estimates), DEC-030 (pipeline frozen,
> test partition read once).

---

## 1. The question

Phase D measured the 16-qubit amplitude-encoded VQC end to end and it lost to the
classical reference in every condition where the comparison is defined: validation
margin −0.264 PR-AUC on the honest oracle condition, test margin −0.165 there and
−0.549 on `predicted/B_localized`. The training objective moved 0.0008–0.0025 off
`ln 2 = 0.693147`.

Three different failures produce that same symptom, and Phase D cannot distinguish
them:

| # | Failure mode | Claim it would support |
|---|---|---|
| **F1** | **Representation failure** | The 65,536-amplitude state does not encode lesion-discriminative information in any measurable way. |
| **F2** | **Measurement failure** | The state encodes it, but `⟨Z₀⟩` is a projection that discards it. |
| **F3** | **Optimizer failure** | The state and readout could expose it, but SPSA on a near-flat objective never found the parameters. |

**Phase E0 exists to separate F1, F2 and F3 experimentally, in that order.** It does
not attempt to fix any of them.

The critical question for E0.1, stated so that it has a falsifiable answer:

> Does the existing 16-qubit state contain class-discriminative information that
> `⟨Z₀⟩` was failing to expose — measurable *before* any trainable circuit
> optimisation, and surviving comparison against a permutation null and against
> classical controls on the same rows?

---

## 2. What this phase explicitly does not do

- **No E1/E2 adaptive-patch architecture.** Not designed, not prototyped, not benchmarked.
- **No new encoding.** The 65,536-amplitude representation is the object under test, not a variable.
- **No ranking loss.** Deferred until after E0.2 has isolated the optimizer.
- **No combined change.** The P0 bundle proposed in discussion (trainable multi-qubit
  head + adjoint gradients + ranking loss, all at once) is **rejected as an
  experiment**, because a single measurement of three simultaneous changes cannot
  attribute the result to any of them.
- **No hard-coded readout scale.** The previously suggested `a ≈ 300` initialisation is
  **withdrawn**: it embeds a measured property of one condition's observable spread as
  a constant, which would silently break on any other condition and would make the
  readout's benefit unattributable. Readout features are standardised using
  **train-partition mean and standard deviation only** (§6.4).
- **No test-partition access.** Zero reads, in every experiment in this document (§10).

---

## 3. Frozen invariants — the control surface

E0.1 changes **only** the measurement/readout. Everything below is held at the exact
Phase D values and is asserted in code, not assumed:

| Invariant | Value | Source of truth |
|---|---|---|
| Encoding | 256×256 grayscale ROI → uint8 0–255 → `/255` → L2 normalise | `backend/ml/pixel_pipeline.py` |
| Amplitudes | exactly **65,536** | `V1_PIXEL_COUNT` |
| Qubits | exactly **16** | `V1_QUBIT_COUNT` |
| Pixel→index map | `index = row*256 + col` (C order) | `pixel_pipeline.py:212` |
| Basis convention | Qiskit little-endian: qubit *k* = bit *k* of the index | `quantum_ml/backends.py:81` |
| Ansatz | L1: `ry(θ)`+`rz(ψ)` on all 16 qubits, then linear nearest-neighbour CNOT cascade with circular closure | `vqc_classifier.py:171` |
| Circuit parameters | exactly **32** | artifact `n_parameters` |
| Circuit weights | the **frozen Phase D weights**, plus a fixed untrained seeded draw | sha256-pinned, §11 |
| Split | train 230 pat/1,692 img · validation 50/363 · test 48/381 | manifest `fe45df6d8900…` |
| Preprocessing | `carescan-v1-pixel-1`, ROI edge 256, lanczos, cache `v1-pixels-1` | artifact `preprocessing` block |
| Seed | 42 (numpy, `algorithm_globals`, weight-init `Generator`) | artifact `seeds` block |
| Loss (E0.1) | class-weighted BCE, inverse-frequency weights normalised to mean 1 | `vqc_classifier.py:370` |
| Execution | exact statevector, `shots: null` | artifact `quantum` block |

Any run whose loaded artifacts do not hash to the recorded values must **refuse to
start**, not warn.

---

## 4. What `⟨Z_S⟩` actually measures on this encoding

This section is the reason E0.1 is worth running at all, and it must be verified
numerically (§12) rather than trusted.

For a Pauli-Z string over qubit subset `S ⊆ {0…15}`, the expectation on any state with
computational-basis probabilities `p(x)` is

```
⟨Z_S⟩ = Σ_x p(x) · (−1)^popcount(x AND S)
```

which is exactly the **Walsh–Hadamard transform of `p` evaluated at `S`**. Two
consequences, both load-bearing:

**4.1 On the bare encoded state (before the ansatz),** `p(x) = pixel(x)² / ‖pixel‖²`, so
the Z-string expectations *are* the normalised 2D Walsh–Hadamard spectrum of the
squared-intensity ROI. With qubits 0–7 = column bits and 8–15 = row bits:

| Observable | Spatial meaning on the 256×256 ROI |
|---|---|
| `Z₀` | energy difference between **even and odd columns** (finest horizontal parity) |
| `Z₇` | energy difference between the **left and right halves** |
| `Z₈` | energy difference between **even and odd rows** |
| `Z₁₅` | energy difference between the **top and bottom halves** |
| `Z_k`, k=0…7 | horizontal Walsh parity at dyadic scale `2^(k+1)` columns |
| `Z_{8+j}`, j=0…7 | vertical Walsh parity at dyadic scale `2^(j+1)` rows |
| `Z_i Z_{i+8}` | **diagonal (quadrant-contrast) Haar-like feature at matched x/y scale** — for `i=7`, `(top-left + bottom-right) − (top-right + bottom-left)` |

This predicts the Phase D result mechanically: **`Z₀` is the single highest-frequency
horizontal Walsh coefficient of the energy image**, which for any natural photograph is
≈ 0. The measured spread `⟨Z₀⟩ ∈ [−0.0094, +0.0091]`, sd 0.0028, is what that
prediction looks like. It is a statement about the *choice of observable*, not about the
information content of the state — which is precisely the F1-vs-F2 ambiguity E0.1 must
resolve.

**4.2 Computationally,** one Fast Walsh–Hadamard Transform of the 65,536-length output
probability vector yields **all 65,536 Z-string expectations simultaneously**, at
`n·2ⁿ ≈ 1.05 M` operations per sample. So variants B, C and D below cost essentially
nothing beyond the single forward pass that variant A already requires. This is an
implementation convenience only: it must be proven equivalent to Aer's
`save_expectation_value` on a set of Z-strings (§12) before any reported number depends
on it, because constraint 8 of the governing spec requires the simulator to stay
mathematically equivalent to the defined circuit.

**4.3 The multiplicity hazard this creates.** Cheap access to 65,536 observables is a
multiple-comparisons trap. Therefore every observable set in E0.1 is
**pre-registered in this document**, fixed before any run, and never searched. No
variant may be defined by choosing observables that scored well. Any future
observable *selection* must be train-only, with the selection multiplicity reported and
validation used exactly once (§9.3).

---

## 5. Partitions and ROI conditions

E0 runs on train + validation only, with ROI conditions kept separate exactly as in
Phase D — never merged, never pooled into one ROI statistic.

| Role | Condition | train | validation |
|---|---|---|---|
| **Primary** | `oracle / A_lesion_polygon` | 215 img / 103 pos / 112 pat | 52 / 21 / 25 |
| **Realism check** | `predicted / B_localized` | 1,620 / 96 / 229 | 353 / 21 / 50 |
| Secondary, caveated | `oracle / A_all` | 1,692 / 104 / 230 | 363 / 21 / 50 |

`A_lesion_polygon` is primary because it is the only condition where every row has an
annotator lesion box, it is near-balanced (47.9% train prevalence), and it is cheap.
`B_localized` answers whether any finding survives realistic localisation.

**`A_all` numbers must never be reported without the area-only geometry leak beside
them** (DEC-024): an ROI-geometry-only classifier reaches ROC-AUC **0.8477** on those
validation rows, so an `A_all` figure below that is measuring the confound, not the
quantum model. `A_center_crop` and `A_region_polygon` have 0 validation positives and
are reported as counts with `null` metrics.

---

## 6. Experiment E0.1 — Multi-observable readout

### 6.1 Design

Two circuit-weight settings × four readout variants × three conditions, with everything
in §3 frozen. The two weight settings are deliberate:

- **W-frozen** — the selected Phase D weights (sha256-pinned per condition). Answers:
  *given the circuit we actually trained, was the information there and unread?*
- **W-untrained** — the seeded initial draw, seed 42. Answers the same question with
  the trained circuit removed, so a positive result cannot be credited to training.

Plus one setting that removes the circuit entirely:

- **W-identity (pre-ansatz probe)** — observables read directly off the encoded state,
  no ansatz at all. This is the cleanest available separation of F1 from F2: if the
  encoded state carries no signal under *any* pre-registered observable set with no
  circuit in the way, that is evidence for representation failure independent of both
  ansatz and optimizer.

### 6.2 Readout variants (pre-registered)

| Variant | Observables | Count | Readout | Trainable params |
|---|---|---|---|---|
| **A0** | `Z₀` | 1 | fixed `(1 − ⟨Z₀⟩)/2` — the exact Phase D map | 0 |
| **A1** | `Z₀` | 1 | standardised → logistic | 2 |
| **B** | `Z_k`, k = 0…15 | 16 | standardised → logistic | 17 |
| **C** | `Z_k` (16) + `Z_i Z_{i+8}`, i = 0…7 (8) | 24 | standardised → logistic | 25 |
| **D** | `Z_k` (16) + all `Z_i Z_j`, i<j (120) | 136 | standardised → logistic | 137 |

**A1 is a deliberate addition to the requested list, and it is not optional.** Without
it, any gain of B over A0 is confounded between *more observables* and *the readout
being fitted at all*. A1 isolates the second, so `B − A1` attributes cleanly to
observable count. Report all four.

Variant C's pairing is spatially motivated per §4.1: `Z_i Z_{i+8}` couples the column
bit and the row bit **at the same dyadic scale**, giving a diagonal quadrant contrast
rather than an arbitrary qubit pair. Variant D is the unrestricted 2-local set,
included as a capacity ceiling — with 136 features against 103 train positives it is
expected to overfit, and the permutation null (§6.5) is what makes that visible instead
of persuasive.

### 6.3 The signal-before-training probe

This is the decisive measurement and it runs **before** any trainable optimisation of
circuit parameters:

1. **Univariate.** Per observable: train and validation ROC-AUC with sign, mean and sd
   by class, standardised mean gap, and the `p05/median/p95` of the observable's
   distribution. Report the full distribution across observables and `max |AUC − 0.5|`,
   not just the best one.
2. **Multivariate.** Plain and L2-regularised logistic regression on the standardised
   observables, **fitted on train only**, with any regularisation strength chosen by
   cross-validation *inside train*. Evaluated once on validation.
3. **Comparison to A0.** The same protocol on the single `Z₀` feature, so the answer is
   a difference, not an absolute.

If step 2 produces validation PR-AUC materially above both the A0 result and the
permutation null, the answer to the critical question is **yes** — the state contained
signal `Z₀` discarded — and it was obtained with the circuit frozen, which rules out F3
as the explanation for *that* portion of the gap.

### 6.4 Normalisation discipline

- Standardisation uses **train-partition mean and standard deviation only**. The scaler
  never sees validation rows; a test asserts this by construction, not by review.
- Observables with train sd below a pre-declared variance floor (`1e-9`) are **excluded
  and counted in the report**, because standardising them amplifies floating-point noise
  into apparent features. The count of excluded observables is a reported number.
- Per condition, the scaler is refit on that condition's train rows. Scalers are never
  shared across conditions.
- The scaler is persisted with the results so any reported number can be recomputed.

### 6.5 Permutation null — mandatory

136 observables against 103 positives can produce a flattering validation number from
capacity alone. For every (variant × weight setting × condition):

- Shuffle train labels within the condition (patient-blocked), refit the readout,
  evaluate on validation. **200 permutations**, seeds `42…241`, recorded.
- Report the null distribution of validation PR-AUC and ROC-AUC (`mean`, `sd`, `p95`,
  `p99`) and the observed value's rank within it.

A variant whose observed PR-AUC falls inside its own null is reported as **no detected
signal**, regardless of how it compares to A0.

### 6.6 Required reporting, per (variant × weight setting × condition)

Non-negotiable, and `null` where undefined rather than filled in:

- **Observable distributions** — per observable: `n`, mean, sd, min, `p05`, median,
  `p95`, max, split all / positive / negative.
- **Separability** — per-observable train and validation ROC-AUC with sign;
  standardised class mean gap; count of observables exceeding the null `p95`.
- **Discrimination** — PR-AUC, ROC-AUC on train and validation, each with partition,
  condition, `n_samples`, `n_positive`, and **prevalence** attached. PR-AUC is the
  primary metric and is read against prevalence, never against 0.5.
- **Calibration** — Brier and ECE, plus the calibration curve bins.
- **Threshold-dependent** — sensitivity, specificity, precision, F1, confusion matrix,
  at a threshold selected **on validation only** (or on train, stated explicitly).
- **Gradient magnitude** — `‖∇_θ L‖₂` over the 32 circuit parameters by parameter-shift
  (exact; 2 evaluations per parameter), plus the per-parameter magnitude distribution,
  at both weight settings. Additionally, the barren-plateau indicator: variance of
  `∂L/∂θ_k` across **20 independent random weight draws**, reported per parameter. This
  is the quantitative bridge to E0.2 — it measures whether a readout change *creates* a
  trainable gradient where A0 had none.
- **Cost** — wall-clock for the forward pass over the condition's train rows, FWHT time,
  peak memory, number of observables, trainable parameter count. **Measured both cold
  and under sustained load**, and both reported: per DEC-026 the machine thermally
  throttles ~5×, and a single cold scalar previously drove two wrong decisions in
  opposite directions.
- **Signal-before-training verdict** — §6.3 result with its permutation-null rank.

### 6.7 Interpretation rules

- `B/C/D` beating `A0` establishes **F2 (measurement failure) contributed**. It does
  **not** establish that the amplitude encoding is adequate (§9).
- `A1 ≈ A0` with `B > A1` attributes the gain to observable count. `A1 > A0` alone means
  the fixed score map was the binding constraint, which is a distinct finding.
- All variants inside the permutation null, at all three weight settings including
  W-identity, is evidence for **F1 (representation failure)** — the strongest available
  without changing the encoding.
- A gain visible only at W-frozen and absent at W-untrained and W-identity is
  attributable to the trained circuit, not the readout, and must be reported as such.

---

## 7. Experiment E0.2 — Optimizer isolation (gated)

**Runs only if E0.1 shows signal above the permutation null in at least one variant on
the primary condition.** If E0.1 is negative, E0.2 is not run and the decision gate
(§9) is entered directly.

Architecture and readout are **fixed** at the best validation-selected E0.1 variant.
Only the optimizer changes. **No ranking loss** — the loss stays class-weighted BCE, so
that E0.2 measures optimisation and nothing else.

| Arm | Method | Notes |
|---|---|---|
| **O1** | SPSA | the Phase D baseline, same budget, for reference |
| **O2** | Parameter-shift gradient + Adam / L-BFGS-B | exact gradients, 64 evaluations per full gradient at 32 parameters |
| **O3** | Adjoint / reverse-mode | **only if benchmarked and verified** — see below |

**O3 admission criteria, all required before any O3 number is reported:**

1. **Numerical equivalence** — gradients agree with parameter-shift to ≤ `1e-8` relative
   on a fixed set of random weight draws, and forward expectations agree with Aer's
   `save_expectation_value` to ≤ `1e-10`.
2. **Measured speedup** — benchmarked cold *and* under sustained load, against O2 and
   O1 on the same rows. No projected or extrapolated figure may be used to justify a
   budget. This is a direct consequence of DEC-026.
3. **Recorded as a simulator method, not a hardware claim** — reverse-mode
   differentiation of a statevector has no hardware analogue. It must be reported
   separately from feasibility, exactly as `set_statevector` is.

Report per arm: loss trajectory (initial, final, best, full history), displacement from
`ln 2`, objective and circuit evaluation counts, wall-clock, validation PR-AUC/ROC-AUC
at matched *evaluation* budget **and** at matched *wall-clock* budget (they rank
differently and both are reportable), gradient norm over training, and convergence
diagnostics.

**The decisive comparison is loss displacement.** If O2/O3 move the objective by orders
of magnitude more than SPSA's 0.0008–0.0025 and validation metrics improve, F3 is
confirmed as a contributor. If exact gradients also stall, the objective is genuinely
flat at these parameters and F3 is excluded — a stronger and more useful result than
Phase D could produce.

---

## 8. Experiment E0.3 — Local / 2D-aware measurement (gated)

**Runs only after E0.1 and E0.2 have reported.** Extends the observable set with
spatially motivated structures, each of which must carry a written justification in
terms of §4.1 before it is measured — no observable enters by having scored well.

Pre-registered candidates, with their spatial motivation:

| Observable family | Spatial meaning | Motivation |
|---|---|---|
| `Z_k` single, by scale | 1D Walsh parity at dyadic scale, per axis | scale sweep: which spatial frequency of the energy image carries class signal |
| `Z_i Z_{i+8}` | diagonal quadrant contrast at matched x/y scale | a lesion is a compact 2D region; matched-scale diagonal contrast is the lowest-order 2D feature the Z basis admits |
| `Z_i Z_j`, same axis (`i,j ≤ 7` or `i,j ≥ 8`) | product of two parities on one axis = finer 1D partition | separates 1D multiscale structure from genuine 2D structure |
| Coarse-scale-only subsets (`Z₆,Z₇,Z₁₄,Z₁₅` and their products) | half/quarter-image contrasts | lesions are coarse relative to 256×256; high-frequency parities are predicted uninformative and this tests that prediction |
| Weighted sums over a Z-string band | a Walsh frequency *band* rather than a single coefficient | single Walsh coefficients are noisy; band energy is the natural robust analogue |

Every family is a **fixed set**. Selection among families is train-only, with the
number of families considered reported alongside the result, and validation consulted
once. A family that requires searching to look good is reported as searched.

E0.3 concludes with an explicit written answer to: **is the whole-image 65,536-amplitude
representation worth retaining?**

---

## 9. Classical controls and the decision gate

### 9.1 Controls — all on identical rows, per condition

| Control | Definition | Discipline |
|---|---|---|
| **RF (current classical reference)** | frozen `random_forest` record, 181 fused descriptor + clinical features, PCA-reduced | refit from `BaselineRecord` on train; validation scored, never fitted |
| **Linear on the same amplitudes** | linear-kernel SVM / logistic on the 65,536-dim amplitude vectors | **must be redone with train-only hyperparameter selection.** The existing figures (PR-AUC 0.128 oracle / 0.263 predicted) had `C` chosen on the evaluation partition and are therefore optimistic for the control — using them as-is would understate the bar |
| **RBF SVM on amplitudes** | precomputed Gram, γ by median-heuristic on **train** pairwise distances, `C` by within-train CV | 1,692² Gram is tractable; if it is not for a condition, say so rather than dropping it silently |
| **Phase D VQC** | the three frozen artifacts, sha256-pinned | numbers already measured; no re-run needed |

Row alignment is mandatory: PR-AUC is prevalence-sensitive, so every comparison is made
on the same image ids with the same prevalence, exactly as `pixel_comparison` already
enforces.

### 9.2 Complementarity test

The governing instruction is explicit: a readout improvement does **not** validate the
amplitude encoding. The encoding must beat a strong classical control **or provide
complementary information**. The second is operationalised, not asserted:

- Stack: logistic regression on `[RF score, best-E0 quantum readout score]`, fitted on
  **train only**, evaluated once on validation.
- Report validation PR-AUC of the stack against RF alone, with the permutation null for
  the stack.
- Report the correlation between the two scores. A quantum score highly correlated with
  the RF score adds nothing even if its standalone PR-AUC is respectable.

Complementarity is claimed only if the stack beats RF alone **and** the stack's gain
survives its own permutation null.

### 9.3 Decision gate — entered only after E0.1 (and E0.2/E0.3 if run) have reported

No absolute numeric thresholds are invented here. Per the standing project rule, where
no explicit go/no-go criterion exists in the project documents, none is manufactured;
the criteria below are **comparative against measured controls and nulls**, which is
what the data can actually support.

| Outcome | Reading | Decision |
|---|---|---|
| Readout gain over A0, survives null, and beats-or-complements the amplitude-space classical controls | F2 was a real contributor and the representation has exploitable content | **A — repair the existing amplitude-encoded architecture** |
| Signal present but confined to coarse/2D-structured observables, and 1D whole-image observables are inside the null | the information is spatial and the whole-image amplitude state expresses it poorly | **B — move to spatial quantum processing** |
| Observables separate classes but no trainable circuit configuration exploits it (E0.2 stalls with exact gradients) | the state is informative but the variational circuit is the wrong extractor | **C — move to a quantum kernel** |
| No variant, at any weight setting including W-identity, exceeds its null on the primary condition | representation failure (F1) | **D — different quantum visual representation** |

A tie, or a result that supports two rows, is reported as a tie. E1/E2 remains
unspecified and unimplemented until this gate is passed with a written decision.

---

## 10. Test isolation

- **Every experiment in this document reads train and validation only.** The test
  partition is not read — not once, not for a sanity check.
- Enforced the same way Phase D enforces it: the E0 module contains no test-partition
  access, asserted by source inspection (`tests/` pattern already established in
  `tests/test_pixel_comparison.py::TestNoTestPartition`), and the payload carries
  `test_partition_used: False`.
- The Phase D test read already happened once (DEC-030). If E0 leads to a new frozen
  configuration, that configuration gets **exactly one** new test read, after freezing,
  via `backend.evaluation.pixel_final_test --confirm-frozen`. Repeated reads would
  convert the held-out set into a second validation set, which is the specific failure
  the Phase D structure was built to prevent.
- No threshold, regularisation strength, observable set, variant, optimizer or scaler
  may be selected using test rows.

---

## 11. Artifacts and reproducibility

Persist per run, JSON or `.npz` only — **no pickle**:

- `e0_diagnostics_version` (proposed `v1-e0-readout-1`), UTC timestamp,
  `test_partition_used: False`.
- Input provenance: split manifest path + sha256 (`fe45df6d8900…`), pixel cache
  metadata block, and the **sha256 of every Phase D VQC artifact whose weights are
  loaded**, so a reported number traces to exact weights.
- Full configuration: variant definitions with their explicit observable index lists,
  weight setting, condition, seeds, variance floor, permutation count and seed range.
- Fitted readout coefficients and the train-only scaler (mean, sd per observable).
- All metrics from §6.6, per (variant × weight setting × condition), with partition,
  `n_samples`, `n_positive` and prevalence attached to every metric.
- Permutation null distributions.
- Environment: Python, numpy, qiskit, qiskit-aer versions; platform.
- Runtime measurements, cold and loaded, labelled as such.

Existing frozen artifacts are read-only. **The Phase C localizer artifact and the three
Phase D VQC artifacts must not be modified or overwritten.**

---

## 12. Required tests

To be added alongside the implementation, in the style of the existing Phase D suites:

**Correctness of the fast path**
- FWHT-derived `⟨Z_S⟩` matches Aer `save_expectation_value` to ≤ `1e-10` for a fixed set
  of Z-strings (singles, matched-scale pairs, random strings) on random normalised
  states. *This is the constraint-8 equivalence proof and it gates every B/C/D number.*
- `Z₀` from the FWHT path equals the existing `z0_diagonal` path exactly.
- Parameter-shift gradients match finite differences to ≤ `1e-6`.

**Invariants**
- Exactly 16 qubits and exactly 65,536 amplitudes at every stage.
- L2 norm 1 within `1e-12`; the raw uint8 representation is not mutated.
- Observable counts are exactly 1 / 1 / 16 / 24 / 136 for A0 / A1 / B / C / D.
- Circuit parameter count stays 32 in every E0.1 variant.
- Loaded artifact hashes match the recorded values, else refuse.

**Leakage and discipline**
- The scaler's fitted mean/sd are bit-identical whether or not validation rows exist in
  the loaded dataset (proves train-only fitting).
- No test-partition access, by source inspection.
- Determinism: two runs at seed 42 produce identical observables, coefficients and
  metrics.
- ROI conditions stay separately identifiable; fallback rows are never merged.
- A single-class subset yields `null` metrics, not a fabricated number.
- Permutation-null harness sanity: on synthetic data with labels independent of
  features, the observed value lands inside the null.

**Regression**
- All existing tests continue to pass (currently **219 passed**), plus flake8 at the
  established repo baseline (54 pre-existing findings, no new ones).

---

## 13. Threats to validity, recorded up front

1. **Multiplicity.** 65,536 observables are cheaply accessible; only pre-registered sets
   are measured, and the permutation null is mandatory. Any post-hoc observable choice
   must be reported as searched.
2. **Small positive counts.** 21 validation positives in every condition. Validation
   PR-AUC has wide uncertainty; report it with the null distribution and do not treat
   small differences as ordering.
3. **Oracle-ROI geometry confound.** DEC-024 stands. `A_all` numbers travel with the
   area-only leak (ROC-AUC 0.8477) or they are not reported.
4. **Capacity in variant D.** 136 features on 103 train positives will overfit; the null
   and the train-vs-validation gap are what expose it.
5. **Patient-level dependence.** Multiple images per patient; permutations are
   patient-blocked so the null does not leak patient identity.
6. **Cost estimates.** Two prior estimates in this project were wrong in opposite
   directions (DEC-026). No budget decision in E0 may rest on an unmeasured
   extrapolation, and every runtime figure is reported cold and under load.
7. **Hardware feasibility is unchanged by anything in E0.** State preparation remains
   ≈ 62,940 CX at 16 qubits against the Shende–Bullock–Markov bound of 131,038 — ≈ 3,934×
   the L1 ansatz's 16 CX. A readout improvement changes no part of that, and no E0
   result may be presented as progress toward hardware viability.

---

## 14. What would falsify the amplitude encoding

Stated in advance so that a negative outcome is a result rather than a disappointment:

> If no pre-registered observable set — including the pre-ansatz W-identity probe, which
> removes the circuit and the optimizer from the question entirely — exceeds its
> patient-blocked permutation null on the primary condition, then the 65,536-amplitude
> whole-image state does not expose lesion-discriminative structure to any 2-local Z
> measurement, and the Phase D result is a **representation** limitation rather than a
> readout or optimizer limitation.

That outcome selects decision **D** and is reported plainly, with the numbers, exactly
as the Phase D negative result was.
