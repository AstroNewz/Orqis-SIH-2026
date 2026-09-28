# Phase E3 — Quantum-Advantage Research: Does the E2 Representation Add Information Beyond the Strongest Classical Baseline?

**Status:** Rounds 1–4 complete — **four independent null results (mission §29B), reported as measured.** Stop-condition (B) reached exhaustively across every qualitatively distinct quantum representation the mission named (§ "The research ladder").
**Round 1 (fixed-map fusion):** `v1-e3-hybrid-fusion-1` · [`backend/evaluation/e3_hybrid_fusion.py`](../backend/evaluation/e3_hybrid_fusion.py) · [`reports_e3_hybrid_fusion.json`](../reports_e3_hybrid_fusion.json) · DEC-037.
**Round 2 (quantum kernel / QSVM):** `v1-e3-quantum-kernel-1` · [`backend/evaluation/e3_quantum_kernel.py`](../backend/evaluation/e3_quantum_kernel.py) · [`reports_e3_quantum_kernel.json`](../reports_e3_quantum_kernel.json) · DEC-038.
**Round 3 (trainable shallow VQC):** `v1-e3-trainable-vqc-1` · [`backend/evaluation/e3_trainable_vqc.py`](../backend/evaluation/e3_trainable_vqc.py) · [`reports_e3_trainable_vqc.json`](../reports_e3_trainable_vqc.json) · DEC-039.
**Round 4 (richer feature map + structured observables):** `v1-e3-feature-map-1` · [`backend/evaluation/e3_feature_map.py`](../backend/evaluation/e3_feature_map.py) · [`reports_e3_feature_map.json`](../reports_e3_feature_map.json) · DEC-040.
**Seed:** 42 · **Folds:** 5-fold patient-grouped `StratifiedGroupKFold` on TRAIN · **Condition:** primary `A_lesion_polygon` only · **Test partition:** never read (`test_partition_used: false`).

---

## 1. The question this phase exists to answer

E2 (DEC-035/DEC-036) already answered *"is the quantum feature map a real, honest, non-degenerate computation that clears its own null?"* — yes (validation PR-AUC 0.879, KEEP). It did **not** answer the harder mission question (§30):

> **Does a genuine quantum / hybrid configuration carry information the *best classical baseline* does not already have?**

The best classical baseline is **C7** — the E1.1 data-driven fusion of the two strongest non-descriptor feature families:

```
C7 = hstack([ C6 = MobileNetV3-Small (576-d),
              C5 = descriptor(163) + multiscale(456) = 619-d ]) = 1195-d, logistic (C=0.01)
```

C7 scores **validation PR-AUC 0.913038** on the primary leak-free condition (`reports_e1_failure_map.json`). That is the bar. Beating it — or adding defensible incremental information to it — is what would count as a quantum-enhanced result.

## 2. Method (almost entirely reused, by design)

The mission requires "identical patient-grouped folds and train-only fitting" and a test of "whether quantum features add incremental information." E1.1 already ships exactly that instrument: `_increment_null`, the pre-registered two-column stacking test that gated C7 itself. E3 points the identical machinery at **C7 → {quantum, RFF, PCA}** instead of E1's C1 → candidate. Reused verbatim: `grouped_folds`, `_standardise`, `_logistic_arm`, `_cv_select`, `_metrics`, `bootstrap_pr_auc`, `_increment_null`, `build_inputs`, `candidate_matrices`. A difference between E3 and E1 is therefore a difference in **representation**, never in harness.

The one genuinely new piece is a **fold-honest arm scorer** (`_score_arm`): the quantum / RFF / PCA pipelines all fit a PCA (and angle scaler / RFF map) that the mission forbids fitting on any row about to be scored. So the entire preprocessing pipeline is **refit inside each fold-train only** before the out-of-fold column is built. This makes the quantum out-of-fold column *stricter* than C7's (whose raw features carry no fitted preprocessing) — the conservative direction: it can only make a quantum "win" harder, never easier.

**Correctness gates (not results):**
- **C7 anchor reproduced 0.913038 exactly** (target 0.913038, tol 1e-6) — the C7 assembly is byte-identical to E1's scoring path.
- **Quantum feature map vs Aer `save_expectation_value`: max abs deviation 8.9e-16** (tol 1e-10) — before a single quantum feature is used.

**Rows (primary condition):** 215 train / 52 validation; 103 / 21 positive; 112 / 25 patients; **patient overlap 0**.

## 3. Standalone leaderboard

| Rank | Arm | Representation | Validation PR-AUC | Train-CV OOF PR-AUC | Val bootstrap 95% CI | Selected C |
|---|---|---|---|---|---|---|
| 1 | **C7** | classical fusion, 1195-d | **0.913038** | **0.766915** | [0.761, 0.989] | 0.01 |
| 2 | quantum | MobileNet→PCA-8→**8-qubit map**→16 obs | 0.868335 | 0.697764 | [0.675, 0.968] | 0.1 |
| 3 | pca | MobileNet→PCA-8 (raw linear control) | 0.838846 | 0.653770 | [0.616, 0.967] | 0.01 |
| 4 | rff | MobileNet→PCA-8→**RFF-16** (matched control) | 0.795360 | 0.630197 | [0.522, 0.980] | 100.0 |

Two facts, both measured, both on the same ordering in the better-powered TRAIN-CV surface as on validation:

1. **Quantum does not beat C7.** 0.868 < 0.913 on validation; 0.698 < 0.767 on train-CV OOF. Not a quantum advantage over the best classical baseline.
2. **Quantum is the best of the three matched low-dimensional MobileNet compressors** — above raw PCA-8 (0.839) and above the equal-width RFF-16 control (0.795), on both surfaces. The fixed 8-qubit map extracts a *more useful* 16-d representation from the PCA-8 angles than its classical analogue does. This is a genuine, honest positive about the map's quality; it is **not** a claim of advantage over C7, and it is not asserted as one.

> Note on the RFF control's absolute value: under E2's `fit_readout` scaler (DEC-036) RFF-16 scored 0.871; under E3's stricter fold-honest PCA + `_standardise` it scores 0.795. The RFF number is protocol-sensitive; what is robust across both protocols is that **all three low-dim representations sit well below C7's 0.913**, and quantum is never worse than its controls.

## 4. The incremental-information test (the gating question)

For each candidate, stack its leakage-free out-of-fold TRAIN prediction column on top of C7's, fit a 2-parameter logistic, score validation, and ask whether the real candidate column beats a **patient-blocked column-permutation** of itself (shuffled in whole patient blocks across TRAIN and VALIDATION — labels and the C7 column untouched). Gate = observed increment exceeds the null's p95.

| Candidate | C7→cand stack PR-AUC | Increment over C7 | Column-perm null empirical p | Exceeds null p95? | Score corr. w/ C7 | **Survives gate?** |
|---|---|---|---|---|---|---|
| quantum | 0.930154 | +0.017116 | 0.0945 | No | 0.778 | **No** |
| rff | 0.931658 | +0.018620 | 0.0697 | No | 0.731 | No |
| pca | 0.929254 | +0.016216 | 0.0796 | No | 0.760 | No |

**Reading:** all three candidates add a numerically similar small stack gain (~+0.016–0.019), and **none clears the pre-registered p95 significance gate.** Decisively, the quantum increment's p-value (0.0945) is the *weakest* of the three — a matched RFF control (0.0697) and raw PCA (0.0796) each produce an equal-or-larger, equal-or-more-significant increment. Whatever faint complementary signal exists in appending a 16-/8-d MobileNet-derived column to C7 is a **generic extra-dimensions effect, not quantum-specific.** (The label-permutation null is reported too but does not gate — it is structurally low-powered for an increment, flooring near p≈0.25, exactly as E0 documented.)

## 5. Conclusion — round 1

**On this evidence the E2 quantum representation adds no defensible information beyond the strongest classical baseline C7.** This is a clean mission-§29B null result:
- `quantum_beats_c7_on_validation`: **false**
- `quantum_beats_c7_on_train_oof`: **false**
- `defensible_quantum_advantage_over_c7`: **false**
- `quantum_specific_advantage`: **false**

It is consistent with — and quantitatively sharper than — DEC-035's pre-disclosed prior (Schreiber 2023 / Bowles 2024: fixed re-uploading maps admit efficient classical surrogates and typically tie-to-lose). No advantage is fabricated to satisfy the aspirational target. C7 remains the champion; the E2 KEEP demonstrator is unaffected (this experiment reads the same TRAIN/VALIDATION surface and changes nothing in the E2 pipeline).

**Recorded as:** DEC-037.

## 6. Round 2 — quantum kernel / QSVM (DEC-038)

Round 1's linear head reads an 8-qubit state through only 16 expectation values. The next rung keeps the identical fold-honest MobileNet→PCA-8→angle encoding but replaces the reader with the **fidelity quantum kernel** (Havlíček 2019) fed to an SVM:

```
K(x, x') = |⟨ψ(x) | ψ(x')⟩|²   over the exact 8-qubit E2 statevectors (256-d), precomputed-kernel SVC
```

An SVM on that Gram can carve boundaries a linear head on 16 features cannot. **Matched control (DEC-033, at the kernel level):** an RBF-SVM on the *same* L2-normalised PCA-8 input, median-heuristic bandwidth — the classical kernel analogue of the quantum one. Same correctness gates (C7 reproduces 0.913038; Aer 8.9e-16), same fold-honest per-fold refit, same `_increment_null` gate. Scores are the SVM `decision_function` (rank-based, so PR-AUC / bootstrap / the standardised stack are all exactly correct on an uncalibrated margin).

**The honest hazard, measured not assumed.** Fidelity kernels *exponentially concentrate* as qubits grow (Thanasilp 2022): the Gram collapses toward the identity and the SVM degenerates. Round 2 reports the **off-diagonal spread of the train Gram** as a concentration witness. Measured: mean **0.233**, sd **0.185**, range **4.8e-9 → 0.969**. The kernel is **not** concentrated — it is a healthy, non-trivial kernel. So the result below is *not* the trivial degeneracy failure; it is a genuine "this kernel geometry is less useful for this task than a plain RBF."

| Rank | Arm | Representation | Validation PR-AUC | Val ROC-AUC | Train-CV OOF PR-AUC | Val bootstrap 95% CI | Selected C |
|---|---|---|---|---|---|---|---|
| 1 | **C7** | classical fusion, 1195-d, logistic | **0.913038** | 0.934 | 0.766915 | [0.761, 0.989] | 0.01 |
| 2 | rbf_svm (matched control) | PCA-8 → RBF-SVM | 0.803026 | 0.865 | 0.691096 | [0.566, 0.971] | 0.1 (γ 0.492) |
| 3 | quantum_kernel | 8-qubit fidelity kernel → SVM | 0.772148 | 0.829 | 0.692485 | [0.507, 0.952] | 1.0 |

**Increment test (C7 → candidate), patient-blocked column-permutation null (gates the claim):**

| Candidate | C7→cand stack PR-AUC | Increment over C7 | Column-perm null empirical p | Exceeds null p95? | Score corr. w/ C7 | **Survives gate?** |
|---|---|---|---|---|---|---|
| quantum_kernel | 0.912375 | **−0.000663** | 0.5920 | No | 0.792 | **No** |
| rbf_svm | 0.926194 | +0.013156 | 0.1294 | No | 0.744 | No |

**Reading — a sharper negative than Round 1:**

1. **Quantum kernel does not beat C7** (0.772 vs 0.913): a wide gap on both validation and train-CV OOF.
2. **Quantum kernel underperforms its matched classical control** — RBF-SVM 0.803 > quantum 0.772 on validation, and they tie on train-CV OOF (0.691 vs 0.692). Round 1's quantum-linear map *beat* its controls; here the quantum kernel *loses* to the classical kernel on identical PCA-8 inputs. There is **no kernel-level quantum advantage** — if anything, the induced fidelity geometry is slightly worse than an ordinary RBF.
3. **Zero incremental information over C7.** The C7→quantum-kernel increment is **−0.0007** and sits at p≈0.59 — dead centre of the null. The quantum kernel adds literally nothing to C7. Even the RBF control adds only +0.013 (p=0.129), not clearing the p95 gate.
4. This is **not** the degeneracy failure: the witness shows a healthy kernel (sd 0.185). The kernel is real; its geometry is simply not more discriminative than classical alternatives for this task.

**Conclusion — round 2.** `quantum_kernel_beats_c7: false` · `quantum_kernel_beats_rbf_control: false` · `quantum_kernel_concentrated: false` · `defensible_quantum_advantage_over_c7: false` · `quantum_specific_advantage: false`. A clean mission-§29B null, and a second, independent data point toward stop-condition (B). **Recorded as:** DEC-038.

## 7. Round 3 — trainable shallow VQC (DEC-039)

Rounds 1–2 kept the E2 map's parameters frozen and varied only the *reader* (a linear head, then a fidelity kernel). Round 3 tests the last remaining degree of freedom: **let the circuit itself train.** Keeping the identical fold-honest MobileNet→PCA-8→angle encoding into the 8-qubit state, a shallow variational ansatz of `L` trainable layers is trained end-to-end:

```
|ψ(x)⟩  →  L × { RY(θ) on every qubit ; CZ ring }  →  8 local ⟨Z_i⟩  →  logistic head
          Adam on class-weighted BCE; depth L ∈ {1, 2} selected on TRAIN OOF
```

Because the E2 state and every gate here are **real** (RY is a real rotation, CZ a real ±1 diagonal), the whole forward map is real-valued and differentiated by exact autodiff in torch float64 — no parameter-shift, no shot noise. **Matched control (DEC-033):** a capacity-matched `tanh` MLP on the *same* L2-normalised PCA-8 input. **Ablation control:** the identical VQC with the CZ ring removed (a product-state circuit) — isolating what entanglement, specifically, contributes.

**Two honest hazards, both guarded and one witnessed:**
- **Overfitting a trainable quantum model.** The generalisation bound `sqrt(T/N)` (Caro 2022; T = 8L + 9 trainable parameters, N = 215) is reported per arm: **0.281** at the selected depth 1 — a genuinely shallow, well-bounded circuit.
- **Barren plateaus** (McClean 2018, Cerezo 2021): a random-init landscape whose gradients vanish exponentially, making training a mirage. Guarded *architecturally* (shallow depth + local single-qubit observables) and then **measured** — the variance of the bare ⟨Z₀⟩ gradient across 48 random initialisations: **0.0137** (abs-mean 0.0316), three orders of magnitude above the vanishing floor, `vanishing: false`. Training is real, not a plateau artifact.

**Correctness gates (before any VQC number):** C7 reproduces **0.913038** to 1e-6; the E2 map matches Aer to **8.9e-16**; and the torch ansatz forward matches an independent **Qiskit `Statevector.evolve`** reference to **4.3e-16 ≤ 1e-10** across 24 checks (both depths × {entangled, product} × samples).

**A structural finding that makes the wiring auditable — depth-1 entanglement is invisible to this readout.** A CZ ring applied *after* the last RY layer is diagonal, so it cannot change any |amplitude|² and therefore cannot change a diagonal ⟨Z_q⟩ expectation. At depth 1 the entangling VQC and its no-entangler ablation are provably identical at the readout — and the measured TRAIN-OOF confirms it to the digit (**both 0.718104**). Entanglement only becomes observable at depth ≥ 2. This is not a defect; it is a clean, verifiable property that also says exactly where to look for an entanglement effect.

| Rank | Arm | Representation | Validation PR-AUC | Val ROC-AUC | Train-CV OOF PR-AUC | Trainable params | sqrt(T/N) |
|---|---|---|---|---|---|---|---|
| 1 | **C7** | classical fusion, 1195-d, logistic | **0.913038** | 0.934 | 0.766915 | 1196 | — |
| 2 | trainable_vqc_no_entangler (ablation) | PCA-8 → product-state VQC (depth 2) → 8 ⟨Z⟩ | 0.778527 | 0.823 | 0.723102 | 25 | 0.341 |
| 3 | trainable_vqc | PCA-8 → **entangling VQC (depth 1)** → 8 ⟨Z⟩ | 0.770660 | 0.819 | 0.718104 | 17 | 0.281 |
| 4 | classical_mlp (matched control) | PCA-8 → tanh-MLP (hidden 2) | 0.768450 | 0.822 | 0.675469 | 21 | — |

(C7's "trainable params" is its 1195 logistic weights + intercept, shown only for scale; `sqrt(T/N)` is a small-circuit diagnostic and is not meaningful for it.)

**Increment test (C7 → candidate), patient-blocked column-permutation null (gates the claim):**

| Candidate | C7→cand stack PR-AUC | Increment over C7 | Column-perm null empirical p | Exceeds null p95? | Score corr. w/ C7 | **Survives gate?** |
|---|---|---|---|---|---|---|
| trainable_vqc | 0.863104 | **−0.049934** | 0.9552 (z −2.15) | No | 0.689 | **No** |
| classical_mlp | 0.912826 | −0.000212 | 0.5572 (z +0.16) | No | 0.631 | No |

**Reading — the sharpest negative of the three rounds:**

1. **The trainable VQC does not beat C7** — 0.771 vs 0.913 on validation, 0.718 vs 0.767 on train-CV OOF: a wide gap on both surfaces, despite the circuit now being free to shape its own state.
2. **Entanglement does not help — it is either invisible or harmful.** TRAIN-OOF *selected depth 1*, exactly the configuration where entanglement is architecturally invisible to the readout (entangling and ablation arms tie at 0.718104). Given the chance to use *observable* entanglement at depth 2, the entangling circuit (0.713016) **underperforms** its own product-state ablation (0.723102). The model's own cross-validation preferred no *effective* entanglement. `trainable_vqc_beats_no_entangler: false`.
3. **The one narrow positive is not quantum.** The VQC edges its matched classical MLP on both surfaces (validation 0.770660 vs 0.768450; train-CV OOF 0.718104 vs 0.675469 → `trainable_vqc_beats_mlp_control: true`), but the validation margin is 0.002 inside a ±0.11 bootstrap SE, and — decisively — the **no-entangler ablation outscores both the entangling VQC and the MLP on both surfaces** (0.778527 / 0.723102). So the VQC's edge over the MLP owes nothing to entanglement: a product-state circuit does it better. It is not asserted as an advantage of any kind.
4. **Zero incremental information over C7 — actively negative.** Appending the VQC's leakage-free OOF column to C7 *lowers* the stack to 0.863 (increment **−0.050**, empirical p 0.955, z −2.15 — the real column does *worse* than almost every patient-blocked permutation of itself). The matched MLP's increment is −0.0002 (p 0.557). Neither clears the pre-registered p95 gate; the VQC's sits decisively on the wrong side of zero. (The label-permutation null is reported but does not gate — structurally low-powered for an increment, flooring near p≈0.25, exactly as E0 documented.)
5. This is **not** a trainability failure: the witness shows healthy gradients (var 0.0137, `vanishing: false`) and the loss falls (all-train BCE 0.688 → 0.519). The circuit trained fine; the representation it learned simply carries no information C7 lacks.

**Conclusion — round 3.** `trainable_vqc_beats_c7: false` · `trainable_vqc_beats_no_entangler: false` · `trainable_vqc_barren_plateau: false` · `trainable_vqc_increment_survives_gating_null: false` · `defensible_quantum_advantage_over_c7: false` · `quantum_specific_advantage: false`. A clean mission-§29B null, and the **third independent** data point toward stop-condition (B): a fixed map read linearly (Round 1), the kernel it induces (Round 2), and now a *trainable* shallow circuit (Round 3) each fail to add defensible information beyond C7. **Recorded as:** DEC-039.

## 8. Round 4 — richer feature map + structured observables (DEC-040)

Rounds 1–3 all held three things fixed: the E2 feature map (RY + CZ ring), the 16 local-Z observables, and the variance-optimal PCA-8 selection. Round 4 changes all three at once — the one qualitatively distinct representation strategy the ladder still named. The map is the **Havlicek et al. (2019) ZZ feature map**, the canonical "classically-hard kernel" encoding:

```
x → {PCA-8 | MI-top-8}(MobileNet-576)  →  reps × { H⊗ⁿ ; P(2xᵢ) ; ∏_{i<j} exp(i(π−xᵢ)(π−xⱼ) ZᵢZⱼ) }
     selected on TRAIN OOF                →  36 observables (8 ⟨Zᵢ⟩ + 28 ⟨ZᵢZⱼ⟩)  →  logistic head
     reps ∈ {1, 2}, entanglement = full
```

Two feature selections are searched on TRAIN OOF — variance-optimal **PCA-8** (the Rounds 1–3 input) and label-aware **mutual-information top-8** of the raw MobileNet-576 (a "quantum-suitable" selection: encode the most class-discriminative directions, not the highest-variance) — crossed with `reps ∈ {1, 2}`. The observable set is **structured, not score-chosen**: all 8 singles + all 28 pairs, richer than E2's nearest-neighbour 16 because the ZZ map entangles *every* pair. **Matched control (DEC-033), at equal output width:** an **RFF-36** on the same PCA-8 input (median-heuristic bandwidth) — the honest "would *any* 36-d classical nonlinear map of these features do the same?"

**The hazard here is accidental separability, and it is *measured*.** An **entanglement witness** — the TRAIN mean/max connected correlation `Cᵢⱼ = ⟨ZᵢZⱼ⟩ − ⟨Zᵢ⟩⟨Zⱼ⟩` over all 28 pairs (identically 0 on a product state) — is reported: **mean |C| 0.052, max 0.689** → `rich_quantum_entangling: true`. The reps-2 map genuinely entangles, so any null is "richer entanglement did not help," not "the map degenerated to separable."

**Correctness gates (before any quantum number):** C7 reproduces **0.913038** to 1e-6 (through Rounds 1–3's exact fold-honest identity path — byte-identical train-CV OOF 0.766915); the fast `|ψ|²@diag` readout matches Aer `save_expectation_value` to **7.2e-16**; and the hand-built ZZ circuit (native h/p/cx) matches qiskit's canonical `ZZFeatureMap` to **1.8e-15 ≤ 1e-10** — so "richer quantum map" provably means the actual Havlicek encoding, not a look-alike.

**A structural finding that makes the wiring auditable — a one-block ZZ map read by Z-strings is identically zero.** The reps-1 circuit is `H⊗ⁿ` followed only by Z-diagonal gates (`P`, `ZZ`), so its state is `U|+⟩⊗ⁿ` with `U` diagonal in Z; every Z-string expectation is therefore `⟨+|⊗ⁿ Z_S |+⟩⊗ⁿ = 0` (verified to **1.4e-16**). The reps-1 feature vector is the **zero vector** — which is exactly why both reps-1 configs below score the class base rate to the digit (train 103/215 = 0.479070, val 21/52 = 0.403846). Only *re-uploading* (reps 2) inserts a second `H` that breaks Z-diagonality and yields any nonzero readout — so the ZZ map's entire usable signal lives in its re-uploading depth. This is not a defect; it is a clean, verifiable property.

| Rank | Arm | Representation | Validation PR-AUC | Val ROC-AUC | Train-CV OOF PR-AUC | Selected C |
|---|---|---|---|---|---|---|
| 1 | **C7** | classical fusion, 1195-d, logistic | **0.913038** | 0.934 | 0.766915 | 0.01 |
| 2 | rff36 (matched control) | PCA-8 → RFF-36 | 0.726207 | 0.836 | 0.693370 | 0.01 |
| 3 | rich_quantum | PCA-8 → **ZZ map (reps 2)** → 36 ⟨Z⟩ | 0.390688 | 0.390 | 0.506997 | 0.001 |

**Config grid (all four, selected on TRAIN OOF — the conservative direction):**

| Selector | reps | Train-CV OOF PR-AUC | Validation PR-AUC |
|---|---|---|---|
| PCA-8 | 1 | 0.479070 | 0.403846 |
| **PCA-8** | **2** | **0.506997** | **0.390688** |
| MI-top-8 | 1 | 0.479070 | 0.403846 |
| MI-top-8 | 2 | 0.503239 | 0.385902 |

(The reps-1 rows equal the class base rates exactly — see the zero-readout property above. reps-2/PCA-8 was selected as best on TRAIN OOF.)

**Increment test (C7 → candidate), patient-blocked column-permutation null (gates the claim):**

| Candidate | C7→cand stack PR-AUC | Increment over C7 | Column-perm null empirical p | Exceeds null p95? | Score corr. w/ C7 | **Survives gate?** |
|---|---|---|---|---|---|---|
| rich_quantum | 0.909536 | **−0.003502** | 0.7164 (z −0.01) | No | −0.140 | **No** |
| rff36 | 0.920187 | +0.007149 | 0.2637 (z +0.60) | No | 0.710 | No |

**Reading — the sharpest negative of the four rounds:**

1. **The richer quantum map does not beat C7** — 0.391 vs 0.913 on validation, 0.507 vs 0.767 on train-CV OOF. The best config's validation PR-AUC is essentially the class base rate (0.404).
2. **It is worse than its own matched classical control.** RFF-36 on the identical PCA-8 input beats the quantum map on every surface — validation 0.726 vs 0.391, ROC 0.836 vs 0.390, train-CV OOF 0.693 vs 0.507. The ZZ encoding *actively destroys* linearly-usable signal a same-width classical random-feature map preserves, so the failure is specific to the quantum representation, not to the 8→36 width or the PCA-8 input.
3. **The map genuinely entangles, yet anti-generalizes.** The witness (mean |C| 0.052, max 0.689) proves the reps-2 entanglers produced real correlations — not a product state — but the best config's validation ROC is **0.390 < 0.5**: the tiny TRAIN-CV signal (0.507 > base 0.479) inverts out of sample. The null is *representational*, not a degeneracy artifact.
4. **Label-aware selection does not rescue it.** MI-top-8 (the "quantum-suitable" selection) ties PCA at reps 1 (both zero-readout → base rate) and is slightly *worse* at reps 2 (0.503 vs 0.507 OOF). Choosing the most class-discriminative inputs for the ZZ map to encode does not help.
5. **Zero (negative) incremental information over C7.** Appending the quantum OOF column to C7 *lowers* the stack to 0.9095 (increment **−0.0035**, empirical p 0.716, score-corr −0.14 — anti-correlated with C7's own scores). Even the RFF control's +0.0071 (p 0.264) does not clear the pre-registered p95 gate.

**Conclusion — round 4.** `rich_quantum_beats_c7: false` · `rich_quantum_entangling: true` · `rich_quantum_increment_survives_gating_null: false` · `rff36_increment_survives_gating_null: false` · `defensible_quantum_advantage_over_c7: false` · `quantum_specific_advantage: false`. A clean mission-§29B null, and the **fourth independent** data point toward stop-condition (B) — a richer, provably-canonical, genuinely-entangling quantum map, read by structured observables with label-aware selection, not only fails to beat C7 but is worse than its matched classical control. **Recorded as:** DEC-040.

## 9. The research ladder (§30/§29) — stop-condition (B) reached exhaustively

All four rungs the mission named have now been tested — a fixed map read linearly (Round 1), the fidelity kernel it induces (Round 2), a trainable shallow circuit (Round 3), and a richer entangling map with structured observables and label-aware selection (Round 4). Each was additive, versioned, TRAIN-only, matched-control, and answered the same question — "did quantum add information beyond C7?" — with `_increment_null`:

1. ~~**Quantum kernel / QSVM.**~~ **DONE (Round 2, DEC-038).** The fidelity quantum kernel `K(x,x') = |⟨ψ(x)|ψ(x')⟩|²` over the exact 8-qubit states, fed to a precomputed-kernel SVM, was compared against C7 and a matched RBF-SVM control on identical folds. Result: 0.772 val PR-AUC — below C7 (0.913) *and* below its own RBF control (0.803); increment over C7 −0.0007 at p≈0.59; kernel healthy (not concentrated, sd 0.185). No kernel-level quantum advantage.
2. ~~**Trainable shallow VQC.**~~ **DONE (Round 3, DEC-039).** A shallow real-arithmetic ansatz (RY + CZ ring, depth ∈ {1,2} selected on TRAIN OOF), `sqrt(T/N)`-bounded (0.281) and barren-plateau-witnessed (gradient variance 0.0137, not vanishing), evaluated against C7, a capacity-matched tanh-MLP, and a no-entangler ablation. Result: 0.771 val PR-AUC — far below C7 (0.913); TRAIN-OOF selected depth 1 (where entanglement is invisible to the readout) and at depth 2 the entangler *underperforms* its own ablation; increment over C7 −0.050 at p 0.955. No advantage — entanglement neither visible nor helpful.
3. ~~**Better / structured feature maps and observables.**~~ **DONE (Round 4, DEC-040).** The Havlicek ZZ feature map (reps ∈ {1,2}, entanglement=full), read by 36 structured observables (8 singles + 28 pairs), with variance-optimal PCA-8 and label-aware MI-top-8 selection, evaluated against C7 and a matched RFF-36 control. Result: 0.391 val PR-AUC — far below C7 (0.913) *and* below its own RFF-36 control (0.726); the map genuinely entangles (witness mean |C| 0.052) yet anti-generalizes (val ROC 0.390 < 0.5); increment over C7 −0.0035 at p 0.716. No advantage — the richer map is *worse* than its classical analogue.

**Stop condition (§29): (B) reached exhaustively.** The mission directed continuing until either **(A)** a genuine quantum/hybrid model beats C7 with statistically defensible evidence [preferred] — **never fabricated** — or **(B)** multiple strong quantum approaches are rigorously tested with no improvement. Rounds 1–4 are now **four independent data points** spanning *every* qualitatively distinct quantum representation strategy the mission named: none beats or adds defensible information beyond C7 (0.913038), Round 3 *measured* entanglement to be invisible-or-harmful, and Round 4 was worse than its matched classical control. No untested rung of the same kind remains — **(B) is satisfied exhaustively, not merely strongly.** Any future quantum attempt on this problem would require a *materially new lever* — more data (215 TRAIN / 103 positives is small), a different modality, or hardware the mandate does not currently justify — not a re-run of these four families. Per §29(A) a genuine gate-clearing advantage from such a lever would be frozen and integrated; none is fabricated to keep the search alive, and the aspirational 0.93/0.95/0.98 targets are explicitly *not* forced.

---

*All numbers in this document are read firsthand from the emitted reports — Round 1 from [`reports_e3_hybrid_fusion.json`](../reports_e3_hybrid_fusion.json), Round 2 from [`reports_e3_quantum_kernel.json`](../reports_e3_quantum_kernel.json), Round 3 from [`reports_e3_trainable_vqc.json`](../reports_e3_trainable_vqc.json), Round 4 from [`reports_e3_feature_map.json`](../reports_e3_feature_map.json) (each seed 42, 2000 bootstrap draws, 200 permutations, run 2026-09-14). Nothing here is estimated, rounded up, or projected.*
