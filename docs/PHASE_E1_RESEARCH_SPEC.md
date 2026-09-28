# Phase E1 — Classical Failure Map and Quantum Target Research

**Status:** research specification. **No implementation is authorised by this document.**
**Predecessors:** DEC-031 (E0 pre-registration), DEC-032 (E0.1 verdict), DEC-024 (ROI geometry
leak), DEC-020 (localizer trained on lesion-annotated images only), DEC-030 (single frozen test
read).
**Test partition:** frozen. Nothing in this document reads it. See §10.

---

## 0. What E0 established, and what it did not

E0.1 answered one question and answered it cleanly. The question was *does the 65,536-amplitude
whole-image state contain class information that `⟨Z₀⟩` was merely failing to expose?* The answer
was no: `n_variant_settings_above_null_p95_at_p05 = 0` across 15 variant×weight settings, and the
failure reproduced at **W-identity** — observables read straight off the encoded state with no
circuit and no optimizer in the question. That localises the failure to the representation (F1),
not the measurement (F2) or the optimizer (F3). DEC-032 records the full numbers.

Three things E0 did **not** establish, and which this document must not silently assume:

1. **The amplitude vector is not empty.** An RBF-SVM on the same 65,536-dimensional vectors scored
   PR-AUC 0.549501 against a prevalence floor of 0.403846 on the primary condition. That is weak
   and it did not generalise usefully, but it is not zero. The correct statement is that the
   *information density per unit of quantum resource* was too low for the proposed readout, not
   that the pixels are uninformative.
2. **Nothing about quantum computing in general.** E0 tested one encoding at one width with one
   ansatz family. It is a negative result about a representation.
3. **Nothing about where the classical pipeline is weak.** E0 measured the quantum path against a
   classical reference; it did not decompose the classical reference. That decomposition is E1.1
   and it has **not been run**. Every claim below is labelled as one of:

| Label | Meaning |
|---|---|
| **[STATIC]** | Derived by reading the code. Certain about what the pipeline computes; says nothing about whether the omission costs accuracy. |
| **[MEASURED]** | Read from an existing committed artifact. Cited with the artifact path. |
| **[OPEN]** | Not yet measured. E1 exists to measure it. Must not be reported as a finding. |

This distinction is the whole point of the phase. E0's most expensive mistake was an earlier leaky
linear control that read PR-AUC 0.128 and, once refitted with train-only `C`, read 0.463627 — the
bar had been understated by 0.336. A failure map built on unmeasured intuition would repeat that
error in the opposite direction.

---

## 1. The classical failure map

### 1.1 What the 181-feature representation provably is  **[STATIC]**

From `backend/ml/features_image.py` (`HANDCRAFTED_VERSION = "1"`, `_GRID = 4`,
`_HIST_BINS = 12`) and `backend/ml/fusion.py`:

| Block | Dims | Construction |
|---|---|---|
| Per-cell CIELAB first/second moments | 96 | 4×4 grid × 3 channels × {mean, std} |
| Global per-channel histograms | 36 | 3 channels × 12 bins, ranges `((0,100),(-100,100),(-100,100))` |
| Per-cell gradient **magnitude** mean | 16 | 4×4 grid, single scale |
| Global scalars | 15 | `grad_mean/std/p90`, `edge_density`, `chroma_mean/std`, `redness_mean/std`, `yellowness_mean/std`, `lightness_mean/std/p10/p90`, `specular_fraction` |
| **Image subtotal** | **163** | `features_handcrafted_lab.npz`, `matrix (2436, 163)` |
| Clinical | 18 | `features_clinical.py`: smoking / alcohol / betel-quid frequency+duration z-scores, habit gates, type/form tokens, age, sex |
| **Total** | **181** | confirmed `n_reduced_features = 181` in `baselines.json` and `evaluation.json` |

The descriptor is computed on the ROI crop resized to 224×224 (`pipeline.py:89`,
`preprocessing.py:127`), so it is already a *localized* descriptor. The ROI is chosen by
`backend/ml/roi.py`: lesion polygon → region polygon → centre crop at `CENTER_CROP_FRACTION = 0.80`,
`POLYGON_MARGIN_FRACTION = 0.12`, `MIN_ROI_EDGE_PX = 64`.

### 1.2 What that representation therefore cannot contain  **[STATIC]**

These are not conjectures. They follow from the construction above.

| # | User's investigation level | Present in the 181? | Precise gap |
|---|---|---|---|
| 1 | Raw / local pixel structure | Partially | 4×4 = 16 cells over 224×224 means each cell is a 56×56 block collapsed to 6 numbers. Any structure below 56 px is averaged away. |
| 2 | Texture | Magnitude only | `np.gradient` magnitude means, plus `edge_density`. **No gradient orientation** (no HOG), **no co-occurrence** (no GLCM), **no local binary patterns**, **no oriented filter bank** (no Gabor). Texture that differs only in *direction* or in *joint* pixel statistics is invisible. |
| 3 | Boundary / shape | Absent | No contour extraction, no eccentricity, no compactness, no border-irregularity measure. The only geometry that reaches the model is ROI area — and that is a leak, not a feature (§1.4). |
| 4 | Local spatial relationships | First-order only | Each of the 16 cells contributes independent summaries. No cell-to-cell difference, ratio, gradient-across-grid, or centre-versus-periphery term. |
| 5 | Multiscale structure | Absent | One grid (4×4), one gradient scale, one resize (224×224). No pyramid, no scale-space. |
| 6 | High-order feature interactions | Not in the representation | Available only to the extent the *classifier* builds them: `random_forest` and `gradient_boosting` construct axis-aligned conjunctions; `logistic_regression` cannot. No explicit product/interaction terms. |
| 7 | Small-region / patch-level patterns | Coarse | The 56×56 cell is the finest spatial unit. Lesion-scale detail finer than that has no channel. |
| 8 | Lesion-vs-background relationship | Fixed and implicit | Background enters only through the 12% polygon margin. There is no explicit lesion-interior-versus-surround contrast term, and for the 610 centre-crop / 1510 region-polygon rows the "lesion" is not delineated at all. |

Levels 2, 3, 4, 5 and 8 are **structurally absent**. Levels 1, 6, 7 are **present but coarse**.

**This is the honest limit of what static inspection can say:** an absent feature family is only a
bottleneck if adding it improves validation PR-AUC over a matched control. That is **[OPEN]** and
is exactly what E1.1 measures.

### 1.3 What the classical pipeline currently achieves  **[MEASURED]**

Validation partition, 363 images / 50 patients / 21 positives / prevalence 0.057851, from
`backend/artifacts/models/v1-handcrafted/baselines.json`. Hyperparameters were selected on
validation, so these are optimistic — the artifact says so itself.

| Model | PR-AUC | ROC-AUC | Brier | ECE | Sens | Spec |
|---|---|---|---|---|---|---|
| `random_forest` | **0.636768** | 0.963659 | 0.045590 | 0.100395 | 0.428571 | 0.979532 |
| `gradient_boosting` | 0.598644 | 0.965052 | 0.034755 | 0.026389 | 0.571429 | 0.970760 |
| `logistic_regression` | 0.587325 | 0.965608 | 0.070742 | 0.132113 | 1.000000 | 0.921053 |

On the lesion-polygon subset (52 validation images, prevalence 0.403846), from
`reports_cmp_oracle.json` and `reports_ctl_oracle_primary.json`:

| Model | Input | PR-AUC |
|---|---|---|
| `random_forest` (E0 control, train-only tuning) | 181 features | **0.790313** |
| `logistic_regression` | 181 features | 0.742846 |
| RBF-SVM | 65,536 amplitudes | 0.549501 |
| Phase-D VQC | 65,536 amplitudes | 0.526376 |
| Linear SVM | 65,536 amplitudes | 0.463627 |

**The shape of this result matters for E1.** ROC-AUC ≈ 0.96 with PR-AUC ≈ 0.60 is not a ranking
failure. The model orders images well; what it lacks is *precision in the high-sensitivity region*.
`logistic_regression` reaches sensitivity 1.0 only at specificity 0.921 — 27 false positives for 21
true positives. A screening tool that flags 48 of 363 images to catch 21 is defensible but not
good. **The operating characteristic, not the ranking, is where the headroom is.** Any E1 candidate
that improves ROC-AUC by 0.01 while leaving the high-sensitivity precision unchanged has not
addressed the actual weakness.

### 1.4 Two confounds that any E1 measurement must carry  **[MEASURED]**

**Confound 1 — ROI geometry leak (DEC-024).** From
`backend/evaluation/state_diagnostics.py:roi_geometry_leak` and the module docstring: every
positive in the oracle cache has a lesion-polygon ROI, and every region-polygon and centre-crop row
is negative. ROI area alone scores the label at ROC-AUC **0.78 train / 0.72 validation / 0.85 test**.
The oracle cache source counts confirm the mechanism: `lesion_polygon` 316, `region_polygon` 1510,
`center_crop` 610.

Consequence: on condition A, **any** model reading these crops scores above chance without learning
anything about tissue. Every E1 number computed on oracle ROIs must be reported against 0.72, not
against 0.5. This is why the E0 primary condition was `A_lesion_polygon` — restricting to one ROI
source removes the area channel by construction.

**Confound 2 — localizer quality gap.** From
`backend/artifacts/models/localizer/carescan-localizer-1/localization_rates.json`:

| Partition | n | localization rate | fallback | rejected | IoU mean | IoU median | ≥0.5 | ≥0.75 |
|---|---|---|---|---|---|---|---|---|
| train | 1692 | 0.9574 | 0.0426 | 0.0 | 0.6886 | 0.7375 | 0.8385 | 0.4531 |
| validation | 363 | 0.9725 | 0.0275 | 0.0 | **0.5285** | 0.5656 | **0.58** | **0.16** |

Validation IoU mean 0.5285 against train 0.6886 is a real generalisation gap in the *localizer*, and
only 16% of validation boxes reach IoU 0.75. Condition B ROIs are therefore materially looser than
condition A ROIs. Any E1 result must be reported per condition and never pooled across A/B/C.

### 1.5 The statistical wall, stated before any experiment is designed  **[MEASURED]**

| Partition | images | patients | positives | prevalence |
|---|---|---|---|---|
| train | 1692 | 230 | 104 | 0.0615 |
| validation | 363 | 50 | 21 | 0.0579 |
| test | 381 | 48 | 18 | 0.0472 |

Lesion-polygon subset: 215 train / 52 validation rows, 103 train positives (prevalence 0.47907).

> **[MEASURED] correction — 2026-09-10.** An earlier draft of this line stated *87* train
> positives (the same figure appears in the frozen E0 records DEC-031/DEC-032). Under the frozen
> split manifest `fe45df6d…` the measured count is **103** (103 / 215 = 0.47907) — already
> recorded in DEC-024 for this exact `oracle/A_lesion_polygon` subset and reproduced by the E1.1
> driver (`reports_e1_failure_map.json` → `conditions.A_lesion_polygon.rows.n_train_positive =
> 103`). 87 was a spec error; 103 is authoritative and is used wherever this document cites the
> primary-condition train-positive count. The correction only *raises* N, which *loosens* the
> √(T/N) bound below, so it cannot manufacture a bottleneck. The immutable DEC-031/DEC-032 entries
> are left as written and are superseded on this point by DEC-034.

**21 validation positives.** A PR-AUC difference of ±0.05 on 21 positives is inside sampling noise.
This governs the entire design:

- Any candidate must clear the control by a margin large relative to a resampling interval, not by
  a decimal place (§9.4 sets the threshold and justifies it).
- The number of candidate representations must stay small. 136 observables against 103 train
  positives is precisely what produced E0's multiplicity artefact, where 17–35 of 136 observables
  cleared validation AUC 0.60 but were *different* observables from the train-selected ones.
- Caro et al. (2022, *Nat Commun* 13:4919) bound generalization error at O(√(T/N)) in trainable
  gates T and samples N. At N = 104 train positives, a model with T ≳ 100 trainable gates has a
  vacuous bound. **This alone rules out deep parameterised quantum models on this dataset**, and it
  is a stronger constraint than any hardware limit.

---

## 2. Classical baselines for the failure map (E1.1)

Seven candidates: one control (C1), five single-gap probes (C2–C6), one fusion test (C7). Kept
deliberately small because of §1.5.

**Common protocol for all seven, non-negotiable:**

- **Partitions:** train fits, validation scores. Test untouched (§10).
- **Grouping:** patient-level. Reuse `backend/dataset/split.py` (`split_manifest.json`,
  `build_patient_level_split`) and the existing `grouped_folds` helper in
  `backend/evaluation/e0_controls.py`. Do not write a new splitter.
- **Tuning:** every hyperparameter chosen by patient-grouped CV **inside train only**. No
  validation row may influence any fitted quantity, including SVM `C`, RBF `gamma`, tree depth, or
  any calibration parameter. E0's control code already enforces this; reuse it.
- **RBF width:** median heuristic on strict-upper-triangle train squared distances only
  (`_median_heuristic` in `e0_controls.py`).
- **Metrics:** `backend/evaluation/metrics.py:evaluate_predictions`. Keys are `pr_auc` (primary),
  `roc_auc`, `brier`, `ece`, `sensitivity`, `specificity`. Note the key names: `brier` and `ece`,
  not `brier_score`.
- **Conditions:** report separately for `A_lesion_polygon` (primary, prevalence 0.403846, geometry
  leak removed by construction) and `A_all` (secondary, geometry leak present at ROC-AUC 0.72 — say
  so in the payload). Do not pool.
- **Reference bars:** the §1.3 numbers. `random_forest` on 181 features at PR-AUC 0.790313 on the
  primary condition is the bar to beat, not 0.5 and not prevalence.

| # | Candidate | Input | Dim | Preprocessing | Classifier | Train-only tuning | Gap tested (§1.2) | Expected failure mode being probed |
|---|---|---|---|---|---|---|---|---|
| **C1** | **Baseline replication** | existing 181 | 181 | persisted pipeline, refit on train | RF, GB, linear SVM | tree depth / `n_estimators` / `C` | — | Control. Establishes the bar under *train-only* tuning, so §1.3's optimistic validation-tuned numbers are not the comparison. |
| **C2** | **Downsampled ROI pixels** | `v1_pixels_*.npz` `raw (n, 65536)` uint8 | 1024 (32×32) then 256 (16×16) | mean-pool from the existing 256×256 cache; L2-normalise | RBF-SVM, linear SVM | `C`, `gamma` (median heuristic) | 1, 7 | Does *any* pixel-level representation beat the descriptor? E0 already measured this at full 65,536 (0.549501). If 1024 dims does markedly better, the E0 failure was partly dimensional, not purely representational — a result that would change §4's premise. |
| **C3** | **HOG** | ROI crop 224×224 grayscale | ~1764 (8×8 cell, 2×2 block, 9 bins) | reuse `crop_to_roi` + `normalise_crop` | linear SVM, RBF-SVM, RF | `C`, `gamma`, orientation-bin count ∈ {9} fixed | **2, 4** | Gradient **orientation** is the single largest structural absence. If HOG ≫ C1 on the primary condition, orientation is the bottleneck. If HOG ≈ C1, the descriptor's magnitude-only texture was sufficient and level 2 is *not* a bottleneck. |
| **C4** | **LBP + GLCM** | ROI crop 224×224, per CIELAB channel | ~150 (uniform LBP P=8 R=1 and R=3 histograms + GLCM contrast/homogeneity/energy/correlation at 4 angles) | as C3 | RF, RBF-SVM | tree depth, `C`, `gamma` | **2, 5** | Joint / co-occurrence texture and a second radius. Two radii give a minimal multiscale probe at near-zero cost. |
| **C5** | **Multiscale grid + cross-cell interaction** | same CIELAB image as the 181 path | 163 + 8×8 grid moments + centre-vs-surround contrast ≈ 550 | identical to the existing descriptor path, additional grids only | RF, GB | tree depth, `n_estimators` | **4, 5, 8** | Isolates spatial resolution and lesion-vs-surround contrast *within the existing feature philosophy*. If C5 ≫ C1, the bottleneck is grid coarseness — a purely classical fix, and quantum would be unjustified for that gap. **This is the most important candidate to run, because it is the cheapest way to kill the quantum motivation.** |
| **C6** | **MobileNetV3-Small embedding** | ROI crop 224×224 | 576 | existing `mobilenet_v3_small` extractor, `IMAGENET1K_V1`, frozen | linear SVM, RBF-SVM, RF | `C`, `gamma` | 1, 2, 4, 5, 6, 7 | The strong-learned-representation ceiling. A frozen ImageNet trunk captures orientation, multiscale and local structure jointly. **If C6 ≫ everything, the bottleneck is "handcrafted features" and the fix is a CNN, not a quantum circuit.** |
| **C7** | **Fusion** | best two of C3–C6, standardised and concatenated | ≤ 2500 | `fusion.py` `StandardScaler`, train-fitted | RF, RBF-SVM | as above | 6 | Are the families complementary or redundant? Uses the existing patient-blocked column-permutation increment test from `e0_controls.py:complementarity` (§9.2). |

Resource note: C2–C7 all read caches that already exist (`v1_pixels_oracle.npz`,
`v1_pixels_predicted.npz`, `features_handcrafted_lab.npz`) or reuse extractors already in
`backend/ml/`. No new dataset pass and no new preprocessing version is required. Estimated total
E1.1 runtime is minutes to low tens of minutes on CPU.

**Multiplicity:** 7 candidates × ≤3 classifiers × 2 conditions. Selection happens on validation
PR-AUC, which is a selection surface — so §9 requires the winner's margin over C1 to be reported
against a Benjamini–Hochberg correction across the whole candidate grid, and the *count* of
candidates must be frozen in the payload before any of them runs.

---

## 3. How the bottleneck will be classified

Each family is assigned exactly one verdict, from the pre-registered rules below. The rules are
written now so they cannot be adjusted after seeing numbers.

| Verdict | Rule (primary condition `A_lesion_polygon`, validation PR-AUC, train-only tuning) |
|---|---|
| **A. Already solved** | Family ≤ C1 + resampling interval, and C1 itself is well above the geometry-leak bar. The gap is real in the representation but costs nothing measurable. |
| **B. Signal present, model fails to exploit** | Family beats C1 with a *nonlinear* classifier by more than the interval, but not with a linear one, **and** the increment survives the patient-blocked column-permutation null. This is the only verdict that motivates a *transformation* rather than more features. |
| **C. Weak / non-generalizable** | Family beats C1 on train CV but not on validation, or the validation gain does not survive the null. This is E0's signature and must be called out as such, not reported as a positive. |
| **D. Redundant** | Family alone ≈ C1, and fusion (C7) adds no increment over C1 by the §9.2 column-permutation test. |

**Guard rails, stated before measurement:**

- One classifier performing badly is not a classical failure. A family is only declared a failure
  after both a linear and a nonlinear classifier have been tuned on train and both fall short.
- A family that C5 or C6 already captures is not a bottleneck. If the coarse-grid or CNN control
  closes the gap, the correct conclusion is *use the classical fix*, and the quantum hypothesis
  loses its premise. This document treats that as a **likely and acceptable** outcome.
- Verdict **B** is the only one that survives into §4. **A**, **C** and **D** each independently
  force "NO QUANTUM IMPLEMENTATION JUSTIFIED YET".

---

## 4. Quantum candidates, conditional on a verdict-B family

Assessed against §1.5 (N = 103–104 train positives), §5 (literature) and §11 (resources). Every
candidate is evaluated on whether it is *disqualified*, and most are.

### 4.1 Screening constraints applied to all candidates

1. **Trainable-gate ceiling.** Caro et al.: error ~ √(T/N). At N = 103 primary-condition train
   positives, T must be O(10) for the bound to mean anything. Anything with dozens of trainable
   parameters is out on generalization grounds alone, before hardware.
2. **State-preparation ceiling.** §11. Arbitrary amplitude encoding is O(2ⁿ) CX. Only encodings
   whose CX cost is O(n) or O(n²) qualify.
3. **Local observables only.** Cerezo et al. (2021, *Nat Commun* 12:1791): global observables give
   barren plateaus even at shallow depth; local observables are at worst polynomial for depth
   O(log n). E0's variant-D global Walsh observables are consistent with this.
4. **Matched control must exist and must be strong.** Not "beats chance".

### 4.2 Candidate assessment

| # | Candidate | Targets | Input to circuit | Qubits | Encoding | Depth | Measurements | Training | Hardware CX | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **Q1** | **Projected quantum kernel** on a compact patch descriptor | verdict-B nonlinear-interaction gap | 6–8 scalars per row from the winning family, standardised to [0, π] on train stats | 6–8 | angle (Rᵢ per qubit) + ≤2 re-uploading blocks | ≤ 12 two-qubit layers | 1- and 2-body reduced density matrices (local, per Cerezo) | **none** — kernel is fixed, only the downstream SVM `C` is fitted | O(n) per layer, ~20–100 CX total | **Viable.** T = 0 trainable gates, so §4.1.1 is satisfied exactly. Huang et al. (2021) give a *pre-computable* go/no-go test (§4.3). |
| **Q2** | Local / patch-level quantum feature extractor (quanvolution) | levels 1, 7 | k×k patches, 4–9 pixels | 4–9 | angle | shallow, random | local Z per qubit | random circuit, untrained | small | **Disqualified as primary.** Henderson et al. (2019) is arXiv-only, MNIST, and its own control (CNN + extra non-linearity) is unresolved. Bowles et al. (2024) find removing entanglement often matches or beats. A random shallow quantum filter is a random nonlinear filter; the honest matched control is a random *classical* filter bank of equal count, which would very likely tie. Retained only as a **secondary** arm if Q1 clears its gate. |
| **Q3** | Data re-uploading classifier | level 6 | 6–8 scalars | 1–8 | repeated angle | L re-uploads | local Z | gradient / SPSA | small | **Disqualified.** Schreiber et al. (2023, *PRL* 131:100803) show large classes of re-uploading models admit efficient **classical surrogates**, and their numerics found "no advantage in performance or trainability". The surrogate *is* the matched control, and it is trainable directly. Running Q3 without the surrogate control would be indefensible; running it with the control makes it a strictly worse Q1. |
| **Q4** | Quantum reservoir / extreme learning | level 6 | 6–8 scalars | 6–8 | angle | fixed random | local moments | linear readout only | small | **Weak.** Mujal et al. (2021, *Adv. Quantum Technol.*) is a review, not evidence. A fixed random nonlinear map with a linear readout is a random-features model; the matched control (classical random Fourier features at equal output dimension) is cheap, well understood, and would need to be beaten. Functionally a degenerate Q1 with a worse-characterised kernel. Not selected. |
| **Q5** | Quantum relational feature processing (explicit cross-region entangling structure) | levels 4, 8 | per-cell summaries from ≥2 spatial cells | 6–8 | angle, one qubit per cell | ≤ 8 layers | 2-body correlators between cell-qubits | none (kernel) or O(10) params | small | **Merged into Q1.** This is a *choice of which scalars to feed Q1 and which correlators to project onto*, not a separate architecture. Cell-pair correlators are the natural Q1 projection when the verdict-B family is level 4 or 8. |
| **Q6** | Localization-confidence-adaptive quantum processing | resource allocation | localizer confidence gates region selection / evaluation count | varies | — | — | — | — | **Deferred.** §1.4 shows validation IoU 0.5285 with only 16% of boxes at IoU ≥ 0.75, so confidence is informative about crop quality. But an adaptive policy is a *second* hypothesis stacked on an unvalidated first one, and it multiplies the selection surface. Revisit only after Q1 has a verdict. |
| **Q7** | Deeper / wider VQC on amplitudes | — | 65,536 amplitudes | 16 | amplitude | — | — | — | **Forbidden.** DEC-032 and the user's E1 constraint. This is the negative control now. |

### 4.3 The Q1 pre-screen — a cheap classical test that runs *before* any circuit

Huang et al. (2021, *Nat Commun* 12:2631) provide the geometric difference

  g_CQ = √( ‖ √K_Q · K_C⁻¹ · √K_Q ‖_∞ )

between a quantum kernel K_Q and the best classical kernel K_C on the *same* inputs. When g_CQ is
small, **no** quantum advantage is possible on that dataset regardless of training, and the test
requires only the two Gram matrices — no optimizer, no shots, no hardware.

On 215 train rows a Gram matrix is 215×215. Computing g_CQ for the shortlisted embeddings costs
seconds. **This is the gate:** if g_CQ is small for every shortlisted embedding, Q1 is killed
before a single training run, and the phase concludes negatively at a fraction of E0's cost. E0
spent 200 permutation draws × 15 settings to learn that a representation was inadequate. E1 should
learn the equivalent from a Gram-matrix norm.

Q1 is also the only candidate where the matched classical control is *exact* rather than
approximate: the same downstream SVM, the same rows, the same train-only `C` protocol, the same
median-heuristic width discipline — only the kernel matrix differs. That is what §8 demands.

---

## 5. Literature reviewed

Classified by evidence strength. **Venue is recorded because arXiv-only preprints are not treated
as established, and a reported advantage is not treated as a demonstrated one.**

### 5.1 Constraints on quantum advantage — peer-reviewed, and load-bearing here

| Work | Venue | What it constrains | Effect on this project |
|---|---|---|---|
| Huang et al., *Power of data in quantum machine learning* | **Nature Communications 12:2631 (2021)** | Classical models learning *from data* compete with quantum models even on classically hard processes. Supplies the geometric-difference test and prediction-error bounds; up to 30 qubits numerically. | Provides the §4.3 pre-screen. Also the direct rebuttal to "the process is complex, so quantum should help". |
| Kübler, Buchholz, Schölkopf, *The inductive bias of quantum kernels* | **NeurIPS 34 (2021)** | Advantage requires the quantum RKHS to be **low-dimensional and aligned** with the target. A large feature space makes generalization *harder*, and useful kernels tend to need exponentially many measurements. | Kills any "expressive embedding" strategy. Forces Q1 to few qubits and a deliberately *restricted* embedding chosen to match a measured bottleneck. |
| Caro et al., *Generalization in QML from few training data* | **Nature Communications 13:4919 (2022)** | Error ~ O(√(T/N)) in trainable gates T, tightening to √(K/N) when only K gates move. | With N = 103–104 positives this is the binding constraint. Directly motivates Q1's **T = 0**. |
| Cerezo et al., *Cost function dependent barren plateaus in shallow parametrized circuits* | **Nature Communications 12:1791 (2021)** | Global observables → barren plateaus even when shallow; local observables → at worst polynomial for depth O(log n). Numerics to 100 qubits. | Local-observable-only rule (§4.1.3). Consistent with E0's global Walsh variants being the worst performers. |
| Schreiber, Eisert, Meyer, *Classical surrogates for quantum learning models* | **Phys. Rev. Lett. 131:100803 (2023)** | Large classes of re-uploading models admit efficiently extractable classical surrogates; numerics found **no advantage in performance or trainability**. | Disqualifies Q3 as a primary direction. |
| Cerezo et al., *Does provable absence of barren plateaus imply classical simulability?* | **Nature Communications 16:7907 (2025)** | The structure imposed to avoid barren plateaus tends to be the same structure that permits classical simulation. | Warns that a *trainable* variational model is suspect by construction. Reinforces the fixed-kernel choice over a variational one. |
| Thanasilp, Wang, Cerezo, Holmes, *Exponential concentration in quantum kernel methods* | arXiv:2208.11060 (2022, rev. 2024) — **preprint** | Kernel values concentrate exponentially in qubit count via four mechanisms: embedding expressivity, global measurements, entanglement, noise. | Treated as a **design constraint, not a fact**: few qubits, local measurements, restrained entanglement. Its prescriptions coincide with Kübler's and Cerezo's, which is why they are adopted. |
| Shaydulin & Wild, *Importance of kernel bandwidth in QML* | **Phys. Rev. A 106:042407 (2022)** | A bandwidth hyperparameter governs quantum-kernel expressivity; tuning it counteracts exponential decay of kernel values with qubit count and reproduces then reverses prior negative scaling results. | Bandwidth must be a **tuned, train-only** hyperparameter of Q1, exactly like RBF `gamma`. Omitting it would produce an unfairly weak quantum arm — the mirror image of E0's leaky-control error. |
| Schuld, Sweke, Meyer, *Effect of data encoding on expressive power* | **Phys. Rev. A 103:032430 (2021)** | A quantum model is a partial Fourier series in the data; the **encoding**, not the variational part, fixes the accessible frequency spectrum. Re-uploading enriches it. | The encoding is the design decision. Justifies spending E1's effort on *what enters the circuit* rather than on ansatz depth — the same conclusion E0 reached empirically. |

### 5.2 Benchmarking evidence

| Work | Venue | Finding |
|---|---|---|
| Bowles, Ahmed, Schuld, *Better than classical? The subtle art of benchmarking QML* | arXiv:2403.07059 (2024) — **preprint** | 12 QML models × 6 tasks × 160 datasets: out-of-the-box classical models **outperform** the quantum classifiers, and removing entanglement is "often as good or better". |
| Peters et al., *Machine learning of high-dimensional data on a noisy quantum processor* | arXiv:2101.09581 (2021), FERMILAB-PUB-20-624-QIS — **preprint, hardware** | 17 qubits on Sycamore, 67 raw dimensions, no dimensionality reduction. Result is **parity**: "competitive with noiseless simulation and comparable classical techniques". Flags shot statistics and mean kernel element size as decisive. |
| Havlíček et al., *Supervised learning with quantum-enhanced feature spaces* | **Nature 567:209–212 (2019)** | Hardware implementation of a variational quantum classifier and a quantum kernel estimator. Foundational; the abstract does not report a matched classical comparison. |
| Glick et al., *Covariant quantum kernels for data with group structure* | **Nature Physics 20:479–483 (2024)** | 27-qubit hardware, kernel built from a group representation with alignment-tuned fiducial state. Advantage is tied **by construction** to group-structured data — a structure this dataset does not have. |
| Pesah et al., *Absence of barren plateaus in QCNNs* | **Phys. Rev. X 11:041011 (2021)** | QCNN gradient variance decays at worst polynomially. Trainability only — no advantage claim. Read together with the 2025 *Nat Commun* result, polynomial trainability is not evidence of usefulness. |
| Mari et al., *Transfer learning in hybrid classical-quantum networks* | **Quantum 4:340 (2020)** | Classical trunk → small variational head, run on IBM and Rigetti hardware. Proof of concept; no equal-capacity classical head control, and no medical-imaging task. |
| Henderson et al., *Quanvolutional neural networks* | arXiv:1904.04767 (2019) — **preprint** | Random quantum circuits as convolutional filters on MNIST. Reports higher accuracy than plain CNNs; the crucial control (CNN with added non-linearity) is not resolved in the abstract. Not an advantage demonstration. |
| Pérez-Salinas et al., *Data re-uploading for a universal quantum classifier* | **Quantum 4:226 (2020)** | Single-qubit universality via repeated encoding. Theory + benchmarks; no named classical baseline. Superseded for our purposes by the *PRL* surrogate result. |
| Mujal et al., *Opportunities in QRC and QELM* | **Adv. Quantum Technol. 2100027 (2021)** | Review. Fixed reservoir, trained readout only. No matched benchmarks reported at abstract level. |
| Mohsen & Tiwari, *Image compression and classification using qubits* | arXiv:2110.05476 (2021) — **preprint** | Argues prior quantum image work is confined to ≤4×4 inputs because standard encodings "necessitate more qubits than are physically realizable"; reaches 16×16 MNIST with accuracy "comparable to classical networks with the same number of learnable parameters". |
| Shende, Bullock, Markov, *Synthesis of quantum logic circuits* | **IEEE TCAD 25(6):1000–1010 (2006)** | Generic n-qubit unitary synthesis at ≈(23/48)·4ⁿ CNOTs, within a factor 2 of optimal. State preparation is O(2ⁿ). | 

### 5.3 What is *not* in this literature

**No peer-reviewed demonstration of empirical quantum advantage on a real medical image
classification task.** Every hardware result located here is parity-at-best (Peters et al.) or
proof-of-concept (Mari et al., Havlíček et al.). The one clear advantage result with a rigorous
separation (Huang et al.) is on **engineered** datasets built to maximise it, and the same paper
supplies the tool showing why natural datasets usually do not qualify. Glick et al.'s advantage is
structural and tied to group data.

Any claim in this project must therefore be a claim about *this dataset under this control*, never
a general one.

### 5.4 How E0 changes the choice of representation

E0 measured F1 (representation) as the failure. Independently, §5.1 says the encoding fixes the
accessible function class (Schuld), that expressive embeddings are counterproductive (Kübler,
Thanasilp), and that trainable-gate count must be tiny at N ≈ 100 (Caro). These converge on the
same prescription, and it is the opposite of E0's design in every axis:

| Axis | E0 (measured negative) | E1 prescription |
|---|---|---|
| Input | whole-image 65,536 raw amplitudes | 6–8 scalars from a *measured* verdict-B family |
| Qubits | 16 | 6–8 |
| Encoding | arbitrary amplitude, ~62,940 CX | angle, O(n) CX |
| Trainable gates | 32 params, SPSA | **0** (fixed kernel) |
| Observables | up to 136, incl. global Walsh | local 1- and 2-body RDMs |
| Selection surface | 15 settings × 200-draw nulls | pre-screened by a Gram-matrix norm (§4.3) |
| Motivation | representation assumed informative | representation *measured* first (E1.1) |

The single most important change is the last one. E0 assumed a representation and tested a readout.
E1 measures the classical representation first and only then asks whether a quantum transformation
of a *specific measured gap* helps.

---

## 6. Primary direction

### Verdict: **NO QUANTUM IMPLEMENTATION JUSTIFIED YET.**

Section 6 requires the primary direction to target "a measured or strongly justified classical
bottleneck". §1.2's gaps are **[STATIC]** — certain about what the code omits, silent about whether
the omission costs anything. **E1.1 has not been run.** There is at present no measured bottleneck,
therefore criterion 1 is unmet, therefore no quantum implementation is authorised.

This is not a stall. It is the same discipline that made E0 informative: DEC-031 pre-registered the
gate before measuring, and the gate held when the answer was unwelcome.

### Pre-registered conditional selection

So that the decision cannot be reverse-engineered from E1.1's results, the follow-on is fixed now.
**If and only if** E1.1 returns a verdict-**B** family under the §3 rules, and **if and only if**
the §4.3 geometric-difference pre-screen is favourable, the primary direction is:

> **Q1 — a projected quantum kernel on the 6–8 scalar summaries of the verdict-B family, 6–8
> qubits, angle encoding, ≤2 re-uploading blocks, local 1- and 2-body RDM projections, zero
> trainable gates, tuned bandwidth, feeding the same SVM as its matched classical control.**

Against the six criteria:

| Criterion | Status |
|---|---|
| 1. Targets a measured bottleneck | **Conditional.** Requires E1.1 verdict B. Unmet today. |
| 2. Quantum processing on real image-derived visual information | Met by construction — the inputs are the verdict-B visual descriptors. |
| 3. Matched classical control exists | Met, and *exactly* matched: same rows, same SVM, same tuning protocol, only the Gram matrix differs (§8). |
| 4. Computationally feasible | Met. 215×215 Gram matrices, 6–8 qubits, statevector. Minutes. |
| 5. Clear falsification test | Met (§9.4). |
| 6. Meaningfully different from the failed approach | Met on every axis in §5.4's table. |

**Any of these routes ends the phase negatively, and each is a legitimate result:** E1.1 returns A,
C or D for every family; or C5 (coarse-grid/contrast) or C6 (CNN embedding) closes the gap, making
the fix classical; or the §4.3 pre-screen is unfavourable.

---

## 7. Proposed architecture (conditional on §6)

```
image
  → MobileNetV3-Small localizer            [FROZEN, Phase C — carescan-localizer-1]
  → ROI + status + confidence              [roi.py; A/B/C conditions kept separate]
  → classical descriptor                   [existing 163-dim path, unchanged]
  → verdict-B family extractor             [NEW, classical: the one family E1.1 identified]
  → 6-8 scalar compact summary             [NEW, classical: train-fitted standardisation to [0, pi]]
       |
       +-- classical control kernel  ------+   <-- identical rows, identical downstream
       +-- quantum projected kernel   -----+       (§8: only this line differs)
                                           |
  → SVM (train-only C)                     [same estimator both arms]
  → fuse with 18 clinical features          [existing fusion.py, train-fitted scalers]
  → Platt calibration                      [existing calibration path, train-fitted]
  → risk output
```

Three properties of this shape are deliberate:

- **The quantum component transforms a visual representation.** It sits between the descriptor and
  the classifier, not after it. It is not a decorative final layer.
- **The classical arm is the same pipeline with one line swapped.** That is what makes §8's control
  exact rather than approximate.
- **Nothing upstream is modified.** The localizer stays frozen (DEC-020), the ROI rules stay frozen,
  the 163-dim descriptor path stays frozen, the fusion scalers and calibration stay as they are. No
  new preprocessing version. No new cache pass.

**Localization-confidence adaptivity (Q6) is excluded from V1.** §1.4 shows confidence carries real
information about crop quality, so the idea is not baseless — but adding an adaptive policy on top
of an unvalidated quantum hypothesis compounds two untested things and widens the selection surface
that E0 already showed to be dangerous at this sample size. Revisit only after Q1 has a verdict.

---

## 8. Matched control design

The experiment must answer *"does the quantum transformation add measurable value over the
strongest reasonable classical transformation of the same information?"* — not *"can it beat
chance?"*

Held identical between arms:

| Held identical | Detail |
|---|---|
| Rows | The same row indices in the same order, per condition. No arm may drop a row the other keeps; any drop is logged and applied to both. |
| Input | The same 6–8 standardised scalars, standardised with **train-fitted** mean/scale. |
| Split | `split_manifest.json`, patient-level, unchanged. Manifest SHA-256 recorded in the payload. |
| Dimensionality | Matched where possible: the classical control kernel is built from the same scalar count. Where the quantum projection yields more features than the classical map, the *classical* arm is given the equal-or-greater budget, never less. |
| Downstream classifier | The same precomputed-kernel SVM implementation. |
| Hyperparameter protocol | `C` and bandwidth by patient-grouped CV **inside train only**, identical fold assignment for both arms, identical grid size. |
| Calibration | Same method, fitted on train only, both arms. |
| Metrics | Same `evaluate_predictions` call, same threshold policy. |

**Only the kernel matrix differs.**

The classical control must be the *strongest* reasonable option, not a token one. It is the best of:
RBF (median-heuristic width, tuned `C`), polynomial (degree tuned on train), and — because Kübler's
result says the comparison that matters is against an aligned classical kernel — a bandwidth-tuned
RBF grid matching the quantum arm's bandwidth grid point for point. **E0's leaky-linear-control
error establishes the direction of the risk:** understating the classical bar by 0.336 would have
manufactured a quantum "win". The control gets every advantage the quantum arm gets, and where
there is doubt, it gets more.

---

## 9. Null tests, multiplicity and falsification — fixed before implementation

### 9.1 Patient-blocked label permutation

Reuse `_patient_blocked_column_shuffle` and the permutation machinery in
`backend/evaluation/e0_controls.py`. Permute labels in whole patient blocks, refit everything that
is fitted, rescore. Draw count frozen in the payload before the run.

### 9.2 The increment null — column permutation, not label permutation

E0 established this the hard way and it must not be re-litigated. A label-permutation null on a
*stack increment* has a structural power ceiling: refitting on permuted labels randomises the two
coefficient signs while still scoring against real validation labels, so roughly a quarter of draws
land on (classical+, quantum+) and reproduce the observed increment by luck. The empirical p floors
near 0.25. **Measured in E0: a genuinely complementary +0.163 increment scored p = 0.11.**

The gating null is therefore the **patient-blocked column permutation**: hold labels and the
classical arm fixed, shuffle only the quantum column in whole patient blocks. The classical-only
model is invariant to that shuffle, so its score is constant across draws and each draw measures
purely what a same-shaped unrelated column would have added. Blocks, not i.i.d. rows, because an
i.i.d. shuffle would additionally destroy within-patient autocorrelation and make the shuffled
column noisier than any real feature — an artificially easy null.

Both nulls are reported. `column_permutation_null` carries `gates_the_claim: true`;
`permutation_null` carries `gates_the_claim: false` and a `power_note`. This mirrors the existing
E0 payload contract exactly.

### 9.3 Negative controls and leakage checks

| Check | Mechanism |
|---|---|
| **Negative control representation** | The E0 amplitude path is now the designated negative control (DEC-032). Any E1 candidate that fails to beat it has failed outright. |
| **Geometry-leak control** | Report every number against the DEC-024 area-only bar (0.72 validation), never against 0.5. Primary condition is `A_lesion_polygon` specifically to remove the area channel. |
| **Shuffled-input control** | Rerun the winning arm on a patient-blocked shuffle of its own input scalars; must collapse to the leak bar. |
| **Patient isolation** | Reuse `test_dataset_split.py:test_real_dataset_has_no_patient_leakage` and `test_pixel_vqc.py:test_real_cache_has_no_patient_overlap`. Assert zero train∩validation patient intersection in the payload itself, not only in tests. |
| **Fit-provenance assertion** | Every fitted object records the partition it was fitted on. Tests in the style of `TestNoTuningAgainstTheEvaluationPartition` must assert no validation row reached any fitted quantity. |

### 9.4 KEEP / KILL thresholds, and how the threshold is chosen

**The threshold is derived from the data, not asserted.** With 21 validation positives (and 21 of
52 rows on the primary condition), the sampling variability of validation PR-AUC is the governing
quantity. Procedure, fixed now:

1. Before any quantum run, compute a **patient-level bootstrap** distribution of the matched
   classical control's validation PR-AUC (resample patients with replacement, ≥2000 draws). Record
   its 2.5th and 97.5th percentiles and its standard error, `se_control`.
2. The **KEEP margin** is `δ = 2 × se_control`, rounded up, **recorded in the payload before the
   quantum arm runs.** This ties the bar to measured variability of this dataset at this
   prevalence, rather than to a number chosen for convenience. It also makes the bar honest about
   §1.5: if `se_control` turns out large, δ is large, and a small quantum gain correctly fails.
3. Multiplicity: if more than one embedding, bandwidth or projection is evaluated, apply
   Benjamini–Hochberg across the **full pre-registered count**, and freeze that count in the
   payload before the first run. E0's 17-of-136 artefact is the reason.

**KEEP** requires *all* of:

- Validation PR-AUC (quantum) ≥ Validation PR-AUC (matched classical control) + δ, on the primary
  condition `A_lesion_polygon`;
- the increment survives the §9.2 patient-blocked **column-permutation** null at BH-adjusted
  p < 0.05;
- the label-permutation null (§9.1) does not contradict it;
- the winning arm collapses to the geometry-leak bar under the §9.3 shuffled-input control;
- the resource budget (§11) is not exceeded;
- no fitted quantity touched a validation row.

**KILL** on any of:

- increment < δ;
- increment does not survive the column-permutation null;
- the classical control C5 or C6 already closes the same gap — the fix is classical;
- the §4.3 geometric-difference pre-screen is unfavourable (kills before training);
- the gain appears only on a non-primary condition, or only after pooling A/B/C;
- resource budget exceeded;
- any leakage check fails.

**A KILL is a publishable result and closes the phase.** DEC-032 is precedent.

---

## 10. Test-partition firewall

The test partition (381 images / 48 patients / 18 positives) is frozen and is **not read anywhere
in E1**. No architecture selection, no hyperparameter tuning, no threshold selection, no feature
selection, no repeated experimentation.

- Every E1 payload carries `test_partition_used: false`, asserted by a test, matching the existing
  contract in `reports_e0_oracle_primary.json` and `reports_ctl_oracle_primary.json`.
- `backend/evaluation/state_diagnostics.py` already requires an explicit `include_test=True` that
  only the frozen final evaluation passes. E1 code must not pass it.
- The Phase-D frozen artifacts — `reports_final_oracle.json`, `reports_final_predicted.json`,
  `evaluation.json` (`times_scored: 3`), and the localizer's `test_evaluation_report.json` — are
  read-only. DEC-030's single test read stands as the only one.
- E0 artifacts are preserved unmodified: `reports_e0_*.json`, `reports_ctl_*.json`,
  `reports_cmp_*.json`, `backend/evaluation/e0_readout.py`, `e0_controls.py`,
  `state_diagnostics.py`, `quantum_ml/readout.py`, both pixel caches.

---

## 11. Resource reality

| Quantity | E0 (16-qubit amplitude) | E1 Q1 target | Basis |
|---|---|---|---|
| Qubits | 16 | 6–8 | §4.1, Kübler / Thanasilp |
| State-preparation CX | **~62,940** | **~0** for angle encoding | Arbitrary state prep is O(2ⁿ) (Shende et al., IEEE TCAD 2006). Angle encoding uses one single-qubit rotation per feature — **zero** two-qubit gates. |
| Entangling CX per circuit | — | ~20–100 total | ≤12 shallow layers on 6–8 qubits |
| Trainable parameters | 32 | **0** | Fixed kernel; only downstream `C` and bandwidth are fitted, both on train |
| Objective evaluations | SPSA iterations × circuits | 0 | No variational loop at all |
| Circuit evaluations | per-image, per-iteration | O(n²) Gram entries, once per embedding | 215×215 train Gram, 52×215 validation Gram |
| Pre-screen cost | — | seconds | §4.3 geometric difference from Gram matrices only |

**Budget, binding:** total E1 quantum wall-clock ≤ 2 hours CPU statevector for the full experiment
matrix. Any candidate exceeding it is KILLed rather than granted more time. E1.1's classical failure
map is minutes.

Two disciplines carried forward explicitly:

- **Aer statevector performance is not hardware feasibility.** E0's amplitude path ran fine in
  simulation and is hardware-infeasible. Every E1 candidate reports an estimated two-qubit gate
  count alongside its simulator runtime, and the gate count is the number that gates it.
- **Do not optimise for qubit count.** The figure of merit is useful quantum processing per unit of
  hardware and compute cost. Dropping from 16 to 8 qubits while removing ~62,940 CX and all 32
  trainable parameters is a strict improvement in that ratio even if it reduces "quantumness" by
  any naive measure.

---

## 12. Novelty audit

Honest classification of the proposed Q1 architecture.

| Component | Classification |
|---|---|
| Projected quantum kernel | **Standard.** Huang et al. (2021). |
| Angle encoding | **Standard.** |
| Data re-uploading blocks | **Standard.** Pérez-Salinas et al. (2020); Schuld et al. (2021). |
| Local RDM observables | **Standard.** Cerezo et al. (2021). |
| Bandwidth as a tuned hyperparameter | **Demonstrated before.** Shaydulin & Wild (2022). |
| Precomputed-kernel SVM downstream | **Standard.** |
| Hybrid classical-descriptor → quantum-kernel → classical-head pipeline | **Recombination.** Not novel. |
| Geometric-difference pre-screen used as a **pre-registered go/no-go gate** before any quantum training | **Methodological, arguably a contribution.** The test is Huang et al.'s. Using it as a *hard gate* in an applied pipeline — kill the quantum arm on a Gram-matrix norm before spending a single training run — is a discipline choice, not a new algorithm. |
| Selecting the quantum input from a **measured** classical failure map, with the failure map's verdict rules pre-registered | **Methodological, arguably a contribution.** The rarity in the applied QML literature is not the architecture; it is refusing to build one until a bottleneck is measured, and pre-committing to the rules that decide. |
| The E0 → E1 sequence itself: a pre-registered negative result on one representation used as the designated negative control for the next | **Methodological.** DEC-031/DEC-032 as precedent. |

**Not novel, and must not be described as such:** the VQC, the QCNN, the quantum kernel, the hybrid
CNN+VQC. If Q1 is implemented and works, the contribution is the *evidence and the protocol*, not
the circuit. If it does not work, the contribution is a clean negative result on a real medical
imaging dataset — of which §5.3 shows the literature has very few.

---

## 13. Experiment matrix

Five experiments. E1.1 gates everything after it.

| ID | Experiment | Input | Classical control | Quantum method | Qubits | Depth | Primary metric | Null test | Resource budget | KEEP / KILL |
|---|---|---|---|---|---|---|---|---|---|---|
| **E1.1** | **Classical failure map** (gates all others) | C1–C7 of §2 | C1 = 181 features, train-only tuning | none | — | — | validation PR-AUC, primary condition `A_lesion_polygon` | patient-blocked label permutation; column-permutation increment for C7 | ≤ 30 min CPU | KEEP a family only on verdict **B** (§3). If no family is B → **NO QUANTUM IMPLEMENTATION JUSTIFIED**, phase ends. |
| **E1.2** | **Geometric-difference pre-screen** (gated on E1.1 = B) | 6–8 scalars of the B family | best classical kernel of §8 | Gram matrices only, no training | 6–8 | ≤ 12 layers | g_CQ | none (deterministic) | ≤ 5 min | KILL Q1 if g_CQ is small for every shortlisted embedding. Cheapest possible kill. |
| **E1.3** | **Q1 primary — projected quantum kernel** (gated on E1.2) | same 6–8 scalars | same SVM, same rows, same protocol; best of RBF / polynomial / bandwidth-matched RBF | projected quantum kernel, **0 trainable gates** | 6–8 | ≤ 12 layers | validation PR-AUC | **column-permutation increment null** (gating) + label permutation + shuffled-input control | ≤ 60 min CPU; ≤ 100 CX/circuit | KEEP only on all §9.4 conditions incl. margin ≥ δ = 2·se_control and BH-adjusted p < 0.05. |
| **E1.4** | **Q2 secondary — patch quantum filter** (only if E1.3 KEEPs) | k×k patches from `v1_pixels_*.npz` | **random classical filter bank of equal count and equal output dimension** | shallow random circuit per patch | 4–9 | shallow | validation PR-AUC | column permutation | ≤ 30 min CPU | KILL if it does not beat the matched random *classical* filter bank. Bowles et al. predicts it will not. |
| **E1.5** | **Condition-B replication** (only if E1.3 KEEPs) | same, on predicted ROIs | same control on the same rows | as E1.3, no refit of anything | 6–8 | ≤ 12 layers | validation PR-AUC | column permutation | ≤ 20 min CPU | KILL the claim if the gain exists only under oracle ROIs. §1.4's IoU 0.5285 makes this a real risk, and a result that survives only with oracle annotations is not deployable. |

No experiment reads the test partition. Every payload asserts `test_partition_used: false`.

---

## 14. Final recommendation

**A. What classical bottleneck was identified?**
None *measured*. What is established **[STATIC]** is that the 181-feature representation provably
contains no gradient orientation, no co-occurrence or LBP texture, no multiscale pyramid, no
cross-cell interaction terms, no boundary geometry, and no explicit lesion-versus-surround contrast
beyond a fixed 12% margin. What is established **[MEASURED]** is that the pipeline achieves
ROC-AUC ≈ 0.96 with PR-AUC ≈ 0.60 on validation — a precision problem in the high-sensitivity
region, not a ranking problem. Whether the structural gaps *cause* that precision ceiling is
**[OPEN]** and is what E1.1 measures.

**B. Why the bottleneck matters?**
Because the operating characteristic is what a screening tool is judged on. Reaching sensitivity
1.0 currently costs specificity 0.921 — 27 false positives for 21 true positives. And because the
alternative explanations are cheap to rule out: if C5 (finer grid + centre-versus-surround
contrast) or C6 (frozen MobileNet embedding) closes the gap, the fix is classical and the quantum
question does not arise. That test costs minutes and must be run first.

**C. What quantum mechanism specifically targets it?**
Conditionally, Q1: a projected quantum kernel on 6–8 scalar summaries of the verdict-B family, 6–8
qubits, angle encoding, ≤2 re-uploading blocks, local 1- and 2-body RDM projections, **zero
trainable gates**, tuned bandwidth. Zero trainable gates is the point — Caro et al.'s √(T/N) bound
is the binding constraint at 103–104 train positives, and a fixed kernel sets T = 0. The mechanism
targets nonlinear interaction structure among a *small, measured-to-matter* set of visual
descriptors, which is the one regime where Kübler et al. allow that a quantum kernel's inductive
bias could be aligned rather than diluted.

**D. What is the strongest classical control?**
The same precomputed-kernel SVM on the same rows with the same train-only protocol, using the best
of RBF (median-heuristic width), polynomial (train-tuned degree), and a bandwidth-tuned RBF grid
matched point-for-point to the quantum arm's bandwidth grid. Only the Gram matrix differs. E0's
leaky control — 0.128 reported, 0.463627 honest — is the standing reminder that the control gets
every advantage the quantum arm gets.

**E. What experiment will decide whether quantum is useful?**
E1.1 first, and it may end the phase on its own. Then E1.2, a geometric-difference pre-screen that
can kill Q1 from two Gram matrices in seconds. Then E1.3, the matched-control comparison gated on a
patient-blocked column-permutation increment null with a margin δ = 2·se_control derived from a
patient-level bootstrap of the control **before the quantum arm runs**.

**F. What result would kill the idea?**
Every family in E1.1 returning verdict A, C or D. C5 or C6 closing the gap classically. An
unfavourable g_CQ. An increment below δ. An increment that does not survive the column-permutation
null. A gain that appears only under oracle ROIs (E1.5) or only after pooling A/B/C. Any leakage
check failing. Any breach of the resource budget.

**G. What result would justify proceeding to implementation?**
All of: E1.1 identifies a verdict-B family under the pre-registered §3 rules; E1.2's g_CQ is
favourable; E1.3's validation PR-AUC on `A_lesion_polygon` exceeds the matched classical control by
at least δ, with the increment surviving the column-permutation null at BH-adjusted p < 0.05 across
the frozen candidate count; the shuffled-input control collapses to the geometry-leak bar; E1.5
reproduces the gain on predicted ROIs; and the resource budget holds.

---

**Standing verdict of this document: NO QUANTUM IMPLEMENTATION JUSTIFIED YET.**
E1.1 — the classical failure map — is the only work authorised next, and it is entirely classical.
