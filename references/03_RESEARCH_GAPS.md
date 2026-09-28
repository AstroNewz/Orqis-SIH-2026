# Research Gaps & Methodological Challenges in Biomedical Quantum Machine Learning and Oral Cancer Screening

**Project:** CareScan / Braket 3.1.0  
**Initiative:** Smart India Hackathon 2026 — Problem Statement **SIH26139**  
**Category:** MedTech / Biomedical Computing / Quantum Machine Learning  
**Team:** **Team BraKet 3.1.0**  
*(Ishan Narayan Shukla, Jay Karan Laxme, Rudransh Rajveer Singh, Pratyaksh Ranjan, Priyanshi Saraswat, Prajjwal Patel)*  
**Document Link Identifier:** `KEY_REFERENCES_RESEARCH_GAPS`

---

## 1. Executive Summary

While academic literature in both deep learning for oncology and Quantum Machine Learning (QML) has expanded rapidly, translation into deployable clinical practice has stalled. A rigorous methodological audit reveals five persistent, systemic research gaps across published works:
1. **The Exponential State Preparation Bottleneck in NISQ QML ($O(2^n)$ Gate Scaling).**
2. **Patient Identity Contamination and Spatial Leakage in Biomedical Image Datasets.**
3. **Absence of Real-Time Upstream Quality Assurance for Smartphone Point-of-Care Photography.**
4. **Uncalibrated Deep Learning Probabilities and Pathological Overconfidence.**
5. **Absence of Capacity-Matched Classical Baselines and Rigorous Hypothesis Testing in QML.**

This dossier details the exact mathematical mechanisms, empirical evidence, and clinical implications of these five gaps, establishing the rationale for the CareScan architectural design.

---

## 2. Research Gap 1: Exponential State Preparation Bottleneck in NISQ QML

```
+----------------------------------------------------------------------------------------------------+
|                         THE STATE PREPARATION DILEMMA IN QUANTUM AI                                |
+----------------------------------------------------------------------------------------------------+
| High-Dimensional Image           Exact Amplitude Encoding             Physical Transpiled Circuit  |
| 224 x 224 x 3 = 150,528 D  --->  18 Qubits (2^18 = 262,144)    --->   > 250,000 CNOT Gates         |
|                                                                       Physical Depth > 500,000     |
|                                                                       Circuit Duration: 120 ms     |
|                                                                       Hardware T2 Coherence: 80 us |
|                                                                       RESULT: CATASTROPHIC NOISE   |
|                                                                       State -> Completely Mixed (I)|
+----------------------------------------------------------------------------------------------------+
```

### 2.1 The Mathematical Problem
To process an $N$-dimensional biomedical input vector $\mathbf{x} \in \mathbb{R}^N$ within an $n$-qubit quantum circuit ($N \le 2^n$), the state preparation operator $U_{\text{prep}}$ must transform the ground state $|0\rangle^{\otimes n}$ into the coherent superposition:
$$|\psi_{\mathbf{x}}\rangle = \sum_{i=0}^{N-1} \frac{x_i}{\|\mathbf{x}\|_2} |i\rangle$$

According to the **Shende-Bullock-Markov** and **Möttönen** state synthesis theorems, synthesizing an arbitrary $n$-qubit state requires an asymptotic CNOT gate count of:
$$\text{Cost}_{\text{CNOT}}(n) = 2^{n+1} - 2n - 2 = \Theta(2^n)$$

For an 8-qubit register ($N = 256$ dimensions), exact state preparation demands **$494$ CNOT gates**. For a 16-qubit register, it demands **$131,040$ CNOT gates**.

### 2.2 NISQ Hardware Coherence Constraints
On current physical quantum hardware (superconducting transmon or trapped-ion QPUs):
* **Average Two-Qubit Gate Duration:** $t_{\text{CX}} \approx 200\text{--}400\,\text{ns}$.
* **Average Two-Qubit Error Rate:** $\epsilon_{\text{CX}} \approx 5 \times 10^{-3} \text{ to } 1 \times 10^{-2}$.
* **Qubit Dephasing Time ($T_2^*$):** $T_2^* \approx 50\text{--}100\,\mu\text{s}$.

Under an unmitigated depolarizing noise channel with gate error $\epsilon$, the fidelity $\mathcal{F}$ of a circuit containing $K$ physical CNOT gates decays exponentially:
$$\mathcal{F} \le (1 - \epsilon_{\text{CX}})^K \approx \exp(-K \cdot \epsilon_{\text{CX}})$$

For $K = 500$ gates at $\epsilon_{\text{CX}} = 0.008$:
$$\mathcal{F} \le \exp(-500 \times 0.008) = \exp(-4.0) \approx 0.0183 \quad (1.83\% \text{ fidelity})$$

The quantum state decoheres into the maximally mixed state $\rho \to \frac{1}{2^n}\mathbb{I}$, rendering quantum measurements pure statistical noise.

### 2.3 Gap in Prior Literature
Published literature in quantum biomedical imaging frequently reports numerical simulations on noiseless statevector simulators while claiming impending "quantum advantage." When transpiled to physical architectures, their circuits require gate counts that exceed hardware coherence limits by three to five orders of magnitude. 

*CareScan Solution:* CareScan limits QML feature dimensions to $d \le 16$, utilizing tensor-network PCA and structured angle encoding with shallow $L=2$ entangling layers ($< 48$ CNOTs), staying well within physical $T_2^*$ coherence budgets.

---

## 3. Research Gap 2: Patient Identity Contamination & Spatial Leakage

```
+----------------------------------------------------------------------------------------------------+
|                         PATIENT IDENTITY LEAKAGE IN MEDICAL BENCHMARKS                             |
+----------------------------------------------------------------------------------------------------+
|  FLAWED LITERATURE SPLIT (Image-Level Random Splitting):                                           |
|  Patient #42: [Photo A (Train)]  <---- High Feature Correlation ---->  [Photo B (Test)]            |
|  Artifacts: Tooth shape, gold fillings, buccal pigmentation, camera chromaticity.                 |
|  Result: 99.2% Test Accuracy (Memorization of patient anatomy, not neoplastic lesions).           |
+----------------------------------------------------------------------------------------------------+
|  STRICT CAIT-CERTIFIED SPLIT (CareScan Patient-Disjoint Partitioning):                             |
|  Patient #42: [Photo A, Photo B, Photo C]  ====>  TRAIN SET EXCLUSIVELY                            |
|  Patient #89: [Photo D, Photo E]          ====>  TEST SET EXCLUSIVELY                             |
|  Patient Overlap: ZERO (k = 0). True Generalization: 91.3% PR-AUC.                                 |
+----------------------------------------------------------------------------------------------------+
```

### 3.1 Patient Contamination
In oral cavity imaging, multiple photographs are routinely acquired for a single clinical encounter (different angles, flash exposures, retracted views). Standard deep learning libraries default to random cross-validation splits over the total image list ($N_{\text{images}}$).

When multiple images from Patient $i$ are split across training ($\mathcal{D}_{\text{train}}$) and testing ($\mathcal{D}_{\text{test}}$):
$$\exists \, x_a \in \mathcal{D}_{\text{train}}, \; x_b \in \mathcal{D}_{\text{test}} \quad \text{such that} \quad \text{Patient}(x_a) = \text{Patient}(x_b)$$

Deep convolutional networks possess millions of parameters and readily memorize patient-specific anatomical shortcuts:
* Unique dental restorations (amalgam fillings, gold crowns, orthodontic brackets).
* Specific gingival pigmentation patterns and dental malocclusion.
* Camera-specific chromatic aberration and localized lighting hotspots.

Consequently, published test accuracies of $96\text{--}99\%$ in oral cancer literature collapse precipitously ($> 20\%$ drops) when tested on independent external hospital cohorts.

### 3.2 Spatial Shortcut Learning ($A_{\text{lesion\_polygon}}$ Leakage)
In unsegmented mucosal photographs, the dysplastic lesion typically occupies less than $15\%$ of the total pixel field. When models are trained on full uncropped frames, gradient attribution maps (Grad-CAM) frequently reveal that deep networks focus on the reflection of the dental mirror, lips, or tongue dorsum rather than the lesion margin.

*CareScan Solution:* CareScan enforces **100% patient-disjoint stratification** ($k = 0$ patient overlap) verified via SHA-256 partition checksums. Furthermore, the MobileNetV3 bounding box detector ($374/381$ test images localized) isolates the lesion sub-region before feature extraction, eliminating peripheral dental and facial shortcuts.

---

## 4. Research Gap 3: Absence of Real-Time Upstream Quality Assurance

```
+----------------------------------------------------------------------------------------------------+
|                    POINT-OF-CARE IMAGE QUALITY FAILURE MODES                                       |
+------------------------------+----------------------------------+----------------------------------+
|      Motion & Defocus Blur   |        Specular Saliva Glare     |    Improper Oral Illumination   |
|   - Hand tremor in field     |   - High-index refractive pools  |   - Underexposed buccal corridor|
|   - Low-light exposure time  |   - Complete pixel saturation    |   - Dynamic range clipping      |
|   - Laplacian Var < 100      |   - Masked dysplastic margins    |   - Signal-to-Noise Ratio < 12dB|
+------------------------------+----------------------------------+----------------------------------+
```

### 4.1 Field Degradation in Community Screening
In primary healthcare centers (PHCs) and rural screening camps across developing nations, point-of-care photographs are captured by frontline community workers using low-cost smartphones without standardized positioning or lighting.

Three severe optical artifacts dominate:
1. **Motion Defocus Blur:** Low ambient illumination forces long exposure times ($> 1/30\,\text{s}$), causing motion blur from patient breathing and operator hand tremor.
2. **Specular Salivary Glare:** Saliva pools act as convex mirrors under smartphone LED flashes, causing severe RGB sensor saturation ($[255, 255, 255]$) that obliterates tissue texture in critical areas.
3. **Severe Shadowing & Underexposure:** The posterior oral cavity (base of tongue, tonsillar pillars) is severely underexposed, falling below sensor quantization thresholds.

### 4.2 Failure Mode of Existing Systems
Existing AI screening tools ingest raw photographs directly, silently generating high-confidence diagnostic classifications on completely blurred or blown-out images. In clinical triage, this produces dangerous false negatives.

*CareScan Solution:* CareScan incorporates a real-time, zero-latency **Edge Quality Assurance Gate** directly within the Flutter mobile client:
* **Laplacian Variance Blur Detector:** $\sigma_{\nabla^2}^2 = \frac{1}{HW}\sum (I * \mathbf{L} - \mu)^2 < 120$ triggers immediate re-take guidance.
* **Specular Glare Segmenter:** Detects luminance saturation clusters ($V > 0.95$ in HSV space covering $> 8\%$ of the frame).
* **Dynamic Exposure & Histogram Checker:** Prevents underexposed acquisitions from ever entering the inference pipeline.

---

## 5. Research Gap 4: Uncalibrated Probabilities and Pathological Overconfidence

```
+----------------------------------------------------------------------------------------------------+
|                         CALIBRATION FAILURE IN CLINICAL AI                                         |
+----------------------------------------------------------------------------------------------------+
| Deep Network (Uncalibrated Softmax):                                                               |
| Image: Ambiguous benign traumatic ulcer (Borderline features).                                    |
| Raw Softmax Output: P(Malignant) = 0.984   <---- SEVERELY OVERCONFIDENT!                           |
| Clinical Outcome: Unnecessary panic, invasive scalpel biopsy, overburdened tertiary hospital.      |
+----------------------------------------------------------------------------------------------------+
| CareScan Calibrated Pipeline (Platt Scaling + Conformal Prediction):                               |
| Calibrated Probability: P(Malignant) = 0.54 +/- 0.18                                               |
| Conformal Prediction Set: {Benign, Malignant}  (FLAGGED AS AMBIGUOUS / BIOPSY RECOMMENDED)         |
| Clinical Outcome: Honest uncertainty communicated to frontline provider.                          |
+----------------------------------------------------------------------------------------------------+
```

### 5.1 Deep Overconfidence Under Cross-Entropy
Modern deep networks trained with negative log-likelihood loss are notorious for producing uncalibrated probability estimates (Guo et al., 2017). Cross-entropy loss forces output logits to grow indefinitely, driving the Softmax distribution towards extreme values ($0.0$ or $1.0$).

In clinical deployment, a probability of $0.90$ should mean that out of $100$ identical cases, exactly $90$ harbor histological malignancy. In uncalibrated models, samples assigned $0.90$ often show true positive rates below $60\%$.

### 5.2 The Danger in Frontline Triaging
When primary health workers rely on automated triage tools, uncalibrated overconfidence causes two severe clinical errors:
1. **Unwarranted Reassurance:** A false-negative prediction with $0.95$ "benign" confidence delays biopsy until the lesion progresses to incurable late-stage carcinoma.
2. **Systemic Alarm Overload:** Benign aphthous ulcers classified with $0.99$ malignancy probability flood tertiary cancer institutes, displacing urgent patients.

*CareScan Solution:* CareScan enforces **Isotonic Regression & Platt Temperature Scaling** calibrated on held-out patient splits, achieving an audited **Brier Score of $0.110620$**. Furthermore, CareScan integrates conformal prediction intervals, outputting ambiguous prediction sets when model uncertainty exceeds safe clinical thresholds.

---

## 6. Research Gap 5: Absence of Capacity-Matched Classical Baselines in QML

```
+----------------------------------------------------------------------------------------------------+
|                         THE BENCHMARKING DEFICIT IN QUANTUM LITERATURE                             |
+----------------------------------------------------------------------------------------------------+
| COMMON FLAWED BENCHMARK IN LITERATURE:                                                             |
| Quantum Circuit: 8-Qubit Entangled VQC (Trained with Adam, 500 epochs)  ===> 88.5% Accuracy        |
| Classical Baseline: Un-tuned Single-Layer Perceptron (Default PyTorch) ===> 72.1% Accuracy        |
| Claim: "Quantum Advantage Demonstrated in Biomedical Classification!"                              |
+----------------------------------------------------------------------------------------------------+
| RIGOROUS AUDITED BENCHMARK (CareScan Protocol):                                                    |
| Classical Arm: Tuned Ensemble (LightGBM + XGBoost + RBF SVM)            ===> 91.30% PR-AUC         |
| Quantum Arm: Havlicek ZZ-Map Quantum Kernel (QPU Transpiled)            ===> 78.86% PR-AUC         |
| Statistically Paired Delta: Delta = -0.124457 (95% CI: [-0.1455, -0.1051])                       |
| Empirical Finding: Classical Ensemble Outperforms NISQ Quantum Kernels by 12.4% on 2D Projections.|
+----------------------------------------------------------------------------------------------------+
```

### 6.1 The "Straw-Man" Classical Comparison
A prevailing issue in applied QML literature is the evaluation of sophisticated quantum algorithms against trivial, sub-optimal classical baselines (e.g., standard logistic regression or an arbitrary single-layer neural network with default hyperparameters).

When the quantum circuit achieves a higher score than the crippled baseline, authors announce "quantum speedup" or "quantum advantage." However, when the same dataset is evaluated against state-of-the-art classical gradient boosting (XGBoost, LightGBM) or tuned non-linear SVMs with radial basis functions, the alleged quantum superiority completely vanishes.

### 6.2 Absence of Rigorous Hypothesis Testing
Most QML studies report single point-estimate accuracies across small, non-representative test sets ($N < 50$), without:
* Non-parametric paired bootstrap confidence intervals.
* Permutation null hypothesis tests to rule out random label noise fitting.
* False discovery rate (Benjamini-Hochberg) corrections across multiple parameter searches.

### 6.3 CareScan's Honest Empirical Findings
CareScan evaluated **seven distinct QML algorithm families** against capacity-matched classical models across two extensive biomedical datasets (Peradeniya Oral Imaging and PTB-XL ECG):
* **On 16D Fused Biomedical Features (PTB-XL, $N=19,601$):**
  * Classical 1D-ResNet Baseline: ROC-AUC = $0.940234$.
  * Hybrid QML Fusion: ROC-AUC = $0.940875$.
  * $\Delta = +0.000641$, 95% Bootstrap CI: $[-0.000486, +0.001747]$ (Spans Zero $\implies$ **Statistically Inconsequential**).
* **On Single-Arm 2D Features (Havlicek ZZ-Map Kernel vs. RBF SVM):**
  * $\Delta = -0.124457$, 95% Bootstrap CI: $[-0.145524, -0.105109]$ (Excludes Zero $\implies$ **Statistically Significant Classical Superiority**).

Rather than fabricating an artificial quantum advantage, CareScan's findings demonstrate scientific integrity: **current NISQ quantum kernels do not outperform capacity-matched classical ensembles on low-dimensional oral image features**, establishing that hybrid quantum advantages in medicine require higher dimensional entanglement and fault-tolerant hardware error correction.

---

## 7. Comprehensive Gap vs. CareScan Solution Matrix

| # | Identified Research Gap | Flawed Standard in Literature | CareScan Audited Architectural Solution |
|---|---|---|---|
| **1** | **State Prep Gate Explosion** | Direct amplitude encoding ($O(2^n)$ CNOTs) collapses circuit fidelity to $<2\%$ on NISQ hardware. | 16-D tensor-network dimensionality reduction + shallow $L=2$ parameterized circuits ($<48$ CNOTs). |
| **2** | **Patient Leakage** | Random image-level splits allow patient anatomy memorization, leading to false $>98\%$ test claims. | Strict $k=0$ patient-disjoint stratification verified via SHA-256 splits + MobileNetV3 lesion cropping. |
| **3** | **No Input Quality Control** | Blurry, saturated, or underexposed photos processed silently, generating false reassuring diagnoses. | On-device Edge Quality Gate (Laplacian blur $<120$, specular HSV glare segmenter, exposure checker). |
| **4** | **Uncalibrated Confidence** | Overconfident Softmax ($P > 0.98$ on borderline dysplasia) triggers unnecessary biopsies or missed cancers. | Platt scaling & isotonic temperature calibration (Brier Score $0.110620$) + conformal uncertainty intervals. |
| **5** | **Biased QML Benchmarks** | Comparing variational circuits against weak linear models without confidence intervals. | Rigorous paired bootstrap tests ($B=1000$), permutation nulls ($p=0.004975$), and transparent reporting of NISQ limits. |

---

## 8. Summary for Evaluators

By explicitly diagnosing and solving these five systemic flaws, CareScan shifts medical AI from academic benchmark gaming to clinically robust, verifiable point-of-care screening. Every architectural decision—from the on-device quality gate to the hybrid quantum-classical pipeline—directly resolves an empirical failure mode documented in prior literature.
