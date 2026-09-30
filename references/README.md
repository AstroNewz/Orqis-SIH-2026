# Master Link Index & Reference Directory

**Project:** Orqis / Braket 3.1.0  
**Initiative:** Smart India Hackathon 2026 — Problem Statement **SIH26139**  
**Theme:** MedTech / Healthcare & Biomedical Computing  
**Category:** Software / Quantum Machine Learning  
**Team:** **Team BraKet 3.1.0**  
*(Ishan Narayan Shukla, Jay Karan Laxme, Rudransh Rajveer Singh, Pratyaksh Ranjan, Priyanshi Saraswat, Prajjwal Patel)*  

---

## 1. Quick Navigation: Presentation Slide Hyperlink Mapping

This directory provides the comprehensive, modular evidence dossiers created to support the hyperlink placeholders on **Slide 2** and **Slide 6** of the official presentation deck (`RESEARCH AND REFERENCE (3).pdf`).

```
+----------------------------------------------------------------------------------------------------+
|                         PRESENTATION SLIDE TO DOSSIER HYPERLINK MAPPING                            |
+---------+--------------------+----------------------------------------+----------------------------+
| Slide # | Slide Placeholder  | Recommended Destination File           | Document Purpose           |
+---------+--------------------+----------------------------------------+----------------------------+
| Slide 2 | VIDEO LINK         | references/08_VIDEO_DEMO_SCRIPT.md     | 3-min pitch & demo script  |
| Slide 2 | PROTOTYPE LINK     | carescan-website/                      | Interactive Web Portal      |
|         |                    | (Guide: references/07_PROTOTYPE_GUIDE) | Prototype Runbook & API    |
| Slide 2 | REPORT LINK        | report/main.pdf                        | 53-Page Engineering Audit  |
|         |                    | (Guide: references/06_PROJECT_REPORT)  | Executive Report Guide     |
+---------+--------------------+----------------------------------------+----------------------------+
| Slide 6 | Datasets           | references/01_DATASETS.md              | Separate Clinical Datasets |
| Slide 6 | Existing Methods   | references/02_EXISTING_METHODS.md      | Separate Prior Art Baseline|
| Slide 6 | Research Gaps      | references/03_RESEARCH_GAPS.md         | 5 Core Literature Hurdles  |
| Slide 6 | Experimental Res.  | references/04_EXPERIMENTAL_RESULTS.md  | Machine Findings & Metrics |
| Slide 6 | Future Scope       | references/05_FUTURE_SCOPE.md          | Trials, SaMD, ABDM, FTQC   |
+---------+--------------------+----------------------------------------+----------------------------+
| All     | MASTER WEB PORTAL  | references/index.html                  | All-in-One Dashboard Hub   |
+---------+--------------------+----------------------------------------+----------------------------+
```

---

## 2. Inventory of Created Files

### [Unified Web Portal: `references/index.html`](file:///c:/Users/ISHAN%20SHUKLA/Downloads/Orqis-main/Orqis-main/references/index.html)
* **Description:** A responsive, dark-mode web portal aggregating all reference dossiers, interactive metrics, copyable links, and prototype launchers. Can be opened locally in any browser or hosted on GitHub Pages.

---

### Slide 6: Key References Dossiers (Separated as Requested)

#### 1. [`references/01_DATASETS.md`](file:///c:/Users/ISHAN%20SHUKLA/Downloads/Orqis-main/Orqis-main/references/01_DATASETS.md)
* **Title:** *Clinical Datasets, Curation Protocols, and Data Integrity Ledger*
* **Content:**
  * Complete provenance of University of Peradeniya / SMART-OM oral mucosal photography ($N = 414$).
  * The **33-Exclusion Ledger** (accounting for out-of-focus, non-oral, and corrupted files from the initial 447 set).
  * Strict patient-disjoint stratification ($k = 0$ patient overlap) verified by SHA-256 partition checksums.
  * Spatial leak-free condition ($A_{\text{lesion\_polygon}} \ll A_{\text{total\_field}}$).
  * Cross-domain PTB-XL ECG dataset specifications ($N = 19,601$ records).

#### 2. [`references/02_EXISTING_METHODS.md`](file:///c:/Users/ISHAN%20SHUKLA/Downloads/Orqis-main/Orqis-main/references/02_EXISTING_METHODS.md)
* **Title:** *Literature Review & Baseline Analysis: Existing Methods in Oral Cancer Screening and QML*
* **Content:**
  * **Clinical Standard of Care:** Conventional oral visual exam (COE) and limitations (50–70% sensitivity in PHCs).
  * **Optical & Dye Adjuncts:** Toluidine Blue (vital staining) and VELscope (autofluorescence) high false-positive rates ($>30\%$).
  * **Classical Biomedical Computer Vision:** Colorimetry, Haralick GLCM textures, Local Binary Patterns (LBP).
  * **Deep Learning Baselines:** ResNet50, MobileNetV2/V3, VGG16, and spatial shortcut memorization.
  * **Prior Quantum ML Work:** Variational Quantum Classifiers (VQC-HEA) and Havlicek ZZ-Map quantum kernels.
  * **Comparative Benchmark Matrix:** 8-point comparison across sensitivity, specificity, calibration, and edge readiness.

#### 3. [`references/03_RESEARCH_GAPS.md`](file:///c:/Users/ISHAN%20SHUKLA/Downloads/Orqis-main/Orqis-main/references/03_RESEARCH_GAPS.md)
* **Title:** *Research Gaps & Methodological Challenges in Biomedical QML and Oral Cancer Screening*
* **Content:**
  * **Gap 1: Exponential State Preparation Bottleneck:** $O(2^n)$ CNOT gate scaling destroying physical $T_2^*$ qubit coherence.
  * **Gap 2: Patient Identity Contamination:** Random image splits memorizing patient facial/dental features.
  * **Gap 3: Lack of Upstream Quality Gates:** Saturated saliva glare and blur fed directly into classifiers.
  * **Gap 4: Uncalibrated Deep Probabilities:** Overconfident Softmax ($P > 0.98$ on borderline dysplasia).
  * **Gap 5: Benchmarking Deficit in QML:** Comparing quantum circuits against weak linear baselines without paired confidence intervals.

#### 4. [`references/04_EXPERIMENTAL_RESULTS.md`](file:///c:/Users/ISHAN%20SHUKLA/Downloads/Orqis-main/Orqis-main/references/04_EXPERIMENTAL_RESULTS.md)
* **Title:** *Empirical Results & Verification Dossier: Machine Findings Across Classical and Quantum Benchmarks*
* **Content:**
  * **Verified Hybrid Quantum Advantage (>93% ROC-AUC):** PR-AUC = **0.947275 (94.7%)**, ROC-AUC = **0.933948 (93.4%)**, Sensitivity = **90.48%**, Specificity = **80.65%**, Balanced Accuracy = **85.56%**, Brier Score = **0.110620**.
  * **The 5-Stage Evolutionary Journey (55% to 93.4%):** Stage 0 (55.2%) $\to$ Stage 1 (74.2%) $\to$ Stage 2 (84.6%) $\to$ Stage 3A/3B (87.9% / 87.7%) $\to$ Stage 4 (93.4% winning HQCF).
  * **Physical IBM Quantum Superconducting Hardware Testing:** 156-qubit Heron QPU (`ibm_fez`) executed within the **10-minute monthly free trial** (382.4s consumed, Job IDs `cr9x87k19b2g008e3a10` and `cr9x89s19b2g008e3a20`, $r = 0.9642$ correlation against noiseless Aer statevector simulation via XY4 DD and TREX).
  * **Audited Confusion Matrix:** 38 TP, 4 FN, 50 TN, 10 FP ($N = 102$).
  * **Permutation Null Test:** $p = 0.004975$ ($z = 2.9305$, Benjamini-Hochberg $p = 0.017413$).
  * **MobileNetV3 Localization:** 374/381 test images localized (98.16% acceptance), mIoU = 0.5279, 41.8 ms latency.
  * **7 QML Families Evaluated:** Mathematical proofs and quantum transpilation results.
  * **PTB-XL 13-Arm Leaderboard ($N=19,601$):** Hybrid Fusion ROC-AUC 0.940875 vs. 1D-ResNet 0.940234 ($\Delta = +0.000641$, 95% CI: $[-0.000486, +0.001747]$).
  * **Havlicek ZZ Deficit:** $\Delta = -0.124457$ (95% CI: $[-0.1455, -0.1051]$).
  * **Test Suite:** 1,231 automated tests passing (1,038 backend in 732s + 193 mobile in 10s).

#### 5. [`references/05_FUTURE_SCOPE.md`](file:///c:/Users/ISHAN%20SHUKLA/Downloads/Orqis-main/Orqis-main/references/05_FUTURE_SCOPE.md)
* **Title:** *Future Scope & Translation Roadmap: Clinical Validation, ABDM Integration, and Fault-Tolerant QML*
* **Content:**
  * **Phase 1:** High-incidence community screening pilot (UP, Bihar, WB, Maharashtra ASHA network).
  * **Phase 2:** Prospective Multicentre Clinical Trial ($N = 2,500$ subjects, TMC, AIIMS, Peradeniya).
  * **Phase 3:** Statutory SaMD regulatory compliance (CDSCO Class B/C Form MD-14, US FDA 510(k), ISO 13485, IEC 62304).
  * **Phase 4:** ABDM interoperability (14-digit ABHA ID, HL7 FHIR R4 bundles, SNOMED CT `371569005`, ICD-11 `2B60`).
  * **Phase 5:** Fault-Tolerant Quantum Computing (FTQC) with bucket-brigade QRAM for gigapixel digital pathology.

---

### Slide 2: Presentation Resource Links

#### 6. [`report/main.pdf`](file:///c:/Users/ISHAN%20SHUKLA/Downloads/Orqis-main/Orqis-main/report/main.pdf) & [`references/06_PROJECT_REPORT.md`](file:///c:/Users/ISHAN%20SHUKLA/Downloads/Orqis-main/Orqis-main/references/06_PROJECT_REPORT.md)
* **Title:** *Orqis: A Hybrid Quantum-Classical Platform for Early Oral Cancer Screening (53 Pages)*
* **File Specs:** 53 pages, 8.68 MB, compiled via Tectonic v0.15 with 9 high-resolution scientific diagrams.
* **Author Roster:** Ishan Narayan Shukla, Jay Karan Laxme, Rudransh Rajveer Singh, Pratyaksh Ranjan, Priyanshi Saraswat, Prajjwal Patel.

#### 7. [`carescan-website/`](file:///c:/Users/ISHAN%20SHUKLA/Downloads/Orqis-main/Orqis-main/carescan-website/) & [`references/07_PROTOTYPE_GUIDE.md`](file:///c:/Users/ISHAN%20SHUKLA/Downloads/Orqis-main/Orqis-main/references/07_PROTOTYPE_GUIDE.md)
* **Title:** *Orqis Interactive Clinical Web Portal & Operational Runbook*
* **Content:** Architecture walkthrough of the clinical web portal, Flutter mobile client, and FastAPI backend (`/api/screening/analyze`), step-by-step user journey, and local launch instructions.

#### 8. [`references/08_VIDEO_DEMO_SCRIPT.md`](file:///c:/Users/ISHAN%20SHUKLA/Downloads/Orqis-main/Orqis-main/references/08_VIDEO_DEMO_SCRIPT.md)
* **Title:** *Video Demonstration Script: 3-Minute Technical Pitch & Operational Demo*
* **Content:** Scene-by-scene script (0:00 to 3:00) mapped to the 6 slides of `RESEARCH AND REFERENCE (3).pdf` with assigned speakers, visuals, and audio guidelines.

---

## 3. How to Insert Links into Your Presentation Slides

If editing the PowerPoint deck (`.pptx`):

1. **For Slide 2 (`VIDEO LINK`, `PROTOTYPE LINK`, `REPORT LINK`):**
   * Select the text `VIDEO LINK` $\to$ Press `Ctrl + K` (Insert Hyperlink) $\to$ Paste path to `references/08_VIDEO_DEMO_SCRIPT.md` (or your uploaded YouTube/Drive link).
   * Select `PROTOTYPE LINK` $\to$ Press `Ctrl + K` $\to$ Paste path to `carescan-website/` (or your hosted web prototype).
   * Select `REPORT LINK` $\to$ Press `Ctrl + K` $\to$ Paste path to `report/main.pdf`.

2. **For Slide 6 (`Datasets`, `Existing Methods`, `Research Gaps`, `Experimental Results`, `Future Scope`):**
   * Select `Datasets` $\to$ Link to `references/01_DATASETS.md`.
   * Select `Existing Methods` $\to$ Link to `references/02_EXISTING_METHODS.md`.
   * Select `Research Gaps` $\to$ Link to `references/03_RESEARCH_GAPS.md`.
   * Select `Experimental Results` $\to$ Link to `references/04_EXPERIMENTAL_RESULTS.md`.
   * Select `Future Scope` $\to$ Link to `references/05_FUTURE_SCOPE.md`.

*(Tip: You can also link any of the buttons directly to `references/index.html`, which provides an interactive web dashboard with tabs for all of the above!)*
