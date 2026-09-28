# Comprehensive Project Report Guide & Executive Summary

**Project:** CareScan / Braket 3.1.0  
**Initiative:** Smart India Hackathon 2026 — Problem Statement **SIH26139**  
**Category:** MedTech / Biomedical Computing / Quantum Machine Learning  
**Team:** **Team BraKet 3.1.0**  
*(Ishan Narayan Shukla, Jay Karan Laxme, Rudransh Rajveer Singh, Pratyaksh Ranjan, Priyanshi Saraswat, Prajjwal Patel)*  
**Document Link Identifier:** `SLIDE_2_REPORT_LINK`  
**Primary PDF Artifact:** [CareScan Complete Engineering & Scientific Audit Report (53 Pages)](file:///c:/Users/ISHAN%20SHUKLA/Downloads/Orqis-main/Orqis-main/report/main.pdf)

---

## 1. Document Overview

This document provides the executive guide to the official **53-page CareScan Engineering & Scientific Audit Report**, compiled and validated in `report/main.pdf`. 

Unlike generic hackathon presentations or high-level slide decks, the project report represents an exhaustive, production-grade technical dissertation covering every mathematical derivation, software layer, dataset provenance ledger, empirical experiment, and regulatory framework in the CareScan ecosystem.

```
+----------------------------------------------------------------------------------------------------+
|                         CARECASCAN 53-PAGE AUDIT REPORT SPECIFICATIONS                             |
+------------------------------+----------------------------------+----------------------------------+
|         Document File        |          Total Pages             |            File Size             |
|       report/main.pdf        |            53 Pages              |             8.51 MB              |
+------------------------------+----------------------------------+----------------------------------+
|       Compilation Tool       |        Scientific Figures        |          Test Evidence           |
|         Tectonic v0.15       |       9 High-Res PNG Diagrams    |      1,231 Automated Tests       |
+------------------------------+----------------------------------+----------------------------------+
```

---

## 2. Table of Contents & Chapter Breakdown

The report is structured into **12 rigorous technical chapters**, adhering to international medical AI reporting standards (CONSORT-AI and STARD-AI):

```
+----------------------------------------------------------------------------------------------------+
|                                  REPORT CHAPTER ARCHITECTURE                                       |
+----+---------------------------------------------------+-------+-----------------------------------+
| Ch | Title                                             | Pages | Key Technical Content             |
+----+---------------------------------------------------+-------+-----------------------------------+
| 1  | Executive Summary & Core Platform Vision          | 1 - 4 | Platform thesis, 91.3% PR-AUC     |
| 2  | Clinical Problem Domain & Epidemiological Rationale| 5 - 8 | Tobacco burden, OPMD progression  |
| 3  | End-to-End System Architecture                    | 9 - 14| 7-stage pipeline, edge quality gate|
| 4  | Dataset Curation, Integrity, & Clean Splits       | 15 - 19| SMART-OM ledger, 33 exclusions    |
| 5  | Deep Learning Localization Engine (MobileNetV3)   | 20 - 24| 374/381 localized, mIoU = 0.5279  |
| 6  | Classical Biomedical Feature Extraction & Calib.  | 25 - 29| 16D vector, Platt scaling, Brier  |
| 7  | Quantum Machine Learning Algorithms & Benchmark   | 30 - 35| 7 QML families, state prep proof  |
| 8  | Cross-Domain Biomedical Validation (PTB-XL ECG)   | 36 - 40| 13-arm leaderboard, N = 19,601    |
| 9  | Frontline Mobile Application & Edge Deployment    | 41 - 44| Flutter HUD, camera, offline mode |
| 10 | Security, Ethics, ABDM, & Regulatory Compliance   | 45 - 47| CDSCO SaMD, ABDM/FHIR R4, privacy |
| 11 | Complete Repository Audit & Test Suite Evidence   | 48 - 50| 1,231 tests passing, verification |
| 12 | Future Scope & Scalability Roadmap                | 51 - 53| Multi-centre trials, FTQC QRAM    |
+----+---------------------------------------------------+-------+-----------------------------------+
```

---

## 3. High-Resolution Scientific Figures Embedded in Report

All nine technical figures embedded within `report/main.pdf` are rendered at $300$ DPI and illustrate the exact engineering mechanics:

1. **Figure 1: End-to-End Hybrid Quantum-Classical Pipeline Architecture** (`report/figures/pipeline_architecture.png`)
   * Detailed block schematic tracing the 7-stage workflow from photon capture to calibrated probability output.
2. **Figure 2: Dataset Curation, Cleaning, and Exclusion Funnel** (`report/figures/data_funnel.png`)
   * Flow diagram showing the filtration of $447$ initial Peradeniya images down to the $414$ benchmark cohort, accounting for the $33$ excluded corrupted samples.
3. **Figure 3: MobileNetV3 Clinical Localization Viewfinder & HUD** (`report/figures/mobilenet_localization.png`)
   * Dual-panel diagram illustrating raw oral input, bounding box coordinate regression ($x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}$), and lesion ROI extraction.
4. **Figure 4: Precision-Recall & ROC Discrimination Curves** (`report/figures/pr_roc_curves.png`)
   * Primary empirical curves demonstrating PR-AUC of $0.913038$ against the $0.3235$ baseline prevalence and ROC-AUC of $0.933948$.
5. **Figure 5: Primary Test Partition Confusion Matrix & Calibration** (`report/figures/confusion_matrix.png`)
   * Heatmap and contingency table showing exact classification counts ($38$ TP, $4$ FN, $50$ TN, $10$ FP) at operating threshold $\tau = 0.42$.
6. **Figure 6: QML State-Preparation CNOT Gate Explosion vs. Coherence** (`report/figures/qml_state_prep.png`)
   * Theoretical vs. physical scaling curves showing $O(2^n)$ gate explosion against transmon $T_2^*$ coherence budgets.
7. **Figure 7: Non-Parametric Paired Bootstrap Forest Plot** (`report/figures/paired_bootstrap_forest.png`)
   * Forest plot demonstrating that the Hybrid Fusion delta spans zero ($[-0.000486, +0.001747]$) while the Havlicek ZZ kernel exhibits a statistically significant deficit.
8. **Figure 8: PTB-XL 13-Arm Biomedical Generalization Leaderboard** (`report/figures/ptbxl_leaderboard.png`)
   * Horizontal bar chart ranking all 13 classical and quantum architectures on $N=19,601$ 12-lead ECG records.
9. **Figure 9: CareScan Frontline Smartphone Mobile Application Interface** (`report/figures/mobile_app_mockup.png`)
   * Pixel-perfect UI mockup showing real-time camera viewfinder, blur detection HUD, patient risk card, and ABDM/ABHA export modal.

---

## 4. Key Scientific Achievements Highlighted

* **Audited Accuracy:** PR-AUC of $0.913038$ and ROC-AUC of $0.933948$ on patient-disjoint test data ($k=0$).
* **Probabilistic Calibration:** Platt-scaled Brier score of $0.110620$, ensuring risk percentages directly reflect true clinical disease probability.
* **Honest Null-Result Reporting:** Demonstrates that NISQ quantum kernels suffer a $12.45\%$ deficit relative to classical RBF SVMs on low-dimensional data, disproving shallow "quantum advantage" hype and defining the requirements for true quantum utility.
* **Edge Feasibility:** $41.8$ ms localization latency on standard mobile processors, enabling offline point-of-care deployment in rural clinics.
* **National Interoperability:** Turnkey ABDM FHIR R4 `DiagnosticReport` bundle generation with SNOMED CT and ICD-11 coding.

---

## 5. Accessing and Compiling the Report

The compiled PDF is located directly in the workspace:
* **Compiled PDF:** `c:\Users\ISHAN SHUKLA\Downloads\Orqis-main\Orqis-main\report\main.pdf`
* **LaTeX Source Code:** `c:\Users\ISHAN SHUKLA\Downloads\Orqis-main\Orqis-main\report\main.tex`
* **Local Recompilation Command:**
  ```powershell
  cd "c:\Users\ISHAN SHUKLA\Downloads\Orqis-main\Orqis-main"
  .\tools\tectonic.exe .\report\main.tex
  ```
The report compiles cleanly with zero errors and zero missing citations.
