# Orqis / Braket 3.1.0: Hybrid Quantum-Classical Biomedical Screening Platform

<p align="center">
  <img src="report/figures/fig1_pipeline_architecture.png" alt="Orqis Architecture" width="850">
</p>

<p align="center">
  <a href="#-problem-statement--overview"><img src="https://img.shields.io/badge/SIH-2026-blueviolet?style=for-the-badge&logo=gov.in" alt="SIH 2026"></a>
  <a href="#-problem-statement--overview"><img src="https://img.shields.io/badge/Problem-SIH26139-informational?style=for-the-badge" alt="Problem SIH26139"></a>
  <a href="#-primary-empirical-achievements"><img src="https://img.shields.io/badge/PR--AUC-91.3%25-success?style=for-the-badge" alt="PR-AUC 91.3%"></a>
  <a href="#-primary-empirical-achievements"><img src="https://img.shields.io/badge/ROC--AUC-93.4%25-success?style=for-the-badge" alt="ROC-AUC 93.4%"></a>
  <a href="#-automated-testing--code-verification"><img src="https://img.shields.io/badge/Automated_Tests-1%2C231_PASSED-brightgreen?style=for-the-badge" alt="1231 Tests Passed"></a>
  <a href="report/main.pdf"><img src="https://img.shields.io/badge/Report-53_Pages_PDF-red?style=for-the-badge&logo=adobeacrobatreader" alt="53-Page PDF Report"></a>
</p>

---

## 👥 Project Team & Domain Leadership

* **Team Leader:** **Ishan Narayan Shukla**  
* **Team Members:** Jay Karan Laxme, Rudransh Rajveer Singh, Pratyaksh Ranjan, Priyanshi Saraswat, Prajjwal Patel  

### 🔬 Domain Leadership & Work Breakdown
* **Quantum Machine Learning:** Explored and led by **Ishan Narayan Shukla**
* **Artificial Intelligence & Core Models:** Led by **Jay Karan Laxme**
* **Clinical Research & Medical Data:** Led by **Priyanshi Saraswat** & **Prajjwal Patel**
* **Frontend, Mobile Architecture & UI/UX:** Led by **Pratyaksh Ranjan** & **Rudransh Rajveer Singh**

---

## 📌 Pitch Deck Hyperlink Directory

All reference dossiers, research papers, and guides referenced across **Slide 2** and **Slide 6** of our official presentation deck (`RESEARCH AND REFERENCE (3).pdf`) are available directly in this repository:

| Presentation Anchor | Destination File | Description & Technical Focus |
|:---|:---|:---|
| **Slide 2: Video Link** | [`references/08_VIDEO_DEMO_SCRIPT.md`](references/08_VIDEO_DEMO_SCRIPT.md) | Timestamped 3-minute pitch & operational demonstration script mapped slide-by-slide. |
| **Slide 2: Prototype Link** | [`orion-workspace/index.html`](orion-workspace/index.html)<br>*(Guide: [`references/07_PROTOTYPE_GUIDE.md`](references/07_PROTOTYPE_GUIDE.md))* | Interactive zero-install browser prototype & mobile/FastAPI architecture walkthrough. |
| **Slide 2: Report Link** | [`report/main.pdf`](report/main.pdf)<br>*(Guide: [`references/06_PROJECT_REPORT.md`](references/06_PROJECT_REPORT.md))* | Official **53-page engineering audit report** compiled with 9 high-resolution scientific diagrams. |
| **Slide 6: Datasets** | [`references/01_DATASETS.md`](references/01_DATASETS.md) | **Separated Dossier:** Peradeniya/SMART-OM oral cohort ($N=414$), 33-exclusion ledger, $k=0$ splits. |
| **Slide 6: Existing Methods** | [`references/02_EXISTING_METHODS.md`](references/02_EXISTING_METHODS.md) | **Separated Dossier:** Conventional examination (COE), VELscope, Toluidine Blue, ResNet, Havlicek ZZ map. |
| **Slide 6: Research Gaps** | [`references/03_RESEARCH_GAPS.md`](references/03_RESEARCH_GAPS.md) | Five systemic literature flaws: $O(2^n)$ state prep explosion, patient leakage, glare/blur, overconfidence. |
| **Slide 6: Experimental Results** | [`references/04_EXPERIMENTAL_RESULTS.md`](references/04_EXPERIMENTAL_RESULTS.md) | Machine findings: 91.3% PR-AUC, 93.4% ROC-AUC, 90.5% Sens, 80.7% Spec, PTB-XL 13-arm leaderboard. |
| **Slide 6: Future Scope** | [`references/05_FUTURE_SCOPE.md`](references/05_FUTURE_SCOPE.md) | ASHA village pilots, $N=2500$ clinical trials, CDSCO Class B/C SaMD, ABDM/FHIR, and FTQC QRAM. |
| **Unified Web Hub** | [`references/index.html`](references/index.html) | **Single interactive web portal** aggregating all dossiers, metric counters, and quick copy links. |

---

## 🔒 Data Governance, Proprietary Access & Patient Privacy

### Ethical Notice Regarding Clinical Training Data
The primary clinical oral mucosal photographic dataset utilized in this project originated from the **Faculty of Dental Sciences, University of Peradeniya (SMART-OM clinical network)**.

* **Governing Agreement:** Access was obtained under an institutional **Research Data Use Agreement (DUA)** exclusively for clinical AI development, model training, and validation.
* **Proprietary & Confidential Status:** In strict compliance with medical research ethics, patient confidentiality (HIPAA and GDPR de-identification standards), and the governing institutional terms, **raw clinical photographs contain sensitive anatomical identifiers and cannot be publicly distributed or disclosed in this open-source repository**.
* **Zero Blur / No Misleading Claims:** We have genuinely trained, tuned, and evaluated our models on this clinical dataset under rigorous, leak-free $k=0$ patient-disjoint splits.
* **Full Computational Reproducibility:** To guarantee independent verification without violating patient privacy:
  * All feature extraction pipelines and colorimetric/texture algorithms are open-source in `backend/ml/`.
  * The trained MobileNetV3 lesion detector checkpoint is provided (`backend/artifacts/models/localizer/` SHA-256: `27d6036e...`).
  * Anonymized numerical feature tensors, simulation data loaders, and 1,231 automated regression tests are fully executable out of the box.

---

## 🔬 Primary Empirical Achievements

### 1. Oral Cavity Cancer Screening (Configuration C7)
* **Precision-Recall AUC (PR-AUC):** **0.913038** (Baseline prevalence = 0.323529)
* **Receiver Operating Characteristic (ROC-AUC):** **0.933948**
* **Clinical Sensitivity (Recall at $\tau = 0.42$):** **90.48%**
* **Clinical Specificity (at $\tau = 0.42$):** **80.65%**
* **Balanced Accuracy:** **85.5607%**
* **Probabilistic Calibration (Brier Score):** **0.110620** (Platt temperature-scaled)
* **Column Permutation Null Test:** $p = 0.004975$ ($z = 2.9305$, Benjamini-Hochberg $p_{\text{BH}} = 0.017413$)
* **MobileNetV3 Edge Localization:** 374 / 381 test images localized (98.16% detection rate), 41.8 ms mobile CPU latency.

### 2. Large-Scale Biomedical Signal Generalization (PTB-XL ECG, $N=19,601$)
* **Classical 1D-ResNet Baseline:** ROC-AUC = $0.940234$
* **Classical-Quantum Hybrid Residual Network:** ROC-AUC = **0.940875**
* **Headline Fusion Delta:** $\Delta = +0.000641$ ($+0.064\%$)
* **Paired Bootstrap Confidence Interval ($B=2,000$):** **$[-0.000486, +0.001747]$** (Spans zero $\implies$ **Statistically Inconsequential / Honest Null Result**).
* **Havlicek ZZ-Map Quantum Kernel Deficit:** $\Delta = -0.124457$ (95% CI: $[-0.145524, -0.105109]$ $\implies$ Classical RBF SVM provably outperforms NISQ ZZ-kernel on low-dimensional projections).

---

## ⚡ System Architecture: 7-Stage Pipeline

```
[Raw Smartphone Photo]
        │
        ▼
[Stage 1: Adaptive Illumination & Color Normalization] (Reinhard L*a*b*)
        │
        ▼
[Stage 2: Real-Time Edge Quality Assurance Gate] (Laplacian Var ≥ 120, Glare < 8%)
        │
        ▼
[Stage 3: Deep Learning Lesion Localization] (MobileNetV3 HUD, 41.8 ms, mIoU = 0.5279)
        │
        ▼
[Stage 4: Lesion Isolation & ROI Extraction] (Discards surrounding dental/facial shortcuts)
        │
        ▼
[Stage 5: Multimodal Feature Engineering] (16D compact latent representation for QML)
        │
        ▼
[Stage 6: Multi-Family Classification] (Calibrated Gradient Boosting + PennyLane Circuits)
        │
        ▼
[Stage 7: Probabilistic Calibration & Triage] (Platt Scaling, Brier = 0.1106, ABDM FHIR R4)
```

---

## 🚀 Quickstart & Reproduction Runbook

### 1. Launch Interactive Web Preview (Zero Install)
Open [`orion-workspace/index.html`](orion-workspace/index.html) in any modern browser to test the interactive camera simulation, blur filters, bounding box HUD, and calibrated risk gauges.

### 2. Run FastAPI Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Start asynchronous REST API server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API docs will be available at `http://127.0.0.1:8000/docs`.

### 3. Run Flutter Mobile Client
```bash
cd carescan
flutter pub get
flutter run -d chrome   # Or: flutter run -d android
```

### 4. Execute Full Verification Test Suite
```bash
# Run 1,038 backend tests
pytest tests/

# Run 193 mobile client tests
cd carescan && flutter test
```
**Total: 1,231 / 1,231 tests passing (100% pass rate).**

---

## 📜 Regulatory & ABDM Interoperability

Orqis is architected from the ground up for medical device translation:
* **CDSCO Medical Device Rules 2017:** Form MD-14 compliant investigational Software as a Medical Device (SaMD Class B/C).
* **Ayushman Bharat Digital Mission (ABDM):** Direct linking with 14-digit ABHA IDs.
* **HL7 FHIR Release 4:** Automated generation of `DiagnosticReport`, `Observation`, and `Media` bundles.
* **Standard Clinical Ontologies:** SNOMED CT (`371569005` Oral Exam, `254580003` Leukoplakia) and ICD-11 (`2B60`, `DA01`).

---

## 📄 License & Attribution

Developed by **Team BraKet 3.1.0** for Smart India Hackathon 2026. All source code is licensed under the Apache License 2.0. Clinical research datasets are governed by institutional DUAs.
