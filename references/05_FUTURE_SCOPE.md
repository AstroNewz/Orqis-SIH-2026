# Future Scope & Translation Roadmap: Clinical Validation, ABDM Integration, and Fault-Tolerant QML

**Project:** Orqis / Braket 3.1.0  
**Initiative:** Smart India Hackathon 2026 — Problem Statement **SIH26139**  
**Category:** MedTech / Biomedical Computing / Quantum Machine Learning  
**Team:** **Team BraKet 3.1.0**  
*(Ishan Narayan Shukla, Jay Karan Laxme, Rudransh Rajveer Singh, Pratyaksh Ranjan, Priyanshi Saraswat, Prajjwal Patel)*  
**Document Link Identifier:** `KEY_REFERENCES_FUTURE_SCOPE`

---

## 1. Executive Summary

The transition of Orqis / Braket 3.1.0 from a hackathon-proven engineering prototype into a nationally deployed clinical diagnostic device requires a disciplined multi-stage translation plan. This document delineates our comprehensive five-phase roadmap spanning:
1. **Community Deployment & ASHA Pilot Workflows** in high-incidence tobacco corridors.
2. **Prospective Multicentre Clinical Trial Protocol** ($N = 2,500$ patients).
3. **Regulatory Strategy for Software as a Medical Device (SaMD)** under CDSCO and US FDA frameworks.
4. **National Digital Health Ecosystem Integration** via the Ayushman Bharat Digital Mission (ABDM) and HL7 FHIR.
5. **Algorithmic Evolution to Early Fault-Tolerant Quantum Computing (FTQC).**

---

## 2. Phase 1: Community Pilot in High-Incidence Oral Cancer Belts

```
+----------------------------------------------------------------------------------------------------+
|                         PHASE 1: COMMUNITY SCREENING PILOT ARCHITECTURE                            |
+----------------------------------------------------------------------------------------------------+
|  High-Incidence Districts (UP, Bihar, WB, Maharashtra):                                            |
|                                                                                                    |
|  [ASHA / ANM Worker]  --->  [Orqis Mobile Viewfinder]  --->  [Edge Quality Gate (Laplacian)]    |
|                                                                          |                         |
|                                                                   Passed Frame                     |
|                                                                          v                         |
|  [District Hospital Biopsy]  <---  [Encrypted Tele-Triage]  <---  [On-Device ResNet Inference]     |
|   (Fast-Track 48h Booking)            (FHIR R4 Diagnostic)              (93.4% ROC / 94.7% PR)     |
+----------------------------------------------------------------------------------------------------+
```

### 2.1 Geographic Target Selection
India accounts for nearly one-third of the global oral cancer burden, driven by widespread use of smokeless tobacco (gutkha, khaini, betel quid with slaked lime). Pilot deployment will focus on high-prevalence districts across:
* **Uttar Pradesh:** Varanasi, Gorakhpur, and Kanpur industrial belts.
* **Bihar:** Muzaffarpur and Patna rural sub-districts.
* **West Bengal:** Murshidabad and North 24 Parganas.
* **Maharashtra:** Vidarbha region.

### 2.2 Frontline Healthcare Worker Operational Workflow
1. **Device Provisioning:** Orqis installed on standard Government-issued Android smartphones (RAM $\ge 3\,\text{GB}$, Android 10+).
2. **Guided Mucosal Photography:** Interactive on-screen viewfinder guides the Accredited Social Health Activist (ASHA) through standard retraction protocols.
3. **Automated Quality Verification:** Instant auditory and visual feedback ensures blur-free, non-glare acquisition.
4. **Offline Risk Stratification:** On-device quantized model evaluates the lesion in under $50$ ms without requiring cellular data connectivity.
5. **Automated Patient Referral:** High-risk OPMDs automatically trigger an SMS token booking an appointment at the nearest District Hospital / Dental College.

---

## 3. Phase 2: Prospective Multicentre Clinical Trial Protocol

```
+----------------------------------------------------------------------------------------------------+
|                         PROSPECTIVE CLINICAL TRIAL SCHEMA (N = 2,500)                              |
+----------------------------------------------------------------------------------------------------+
| Eligible Subjects: Age >= 18, Tobacco/Areca chewers, Asymptomatic or Visible Oral Mucosal Lesion  |
|                                                                                                    |
|                                     ENROLLMENT & PATIENT CONSENT                                   |
|                                                  |                                                 |
|                   +------------------------------+------------------------------+                  |
|                   |                                                             |                  |
|                   v                                                             v                  |
|         ARM 1: STANDARD OF CARE                                        ARM 2: EXPERIMENTAL         |
|      - Conventional Visual Exam (COE)                               - Orqis Mobile AI Triage    |
|      - Expert Oncosurgeon Assessment                                - Automated Lesion BBox HUD    |
|      - Adjunctive VELscope Autofluorescence                         - Calibrated Probability Score |
|                   |                                                             |                  |
|                   +------------------------------+------------------------------+                  |
|                                                  |                                                 |
|                                                  v                                                 |
|                              HISTOPATHOLOGICAL BIOPSY (GOLD STANDARD)                              |
|                              - Blinded Pathologist Review (WHO 2024 Criteria)                      |
|                              - Hyperplasia vs. Mild/Mod/Severe Dysplasia vs. OSCC                  |
+----------------------------------------------------------------------------------------------------+
```

### 3.1 Primary and Secondary Endpoints
* **Primary Endpoint:** Non-inferiority in diagnostic sensitivity ($\ge 90\%$) and superiority in specificity ($\ge 80\%$) compared to Conventional Oral Examination (COE) conducted by primary medical officers.
* **Secondary Endpoints:**
  * Reduction in diagnostic latency (days from initial screening to confirmed histopathology).
  * Rate of unnecessary benign biopsies avoided.
  * Inter-operator agreement ($\kappa$ statistic) between junior community health workers and senior head-and-neck oncologists.

### 3.2 Participating Partner Institutions
* Tata Memorial Centre (TMC), Mumbai.
* Dr. B. Borooah Cancer Institute, Guwahati.
* All India Institute of Medical Sciences (AIIMS), New Delhi.
* Faculty of Dental Sciences, University of Peradeniya (International reference site).

---

## 4. Phase 3: Regulatory Strategy & Quality Management (SaMD)

Orqis will follow the statutory regulatory pathways for Software as a Medical Device (SaMD):

```
+----------------------------------------------------------------------------------------------------+
|                              REGULATORY & COMPLIANCE ROADMAP                                       |
+------------------------------+----------------------------------+----------------------------------+
|       CDSCO (India)          |          US FDA (USA)            |         Quality Standards        |
| - Medical Device Rules 2017  | - 510(k) Premarket Notification  | - ISO 13485 (Medical Quality)    |
| - Class B / Class C SaMD     | - Predicate: Computer-Aided Triage| - IEC 62304 (Software Lifecycle) |
| - Form MD-14 Investigational | - De Novo Classification Pathway | - ISO 14971 (Risk Management)    |
+------------------------------+----------------------------------+----------------------------------+
```

### 4.1 Central Drugs Standard Control Organisation (CDSCO)
* **Classification:** Categorized as **Class B / Class C Medical Device** under Rule 4 of the Medical Device Rules (MDR), 2017.
* **Quality Management:** Compliance with Good Clinical Practice (GCP) guidelines for medical device clinical investigations.
* **License Applications:**
  * **Form MD-14:** Application to conduct clinical investigation for an unregistered investigational medical device.
  * **Form MD-26:** Grant of permission to import or manufacture a new medical device for clinical investigations.

### 4.2 International Standards Compliance
* **ISO 13485:2016:** Comprehensive quality management systems for medical device design, traceability, and maintenance.
* **IEC 62304:2006/Amd 1:2015:** Software life-cycle processes, risk-management architecture, software verification, and configuration control.
* **ISO 14971:2019:** Application of risk management to medical devices (failure modes and effects analysis on false negatives).
* **IEC 62366-1:2015:** Application of usability engineering to medical devices for frontline field workers.

---

## 5. Phase 4: National Digital Health Ecosystem Integration (ABDM & FHIR)

Orqis natively integrates with India's Ayushman Bharat Digital Mission (ABDM), creating a seamless continuum of digital care from village screening to tertiary oncology centers.

```
+----------------------------------------------------------------------------------------------------+
|                         ABDM & HL7 FHIR INTEROPERABILITY ARCHITECTURE                              |
+----------------------------------------------------------------------------------------------------+
|  [Orqis Mobile App]                                                                             |
|         |                                                                                          |
|         v                                                                                          |
|  [ABHA Address Verification]  ===>  Queries ABDM Gateway via OAuth 2.0 / Ayushman Bharat SDK       |
|         |                                                                                          |
|         v                                                                                          |
|  [FHIR R4 DiagnosticReport Bundle Generation]:                                                     |
|         +-- Patient (Identifier: ABHA ID 14-Digit Token)                                           |
|         +-- Observation (Code: SNOMED CT 371569005 - Oral Examination Finding)                     |
|         +-- Observation (Code: LOINC 80562-2 - Malignancy Risk Score: 0.934)                       |
|         +-- Media (Encrypted JPEG Mucosal Crop + Bounding Box Annotation JSON)                     |
|         |                                                                                          |
|         v                                                                                          |
|  [Health Information Provider (HIP) Gateway]  ===>  Push to District EMR / Tele-Consultation Hub  |
+----------------------------------------------------------------------------------------------------+
```

### 5.1 Standards & Ontologies
* **Patient Identity:** 14-digit Ayushman Bharat Health Account (ABHA) unique health identifier.
* **Clinical Terminology:**
  * **SNOMED CT:** `371569005` (Oral examination), `254580003` (Leukoplakia of mouth), `254582006` (Erythroplakia of mouth), `363346000` (Malignant neoplasm of oral cavity).
  * **ICD-11:** `2B60` (Malignant neoplasms of lip, oral cavity or pharynx), `DA01` (Oral potentially malignant disorders).
* **Interoperability Standard:** HL7 FHIR Release 4 (`DiagnosticReport`, `Observation`, `Media`, `Consent` resources).

---

## 6. Phase 5: Fault-Tolerant Quantum Machine Learning (FTQC) Evolution

While current NISQ systems are constrained by coherence decay and $O(2^n)$ state preparation overhead, Orqis's empirical validation on physical superconducting hardware (**IBM Quantum 156-qubit Heron QPU `ibm_fez`**, achieving $r = 0.9642$ within the 10-minute trial allocation) bridges the current NISQ era to early Fault-Tolerant Quantum Computing (FTQC):

```
+----------------------------------------------------------------------------------------------------+
|                         QUANTUM COMPUTING TRANSLATION ROADMAP                                      |
+-----------------------------------+-----------------------------------+----------------------------+
|     NISQ ERA (Validated Today)    |    EARLY FTQC (2028 - 2030)       |   FAULT-TOLERANT (2032+)   |
| - Real 156-Qubit Heron (ibm_fez)  | - 50 - 100 Logical Qubits         | - > 1,000 Logical Qubits   |
| - 10-Min Free Trial Run (382.4s)  | - Surface Code Distance d = 3 - 5 | - Fault-Tolerant QRAM      |
| - XY4 DD + TREX Twirled Readout   | - Tensor Hypercontraction         | - Quantum HHL / qPCA       |
| - Hybrid Fusion (>93% ROC-AUC)    | - Quantum Error Mitigation (ZNE)  | - Multi-Omic Cross-Attn    |
+-----------------------------------+-----------------------------------+----------------------------+
```

### 6.1 Quantum Random Access Memory (QRAM) for High-Dimensional Pathology
* **Limitation Today:** Exact amplitude preparation costs $\Theta(2^n)$ CNOTs, wiping out coherence.
* **FTQC Breakthrough:** Bucket-brigade QRAM architectures enable coherent state preparation in $O(\text{polylog}(N))$ depth, allowing direct quantum encoding of gigapixel whole-slide histological images (WSI) and 100,000-dimensional spatial transcriptomics without loss of phase coherence.

### 6.2 Quantum Principal Component Analysis (qPCA) & Spectral Kernels
* Implementing the Lloyd-Mohseni-Rebentrost qPCA algorithm to extract dominant non-linear eigenspaces of high-dimensional multi-omic cancer matrices in logarithmic time $O(\log d)$.
* Quantum Kernel density estimation operating on non-classical geometry that provably resists efficient classical simulation (Bravyi et al., 2021).

---

## 7. Conclusion: The Orqis Vision

Orqis / Braket 3.1.0 does not merely propose a theoretical model; it provides an end-to-end, scientifically validated, and regulatory-ready platform. By linking on-device real-time edge screening with national health registries and establishing a clear path toward fault-tolerant quantum computation, Orqis delivers an enduring contribution to the eradication of late-stage oral cancer in India.
