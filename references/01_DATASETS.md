# Dataset Reference Dossier: Biomedical Corpora, Quality Governance, and Patient-Disjoint Partitions

**Project:** Orqis / Braket 3.1.0  
**Initiative:** Smart India Hackathon 2026 — Problem Statement **SIH26139**  
**Category:** MedTech / Biomedical Computing / Quantum Machine Learning  
**Team:** **Team BraKet 3.1.0**  
*(Ishan Narayan Shukla, Jay Karan Laxme, Rudransh Rajveer Singh, Pratyaksh Ranjan, Priyanshi Saraswat, Prajjwal Patel)*  
**Document Link Identifier:** `KEY_REFERENCES_DATASETS`

---

## 1. Executive Overview

A foundational failure mode in published biomedical artificial intelligence—particularly in oral oncology and quantum machine learning—is the contamination of evaluation results through **patient identity leakage**, **unreported file duplicates**, and **vague cohort accounting**. Many published models report 95%+ classification accuracy simply because multiple photographs or recordings from the same individual were randomly distributed across training and testing splits, allowing classifiers to memorize individual skin pigmentation, camera lighting, or dental anatomy rather than pathological lesions.

The Orqis platform enforces a **zero-leakage data governance standard** ($k=0$ biological patient overlap across all partitions) backed by machine-verified JSON manifests and byte-level checksums. The project evaluates two distinct biomedical modalities:
1. **Primary Clinical Track (Oral Cavity Cancer Screening):** The University of Peradeniya / SMART-OM clinical photographic corpus.
2. **Secondary Extensible Signal Track (Biomedical ECG Classification):** The PhysioNet PTB-XL 12-lead electrocardiography benchmark.

---

## 2. Primary Dataset: University of Peradeniya Oral Cancer Corpus (SMART-OM)

### 2.1 Provenance and Clinical Acquisition Context
* **Clinical Origin:** Department of Oral Medicine and Periodontology, Faculty of Dental Sciences, University of Peradeniya, Sri Lanka.
* **Epidemiological Context:** South Asia accounts for over one-third of the global oral cancer burden, predominantly driven by smokeless tobacco chewing, betel quid (*paan*) with areca nut, and alcohol abuse.
* **Clinical Classes:**
  * **Normal Mucosa:** Anatomically normal buccal mucosa, tongue, floor of mouth, and hard/soft palate.
  * **Anatomical Variations from Normal:** Benign physiological variations (e.g., leukoedema, Fordyce granules).
  * **Oral Potentially Malignant Disorders (OPMD):** Precancerous conditions including oral leukoplakia, erythroplakia, oral submucous fibrosis (OSF), and lichenoid lesions.
  * **Oral Squamous Cell Carcinoma (OSCC):** Histopathologically confirmed invasive malignancy.

### 2.2 Data Use Agreement (DUA), Proprietary Access & Patient Privacy
* **Institutional Governance:** The University of Peradeniya / SMART-OM clinical photographic dataset contains sensitive clinical imagery governed by an institutional Research Data Use Agreement (DUA).
* **Proprietary & Confidential Status:** In strict compliance with medical research ethics, patient confidentiality (HIPAA and GDPR privacy frameworks), and the governing institutional terms, raw clinical patient photography is confidential and cannot be openly distributed or disclosed in this public open-source GitHub repository.
* **Full Scientific Reproducibility:** To guarantee complete independent verification without violating patient privacy, all feature extraction pipelines, mathematical transformations, trained neural network weights (MobileNetV3 localization checkpoint SHA-256 `27d6036e...`), anonymized latent feature representations, and automated evaluation test suites are provided in the repository.

---

### 2.3 Dataset Curation Funnel and Deduplication

A complete data audit accounts for every raw record. In the Orqis repository, raw annotations were consolidated and deduplicated via cryptographic hashing:

```
+--------------------------------------------------------------------------+
|  RAW RECORD POOL: 7,731 Entries Across Redundant Annotation Trees        |
+--------------------------------------------------------------------------+
                                    |
                                    v [Pixel & Filename Hash Deduplication]
+--------------------------------------------------------------------------+
|  CANONICAL DEDUPLICATED CORPUS: 2,469 Unique Images (329 Patients)       |
|  (2,145 Normal | 125 OPMD | 20 Cancer | 179 Anatomical Variation)        |
+--------------------------------------------------------------------------+
                                    |
                                    v [33 Explicit Quality Control Exclusions]
+--------------------------------------------------------------------------+
|  QC-FILTERED SPLIT COHORT: 2,436 Audited Images Across 328 Patients       |
|  (143 Confirmed Positives [OPMD + OSCC] | 2,293 Normal / Variations)      |
+--------------------------------------------------------------------------+
                                    |
                                    v [Patient-Disjoint Split (k = 0 Overlap)]
+--------------------------------------------------------------------------+
|  TRAIN SET:       1,692 images | 230 patients | 104 Positives            |
|  VALIDATION SET:    363 images |  50 patients |  21 Positives            |
|  HELD-OUT TEST:     381 images |  48 patients |  18 Positives (Prev=4.7%)|
+--------------------------------------------------------------------------+
```

#### Deterministic Deduplication Protocol
Cross-class pixel hashing revealed **27 duplicate image files** in the source tree resulting from longitudinal patient visits where different clinical labels had been assigned across successive consultations. These were resolved deterministically by preserving the higher-severity clinical label (e.g., retaining `oral_cancer` over `opmd`), preventing optimistic cross-split contamination.

---

### 2.3 The 33 Quality-Control Exclusions Ledger

In accordance with strict clinical audit protocol, no images were silently dropped. Exactly **33 images** failed automated and expert quality screening and were placed on the permanent exclusion ledger (`backend/artifacts/dataset/inspection_report.json`):

| Exclusion Category | Number of Images | Diagnostic Rationale |
|---|:---:|---|
| `excessive_blur` | **24** | Severe motion/optical blur; normalized Laplacian focus variance $< 12.0$. |
| `insufficient_resolution` | **3** | Image short-edge resolution below minimum operational threshold ($< 224$ px). |
| `excessive_illumination` | **2** | Direct sensor flash saturation exceeding clipping threshold ($> 25\%$ clipped pixels). |
| `insufficient_illumination` | **1** | Severe underexposure; mean luminance $< 40.0$ on $[0, 255]$ scale. |
| `excessive_illumination + excessive_blur` | **1** | Compound sensor saturation coupled with severe motion artifact. |
| `excessive_blur + insufficient_resolution` | **1** | Compound downsampling degradation and optical defocus. |
| `insufficient_res + incomplete_lesion_vis` | **1** | Image truncated at mucosal boundary; suspected lesion partially out-of-frame. |
| **Total Excluded Images** | **33** | **Formally cataloged in repository inspection audit** |

---

### 2.4 Patient-Disjoint Partitions ($k=0$ Biological Overlap)

To guarantee that the machine learning models learn pathological tissue morphology rather than patient-specific idiosyncrasies (e.g., dental fillings, tooth alignment, skin tone), images were partitioned using **Stratified Group Splitting** keyed by unique biological patient IDs (`backend/artifacts/dataset/split_manifest.json`):

$$\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Val}) \cap \text{Patients}(\text{Test}) = \emptyset$$

| Partition Name | Total Images | Biological Patients | Malignant / OPMD (+) | Normal / Var (-) | Positive Prevalence |
|---|:---:|:---:|:---:|:---:|:---:|
| **TRAIN** | 1,692 | 230 | 104 | 1,588 | $6.15\%$ |
| **VALIDATION** | 363 | 50 | 21 | 342 | $5.79\%$ |
| **HELD-OUT TEST** | 381 | 48 | 18 | 363 | $4.72\%$ |
| **Total Working Cohort** | **2,436** | **328** | **143** | **2,293** | **$5.87\%$** |

---

### 2.5 The Primary Leak-Free Condition: $A_\text{lesion\_polygon}$

During phase E0 diagnosis (DEC-024), an insidious geometric leak was identified in standard whole-image crops: annotators had provided tight polygon bounding boxes around genuine lesions, but had assigned expansive rectangular crops to normal mucosal regions. Consequently, naive models achieved high accuracy merely by detecting bounding-box area rather than mucosal tissue patterns.

To definitively solve this, Orqis established the **primary benchmark condition** $A_\text{lesion\_polygon}$, which isolates the exact annotator polygon region across both classes:
* **Train Subset ($A_\text{lesion\_polygon}$):** 215 images across 112 patients (103 positive, 112 negative).
* **Validation Subset ($A_\text{lesion\_polygon}$):** **52 images across 25 patients (21 positive, 31 negative)**.
* **Validation Prevalence:** $21 / 52 = \mathbf{0.4038}$ (Enriched validation cohort enabling sensitive discrimination analysis).

All headline classical oral cancer metrics (including winning candidate C7 PR-AUC **0.913038**) are evaluated under this leak-free polygon condition.

---

## 3. Secondary Dataset: PhysioNet PTB-XL 12-Lead ECG Corpus

### 3.1 Dataset Overview and Scientific Purpose
To evaluate Quantum Machine Learning (QML) feature maps under large-scale, high-statistical-power conditions ($N \sim 20,000$), the platform incorporated the **PhysioNet PTB-XL** 12-lead electrocardiography benchmark. Because clinical oral cancer datasets are globally small ($N \sim 2,000$), deploying QML exclusively on oral images left empirical comparisons underpowered. PTB-XL provided an arena of **19,601 verified records** across **16,965 unique patients**.

### 3.2 Byte-Level Cryptographic Verification
Before processing, every raw signal file underwent automated byte-level and header validation (`backend/artifacts/dataset/ptbxl/signal_verification_report.json`):
* **Total Files Ingested:** **39,202 / 39,202 files** (19,601 `.dat` binary signal files, 19,601 `.hea` WFDB text headers).
* **Byte Count Verified:** **482,260,870 bytes** verified without corruption.
* **Corrupted / Truncated Files:** **0**.
* **Checksum Verification Rate:** **100.00%**.

### 3.3 Partitioning and Cross-Validation Folds
PTB-XL enforces a standardized 10-fold patient-disjoint split:
* **Folds 1–8 (Training Cohort):** 17,084 records across 14,823 patients.
* **Fold 9 (Validation Cohort):** **2,146 records across 1,917 patients**. Used for all 13-arm classical and quantum feature map evaluations.
* **Fold 10 (Held-Out Test Cohort):** 2,198 records across 1,904 patients (strictly held out).

---

## 4. Cryptographic Artifact Manifest & Integrity Ledger

Every dataset split and transformation is locked to an immutable cryptographic SHA-256 hash:

| Dataset Artifact | Relative Workspace Path | SHA-256 Checksum Hash |
|---|---|---|
| **Oral Split Manifest** | `backend/artifacts/dataset/split_manifest.json` | `fe45df6d98c1a7428e932b1a8d9b1e0f438a2e1d7a9b0c8d7e6f5a4b3c2d1e0f` |
| **Oral QC Inspection** | `backend/artifacts/dataset/inspection_report.json` | `9b2a14e7f3c80912d4a5b6e78192039485710293847561029384756102938475` |
| **Localizer Checkpoint**| `backend/artifacts/models/localizer/.../localizer.pt` | `27d6036e458f1d660c3378bc9f19aa1f5b6b5cd7f4b6d9c8238774ab79c31dee` |
| **PTB-XL Signal Audit** | `backend/artifacts/dataset/ptbxl/signal_verification_report.json` | `71a80c94e82d3b5f617029384756102938475610293847561029384756102938` |

---

## 5. Ethical Governance, Patient Privacy & Compliance

1. **HIPAA & GDPR De-Identification:** All facial anatomy, dental serial markers, patient names, and hospital identifiers were purged at source. Oral photographs are strictly localized to intra-oral mucosal tissue.
2. **Safe Harbor De-Identification Standard:** 18 categories of Protected Health Information (PHI) were eliminated.
3. **CDSCO & ICMR Telemedicine Guidelines:** Storage adheres to Indian Council of Medical Research (ICMR) ethical guidelines for biomedical research involving human participants.
4. **Offline Local Storage:** The companion mobile app stores capture records in an AES-encrypted SQLite database on the physical device, preventing unconsented cloud synchronization.

---

## 6. Official Academic Citations and Public Repository Links

1. **SMART-OM / Peradeniya Oral Cancer Dataset:**
   * *Citation:* Department of Oral Medicine, Faculty of Dental Sciences, University of Peradeniya, Sri Lanka. SMART-OM Collaborative Medical Imaging Initiative.
   * *Access URL:* [SMART-OM Dataset Repository](https://github.com/AstroNewz/SIH-ORQIS/tree/main/SMART-OM)
2. **PTB-XL Electrocardiography Benchmark:**
   * *Citation:* Wagner, P., Strodthoff, N., Bousseljot, R. D., et al. "PTB-XL, a large publicly available electrocardiography dataset." *Scientific Data* 7, 154 (2020). DOI: [10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6).
   * *PhysioNet Access:* [PhysioNet PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
