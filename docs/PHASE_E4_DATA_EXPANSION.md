# Phase E4 — Data Expansion for Quantum-Advantage Research: Choosing a Better Scientific Arena

**Status:** Candidate audit (FIRST DELIVERABLE) — **PTB-XL selected and locally audited.** Selection criteria pre-registered *before* any modeling. No model has been trained on any candidate; this document ranks datasets by **scientific suitability for discovering a genuine quantum contribution**, never by a preliminary score. The PTB-XL metadata has been downloaded, **checksum-verified against PhysioNet's `SHA256SUMS.txt`**, and audited by the committed `backend/dataset/ptbxl.py` — the audited figures are in §7.1.

**Why this phase exists.** Phase E3 exhausted four qualitatively distinct quantum families — fixed feature map + fusion (DEC-037), fidelity quantum kernel / QSVM (DEC-038), trainable shallow VQC (DEC-039), richer Havlicek ZZ map + structured observables (DEC-040) — on the SMART-OM oral-cancer set. All four nulled against the classical baseline C7 (PR-AUC **0.913038**) under identical patient-blocked evaluation; the richest quantum map was *worse* than its matched classical control. The binding constraint looked like **data scale / task information**, not circuit design: the development set is only **215 images / 103 positives / ~124 patients**, and compressing megapixel images down to 8 qubits discards exactly the structure a quantum map might exploit. E4 asks a different question: **is there a legitimate, richer biomedical dataset that is a better arena to discover a real, statistically-defensible quantum-enhanced representation?** The target is scientific defensibility, not "a pretty number."

---

## 1. Pre-registered selection criteria (fixed before any training — anti-metric-shopping)

A candidate is judged on these, decided *now*, before a single model is fit. This is the guard against dataset shopping (train → bad → discard → train → good → declare victory):

1. **Legitimacy & provenance** — official source, peer-reviewed or institutional release, clear licence. No scraped/unclear-provenance data; no synthetic data as the main benchmark.
2. **Patient/subject identifiers** — must exist, to enforce patient-level train/validation/test separation (the E-series' mandatory leakage rule).
3. **Sample size & positives** — enough positives and patients to support a *frozen* patient-level test partition with non-trivial confidence intervals (materially more than SMART-OM's 103 positives).
4. **Reproducible programmatic download** — scriptable, checksummed, version-pinned; another machine can reconstruct it.
5. **Class labels reliable** — real clinical labels, known label-noise characterised.
6. **Not saturated** — a dataset where the best classical model already sits at ≈99% leaves no head-room to demonstrate quantum complementarity and is a poor arena.
7. **Quantum suitability** — how naturally the data yields a *compact* (≈8–16 feature) representation. Near-term quantum kernels/maps operate in tiny Hilbert spaces; **naturally low-dimensional, clinically-meaningful feature vectors are a far better fit than megapixel images that must be crushed to 8 dims** (the exact bottleneck E3 hit).
8. **Licence permits research use & redistribution of derived features**, ideally with **no credentialing/DUA barrier** (open > gated, for reproducibility).
9. **Relevance to the official problem** — biomedical *early disease detection*, compatible with classical preprocessing → feature engineering → quantum ML → prediction → explainability → benchmarking.

**Selection rule:** pick the candidate that best satisfies 1–9 *as a benchmark*. The preliminary metric a model happens to produce is explicitly **not** a selection input.

---

## 2. Candidate comparison table (top 5)

Provenance legend: **✓** = confirmed this session from the dataset's own primary source (URL in §6); **○** = well-established figure from prior knowledge, **to be confirmed by the committed audit script at download time** (not yet independently re-verified this session due to an environment web/quota constraint — see §7).

| # | Dataset | Modality / task | Samples | Patients | Positives / balance | Patient IDs | Licence | Access / download | Quantum suitability | Key risk |
|---|---------|-----------------|---------|----------|---------------------|-------------|---------|-------------------|---------------------|----------|
| 1 ⭐ | **PTB-XL** | 12-lead ECG; diagnostic super/subclasses (NORM/MI/STTC/CD/HYP), reducible to binary (e.g. MI-vs-NORM, normal-vs-abnormal) | **21,799 records** ✓ | **18,869** ✓ | multi-label; NORM 9,514 (43.6%) ✓, MI 5,469 ✓, STTC 5,235 ✓, CD 4,898 ✓, HYP 2,649 ✓ | **Yes** — `patient_id` + `strat_fold` 1–10, **verified patient-disjoint** ✓ | **CC BY 4.0** ✓ | **Open, no credentialing** ✓; `wget`/`wfdb` from PhysioNet ✓ | **Highest** — ECG → compact morphological/HRV/spectral feature vectors, natively 8–32 dims; no image-crush | Multi-label; 411 unlabeled records; SCP-code label noise; needs a signal-feature pipeline |
| 2 | **SIIM-ISIC 2020 Melanoma** | Dermoscopy; binary malignant/benign | **33,126 train imgs** ✓ | ~2,056 ○ | ~584 malignant, ~1.8% ○ (severe imbalance) | **Yes** — `patient_id` in metadata ✓ | **CC BY-NC** ✓ | Open S3 download ✓ (JPEG 23 GB / DICOM 48.9 GB) ✓ | Moderate — images need CNN→PCA compression (same regime as E3) | Extreme class imbalance; reuses the very bottleneck E3 hit |
| 3 | **NIH ChestX-ray14** | Frontal chest radiograph; 14 findings, multi-label | ~112,120 ○ | ~30,805 ○ | 14 findings, varied prevalence ○ | **Yes** — `Patient ID` + official patient-wise split ○ | Open (NIH terms) ○ | Open (NIH Box / Kaggle) ○ (~42 GB) | Moderate — image compression required | NLP-mined label noise; large download |
| 4 | **BreakHis** | Breast histopathology (H&E, 40–400×); binary benign/malignant | **7,909 imgs** ✓ | **82** ✓ | **2,480 benign / 5,429 malignant** ✓ | **Yes** — encoded in filename ✓ | **CC BY 4.0** (non-commercial research) ✓ | Direct `tar.gz` + form ✓ (~4 GB) | Moderate — image compression required | Only **82 patients** → thin patient-level test; magnification confound |
| 5 | **MedMNIST v2** (e.g. PneumoniaMNIST, BreastMNIST) | Standardized 28×28 biomedical images; per-subset binary/multiclass | subset-dependent (Pneumonia 5,856 ○; Breast 780 ○) | mostly **not exposed** ○ | subset-dependent ○ | **Weak** — fixed official splits, patient IDs generally not surfaced ○ | Apache-2.0 code; subsets inherit (mostly CC BY) ○ | `pip install medmnist` ○ (tiny) | High for *reproducibility* (already 28×28, QML-benchmarked) but low res loses detail | No patient-level grouping → cannot enforce our leakage rule cleanly |

**Also considered (not in top 5):** *PatchCamelyon* (327,680 H&E patches ○, CC0 ○, `tfds` ○ — excellent reproducibility and used in QML, but patient/WSI IDs are stripped at patch level, so patient-level splitting needs mapping back to Camelyon16); *Wisconsin Diagnostic Breast Cancer* (569×30 ○ — the QML tabular classic, but tiny, no patient grouping, and classical models already reach ~97–98% → **saturated**, fails criterion 6); *CheXpert / MIMIC-CXR* (very large, but **registration/credentialing/DUA gated** → fails the open-reproducibility preference).

---

## 3. Ranking & recommendation

**Recommended primary arena: PTB-XL (⭐).** It is the only top candidate that satisfies *all nine* criteria, and it is strongest on the two that matter most for this specific mission — **quantum suitability** and **open reproducibility**:

1. **It escapes the image→8-dim compression bottleneck.** The single clearest lesson of E3 is that crushing a megapixel dermoscopy image into 8 amplitudes destroys the structure a small quantum map could use. ECG is different: standard signal processing yields **compact, clinically-meaningful feature vectors** (QRS morphology, ST/T deviations, intervals, HRV, wavelet/spectral coefficients) that are *natively* in the 8–32 dimensional regime a near-term quantum kernel/feature-map actually operates in. The quantum representation stops being a lossy afterthought and becomes a first-class encoding.
2. **~180× more patients.** 18,869 patients vs SMART-OM's ~124 means a genuinely large **frozen patient-level test** partition and tight bootstrap CIs — the statistical power E3 lacked.
3. **Fully open, CC BY 4.0, no credentialing** (✓ verified) → the reproducible, checksummed, version-pinned download the mission requires, with zero DUA friction.
4. **Ships a patient-respecting 10-fold split** (`strat_fold`) *and* raw `patient_id`, so patient-level separation is native, not reconstructed.
5. **Multi-label structure = several binary tasks** (MI-vs-normal, conduction-disturbance detection, normal-vs-abnormal), giving multiple independent chances to detect quantum complementarity — and cardiac abnormality detection is squarely "early disease detection."

**Secondary (image-pipeline reuse): SIIM-ISIC 2020.** If we want to exercise the *existing* CareScan image → MobileNet → localization stack on a far larger, patient-ID'd set, ISIC 2020 is the best-licensed, most-reproducible option. Caveat: it re-enters the exact image-compression regime E3 already found unfavourable for quantum, and its ~1.8% positive rate is punishing.

**Tertiary (fast reproducible sanity check): MedMNIST.** Trivially reproducible and already used in QML papers, so it is a good *smoke test* for the quantum infrastructure — but its lack of patient-level grouping disqualifies it as the primary scientific benchmark under our leakage rule.

### Why PTB-XL is a more promising arena than SMART-OM for the quantum objective
SMART-OM is small (103 positives), image-only (forcing lossy 8-dim compression), and already has a strong classical baseline (0.913) that four quantum families could not beat or complement. PTB-XL is two orders of magnitude larger in patients, is natively low-dimensional after feature extraction (the regime where near-term quantum methods have their best theoretical shot), is fully open and reproducible, and offers multiple binary sub-tasks. It does **not** guarantee a quantum advantage — nothing does — but it removes the two most likely *confounds* for E3's null (too little data, and a representation forced through an image bottleneck), which is exactly what "a better scientific arena" means.

---

## 4. Test policy for the new dataset (frozen from the start)
- **New independent frozen TEST partition**, patient-disjoint. For PTB-XL the convention is `strat_fold` 1–8 = train, 9 = validation, 10 = test; the committed audit **verifies no `patient_id` spans folds** before the split is trusted, and re-derives a patient-level split with the existing `backend/dataset/split.py` machinery if the shipped folds ever violate it.
- **All** PCA / feature selection / normalization / thresholds / hyperparameters / architecture selection are **TRAIN (or TRAIN+VALIDATION) only.** The test fold is read once, after an architecture is frozen — never for tuning or repeated inspection.
- Duplicate / near-duplicate audit across partitions where practical (PTB-XL: exact `patient_id` collision check; signal near-duplicate audit is a follow-up).

## 5. Data handling & reproducibility
- Raw data lives under the **gitignored** `backend/artifacts/dataset/ptbxl/` (matches the existing `backend/artifacts/dataset/` ignore rule); **large raw datasets are never committed.**
- A reproducible **acquisition + audit** module (`backend/dataset/ptbxl.py`) records source URL, version, download date, per-file checksum (against PhysioNet `SHA256SUMS.txt`), expected vs actual file counts, and emits a formal **audit report** (record/patient counts, class distribution, missing labels, patient-fold collision check, demographic summary).
- **Modeling order is enforced:** dataset audit → classical failure map (simple / feature-engineered / deep-feature / best-reasonable fusion) → *only then* quantum research. No quantum model is built before the classical baseline exists.

## 6. Sources (primary, consulted this session)
- **PTB-XL** — PhysioNet: <https://physionet.org/content/ptb-xl/> — "21799 clinical 12-lead ECGs", "18869 patients", "Creative Commons Attribution 4.0 International Public License", open access, 500 Hz + 100 Hz, superclasses NORM/MI/STTC/CD/HYP. *(✓ quoted verbatim this session.)*
- **SIIM-ISIC 2020** — ISIC Challenge: <https://challenge.isic-archive.com/data/2020/> — "33,126 JPEG images", metadata contains "patient ID, lesion ID, sex, age, and general anatomic site", licence "CC-BY-NC", S3 download (JPEG 23 GB / DICOM 48.9 GB). *(✓ this session; patient count ~2,056 and malignant ~584 are commonly-cited ○ figures not present on that page.)*
- **BreakHis** — UFPR Vision Lab: <https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/> — "7,909" images (breakdown table; intro's "9,109" is an internal inconsistency to verify), "82 patients", "2,480 benign and 5,429 malignant", magnifications 40–400×, patient ID encoded in filename, "Creative Commons Attribution 4.0 International License", `BreaKHis_v1.tar.gz`. *(✓ this session.)*
- **NIH ChestX-ray14 / MedMNIST / PatchCamelyon / WDBC** — figures from prior knowledge (○); to be confirmed against their primary sources (NIH Clinical Center release; medmnist.com; PCam GitHub; UCI ML Repository) when the environment web budget permits.

## 7. Verification status & environment constraints (full disclosure)
- **Primary-source verified this session (✓):** PTB-XL core figures + licence + open access; ISIC 2020 image count + `patient_id` field + licence + download; BreakHis image/patient/class counts + patient-ID-in-filename + licence.
- **Knowledge, verify-pending (○):** ISIC patient/malignant counts; all NIH ChestX-ray14 figures; PCam; MedMNIST subset counts; WDBC. These are flagged in-table and will be confirmed by the acquisition/audit scripts (for the selected dataset, the audit *is* the verification).
- **Environment limits encountered:** the five parallel research sub-agents launched for this deliverable each terminated on an inference-quota error (403 pre-consume quota; sub-agent spawns need ~$0.26–0.39 of quota that was unavailable), and **WebSearch is not provisioned for sub-agents** in this deployment. Main-session `WebFetch` works but the network to PhysioNet is slow (~33 KB/s), so the PTB-XL metadata was downloaded in the background; **it has now completed and been audited (§7.1).** No figure in this document is fabricated: every number is either ✓ primary-verified or ○ explicitly labelled knowledge-pending.

## 7.1 PTB-XL local audit results (verified this session)
Produced by `python -m backend.dataset.ptbxl --root backend/artifacts/dataset/ptbxl` and locked as a regression test in `tests/test_ptbxl_dataset.py::test_real_ptbxl_audit_matches_published_figures`:

| Audited quantity | Value | Note |
|---|---|---|
| Dataset version | **1.0.3** | from `VERSION.txt` |
| `ptbxl_database.csv` SHA-256 | **OK** — `7600de9c…6859d216b` | matches PhysioNet `SHA256SUMS.txt` exactly |
| Records | **21,799** | matches published figure |
| Unique patients | **18,869** | matches published figure |
| Max studies / patient | **10** | most patients have 1 |
| Missing age / missing sex | **0 / 0** | complete demographics (sex 0: 11,354; sex 1: 10,445) |
| Unlabeled records (no diagnostic superclass) | **411** | must be excluded or handled explicitly in any binary task |
| Superclass record counts (multi-label) | NORM 9,514 · MI 5,469 · STTC 5,235 · CD 4,898 · HYP 2,649 | sums exceed 21,799 (records can carry several) |
| Frozen partitions (strat_fold 1–8 / 9 / 10) | train 17,418 rec / 15,023 pt · val 2,183 / 1,942 · test 2,198 / 1,904 | authors' recommended split |
| **Patient-level leakage across folds** | **NONE (`patient_leakage_free: True`)** | re-derived patient→partition from record assignments; **no patient spans folds** → the frozen test fold is safe |

**Interpretation.** Integrity (checksum), scale (21,799 / 18,869), completeness (0 missing demographics), and — the decisive one — **patient-disjoint folds** are all confirmed on the real file, not asserted from the website. The 411 unlabeled records are the one genuine caveat surfaced by the audit and will be handled explicitly (excluded from, or held out of, any binary label) rather than silently dropped. The machine-readable report is written to the gitignored `backend/artifacts/dataset/ptbxl/audit_report.json`.

## 7.2 Signal acquisition — bounded 100 Hz sample (verified this session)
The full `records100` tree is ~1.7 GB, which at the measured ~33 KB/s link is impractical to fetch in one pass. A **bounded 200-record sample** was therefore acquired first, so the reader and feature pipeline could be built and validated against real signals while the bulk download remains a separate, resumable step.

| Audited quantity | Value |
|---|---|
| Records requested / fully downloaded | **200 / 200** |
| Files on disk (`.hea` + `.dat`) | **400 / 400** |
| SHA-256 vs PhysioNet `SHA256SUMS.txt` | **400 OK, 0 mismatch, 0 not-in-manifest** |
| Total size | 4,920,686 bytes (4.7 MiB) |
| Fold composition | train (1–8) **127 rec / 124 pt** · val (9) 42 / 34 · test (10) 31 / 27 |
| Patient overlap train ∩ (val ∪ test) | **0** |

- Acquisition script: `backend/dataset/_dl_ptbxl_sample.sh` — idempotent (skips files already present), 3 retries per file, deletes empty stubs so a rerun is clean, prints an explicit attempted/downloaded tally. It normalises `\r` out of the record list: an earlier attempt failed *all* 398 signal fetches with `curl: (3) URL rejected: Malformed input to a URL function` because `sample_records.txt` is CRLF-terminated and every URL carried a trailing carriage return. That loop also swallowed per-file failures and still exited 0, which is why the script now tallies explicitly.
- **Discipline note.** The sample was drawn as the first 200 stems, which — as the table shows — includes 42 validation and **31 frozen-test** records. All development and validation in §7.3 was therefore restricted to the **127 TRAIN-fold records**; the 73 held-out records were downloaded but deliberately **not read**. Any future sample list should be fold-filtered at construction time.

## 7.3 Representation — the fold-honest ECG feature set (`ecg-v1`)
Two new modules, both numpy/scipy-only (`wfdb`, `neurokit2` and `pywt` are not installable in this environment):

**`backend/dataset/wfdb_reader.py`** — a focused WFDB reader. PTB-XL is a single fully-specified WFDB variant (one `.dat` per record, signal format 16, one sample per frame, no skew, no byte offset), so a ~340-line reader removes an install-time failure mode entirely. It is **deliberately strict**: any header feature it does not implement raises `WfdbError` rather than being ignored, because silently mis-reading a signal would poison every downstream feature. It also verifies the **WFDB per-signal checksum** carried in the header — an integrity check on the decoded samples themselves, independent of the SHA-256 check on the file bytes.

**`backend/ml/features_ecg.py`** — the extractor, `FEATURE_SET_VERSION = "ecg-v1"`, **97 named features**:

| Block | Count | Contents |
|---|---|---|
| Rhythm / HRV | 9 | `hr_bpm`, `rr_mean_ms`, `rr_sdnn_ms`, `rr_rmssd_ms`, `rr_cv`, `rr_min_ms`, `rr_max_ms`, `pnn50`, `n_beats` |
| Global morphology | 4 | `qrs_duration_ms`, `qt_approx_ms`, `qtc_bazett_approx_ms`, `jt_approx_ms` |
| Per-lead morphology | 72 | 12 standard leads × (`q_amp_mv`, `r_amp_mv`, `s_amp_mv`, `st60_mv`, `st_slope_mv_s`, `t_amp_mv`) |
| Frontal axis | 2 | `qrs_axis_deg`, `t_axis_deg` |
| Spectral | 5 | 3 relative band powers (0.5–4 / 4–15 / 15–40 Hz) + `spectral_entropy` + `dominant_freq_hz` |
| Acquisition quality | 5 | `flatline_lead_fraction`, `saturated_sample_fraction`, `baseline_wander_mv`, `hf_noise_mv`, `rr_regularity` |

Pipeline per record: 0.5–40 Hz zero-phase conditioning → Pan-Tompkins-style R-peak detection on the **across-lead RMS composite** (so one dead lead cannot destroy the beat series) → **median** beat template (whole beats only, so edge beats cannot bias it; median so one ectopic beat cannot drag it) → threshold delineation on the template energy envelope → the six blocks above. The 12 standard leads are always emitted in fixed order, `NaN` for any lead a record does not carry, so the vector layout is identical for every record.

**Fold honesty is structural, not a convention.** Every function is a deterministic function of *one record*; no statistic is pooled across records, so extraction cannot leak between train, validation and test. Imputation, scaling, feature selection and PCA — the steps that genuinely must learn from data — are **deliberately absent** from this module and belong to a TRAIN-only fitting stage. An unmeasurable feature is emitted as `NaN` rather than silently filled, precisely so the fill value has to come from a fold-aware imputer.

### TRAIN-fold extraction audit (n = 127, 124 patients, dim = 97)
| Check | Result |
|---|---|
| Extraction | **127/127 succeeded**, 0 failed, **0 quality-flagged**, 0.4 s total (**3 ms/record** → ~70 s for all 21,799) |
| Non-finite cells | **0 / 12,319** |
| Zero-variance features | 2 (`flatline_lead_fraction`, `saturated_sample_fraction`) — expected on clean published data; zero-variance handling is a TRAIN-only downstream step |

Physiological validity against **published adult reference ranges** (median [p5, p95], % inside range):

| Feature | Median | p5 – p95 | In reference range |
|---|---|---|---|
| `hr_bpm` | 68.89 | 51.26 – 98.59 | 99.2% in [40, 140] |
| `rr_mean_ms` | 870.91 | 608.61 – 1171.25 | 94.5% in [600, 1500] |
| `rr_sdnn_ms` | 31.93 | 7.24 – 96.32 | 98.4% in [0, 150] |
| `qrs_duration_ms` | 80.00 | 60.00 – 124.00 | 88.2% in [70, 120] |
| `qt_approx_ms` | 370.00 | 320.00 – 470.00 | 95.3% in [300, 480] |
| `qtc_bazett_approx_ms` | 397.42 | 353.85 – 505.10 | 89.8% in [330, 460] |
| `qrs_axis_deg` | 42.24 | −31.07 – 88.57 | 95.3% in [−45, 110] |
| `t_axis_deg` | 35.64 | −0.42 – 61.65 | 96.9% in [−45, 110] |
| `ii_r_amp_mv` | 0.79 | 0.12 – 1.46 | 95.3% in [0.1, 3.0] |
| `ii_st60_mv` | −0.02 | −0.08 – 0.03 | 100.0% in [−0.2, 0.2] |
| `v2_t_amp_mv` | 0.39 | 0.10 – 0.93 | 100.0% in [−0.5, 1.5] |
| `rr_regularity` | 0.96 | 0.89 – 0.99 | 100.0% in [0.5, 1.0] |

### Instrument calibration — and why it is not metric shopping
The first implementation showed a **systematic low bias** in QRS duration (median 70 ms vs the published normal 80–100 ms): a bare fraction-of-peak threshold truncates the low-amplitude onset/offset slurs, each costing a full 10 ms bin at 100 Hz. Fixed by anchoring the crossing level to the **PR-segment noise floor** instead of to zero, then choosing the excursion fraction under a rule **fixed in advance**:

> the largest fraction whose **median QRS duration falls in the normal adult window (80–100 ms)** while clamping fewer than 5% of records at the 200 ms hard cap — evaluated on **TRAIN folds only**, against **published physiology**, and **never against a label**.

Selected `qrs_onset_fraction = qrs_offset_fraction = 0.08`, giving median QRS 80 ms and **1.6% clamped**. **No classification metric was consulted** — this is instrument calibration against external physiology, not model selection, and it is recorded here because a threshold chosen after seeing a metric would be exactly the metric shopping the mission forbids.

### Honest limitations
- 100 Hz gives a **10 ms sampling grid**, so interval features are quantised to ±10 ms; names carry `_approx` where the estimate is coarse. They are useful as *relative* features across a cohort, **not clinical-grade measurements**.
- Delineation is a threshold-and-template method, **not a validated clinical delineator**. It is deterministic and reproducible, which is what a benchmark needs; no clinical claim is made.
- **No feature here is a diagnosis.** They are inputs to a classifier that is itself evaluated under the patient-level protocol.

### Test coverage
`tests/test_wfdb_reader.py` (18 tests) and `tests/test_features_ecg.py` (35 tests) — synthetic byte-for-byte WFDB records (physical-unit round-trip, baseline/gain handling, checksum verify **and** mismatch, strict rejection of format ≠ 16 / multi-sample-per-frame / skew / byte offset / multi-file / truncation), a synthetic beat train with known rate and lead geometry (beat count and HR recovered at 50/60/75/100 bpm, analytic frontal-axis ratios, median-template outlier rejection, fiducial ordering, layout stability at 97 dims, NaN-not-filled, determinism, record-locality), and **TRAIN-fold-only** real-data regression locks. Full suite **703 green** (was 650 + 53).

## 7.4 Pre-registered task definition and evaluation protocol (fixed before any model)

Written **before a single model of any kind has been fitted on PTB-XL** and before any classification metric exists, for the same reason §1 fixed the dataset criteria in advance: a task or metric chosen after seeing a number is metric shopping, whatever it is called afterwards. Everything in this section is a commitment, not a summary.

**Primary task — `NORM` vs abnormal (screening framing).** Defined logically from the diagnostic superclasses, so no data inspection was needed to choose it:

| Label | Rule |
|---|---|
| `abnormal` (positive, 1) | at least one of `MI`, `STTC`, `CD`, `HYP` is present |
| `normal` (negative, 0) | `NORM` present **and** none of the four abnormal superclasses |
| excluded | no diagnostic superclass at all (the record carries only non-diagnostic statements) — the label is *unknown*, not negative, and a record with an unknown label must never be silently counted as healthy |

A record carrying `NORM` **and** an abnormal superclass is **positive**: a diagnostic statement was made. This is the screening-conservative direction and is fixed here so it cannot be flipped later to whichever reading scores better.

**Secondary task — `MI` vs `NORM`** (lower prevalence, clinically crisp, closer to E3's imbalanced regime). Reported, never used for model selection.

**Primary metric — ROC-AUC.** PTB-XL's primary task is near-balanced, which is the regime where ROC-AUC is the stable, conventional choice; C7's PR-AUC was primary on SMART-OM precisely because that task was *not* balanced. PR-AUC, balanced accuracy at a TRAIN-selected threshold, sensitivity at fixed specificity, and a per-superclass breakdown are all reported as **secondary** and none of them may be promoted to primary after the fact.

**Partitions.** PTB-XL's own `strat_fold`, already verified patient-disjoint (§7.1): folds **1–8 train**, **9 validation**, **10 frozen test**. All model, hyperparameter, feature-count, threshold and architecture selection happens on train/validation only. Fold 10 is evaluated **once**, at the end, and — see §7.2 — is not even present on this machine during development.

**Unit of analysis.** Metrics are computed per record; **bootstrap resampling is over patients**, not records, because patients contribute multiple ECGs and resampling records would understate the uncertainty through pseudo-replication. 2,000 bootstrap resamples, percentile 95% CIs.

**What would count as a quantum contribution.** All five conditions, fixed now:

1. Validation primary metric exceeds the **strongest** classical baseline (the best of simple, feature-engineered, deep, and fusion — not a weak straw man).
2. The **paired** bootstrap 95% CI of the difference (same patient resamples for both models) excludes 0.
3. A **matched classical control** — same input dimension, same preprocessing, same fitting budget — does *not* also achieve it. A quantum model beating a 97-feature classical model while itself consuming an 8-dimensional projection proves nothing about quantum computation until the 8-dimensional classical control is run too.
4. Label-**permutation** test p < 0.01, so the effect is not an artefact of the pipeline.
5. It **survives the single frozen-test evaluation**, with no re-tuning afterwards.

Failing any of these is a null, and a null reported honestly is an acceptable outcome of this phase.

## 7.5 Bulk signal acquisition and corpus integrity (verified this session)

The bounded sample of §7.2 was enough to build the reader and the extractor; it is not enough to model on. The **fold-filtered** bulk fetch covers folds **1–9 only** — the frozen test fold's signals are deliberately never downloaded, so §7.4's "evaluated once" commitment is enforced by the absence of the files, not only by a flag in the code.

| Audited quantity | Value |
|---|---|
| Records requested / complete | **19,601 / 19,601** (folds 1–9) |
| Files on disk (`.hea` + `.dat`) | **39,202 / 39,202**, 0 missing |
| SHA-256 vs PhysioNet `SHA256SUMS.txt` | **39,202 OK · 0 mismatch · 0 without a published checksum** |
| `fully_verified` | **True** |
| Total size | 482,260,870 bytes (**459.9 MiB**) |
| Fold-10 signals present | **0 — never fetched** |

**Correction to an earlier estimate.** §7.2 and DEC-042 cited "~1.7 GB" for `records100`. That is the size of the *whole* PTB-XL release including the 500 Hz tree; the folds-1–9 100 Hz corpus is **460 MiB**. The fetch completed in 1,476 s at ~96 files/s once the transport problems below were fixed, not the ~14 h the bad estimate and the early measured rate implied.

**Acquisition script** — `backend/dataset/download_ptbxl_signals.sh`. Four things in it are load-bearing, and each exists because of an observed failure:

- **Per-batch progress is recomputed from the filesystem**, never from curl's exit code. An earlier script exited 0 while fetching nothing.
- **`cygpath -m` for curl's `output=` paths.** Native Windows curl cannot interpret a `/c/Users/...` POSIX path and silently created `C:\c\Users\...`.
- **One `find -printf` + one `awk` pass** for the missing-file scan. The obvious per-file `stat` loop took ~20 minutes per scan on this filesystem; the replacement takes 0.066 s.
- **A DNS circuit breaker** (abort after 2 consecutive all-dead batches). A run that reported 13,708 files "missing" turned out to be a **local DNS outage** — 53,608 instances of `curl: (6) Could not resolve host: physionet.org`, with no 403 or 429 anywhere, i.e. not rate limiting. Without the breaker a network fault looks exactly like a dataset fault.

### Presence is not integrity — and neither is a manifest you did not check

The first verification pass returned `fully_verified: False`: 36,935 files OK, **0 mismatch**, but **2,267 files with no published checksum**. The cause was not the corpus. **Our local copy of `SHA256SUMS.txt` was itself a truncated download** — 3,915,407 of 8,284,204 bytes, ending mid-hash with no trailing newline, containing 0 `records500` entries, and stopping at `records100/20000/20643_lr.dat`.

Re-fetched the manifest (**87,203 lines** = 43,598 `records100` + 43,598 `records500` + metadata = 21,799 records × 2 files × 2 sampling rates, SHA-256 `b7224b92b341511ec3ceb13dc6652079b2c36a06504bcb49506f157f51dc695d`) and confirmed with `cmp -n 3915407` that the old file was a **strict byte-prefix** of the new one. That distinction matters: pure truncation means the 36,935 checks that *did* run were against genuine published hashes, so no earlier result is invalidated — only its coverage was incomplete. The partial file is kept as `SHA256SUMS.txt.truncated-partial` rather than deleted, so the diagnosis stays reproducible.

Re-verification against the complete manifest gives the table above: **39,202 / 39,202 OK, 0 mismatch, 0 unknown, `fully_verified: True`**. The verifier deletes any file that fails, so a re-run of the fetch is idempotent; it deleted nothing. Report: gitignored `backend/artifacts/dataset/ptbxl/signal_verification_report.json`.

The generalisable lesson is uncomfortable and worth stating plainly: a checksum pass proves nothing about the files it never covered, so **"verified" must report its own denominator**. Had the verifier printed only "0 mismatch" the corpus would have looked fully checked while 5.8% of it was unexamined.

## 7.6 The cohort matrix and the TRAIN-only fitted transform (verified this session)

Two modules complete the representation pipeline. They are separate on purpose: the first is unfitted and record-local, the second is where every fitted statistic lives. That boundary is the whole fold-honesty argument, so it is enforced by file organisation rather than by discipline.

### `backend/training/prepare_ecg_features.py` — the cohort matrix (`ptbxl-ecg-1`)

Runs the record-local extractor over the corpus and caches one `.npz` holding the matrix **and every identifier an honest evaluation needs**: `ecg_id`, `patient_id`, `strat_fold`, the §7.4 binary label, `label_known`, the five-way superclass matrix, age/sex, beat counts and quality flags. Metadata travels *inside* the archive (a cache cannot be separated from its provenance), nothing is pickled, row alignment is validated on save, and a cache built by a different `feature_set_version` or carrying a different feature layout is **refused on load** — because a stale cache is exactly how a run would end up reporting `ecg-v1` numbers computed by some other extractor.

Three properties are load-bearing: **nothing is fitted here**; **fold 10 is refused** without an explicit `allow_test_fold=True`; and **an absent label stays `-1`**, with `label_known` as its machine-readable form, so no caller can quietly count an unlabeled record as healthy. A record whose signal will not read is named in `failures` and still counted in the denominator (`rows + failures == n_candidates`) — never zero-filled, never silently absent from the tally.

**Full-corpus extraction (folds 1–9):**

| Quantity | Value |
|---|---|
| Extracted / candidates | **19,601 / 19,601**, **0 failed** |
| Patients | **16,965** |
| Dimension | 97 |
| Non-finite cells | **0 / 1,901,297** |
| Quality-flagged records | 29 |
| Runtime | 227.8 s (86 records/s) |
| §7.4 label | **abnormal 11,073 · normal 8,157 · unknown 371** |

Two independent consistency checks, neither of which required looking at the held-out data: fold 9 contains **2,183 records / 1,942 patients**, exactly the figure the DEC-041 metadata audit reported from the CSV alone; and TRAIN's 15,023 patients plus validation's 1,942 sum to **precisely** the 16,965 dev total, which can only happen if no patient spans the two. The label balance (**57.6% / 42.4%** over the 19,230 labeled records) confirms §7.4's pre-registered choice of ROC-AUC as the primary metric was the right call for this task — and that choice was fixed before the count was known.

### `backend/ml/ecg_transform.py` — the fitted half (`ecg-fit-1`)

Coverage filter → TRAIN-median imputation → TRAIN mean/std standardisation → reduction (`pca` / `select` / `identity`) into the compact 8–32 dimensions of the near-term angle-encoded regime. Four guarantees, each tested:

1. **It refuses to fit on a non-TRAIN fold** unless `allow_non_train=True` is passed deliberately, on the grounds that *"fitting a median, a scale or a PCA basis on validation data leaks its statistics into every subsequent evaluation."* The refusal is shown to be **effective rather than decorative** by a test that shifts every fold-9 row by +500 and demonstrates that the fitted medians, means and scales do not move at all.
2. **The fitting cohort is recorded, not asserted.** `train_patient_ids` is persisted *inside the artifact*, so `assert_no_fit_eval_patient_overlap` turns "the folds are patient-disjoint" from a claim about the code path into a property checkable against the saved file after a reload.
3. **Only the step that needs labels sees them,** and it sees only the *labeled* TRAIN rows. A test plants a column that screams a competing signal in the unknown-label rows; if those rows were treated as negatives it would win. It loses.
4. **Nothing is pickled** — plain arrays plus a JSON metadata string, verified by loading with `allow_pickle=False`.

`backend.ml.reduction.DimensionalityReducer` was deliberately **not** reused: it targets `2 ** n_qubits` for amplitude encoding and performs no imputation, whereas E4's regime is angle-encoded, so the output dimension need not be a power of two.

**Full-corpus fit — folds 1–8 only, 17,418 records / 15,023 patients (17,084 labeled):**

| Quantity | Value |
|---|---|
| Dimensions | 97 in → **97 kept** → 16 out (PCA) |
| Features dropped | **0** |
| Variance retained at 16d | **0.7745** |
| Partitions after transform | train 17,418/15,023 → 16d, finite · validation 2,183/1,942 → 16d, finite |
| Fit ∩ eval patient overlap | **0** (guard passed) |

The two features that were zero-variance on the 127-record sample (`flatline_lead_fraction`, `saturated_sample_fraction`) **do vary** across the full corpus, so nothing is dropped. That is worth recording as a small methodological lesson in its own right: a sample-level property of the data did not generalise to the corpus, which is precisely why the filters are fitted rather than hard-coded.

### What the compaction costs — the number E3 never had

TRAIN-only cumulative explained variance (folds 1–8; no validation or test data was read to produce this):

| Components | 2 | 4 | **8** | 12 | **16** | 24 | **32** | 48 | 64 | 97 |
|---|---|---|---|---|---|---|---|---|---|---|
| Cumulative variance | 0.2578 | 0.4103 | **0.6003** | 0.7020 | **0.7745** | 0.8682 | **0.9275** | 0.9781 | 0.9944 | 1.0000 |

90% needs 28 components, 95% needs 37, 99% needs 58.

This quantifies the central hypothesis behind the arena switch instead of merely asserting it. E3's four nulls were obtained after crushing megapixel images to 8 dimensions, and the suspicion recorded at the §29B stop was that the *crush*, not the circuit, was the ceiling — but that suspicion was never measured. Here it is measurable: **8 dimensions retains 60.0% of the representation's variance and 32 retains 92.8%**, so a quantum model working in the near-term regime is no longer being handed a near-destroyed input. Whether that is enough for a genuine quantum contribution remains completely open; it is a statement about the input, not about any result.

A supervised sanity check on the representation, again TRAIN-labeled rows only: the top-16 features by univariate |AUC − 0.5| are `qrs_duration_ms`, `qtc_bazett_approx_ms`, `qrs_axis_deg`, and ST60 / T-wave amplitudes across leads I, II, aVR, aVL, aVF, V4, V5, V6. Those are the conduction and repolarisation markers the MI/STTC/CD/HYP label is actually built from, which is evidence that the representation carries genuine task information rather than acquisition artefacts. **It is not a performance result** — no model has been fitted, and a univariate ranking is not a metric.

### Test coverage
`tests/test_prepare_ecg_features.py` (**44 tests**) — fold-spec parsing (11 accepted forms including `" TRAIN , 9 "`, 8 rejected typos, because a silently misparsed spec would change the cohort a result was computed on), the fold-10 refusal including an attempt to smuggle it in beside a dev fold, label honesty on a record carrying only a non-diagnostic code, NaN-not-zero on a flatline record, failure accounting under a deleted and a truncated `.dat`, and cache round-trips that preserve NaN and reject a stale version or a changed layout. `tests/test_ecg_transform.py` (**40 tests**) — the leakage refusals and the +500 poisoning proof, imputation versus dropping, the PCA/select/identity paths, and the misuse guards. Full suite **803 green** (the 703 baseline, plus 16 tests added to `tests/test_ptbxl_dataset.py` in a prior window that were never recorded, plus the 84 new here).

## 7.7 The classical failure map — tabular rungs and the matched-dimension controls (measured this session)

`backend/evaluation/e4_classical_baseline.py` (`E4_BASELINE_VERSION = "v1-e4-classical-1"`) contains the **first models ever fitted on PTB-XL in this project**. It exists to answer two questions that must both be settled *before* any circuit is written: **how good is classical here**, and **how good is classical at each input dimension a near-term circuit might actually consume**. The second question is §7.4 condition 3, and it had to be answered now — a matched control run *after* seeing a quantum number is not a control, because the knowledge of what it must match is already in the room.

### The grid, frozen in the module before the first fit

**3 models × 4 representations × 2 tasks, plus a trivial prevalence arm.** The three models are chosen for what each one *rules out*:

| Model | What a result here rules out |
|---|---|
| `logistic` | Whether the task is already linearly separable in these 97 features — if it were, nothing non-linear, classical or quantum, has anything to explain. |
| `gbm` (HistGradientBoosting) | What a strong non-linear **tabular** learner achieves. This is the number a quantum model is really competing with. |
| `rbf_svm` | The classical **kernel machine**. A quantum fidelity kernel is the direct analogue of this arm, so a kernel-advantage claim that does not beat it is not a claim at all. |

The four representations — `f97` (identity), `pca32`, `pca16`, `pca08` — exist so that every plausible circuit input width has a classical ceiling measured in advance.

**Every** hyperparameter, threshold and calibrator is selected by **leave-one-`strat_fold`-out cross-validation inside folds 1–8**. PTB-XL's own folds are reused as the inner split rather than inventing a fresh grouping, so the inner CV *inherits* the patient disjointness already verified twice (§7.1, §7.5) instead of asserting a new one. Fold 9 is scored. **Fold 10 is never touched, and `test_partition_used: false` is asserted in both emitted reports.**

One library note worth recording: `SVC(probability=True)` is deprecated in scikit-learn 1.9 (removal 1.11), and `SVC().probability` now defaults to the **string `'deprecated'`**, not `False`. The kernel arm is therefore scored through `decision_function` and **Platt-scaled from the TRAIN out-of-fold decision values already computed for threshold selection** — no extra fits, fitted only in the partition where fitting is legal, and monotone, so ROC-AUC is provably unmoved. The calibrator exists only so Brier / log-loss / ECE and the operating threshold are defined.

### Primary task result (fold 9: 2,146 labeled records / 1,917 patients, prevalence 0.5741)

| Rank | Arm | ROC-AUC | 95% CI (patient bootstrap, 2,000 draws) |
|---|---|---|---|
| 1 | **`gbm@f97`** | **0.940234** | [0.9297, 0.9494] |
| 2 | `rbf_svm@f97` | 0.933624 | |
| 3 | `rbf_svm@pca32` | 0.931961 | |
| 4 | `rbf_svm@pca16` | 0.927408 | |
| 5 | `gbm@pca32` | 0.922905 | |
| 6 | `gbm@pca16` | 0.916917 | |
| 7 | **`rbf_svm@pca08`** | **0.915006** | [0.9024, 0.9265] |
| 8 | `gbm@pca08` | 0.907049 | |
| 9 | `logistic@f97` | 0.900270 | |
| 10 | `logistic@pca32` | 0.895591 | |
| 11 | `logistic@pca16` | 0.879296 | |
| 12 | `logistic@pca08` | 0.869805 | |
| — | `prevalence` (trivial floor) | **0.500000** | exactly — the score/label alignment check |

All **11** paired patient-clustered bootstrap deltas against the strongest arm **exclude zero**, from `rbf_svm@f97` +0.006610 [+0.00159, +0.01157] to `logistic@pca08` +0.070430 [+0.05789, +0.08329]. The ordering is not bootstrap noise. The secondary MI-vs-NORM task is **reported only** (`used_for_selection: false`); its ceiling is 0.973262.

### The matched-dimension ceilings — the numbers a quantum model must clear

| Input dimension | Best classical arm | ROC-AUC | Variance retained (§7.6) |
|---|---|---|---|
| **8** | `rbf_svm@pca08` | **0.915006** | 0.6003 |
| **16** | `rbf_svm@pca16` | **0.927408** | 0.7745 |
| **32** | `rbf_svm@pca32` | **0.931961** | 0.9275 |
| **97** | `gbm@f97` | **0.940234** | 1.0 |

**This is the finding the arena switch was chartered to produce.** At 8 dimensions — the regime an angle-encoded near-term circuit actually works in — classical reaches **97.3% of the full-feature ceiling**. E3's four nulls were obtained after crushing megapixel images into 8 dimensions, and the §29B suspicion was that the *crush*, not the circuit, was the ceiling. On PTB-XL that confound is gone.

Both consequences must be stated, because they point in opposite directions:

- **Favourable:** a null here is genuinely informative *about circuits*, and a positive result here is *attributable* to the circuit rather than to input destruction. That is exactly the condition E4 went looking for.
- **Unfavourable:** the headroom a circuit can claim from dimensionality alone is at most **+0.0252 [+0.0173, +0.0340]**, and to beat the overall classical ceiling it must clear **0.940234** outright. The bar is now a number, and it is high.

### Permutation null — and the one assumption it violates, measured rather than glossed

200 **patient-blocked** label permutations inside TRAIN, with the model **refitted on every draw** (not merely rescored, so the test covers the whole pipeline's capacity to manufacture signal):

| Quantity | Value |
|---|---|
| Observed | 0.940234 |
| Null mean (sd) | 0.507865 (0.023552) |
| Null p97.5 / max | 0.554956 / 0.567394 |
| Draws at or above observed | **0 of 200** |
| p-value, conservative `(k+1)/(n+1)` | **0.004975** — clears §7.4 condition 4 (p < 0.01) |
| Wall clock | 867.3 s |

The run reported **`within_patient_labels_consistent: False`**, and the size of that violation is recorded here rather than buried: of 14,823 TRAIN patients, **1,631 (11.00%) have more than one record**, and **281 (1.90% of all patients, 17.23% of multi-record patients, covering 688 records = 4.03%) genuinely carry mixed labels** — a patient normal at one visit and abnormal at another, which is a fact about medicine, not a defect in the cache. Blocking assigns each patient's first-record label to the whole block, shifting the permuted **record-level** prevalence from the observed 0.5760 to a mean of **0.5525** (sd 0.0018, range [0.5478, 0.5585]). Blocking is kept anyway, because it preserves *more* structure than record-level permutation and therefore produces the **harder** null — the conservative direction.

### Where the classical ceiling fails

Operating threshold **0.5866**, selected on TRAIN out-of-fold predictions and never on fold 9. At that point: **221 missed positives, 84 false alarms**; sensitivity at fixed specificity 0.90 = **0.8255**.

| Failure mode | Number |
|---|---|
| **HYP alone** (single-superclass recall) | **0.400** (n = 65) — missed 3 times in 5 |
| CD alone | 0.7379 (n = 206) |
| MI alone | 0.7468 (n = 233) |
| STTC alone | 0.8147 (n = 259) |
| HYP in any combination | 0.8134 (n = 268) — found when it travels with another abnormality |
| **NORM-carrying positives** (record labelled both normal and abnormal) | **0.3171** (n = 955, specificity 0.9081) |

And a steep **age-dependent operating-point drift** under a single global threshold:

| Age band | Sensitivity | Specificity |
|---|---|---|
| < 40 | 0.5645 | 0.9593 |
| 40–59 | 0.6513 | 0.9403 |
| 60–74 | 0.8479 | 0.8553 |
| 75+ | **0.9327** | **0.7143** |

One threshold behaves as a near-specific rule in the young and a near-sensitive one in the old. Sex is near-neutral (sensitivity 0.8308 vs 0.8093), so there is no disparity on that axis. The 37 unknown-label fold-9 rows are excluded from scoring and **reported unscored**, not silently dropped.

These are **hypothesis-generating observations, not evidence for any mechanism.** That a gradient-boosted model on 97 hand-built features struggles with isolated hypertrophy says nothing yet about whether a circuit would find it easier.

### Test coverage

`tests/test_e4_classical_baseline.py` (**58 tests**) is organised around the four ways this module could lie — **leak**, **miscount**, **understate uncertainty**, **flatter itself** — rather than around its function list. The bootstrap test plants a **per-patient random effect** in the scores (how repeated ECGs from one patient actually behave) and asserts the patient-clustered standard error and CI width *exceed* the naive record-level ones; the paired-delta test asserts an arm against itself is exactly zero and that the paired interval is tighter than an unpaired one; the out-of-fold test proves predictions are held out rather than in-sample; the permutation test asserts the p-value can never be 0; the trivial arm must land on exactly 0.5; and one test walks the serialised report asserting **no raw score vector survives into it**. Full suite **861 green** (803 + 58).

Three fixture bugs are worth recording because in each case the module's own guards caught them. The first run was 8 failed + 14 errors because the fixture assigned `strat_fold` **per record** while emitting two records per patient — patients straddled train and validation and `guard_partitions` correctly refused. The defect was the fixture; the guard was working. The bootstrap test then failed with the grouped standard error *smaller* than the ungrouped, diagnosed by direct measurement as the fixture correlating labels within patient but drawing scores independently, leaving no pseudo-replication to detect. And `test_every_arm_beats_the_trivial_floor` failed at 0.4072 on the secondary task — honest rather than a bug, since the fixture plants a signal separating abnormal from normal but nothing separating MI from other abnormals; it was scoped to the primary task with the reason written into the docstring, not weakened until it passed quietly.

### What this does and does not establish

It establishes a **ceiling and a gate**. It is **not the complete failure map the mission specifies**: four rungs were named — simple, feature-engineered, modern deep, best fusion — and this delivers the first two plus a kernel arm and all matched-dimension controls. **The 1D-CNN on the raw 12-lead signal and the fusion are still owed, so the quantum gate does not open here.** A circuit compared against a tabular-only ceiling could be beating a baseline that a deep model already exceeds, which is precisely the straw man §7.4 condition 1 forbids. Every number above is a **fold-9 validation** number; fold 9 has now been read repeatedly for model selection and is no longer a clean generalisation estimate, which is why fold 10 exists and why its signals are still not on this machine.

## 7.8 The deep rung and the fusion — the gate opens (measured this session)

`backend/evaluation/e4_deep_baseline.py`, version `v1-e4-deep-1`, seed 42. This closes the
two rungs §7.7 left owed: the modern deep model on the raw signal, and the best-reasonable
fusion. Report: `backend/artifacts/reports/e4_deep_baseline.json`. Total **14,061 s**.

TRAIN folds 1–8: 17,084 records / 14,823 patients, prevalence 0.5760.
VALIDATION fold 9: 2,146 records / 1,917 patients, prevalence 0.5741. **Fold 10 not read.**

### Primary task result (fold 9, patient-clustered bootstrap, 2,000 draws)

| Arm | Input | ROC-AUC | 95% CI | bal. acc | sens@sp0.90 | ECE | Brier |
|---|---|---|---|---|---|---|---|
| **fusion@cnn+gbm** | both | **0.946293** | [0.9364, 0.9550] | 0.8700 | **0.8409** | **0.0190** | **0.0925** |
| cnn@resnet_small | 12 × 1000 raw mV | 0.940476 | [0.9297, 0.9501] | 0.8719 | 0.8393 | 0.0867 | 0.1118 |
| gbm@f97 (§7.7 ceiling) | 97 `ecg-v1` | 0.940234 | [0.9297, 0.9494] | 0.8644 | 0.8255 | 0.0401 | 0.0989 |

Paired patient-clustered deltas, same resamples for both members:

| Contrast | Observed Δ | 95% CI | Excludes zero |
|---|---|---|---|
| cnn − gbm | **+0.000242** | [−0.006101, +0.006374] | **No** |
| fusion − cnn | +0.005817 | [+0.002034, +0.009751] | Yes |
| fusion − gbm | +0.006058 | [+0.003693, +0.008641] | Yes |

### A tie that is not a redundancy

126,649 parameters reading 12,000 raw samples land **within 0.00025 ROC-AUC** of 400 boosted
trees reading 97 scipy-computed numbers, and the paired interval comfortably spans zero. The
obvious reading is that the two arms see the same thing. They do not: fusing them beats both
with an interval that excludes zero, and the blender retains **both** members with positive
weight (tabular 0.5637, deep 0.2865, intercept 0.2548). They **tie in aggregate and disagree
per record**. The fusion also calibrates substantially better than either member — ECE 0.019
against 0.040 and 0.087 — which is a second, independent sign that the two carry different
information rather than the same information twice.

This is the most consequential fact the phase has produced, and it cuts both ways (below).

### Architecture selection — on TRAIN fold 8 only, fold 9 never read

| Architecture | Params | Epochs run | s/epoch | Best epoch | Inner ROC-AUC (fold 8) |
|---|---|---|---|---|---|
| **resnet_small** | 126,649 | 25 | 66 | **17** | **0.946739** |
| resnet_long_kernel | 218,809 | 14 | 83 | 6 | 0.946422 |
| resnet_wide | 251,785 | 16 | 640 | 8 | 0.945696 |

The three span **0.0010** on the inner fold. Doubling the channel width buys nothing and costs
six times the wall clock. The search found no architecture-shaped headroom, which is itself a
measurement and not a disappointment. `resnet_small` was selected and refit on folds 1–8 for
its selected 17 epochs; the operating threshold 0.395417 comes from the selection fit
(folds 1–7) and is reused rather than refitted, with its retained fold-8 ROC-AUC asserted
against the selection record.

### Protocol integrity (asserted in code, serialised in the report)

- `test_folds_read: []`, `test_partition_used: false`.
- Patient overlap **0**, `ecg_id` overlap **0** between folds 1–8 and fold 9.
- Signal cache `ptbxl-signal-1`: 19,601 rows, **0 failures**, `fitted_statistics: "none"`.
  Conditioning is a **record-local** 0.5–40 Hz zero-phase band-pass with no amplitude
  normalisation, so it is fold-honest by construction for the same structural reason
  `backend/ml/features_ecg.py` is.
- Architecture, epoch count, threshold **and** the fusion blender are all selected on TRAIN
  fold 8; the blender is fitted there with **both members out-of-sample on it**.
- `gbm@f97` reproduces DEC-044's recorded **0.9402343416976896** exactly
  (`reproduces_recorded_roc_auc: true`, asserted to 1e-9). The deep arm is therefore measured
  against the ceiling **on record**, not against a convenient re-derivation of it.

### The failure map of the strongest arm

`fusion@cnn+gbm` at threshold **0.544001** (TRAIN fold 8). Error rate **0.1356**:
**207 missed positives against 84 false alarms** — the residual error is 2.5 : 1
false-negative, so sensitivity is the weak side at the chosen operating point.

**Isolated versus accompanied** — the diagnostic contrast:

| Superclass | Alone | In any combination | Gap |
|---|---|---|---|
| **HYP** | **0.3692** (n = 65) | 0.8209 (n = 268) | **+0.452** |
| MI | 0.7554 (n = 233) | 0.8833 (n = 540) | +0.128 |
| CD | 0.7524 (n = 206) | 0.8667 (n = 495) | +0.114 |
| STTC | 0.8378 (n = 259) | 0.9129 (n = 528) | +0.075 |

Every superclass is found more often when it travels with another abnormality. For
hypertrophy the model is very largely **not reading hypertrophy at all** — it is detecting the
company hypertrophy keeps. **The deep arm did not fix this**: HYP-alone was 0.400 under the
tabular ceiling and 0.369 under the fusion, a difference well inside noise at n = 65. Raw
waveform access bought nothing on the one failure mode §7.7 flagged as most severe.

**Age drift under one global threshold** persists essentially unchanged:

| Age band | Sensitivity | Specificity | Prevalence |
|---|---|---|---|
| < 40 | 0.5645 | 0.9548 | 0.219 |
| 40–59 | 0.6667 | 0.9303 | 0.394 |
| 60–74 | 0.8580 | 0.8640 | 0.684 |
| 75+ | 0.9447 | 0.7619 | 0.869 |

Neither arm is **given** age: it is not among the 97 features and not a channel of the 12.
The gradient is therefore emergent from waveform morphology, which is physiologically
expected and operationally awkward — one threshold behaves as a near-specific rule in the
young and a near-sensitive one in the old. Sex remains near-neutral (0.8415 / 0.8213).
Quality-flagged records: 3 in fold 9, all found. The 37 unknown-label rows are excluded and
**reported unscored**, never counted as negatives.

### What this does to the quantum gate

**The gate opens.** All four rungs the mission named — simple, feature-engineered, modern
deep, best fusion — are now measured on PTB-XL under patient-level validation. The bar is no
longer 0.940234. It is **0.946293**, the fusion, with matched-dimension controls unchanged at
0.915006 (d = 8), 0.927408 (d = 16), 0.931961 (d = 32).

But the result **reshapes** the question rather than merely raising the bar, and two readings
have to be held at once.

*Against a quantum contribution.* Two function classes with nothing in common — axis-aligned
splits on hand-built scalars, and a convolutional hierarchy on raw millivolts — tie to within
0.00025. Trebling capacity across three architectures moves the inner score by 0.001. When
the representation is varied that widely and the number does not move, the remaining error
looks like a property of **the task** at 100 Hz over 10 seconds, not of any model's expressive
power. A quantum feature map over an 8–32-dimensional reduction of `ecg-v1` carries strictly
**less** information than either arm here, and §7.7's matched controls already price that
loss: 0.915006 at d = 8, some 0.031 below the fusion. Nothing measured this session makes a
quantum advantage on this arena more likely.

*For a sharper experiment.* The fusion gain is real and patient-clustered-significant, which
proves there is residual structure **neither representation captures alone** and that
combining representations pays. That is precisely the mechanism a quantum feature map claims:
a different, higher-order embedding of the same inputs. So the question is falsifiable in a
way it was not before — and the correct control is harsher than the one §7.4 fixed. The
honest test is not only "does a quantum arm beat 0.946293"; it is:

> **Does a quantum arm, fused in this way, add anything beyond what a second *classical*
> representation, fused in this way, already adds?**

The matched control for a quantum fusion is therefore a **classical fusion of the same shape**
— same blender family, same parameter count, same fold, same members-out-of-sample
discipline. §7.4's five conditions stand unaltered; this is an additional control, and it is
**pre-registered here before any quantum arm has been fitted on PTB-XL**, which is the only
moment at which writing it down means anything.

**Where the headroom is, and how small the cells are.** If a quantum arm is to contribute,
the failure map says where to look: not aggregate ROC-AUC, where two classical families have
converged, but isolated HYP (0.369, **n = 65**), isolated CD (0.752, n = 206), isolated MI
(0.755, n = 233), and the under-40s (sensitivity 0.5645, n = 283). These are **small cells**.
A bootstrap interval on n = 65 is wide enough that a headline improvement there would be
indistinguishable from resampling noise without the frozen partition, and fold 9 has now been
read many times. Stating that now, rather than discovering it after a favourable number
appears, is the point of writing it here.

### Cost, and a corrected budget

14,061 s total. `resnet_wide` consumed **10,235 s of it — 73%** — for the *worst* inner score,
running at **640 s/epoch against a predicted 132**. The quadratic-in-width model recorded
after the §7.7-era benchmark underestimates this machine by ~5× at 250 channels; the next grid
must be budgeted from a measured epoch at the actual width, not extrapolated.

### Test coverage

`tests/test_prepare_ecg_signals.py` (**40 tests**) closes the gap left by the waveform cache
shipping without a dedicated suite: the fold-10 refusal and its citation, the lead-permutation
contract, physical-millivolt decoding, ordering by `ecg_id` and its agreement with the feature
cache, the trim-to-written-rows path, the refusal to write an all-failures cache, `rows_for`
raising rather than dropping a miss, and version-mismatch refusal on load. The
no-fitted-statistics property is tested **as a property** — the same record is cached in two
different corpora and the rows are required to be bit-identical — rather than asserted in
prose.

It found one real defect. `_lead_permutation` accepted a **13-channel** record and silently
dropped the extra column, contradicting its own docstring ("a missing or extra lead changes
what the channel axis means"). Fixed by enforcing the length. **No measured number changes**:
the cache reports 0 failures over all 19,601 records and PTB-XL's `records100` tree is
uniformly 12-lead, so nothing in this document was computed through the defective path.

## 7.9 Prior art — what the literature already did, and the one thing it apparently did not

§7.8 pre-registered a classical-fusion control *before* this search was run. That ordering
matters: the control was chosen because the tie-without-redundancy result demanded it, not
because a literature gap made it look novel. What follows either supports the positioning or
it does not, and either way the control stays.

### Method, and its limits stated first

A systematic sweep of the arXiv API, paged to exhaustion:

| Query | Hits | Examined |
|---|---|---|
| `abs:"PTB-XL"` | 103 | **102** (3 pages; 1 entry truncated mid-feed) |
| `all:"quantum" AND all:"ECG"` | 40 returned | 40 (only ~8 genuinely about ECG) |
| `abs:"quantum machine learning" AND abs:"classical baselines"` | 48 | 40 |

**What this audit cannot support.** It is arXiv-only. PubMed, IEEE Xplore, Scopus and Google
Scholar were not searched — `WebSearch` is unavailable in this deployment and one `WebFetch`
query failed on quota — so work published in IEEE/Elsevier venues without a preprint is
invisible to it. `abs:"PTB-XL"` misses any paper that uses PTB-XL without naming it in the
abstract. Everything below is read at abstract level; no full texts were retrieved, so
reported comparisons are as-abstracted. The honest form of any claim here is therefore
**"unattested in this sweep"**, never *"nobody has ever done this"*.

### Finding 1 — quantum ML on PTB-XL is essentially unexplored, and the one attempt did not win

Of **102 PTB-XL papers spanning 2020-04-28 to 2026-09-15, exactly one applies quantum
methods**: arXiv:2603.27269v1 (2026-03-28), *From Foundation ECG Models to NISQ Learners:
Distilling ECGFounder into a VQC Student*, Franco, Mahlow, Cardoso & Fanchini (quant-ph,
cs.AI). Everything else is classical — CNNs, ResNets, transformers, Mamba/SSMs, diffusion,
contrastive and self-supervised learning, federated and TinyML deployment.

That paper: PTB-XL plus MIT-BIH, binary ECG classification. A fine-tuned **ECGFounder**
teacher is distilled into two classical students (ResNet-1D, a lightweight CNN-1D) and one
quantum student — a convolutional autoencoder compressing 256-sample windows into a
low-dimensional latent, feeding a **6-qubit variational circuit** in simulated Qiskit. The
reported outcome is that **the classical teacher delivers the best overall results**, with
the distilled students "competitive despite far fewer trainable parameters". No metrics
appear in the abstract record.

Three consequences, and the first is the uncomfortable one:

- **We cannot claim to be first to put a quantum model on PTB-XL.** Any framing that implies
  otherwise is false, and the phase is not permitted to use it.
- **The one prior attempt did not beat its classical reference** — consistent with E3's four
  nulls and with the methodology cluster below.
- **Its design is bounded above by construction, and ours is not.** A distillation student
  is trained to imitate a teacher; it cannot exceed what the teacher encodes, so that
  experiment could not have detected a quantum advantage even if one existed. It answers
  *"can a 6-qubit circuit compress a foundation model?"*, not *"can a quantum model beat the
  best classical approach?"* The second question is still open on this arena. That is a
  narrower and more defensible novelty statement than "first on PTB-XL", and it is the one
  the evidence actually supports.

### Finding 2 — the methodology critique already exists, and this project is downstream of it

A 2025–2026 cluster converges on precisely the failure mode E4 was chartered against:

| Paper | Date | What it establishes |
|---|---|---|
| 2608.18155 — *How Quantum Is the Advantage?* (network intrusion detection) | 2026-08-13 | Leakage-controlled, calibration-aware benchmark with a **quantum-attribution audit** tracing apparent gains to preprocessing rather than quantum effects; tuned RF/XGBoost match or beat the quantum arms. Commits that the contribution "stands whether quantum wins, ties, or loses." |
| 2607.15815 — *The Fourier Wall* | 2026-07-17 | QML losses on tabular data are **structural**: "QML models usually lose to carefully tuned classical baselines." Screens datasets *before* comparing, against **five tuned classical twins**. |
| 2607.01197 — *Quantum vs. Classical ML: A Unified Empirical Comparison* | 2026-07-01 | Seven matched pairs; the quantum models "do not yet surpass the classical baselines". |
| 2607.04915 — *Cloud Microphysics Stress Test* | 2026-07-06 | Heavily tuned QNNs still beaten by plain fully-connected nets; a null reported honestly. |
| 2605.19233 — *Leakage-Free Evaluation for UAV anomaly detection* | 2026-05-19 | **Random stratified splits inflate scores**; a group-aware protocol is required. The direct analogue of this phase's patient-level grouping. |
| 2601.00921 — *Quantum kernels for COPD muscle outcomes* | 2026-01-01 | Clinical, and unusually careful: the quantum-hybrid improvement is **numerical only, not significant after Holm adjustment**. |
| 2507.11401 — *Stochastic Entanglement Configuration* (cardiac MRI) | 2025-07-15 | Samples **400 entanglement topologies** and reports the 16% that beat the classical baseline, while all four conventional topologies fail. A multiple-comparisons hazard, and a useful negative example of what §7.4's pre-registration exists to prevent. |

The honest reading: **E4's evaluation stance is not novel.** Leakage control, matched
controls and attribution audits are established practice as of 2026, and at least one paper
(2608.18155) has already published the "our contribution stands whether quantum wins or
loses" commitment this phase operates under. Claiming methodological originality here would
be false.

### Finding 3 — the gap that does appear to be real

Two observations that only look interesting together:

1. **Classical ECG work knows ensembles win.** arXiv:2609.12803 (2026-09-11), *Multi-Label
   12-Lead ECG Classification on the PTB-XL Dataset*, compares five architectures plus three
   ensembling schemes and finds **stacking tops macro AUROC/AUPRC/F1**. Independently
   consistent with §7.8: fusion beats every single arm on this dataset.
2. **Quantum work compares against single models.** Across every quantum paper found here,
   the reference is one classical model, a set of *parallel* single models (the Fourier
   Wall's "five tuned classical twins" — twins, not a fused baseline), or a teacher in a
   distillation. **No paper in this sweep compares a quantum arm against a *fused* classical
   baseline of the same shape.**

So the standard the quantum-ECG literature holds itself to is weaker than the standard the
classical ECG literature has already established for itself. A quantum fusion evaluated
against a single classical model can post a gain that a classical fusion would also have
posted — which is not a quantum effect, it is an ensembling effect. **That is the gap
DEC-046's control closes**, and it was pre-registered before this search, on internal
evidence alone.

This is a claim about an *evaluation standard*, not about circuits, and it is deliberately
small. It is also the only novelty claim in this section that the evidence supports.

### Finding 4 — the tie's pessimistic reading is independently corroborated

§7.8 read the deep/tabular tie as evidence that residual error is a property of the **task**
rather than of model capacity. arXiv:2608.14633 (2026-07-27), a leakage-controlled study of
Wolff-Parkinson-White detection at 471:1 imbalance pooling PTB-XL with
Chapman-Shaoxing-Ningbo, reaches the same conclusion from a different direction: **data, not
model capacity, is the limiting factor.** arXiv:2606.08583 adds a related caution — a
spectral audit finds ECG/EEG deep models often lean on broadband 1/f structure rather than
diagnostic morphology.

Neither result makes quantum advantage more likely. Both make it more likely that the
remaining 13.6% error on this task is not waiting for a better function class.

### What this licenses, and what it does not

- **Licensed:** proceeding to T-E4-Q-01 with the §7.4 five conditions plus the §7.8
  classical-fusion control; describing the quantum-on-PTB-XL literature as one distillation
  study that did not beat its classical teacher; describing the classical-fusion control as
  unattested in this sweep.
- **Not licensed:** "first quantum model on PTB-XL" (false — 2603.27269). "Novel evaluation
  methodology" (false — 2608.18155, 2607.15815, 2605.19233). Any claim of exhaustive
  coverage — this is an arXiv-only, abstract-level sweep, and it says so.

## 7.10 The quantum protocol, pre-registered before the first circuit was fitted

Everything in this section was written and committed to the module **before any quantum arm
was scored on any partition**. Its purpose is to make the result unfalsifiable-by-hindsight:
the arms, the controls, the headline comparison and the decision rule are all fixed here, so
a null cannot be rescued afterwards by changing what counted as the comparison.

### The one question this rung exists to answer

§7.9 found that the phase's only surviving novelty claim is the control §7.8 had already
pre-registered: **no paper in that sweep compares a quantum arm against a *fused* classical
baseline of the same shape.** That makes the control the experiment, not a formality. So the
question is deliberately *not* "does a quantum arm beat a classical model" — a second
classical representation already does that, and beating a single model would be an
ensembling effect wearing a quantum costume. The question is:

> Does a quantum feature map contribute anything to a fusion that a **classical map of the
> same shape, blended the same way, on the same folds**, does not also contribute?

### The "same shape" contract

An arm is a map `R^8 -> R^36` fitted on TRAIN rows only, followed by an **identical**
logistic head with an identical `C` grid selected on the identical inner fold. The three maps
differ *only* in how the 36 columns are produced:

| Map | Construction | Trainable params in the map |
|---|---|---|
| `zz` | Havlicek ZZ feature map, exact Aer statevector, 8 single-qubit `<Z_i>` + 28 pair `<Z_iZ_j>` | **0** (the angles are data) |
| `poly2` | the same index set, classically: 8 standardised `x_i` + their 28 pairwise products `x_i x_j` | **0** |
| `rff36` | 36 random Fourier features (RBF), bandwidth from the TRAIN median heuristic | **0** |

`poly2` is the tight control, and it is chosen for a specific reason: the ZZ map's 36
observables are indexed by exactly the 8 singletons and 28 pairs of an 8-element set, which
is precisely the index set of a degree-2 polynomial on the same 8 inputs. The two maps
therefore have the same width, the same input, the same parameter count (zero), and the same
*combinatorial structure* — they differ only in the function placed on each index. If the
quantum arm wins against `poly2`, "second-order interactions were suddenly available" is not
an available explanation. `rff36` is the looser second control: same width, no index-set
correspondence, a standard classical nonlinear expansion.

### Arms, fixed

Single-arm rung (d = 8, one qubit per component): `q@zz`, `c@poly2`, `c@rff36`.
Fusion rung, all four blended by the *same* logistic blender on fold-8 out-of-sample
log-odds, members `cnn@resnet_small` + `gbm@f97` recovered exactly as §7.8 recorded them:

- `fusion@cnn+gbm` — the recorded bar, recomputed here so the blender is identical
- `fusion@cnn+gbm+zz` — the quantum fusion
- **`fusion@cnn+gbm+poly2` — the control that is the experiment**
- `fusion@cnn+gbm+rff36` — the second control

### The headline comparison, and the decision rule

The headline is **not** `fusion@cnn+gbm+zz` versus `fusion@cnn+gbm`. It is:

> paired, patient-clustered bootstrap delta of
> **`fusion@cnn+gbm+zz` − `fusion@cnn+gbm+poly2`**, 2,000 draws, same resampled patients
> for both arms.

Decided in advance:

- **If that interval spans zero** — the honest conclusion is that the quantum map contributed
  nothing a same-shape classical map did not, *regardless of how either arm scores against
  0.946293*. This is recorded as a null and the phase does not propose a quantum arm.
- **If it excludes zero and favours `zz`** — that is a candidate contribution, and it then
  still has to clear §7.4's five conditions and the absolute bar 0.946293 before the word
  "advantage" is used.
- **A quantum arm that beats 0.940234 but not 0.946293 is not an advantage** (§7.4 condition
  1), and a quantum arm that beats `fusion@cnn+gbm` but not `fusion@cnn+gbm+poly2` is an
  ensembling result, not a quantum one.

### Search budget, fixed in advance

Deliberately small, because arXiv:2507.11401 is in the record as the failure mode to avoid:
that paper sampled **400 entanglement topologies** and reported the 16% that beat baseline.
Here the entire quantum search is **`reps ∈ {1, 2}`** and the head's `C ∈ {0.01, 0.1, 1, 10}`
— identical in size to the grid each classical control gets, selected on inner fold 8 only.
No topology search, no ansatz sweep, no angle sweep. If the map needs a 400-cell search to
win, that is a multiple-comparisons artefact and this protocol is built not to find it.

### Why the explicit map and not a fidelity kernel

A fidelity quantum kernel on 17,084 TRAIN rows needs ~292M pairwise circuit evaluations and a
2.3 GB Gram matrix; this machine has ~0.9 GB free. That is a real constraint, but it is not
the reason — the reason is that the same count makes a fidelity kernel **not near-term
executable**, which is one of the mission's own success criteria. The explicit map needs
19,230 circuits total (measured: ~2 min at reps=1, ~3.5 min at reps=2 for the whole corpus),
which is a plausible hardware job. A kernel arm may still be run later as a subsampled
diagnostic; it is not eligible to be the proposed arm.

### Protocol integrity, asserted in code

`test_partition_used: false` in every payload; the transform refuses non-TRAIN folds; the
inner split is patient-disjoint and asserted; the recovered `cnn@resnet_small` and `gbm@f97`
members must reproduce their recorded fold-9 ROC-AUCs (0.9404758944556537 and
0.9402343416976896) **exactly**, or the run raises rather than silently comparing against a
different baseline than the record names. Fold 10 is not read and its signals stay
undownloaded.

### Amendment, made before any arm was scored: `reps = 1` is excluded on a proof

The protocol above pre-registered `reps ∈ {1, 2}`. Building the module found that **`reps = 1`
is a degenerate arm that cannot carry information**, and it is excluded on that structural
ground. The timing matters and is stated plainly: this was found while writing the map, from
a property of the circuit that involves no labels and no partition, and **no arm had been
scored on fold 8 or fold 9 when the decision was made**.

The argument: a Havlicek block is one Hadamard layer followed by `P(2 x_i)` phases and
`CX–P–CX` gadgets. Everything after that Hadamard layer is **diagonal in the computational
basis**, so it changes phases only and leaves `|ψ|²` exactly uniform. Every observable read
here is a function of `|ψ|²` alone. Therefore all 36 outputs are identically zero for every
input, and the only thing a head fitted on them can learn is an intercept.

Measured, and recorded in the report's `entangling_witness` and in the module's constant:

| reps | per-row spread of `\|ψ\|²` | max `\|⟨O⟩\|` | max per-column TRAIN std | mean `\|C_ij\|` |
|---|---|---|---|---|
| 1 | **exactly 0** | 2.2e-16 | 6.8e-17 | **0.0** — product state |
| 2 | 0.032 – 0.086 | 0.9995 | 0.218 | 0.0701 — entangling |

Two things follow, both in the conservative direction. The quantum search shrinks from eight
cells to **four**, which is now *exactly* the four cells each classical control gets — so the
same-shape contract covers the search budget too, not just the map width. And `fit_map` gained
a runtime guard that **refuses** any map whose TRAIN output is constant in every column, so
the exclusion is enforced by code rather than by intention: a future caller that asks for
`reps = 1` gets an exception, not a chance-level number.

This also means E3's `REPS_GRID = (1, 2)` contained a dead cell wherever that grid was
searched with Z-basis readout. That is noted, not retro-fixed; E3's conclusions were nulls,
and a dead cell can only have made a null more likely.

## 7.11 The quantum rung — measured, and the answer is a null (measured this session)

`backend/evaluation/e4_quantum.py` (`E4_QUANTUM_VERSION = "v1-e4-quantum-1"`), 7,127.9 s.
The protocol is §7.10's, unchanged after the fact. The report is `backend/artifacts/reports/e4_quantum.json`.

### The correctness gates, run before any score was kept

| gate | reps=1 | reps=2 | tolerance |
|---|---|---|---|
| readout vs. Aer, max abs deviation over 36 observables | 1.80e-16 | 1.11e-15 | 1e-10 |
| circuit vs. qiskit `ZZFeatureMap`, max infidelity | 6.66e-16 | 9.99e-16 | 1e-10 |

Both passed at machine precision on both settings. The map is the map it claims to be, so a
null cannot be dismissed as a broken circuit.

The entangling witness, measured on 256 TRAIN rows, confirms the §7.10 amendment's proof:

| reps | mean \|C_ij\| | max \|C_ij\| | reading |
|---|---|---|---|
| 1 | **0.00000000** | 0.00000000 | product state — excluded structurally, never scored |
| 2 | **0.07320875** | 0.71480617 | genuinely entangling |

This matters for what the null *means*. A null from a product-state circuit would say only
"the entanglers did nothing here". This null comes from a map whose connected correlations
are demonstrably non-zero, so it says something stronger: the entanglement was present and
did not help.

### The members reproduce the record exactly

| member | recovered here | on record (DEC-046) | |
|---|---|---|---|
| `cnn@resnet_small` fold 9 | 0.9404758944556537 | 0.9404758944556537 | **bit-identical** |
| `gbm@f97` fold 9 | 0.9402343416976896 | 0.9402343416976896 | **bit-identical** |
| `cnn@resnet_small` inner fold 8 | 0.946739 (epoch 17) | 0.946739 (epoch 17) | matched |

Recovery re-ran the original code path — selection fit with early stopping (stopped at epoch
25, best epoch 17), fixed-epoch refit on folds 1–7 for the fold-8 vector, fixed-epoch refit on
folds 1–8 for the fold-9 vector. The quantum fusion is therefore compared against the baseline
that is actually on record, not against a re-derived lookalike.

### The arms

Fold 9, 2,146 records / 1,917 patients, prevalence 0.5741. 95% CIs are patient-clustered
bootstrap, 2,000 draws.

| arm | fold-9 ROC-AUC | 95% CI |
|---|---|---|
| **`q@zz`** (reps 2, C 0.01) | **0.760505** | [0.7401, 0.7796] |
| `c@poly2` (C 10.0) | 0.884961 | [0.8697, 0.8990] |
| `c@rff36` (C 0.1) | 0.897873 | [0.8839, 0.9110] |
| `fusion@cnn+gbm` — **the bar** | 0.946293 | [0.9364, 0.9550] |
| **`fusion@cnn+gbm+zz`** | **0.946329** | [0.9365, 0.9551] |
| `fusion@cnn+gbm+poly2` | 0.945688 | [0.9358, 0.9546] |
| `fusion@cnn+gbm+rff36` | 0.945662 | [0.9357, 0.9546] |

Every family searched exactly 4 cells (`search_budget_cells_per_family: {zz: 4, poly2: 4, rff36: 4}`).

### The headline, and the pre-registered decision rule applied

> **`fusion@cnn+gbm+zz` − `fusion@cnn+gbm+poly2` = +0.000641, 95% CI [−0.000486, +0.001747] — spans zero.**

§7.10 fixed the reading of that interval before it was computed: *if it spans zero, the quantum
map contributed nothing a same-shape classical map did not, regardless of how either arm scores
against 0.946293.* It spans zero. **The verdict is a NULL.**

The rule is doing real work here rather than rubber-stamping an obvious outcome. The quantum
fusion is the highest number in the table — it beats the bar by +0.000036 and beats both
classical-fusion controls. Without the pre-registration, "our quantum fusion set a new best"
would be a defensible-sounding sentence. It is not a defensible claim, and the interval is why:

| comparison | delta | 95% CI | |
|---|---|---|---|
| **headline** — zz fusion vs. poly2 fusion | +0.000641 | [−0.000486, +0.001747] | spans zero |
| `q@zz` vs. `c@poly2`, single arms | **−0.124457** | [−0.145524, −0.105109] | **excludes zero** |
| zz fusion vs. the bar | +0.000036 | [−0.000070, +0.000143] | spans zero |
| poly2 fusion vs. the bar | −0.000605 | [−0.001678, +0.000510] | spans zero |
| rff36 fusion vs. the bar | −0.000631 | [−0.001341, +0.000086] | spans zero |

### What was actually learned

**1. The quantum map is decisively the *worst* of the three maps of its own shape.** `q@zz`
0.7605 against `c@poly2` 0.8850 and `c@rff36` 0.8979 — a paired deficit of −0.124 whose
interval excludes zero comfortably. This is the only interval in the experiment that excludes
zero, and it points the opposite way from the hypothesis. Over the identical index set (8
singletons + 28 pairs, verified column-for-column against `all_pair_masks`), with an identical
head and an identical budget, the plain degree-2 polynomial extracts substantially more
label-relevant structure from the same 8 PCA components than the Havlicek encoding does.

**2. The near-term regime was not the obstacle.** §7.7 measured `rbf_svm@pca08` at 0.915006 —
97.3% of the 97-feature tabular ceiling — so 8 dimensions is a real arena, and the quantum
arm had a fair one. It scored 0.7605 in it.

**3. The fusion is saturated, and that is the finding with the longest reach.** Adding *any*
36-column map to `cnn+gbm` moves it by less than ±0.0007, with every interval spanning zero —
quantum or classical, better or worse standalone. This is §7.8's "the residual error is a
property of the task rather than of model capacity" reading, now confirmed from a third
direction. It also means the classical-fusion control DEC-046 pre-registered was not a
formality: had only `fusion@cnn+gbm+zz` been run, its +0.000036 over the bar would have looked
like a small quantum contribution, when the honest reading is that the slot it fills is inert.

**4. C selected at the grid edge for `q@zz`** (0.01, the most-regularised cell of
`HEAD_C_GRID`). Recorded as a caveat, not repaired: widening the grid for the losing arm alone
would break the same-shape budget contract and is exactly the metric shopping the mission
forbids. The direction is also informative — the head pushing the ZZ features toward zero
weight is consistent with their carrying little label-relevant signal.

### What this does *not* say

It does not say quantum feature maps cannot help on ECG. It says **this** map (Havlicek ZZ,
reps 2, Z-basis readout over 8 PCA components), with **this** head, on **this** task, does not
— and loses to its own classical shadow while doing so. §7.10 fixed a tiny budget precisely so
that this statement stays narrow and honest; arXiv:2507.11401 sampled 400 entanglement
topologies and reported the 16% that won, and the guard against becoming that paper is not
searching until something wins.

Fold 10 was not read. `test_partition_used: false`, `patient_overlap: 0`, `ecg_id_overlap: 0`,
`pca_sees_fold_9: false` in the report.

**A null, reported honestly, is what §7.4 said an acceptable outcome of this phase looks like.**

## 8. Next steps
1. ~~Finish PTB-XL metadata download → run `backend/dataset/ptbxl.py --audit` → append verified audit report.~~ **DONE (§7.1).**
2. ~~Acquire signal files and build a fold-honest ECG feature representation.~~ **DONE for the bounded sample (§7.2, §7.3):** 200 records / 400 files SHA-256-verified; `ecg-v1` 97-feature extractor validated on 127 TRAIN records with 0 failures and 0 non-finite cells.
3. ~~**Bulk signal download** — fold-filtered `records100` fetch in resumable background batches.~~ **DONE (§7.5):** 19,601 records / 39,202 files, **39,202 SHA-256 OK, 0 mismatch, 0 unknown, `fully_verified: True`**, 459.9 MiB. Fold-10 signals deliberately never fetched. A truncated local `SHA256SUMS.txt` was found and replaced.
4. ~~**TRAIN-only fitting stage** — median imputer, scaler, and selection/PCA down to the compact 8–32 dims, as a separate module.~~ **DONE (§7.6):** full 19,601 × 97 matrix built with 0 failures and 0 non-finite cells; `ecg-fit-1` fitted on folds 1–8 only (97 → 16 at 0.7745 variance; 8d = 0.6003, 32d = 0.9275), with the non-TRAIN refusal proved effective and fit/eval patient disjointness checkable from the saved artifact.
5. ~~**Build the classical failure map on PTB-XL** — all four rungs: simple, feature-engineered, modern deep, best fusion, with matched 8/16/32-dim controls, patient-level, permutation-tested.~~ **DONE (§7.7 + §7.8).** Ceiling **0.946293** (`fusion@cnn+gbm`); tabular **0.940234**; deep **0.940476**, which ties the tabular arm with a paired interval spanning zero; matched controls 0.915006 / 0.927408 / 0.931961; permutation p = 0.004975. Failure map of the strongest arm recorded, dominated by false negatives and concentrated in isolated HYP (0.369), isolated CD/MI (~0.75) and the under-40s (0.565).
6. ~~**Focused prior-art / novelty audit for quantum-ML on ECG.**~~ **DONE (§7.9).** Systematic arXiv sweep: **102 of 103 PTB-XL papers (2020-04-28 → 2026-09-15) examined, exactly one quantum** — arXiv:2603.27269, a 6-qubit VQC distillation student whose *classical* teacher won. So "first on PTB-XL" is false and may not be claimed; what is open is the question that paper structurally could not ask, since a distillation student is bounded above by its teacher. The evaluation stance is **not** novel either (2608.18155, 2607.15815, 2605.19233 already publish leakage control, matched twins and attribution audits). The one gap the evidence supports: classical ECG work knows stacking wins (2609.12803), quantum work still compares against **single** models — so no paper in this sweep controls a quantum arm against a *fused* classical baseline. That is exactly the control §7.8 pre-registered, on internal evidence, before this search ran. Limits stated in §7.9: arXiv-only, abstract-level, `WebSearch` unavailable and one query quota-failed.
7. ~~**Then, and only then:** transfer and re-evaluate the E-series quantum infrastructure on the compact ECG representation, under §7.4's five conditions **plus** the classical-fusion control pre-registered in §7.8.~~ **DONE (§7.11), and the answer is a null.** `fusion@cnn+gbm+zz` − `fusion@cnn+gbm+poly2` = **+0.000641, CI [−0.000486, +0.001747]** — spans zero, so by §7.10's pre-registered rule the quantum feature map contributed nothing a classical map of the same shape did not. The rule earned its place: `fusion@cnn+gbm+zz` **0.946329** is the highest number in the phase and beats the §7.8 bar, so "our quantum fusion set a new best" was writable and indefensible, and §7.10 removed the choice before there was anything to choose. The only interval excluding zero runs the other way: `q@zz` 0.760505 vs `c@poly2` 0.884961 = **−0.124457 [−0.145524, −0.105109]** over an identical index set, head and budget. And the fusion slot is **saturated** — any 36-column map moves `cnn+gbm` by <±0.0007 with every interval spanning zero — which is the third independent sign that the residual error belongs to the task rather than to model capacity. Not an artifact: both E3 correctness gates passed at machine precision and the entangling witness measured reps=2 mean \|C_ij\| 0.07320875, so the entanglement was present, was measured, and did not help.
8. **Fold 10 stays closed.** Fold 9 has now been read for tabular arm selection (§7.7) and again for deep/fusion reporting (§7.8); it is a selection surface, not a clean generalisation estimate. The fold-10 signals remain undownloaded, and the single permitted read is reserved for whatever arm the phase finally proposes.

*Guiding constraint: the mission is to find a better scientific arena for discovering a real quantum contribution — not to find data that yields a favourable number. A null on PTB-XL, honestly reported, is a valid and valuable outcome. §7.8 made a null somewhat more likely and considerably more interesting; §7.9 found that the field's own methodology literature expects one; §7.11 measured it. The arena was the right one: unlike E3's 8-dimensional crushed images, 8 dimensions here retain 97.3% of the 97-feature ceiling, so this null says something about the quantum feature map rather than about a destroyed input. That is the deliverable.*
