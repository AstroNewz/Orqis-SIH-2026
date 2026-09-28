# FINAL 20–30 PAGE REPORT PLAN

**Document:** Orqis — Hybrid Quantum–Classical Screening Platform: Engineering Report and Measured Quantum Contribution
**Class:** `report`, 11pt, A4. **Target:** 26 pages body + front matter + bibliography.
**Source:** `report/main.tex` + `report/references.bib`.

## Structure and page budget

| § | Chapter | Pages | Content anchor |
|---|---|---|---|
| — | Title, abstract, ToC, notation, integrity statement | 4 | Metric-naming rule; ORAL vs ECG separation rule |
| 1 | Introduction and Problem Framing | 2 | SIH26139; what is and is not claimed |
| 2 | System Architecture | 3 | 7-stage pipeline; 20 endpoints; additive track layer + its regression test |
| 3 | Datasets and Cohort Construction | 3 | 7,731 → 2,469 → 2,436/328; 33 exclusions; PTB-XL 19,601/16,965 byte-verified |
| 4 | Preprocessing and Feature Engineering | 2 | LAB/HOG/LBP-GLCM/multiscale/MobileNetV3; C7 = 1,195-d; 97 `ecg-v1` |
| 5 | Localization | 2 | Frozen SHA256-pinned regressor; 0.9816 acceptance vs IoU 0.5279 on n=47; DEC-024 geometry leak |
| 6 | Quality Control | 1.5 | Backend gate (7 checks) vs client red-chromatic heuristic; "not an anatomy classifier" |
| 7 | Classical Machine Learning — Oral | 3 | 0.913038 validation; full confusion matrix; held-out test table (4 models) |
| 8 | Quantum Machine Learning — Seven Families | 5 | E0 pixel VQC · E0.1 plateau · E2 reservoir · E3 ×4 · E4 ZZ; circuit proofs; hardware infeasibility |
| 9 | Statistical Methodology | 2.5 | Two null families; why column-permutation gates; patient-clustered paired bootstrap; pre-registration |
| 10 | Results — ECG Arena | 2 | 13-arm leaderboard; deep baseline; the two nulls and the one significant deficit |
| 11 | Mobile Application | 1.5 | 193 tests; analyzer clean; VERIFY WITH STITCH |
| 12 | Testing and Verification | 1 | 1,231 tests passing; the C7 anchor assertion |
| 13 | Security and Privacy | 1.5 | Verified controls table + honest deficiencies table |
| 14 | Feasibility, Cost, Limitations | 2 | 62,940 CX vs 16 CX; cost NOT MEASURED; 14 limitations |
| 15 | Conclusion | 1 | The null as the contribution |
| — | Appendix A: Claims we do not make · Appendix B: Artifact index | 2 | Traceability table: every number → file |

## Figures and tables — all from verified values, none fabricated

**Tables (booktabs/longtable):** cohort funnel · 33-exclusion ledger · C7 validation metrics · oral test 4-model comparison · ECG 13-arm leaderboard · same-shape arm comparison · fusion comparison · paired-bootstrap decisions · circuit validation residuals · entangling witness · state-prep CX scaling · quality thresholds with observed margins · security controls · test counts · artifact index.

**Figures (TikZ/PGFPlots — every plotted value verified):**
1. Pipeline block diagram (TikZ, structural).
2. Cohort funnel (TikZ, structural).
3. Forest plot of the four decisive bootstrap intervals — `single_arm_zz_vs_poly2` [−0.145524, −0.105109], headline [−0.000486, +0.001747], `deep_vs_tabular` [−0.006101, +0.006374], `gbm@f97` [0.929748, 0.949353]. **This is the report's centrepiece figure.**
4. ECG leaderboard bar chart (13 measured ROC-AUCs).
5. State-preparation CX scaling, log-y: measured 6/8/10/12 qubits + extrapolated 16 + SBM bound vs the 16-CX ansatz line.
6. Permutation-null histogram schematic for the ECG label null (observed 0.940234 vs null mean 0.507865, max 0.567394) — drawn from the four published summary statistics, labelled as a summary-statistic rendering, not raw draws.

**No screenshots** (none exist in-repo that can be honestly captioned) and no fabricated plots.

## Editorial rules enforced throughout

1. Every metric carries name + dataset + split + cohort level.
2. ORAL and ECG results never share a table; chapter headers state the modality.
3. Validation and test figures are never compared (prevalence 0.404 vs 0.047).
4. Nulls are stated as results, with intervals, in the same typeface as positives.
5. Implemented / verified-research / future are visually separated in §2 and §14.
6. `VERIFY WITH STITCH` and `NOT MEASURED` appear literally where they apply.
