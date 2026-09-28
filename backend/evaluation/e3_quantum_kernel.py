"""E3 Round 2: does a *quantum kernel* over the 8-qubit states add information beyond C7?

Round 1 (:mod:`backend.evaluation.e3_hybrid_fusion`) fed the 16 local-Z observables of the
E2 map into a *linear* logistic head and found no defensible advantage over the classical
fusion baseline C7 (validation PR-AUC 0.913038). A linear head on 16 expectation values is
a weak reader of an 8-qubit state. The natural next rung of the mission ladder (§30) is the
**fidelity quantum kernel** (Havlicek et al. 2019): keep the *same* fold-honest
MobileNet->PCA-8->angle encoding, but instead of reading 16 observables, use the full
inner product between quantum states as an SVM kernel::

    K(x, x') = |<psi(x) | psi(x')>|^2

Because the E2 statevectors are exact and cheap (256-dim), the whole Gram matrix is a
single ``|S S^H|^2`` GEMM -- no sampling, no estimator noise. An SVM on that kernel can
carve decision boundaries a linear head on 16 features cannot.

The honest hazard this module measures rather than assumes
----------------------------------------------------------
Fidelity quantum kernels are known to **exponentially concentrate** as qubit count grows
(Thanasilp et al. 2022, Huang et al. 2021): every off-diagonal entry collapses toward a
constant, the Gram matrix approaches the identity, and the SVM degenerates into a
nearest-neighbour-on-nothing classifier that "learns" only the training labels. At 8 qubits
with angle encoding in ``[0, pi/2]`` this may or may not bite -- so the module reports the
**off-diagonal spread of the train Gram** (mean, sd, min, max) as a concentration witness,
exactly as E2 reports connected correlations as an entanglement witness. A near-constant
off-diagonal is an honest negative result about the kernel, not something to hide behind an
accuracy number.

Matched classical control (DEC-033 discipline, at the kernel level)
-------------------------------------------------------------------
The quantum kernel is never reported alone. The matched control is an **RBF-SVM on the same
PCA-8 input** (L2-normalised, median-heuristic bandwidth -- the same convention the RFF
control uses), i.e. the classical kernel analogue of the quantum one. If the quantum kernel
does not beat an ordinary RBF kernel on identical inputs and folds, there is no
kernel-level quantum advantage, and the module says so.

Discipline (identical to Round 1 / E1 / E2)
-------------------------------------------
C7 is rebuilt and must reproduce 0.913038 to 1e-6 before any conclusion; the quantum map is
validated against Aer to <=1e-10; the PCA/angle encoding is refit **inside each fold-train
only** so no kernel row is built from a held-out point's fitted statistics; scores are the
SVM ``decision_function`` (rank-based -- PR-AUC, the patient bootstrap, and the
``_increment_null`` two-column stack are all rank/standardise based, so an uncalibrated
margin is exactly right and no arbitrary probability squashing is introduced); the gating
test is E1.1's ``_increment_null`` with C7 as reference; TEST is never read. No number is
fabricated -- a concentrated kernel or a sub-C7 result is reported as measured (§29B).

Usage::

    python -m backend.evaluation.e3_quantum_kernel --out reports_e3_quantum_kernel.json
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.core.config import Settings
from backend.evaluation.e0_controls import C_GRID, N_FOLDS, grouped_folds, median_heuristic_gamma
from backend.evaluation.e1_failure_map import (
    BOOTSTRAP_DRAWS,
    INCREMENT_PERMUTATIONS,
    PRIMARY,
    SEED,
    E1Inputs,
    _average_precision,
    _cv_select,
    _increment_null,
    _logistic_arm,
    _standardise,
    bootstrap_pr_auc,
    build_inputs,
    candidate_matrices,
)
from backend.evaluation.e3_hybrid_fusion import (
    C7_ANCHOR_TOLERANCE,
    C7_DIM,
    C7_REFERENCE_PR_AUC,
    C5_DIM,
    MOBILENET_DIM,
    QUANTUM_VALIDATION_SAMPLE,
    HybridFusionError,
)
from backend.ml.quantum_visual import PCA_COMPONENTS, QuantumVisualPreprocessor, _l2_normalise
from quantum_ml.visual_circuit import QuantumVisualFeatureMap, validate_against_aer

logger = logging.getLogger(__name__)

#: Bump when the Round-2 protocol semantics change.
E3_QUANTUM_KERNEL_VERSION = "v1-e3-quantum-kernel-1"


# ------------------------------------------------------------------------ kernels
def _fidelity_gram(states_rows: np.ndarray, states_cols: np.ndarray) -> np.ndarray:
    """``|<psi_i | phi_j>|^2`` for two batches of statevectors -> real ``(n_rows, n_cols)``.

    Exact: the E2 statevectors are exact Aer amplitudes, so this is the true fidelity
    kernel with no estimator noise. The diagonal of ``_fidelity_gram(S, S)`` is 1 to
    floating point (a normalised state overlaps itself perfectly).
    """
    overlap = np.asarray(states_rows) @ np.conjugate(np.asarray(states_cols)).T
    return np.abs(overlap) ** 2


def _offdiagonal_summary(gram: np.ndarray) -> Dict[str, float]:
    """Concentration witness: spread of the train Gram's off-diagonal entries.

    A healthy kernel has meaningful spread; a concentrated one collapses toward a constant
    (sd -> 0, mean -> a fixed floor), which makes the SVM degenerate. Reported, never
    hidden.
    """
    n = gram.shape[0]
    if n < 2:
        return {"n_offdiagonal": 0, "mean": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0}
    off = gram[~np.eye(n, dtype=bool)]
    return {
        "n_offdiagonal": int(off.size),
        "mean": float(np.mean(off)),
        "sd": float(np.std(off)),
        "min": float(np.min(off)),
        "max": float(np.max(off)),
    }


# --------------------------------------------------------------- kernel arm scorers
def _score_quantum_kernel_arm(
    source_train: np.ndarray,
    source_validation: np.ndarray,
    y_train: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    feature_map: QuantumVisualFeatureMap,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Fold-honest precomputed-kernel SVM on the fidelity quantum kernel.

    Per fold the PCA/angle encoding is refit on the fold-train source only, states are
    computed for fold-train and fold-test, and the SVM is fit on ``K(train, train)`` and
    scored on ``K(test, train)`` -- so no kernel entry a fold is scored on was built from a
    held-out point's fitted preprocessing. Scores are ``decision_function`` (rank-based).
    """
    from sklearn.svm import SVC

    y = np.asarray(y_train, dtype=int)
    source_train = np.asarray(source_train, dtype=np.float64)
    source_validation = np.asarray(source_validation, dtype=np.float64)

    fold_kernels: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in folds:
        pre = QuantumVisualPreprocessor.fit(source_train[train_idx], seed=seed)
        states_train = feature_map.statevectors(pre.to_angles(source_train[train_idx]))
        states_test = feature_map.statevectors(pre.to_angles(source_train[test_idx]))
        k_train = _fidelity_gram(states_train, states_train)
        k_test = _fidelity_gram(states_test, states_train)
        fold_kernels.append((train_idx, test_idx, k_train, k_test))

    def _oof_at(c: float) -> np.ndarray:
        oof = np.full(y.size, np.nan, dtype=np.float64)
        for train_idx, test_idx, k_train, k_test in fold_kernels:
            if len(set(y[train_idx].tolist())) < 2:
                continue
            svc = SVC(C=float(c), kernel="precomputed", class_weight="balanced", random_state=seed)
            svc.fit(k_train, y[train_idx])
            oof[test_idx] = svc.decision_function(k_test)
        return oof

    def _score(c: float) -> Optional[float]:
        oof = _oof_at(c)
        usable = np.isfinite(oof)
        if usable.sum() < 2:
            return None
        return _average_precision(y[usable], oof[usable])

    best_c, by_c = _cv_select(C_GRID, _score, label="e3 quantum-kernel C")
    oof = _oof_at(float(best_c))
    usable = np.isfinite(oof)
    oof_pr_auc = None if usable.sum() < 2 else _average_precision(y[usable], oof[usable])

    # Validation: fit the encoding + SVM on all TRAIN, score validation once.
    pre_all = QuantumVisualPreprocessor.fit(source_train, seed=seed)
    states_all = feature_map.statevectors(pre_all.to_angles(source_train))
    states_val = feature_map.statevectors(pre_all.to_angles(source_validation))
    k_all = _fidelity_gram(states_all, states_all)
    k_val = _fidelity_gram(states_val, states_all)
    svc = SVC(C=float(best_c), kernel="precomputed", class_weight="balanced", random_state=seed)
    svc.fit(k_all, y)
    validation_score = svc.decision_function(k_val)

    return {
        "kernel": "fidelity_quantum",
        "representation_dim": int(states_all.shape[1]),  # Hilbert-space dimension 2**8
        "selected_c": float(best_c),
        "cv_average_precision_by_setting": by_c,
        "train_out_of_fold": oof,
        "train_cv_pr_auc": None if oof_pr_auc is None else round(oof_pr_auc, 6),
        "validation_score": validation_score,
        "n_support_vectors": int(svc.n_support_.sum()),
        "kernel_offdiagonal_train": _offdiagonal_summary(k_all),
    }


def _score_rbf_arm(
    source_train: np.ndarray,
    source_validation: np.ndarray,
    y_train: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Matched classical control: RBF-SVM on the same L2-normalised PCA-8 input.

    The classical kernel analogue of the quantum kernel -- same PCA-8 features, same
    L2-normalisation and median-heuristic bandwidth as the RFF control (DEC-033), but an
    exact RBF kernel instead of its random-Fourier approximation. Fold-honest: PCA and the
    bandwidth are fit on the fold-train source only.
    """
    from sklearn.svm import SVC

    y = np.asarray(y_train, dtype=int)
    source_train = np.asarray(source_train, dtype=np.float64)
    source_validation = np.asarray(source_validation, dtype=np.float64)

    fold_inputs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
    for train_idx, test_idx in folds:
        pre = QuantumVisualPreprocessor.fit(source_train[train_idx], seed=seed)
        xn_train = _l2_normalise(pre.pca_scores(source_train[train_idx]))
        xn_test = _l2_normalise(pre.pca_scores(source_train[test_idx]))
        gamma, _ = median_heuristic_gamma(xn_train @ xn_train.T)
        fold_inputs.append((train_idx, test_idx, xn_train, xn_test, float(gamma)))

    def _oof_at(c: float) -> np.ndarray:
        oof = np.full(y.size, np.nan, dtype=np.float64)
        for train_idx, test_idx, xn_train, xn_test, gamma in fold_inputs:
            if len(set(y[train_idx].tolist())) < 2:
                continue
            svc = SVC(C=float(c), kernel="rbf", gamma=gamma, class_weight="balanced", random_state=seed)
            svc.fit(xn_train, y[train_idx])
            oof[test_idx] = svc.decision_function(xn_test)
        return oof

    def _score(c: float) -> Optional[float]:
        oof = _oof_at(c)
        usable = np.isfinite(oof)
        if usable.sum() < 2:
            return None
        return _average_precision(y[usable], oof[usable])

    best_c, by_c = _cv_select(C_GRID, _score, label="e3 rbf-svm C")
    oof = _oof_at(float(best_c))
    usable = np.isfinite(oof)
    oof_pr_auc = None if usable.sum() < 2 else _average_precision(y[usable], oof[usable])

    pre_all = QuantumVisualPreprocessor.fit(source_train, seed=seed)
    xn_all = _l2_normalise(pre_all.pca_scores(source_train))
    xn_val = _l2_normalise(pre_all.pca_scores(source_validation))
    gamma_all, _ = median_heuristic_gamma(xn_all @ xn_all.T)
    svc = SVC(C=float(best_c), kernel="rbf", gamma=float(gamma_all), class_weight="balanced", random_state=seed)
    svc.fit(xn_all, y)
    validation_score = svc.decision_function(xn_val)

    return {
        "kernel": "rbf",
        "representation_dim": int(xn_all.shape[1]),
        "selected_c": float(best_c),
        "selected_gamma": float(gamma_all),
        "cv_average_precision_by_setting": by_c,
        "train_out_of_fold": oof,
        "train_cv_pr_auc": None if oof_pr_auc is None else round(oof_pr_auc, 6),
        "validation_score": validation_score,
        "n_support_vectors": int(svc.n_support_.sum()),
    }


def _roc_auc(y: np.ndarray, scores: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, np.asarray(scores, dtype=np.float64)))


def _kernel_arm_summary(
    name: str,
    fitted: Dict[str, Any],
    inputs: E1Inputs,
    *,
    bootstrap_draws: int,
    seed: int,
) -> Dict[str, Any]:
    """PR-AUC + ROC-AUC + patient bootstrap for a decision-function-scored SVM arm.

    Only rank-based metrics are reported: the SVM ``decision_function`` is an uncalibrated
    margin, so a 0.5-threshold accuracy would be meaningless -- PR-AUC and ROC-AUC are
    rank-invariant and exactly correct on it, and are what the mission compares.
    """
    scores = np.asarray(fitted["validation_score"], dtype=np.float64)
    validation_pr_auc = _average_precision(inputs.y_validation, scores)
    bootstrap = bootstrap_pr_auc(
        inputs.y_validation, scores, inputs.patients_validation, draws=bootstrap_draws, seed=seed
    )
    summary = {
        "arm": name,
        "kernel": fitted["kernel"],
        "representation_dim": fitted["representation_dim"],
        "selected_c": fitted["selected_c"],
        "cv_average_precision_by_setting": fitted["cv_average_precision_by_setting"],
        "train_cv_pr_auc": fitted["train_cv_pr_auc"],
        "validation_pr_auc": None if validation_pr_auc is None else round(validation_pr_auc, 6),
        "validation_roc_auc": _roc_auc(inputs.y_validation, scores),
        "n_support_vectors": fitted["n_support_vectors"],
        "score_note": "SVM decision_function (uncalibrated margin); PR/ROC-AUC are rank-based",
        "validation_bootstrap": bootstrap,
    }
    if "selected_gamma" in fitted:
        summary["selected_gamma"] = fitted["selected_gamma"]
    if "kernel_offdiagonal_train" in fitted:
        summary["kernel_offdiagonal_train"] = fitted["kernel_offdiagonal_train"]
    return summary


# ------------------------------------------------------------------------- driver
def run(
    *,
    config: Optional[Settings] = None,
    seed: int = SEED,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    increment_permutations: int = INCREMENT_PERMUTATIONS,
    quantum_validation_sample: int = QUANTUM_VALIDATION_SAMPLE,
) -> Dict[str, Any]:
    """Run the E3 Round-2 quantum-kernel experiment on the primary leak-free condition."""
    started = time.perf_counter()

    per_condition, provenance = build_inputs(config=config, conditions=(PRIMARY,))
    inputs = per_condition[PRIMARY]
    if set(inputs.patients_train) & set(inputs.patients_validation):
        raise HybridFusionError("Patient overlap between train and validation; refusing to measure.")

    folds, cv_note = grouped_folds(inputs.y_train, inputs.patients_train, seed=seed, n_splits=N_FOLDS)
    logger.info(
        "E3-QK primary %s: %d train / %d validation (%d / %d positive); %s",
        PRIMARY, inputs.y_train.size, inputs.y_validation.size,
        int(inputs.y_train.sum()), int(inputs.y_validation.sum()), cv_note,
    )

    # --- Rebuild C7 exactly as Round 1 / e1_failure_map.run() does, and anchor it.
    specs = candidate_matrices(inputs)
    c6_train, c6_validation = specs["C6"]["train"], specs["C6"]["validation"]
    c5_train, c5_validation = specs["C5"]["train"], specs["C5"]["validation"]
    if c6_train.shape[1] != MOBILENET_DIM or c5_train.shape[1] != C5_DIM:
        raise HybridFusionError(
            f"Unexpected C6/C5 dims ({c6_train.shape[1]}/{c5_train.shape[1]}); expected "
            f"{MOBILENET_DIM}/{C5_DIM}."
        )
    c7_train = np.hstack([c6_train, c5_train]).astype(np.float64)
    c7_validation = np.hstack([c6_validation, c5_validation]).astype(np.float64)
    if c7_train.shape[1] != C7_DIM:
        raise HybridFusionError(f"C7 has {c7_train.shape[1]} columns, expected {C7_DIM}.")

    c7_xt, c7_xv = _standardise(c7_train, c7_validation)
    c7_fit = _logistic_arm(c7_xt, inputs.y_train, c7_xv, folds, seed=seed)
    c7_pr_auc = _average_precision(inputs.y_validation, c7_fit["validation_probability"])
    anchor_ok = c7_pr_auc is not None and abs(float(c7_pr_auc) - C7_REFERENCE_PR_AUC) <= C7_ANCHOR_TOLERANCE
    anchor = {
        "target_pr_auc": C7_REFERENCE_PR_AUC,
        "reproduced_pr_auc": None if c7_pr_auc is None else round(float(c7_pr_auc), 6),
        "selected_c": c7_fit["hyperparameters"].get("C"),
        "tolerance": C7_ANCHOR_TOLERANCE,
        "reproduced": bool(anchor_ok),
    }
    if not anchor_ok:
        logger.warning("C7 anchor NOT reproduced (%s vs %s); conclusions suppressed.",
                       anchor["reproduced_pr_auc"], C7_REFERENCE_PR_AUC)

    # --- Quantum feature map, Aer-validated before any quantum number is used.
    mobilenet_train = np.asarray(inputs.e1_train["mobilenet"], dtype=np.float64)
    mobilenet_validation = np.asarray(inputs.e1_validation["mobilenet"], dtype=np.float64)
    feature_map = QuantumVisualFeatureMap()
    pre_check = QuantumVisualPreprocessor.fit(mobilenet_train, seed=seed)
    sample = min(int(quantum_validation_sample), mobilenet_train.shape[0])
    aer_check = validate_against_aer(feature_map, pre_check.to_angles(mobilenet_train[:sample]))

    # --- Arms.
    logger.info("  arm quantum_kernel ...")
    qk_fit = _score_quantum_kernel_arm(
        mobilenet_train, mobilenet_validation, inputs.y_train, folds, feature_map=feature_map, seed=seed
    )
    logger.info("  arm rbf_svm ...")
    rbf_fit = _score_rbf_arm(mobilenet_train, mobilenet_validation, inputs.y_train, folds, seed=seed)

    arms = {
        "quantum_kernel": _kernel_arm_summary("quantum_kernel", qk_fit, inputs, bootstrap_draws=bootstrap_draws, seed=seed),
        "rbf_svm": _kernel_arm_summary("rbf_svm", rbf_fit, inputs, bootstrap_draws=bootstrap_draws, seed=seed),
    }
    # C7 reference arm (logistic probability) for the leaderboard.
    c7_boot = bootstrap_pr_auc(
        inputs.y_validation, c7_fit["validation_probability"], inputs.patients_validation,
        draws=bootstrap_draws, seed=seed,
    )
    arms["c7"] = {
        "arm": "c7",
        "kernel": "linear_logistic",
        "representation_dim": C7_DIM,
        "selected_c": c7_fit["hyperparameters"].get("C"),
        "train_cv_pr_auc": c7_fit.get("train_score"),
        "validation_pr_auc": None if c7_pr_auc is None else round(float(c7_pr_auc), 6),
        "validation_roc_auc": _roc_auc(inputs.y_validation, c7_fit["validation_probability"]),
        "validation_bootstrap": c7_boot,
    }

    # --- Increment tests: C7 -> {quantum_kernel, rbf_svm}.
    increments: Dict[str, Dict[str, Any]] = {}
    reference_val = np.asarray(c7_fit["validation_probability"], dtype=np.float64)
    reference_oof = np.asarray(c7_fit["train_out_of_fold"], dtype=np.float64)
    for name, fit in (("quantum_kernel", qk_fit), ("rbf_svm", rbf_fit)):
        logger.info("  increment C7 -> %s ...", name)
        increments[name] = _increment_null(
            reference_val,
            np.asarray(fit["validation_score"], dtype=np.float64),
            reference_oof,
            np.asarray(fit["train_out_of_fold"], dtype=np.float64),
            inputs.y_train,
            inputs.y_validation,
            inputs.patients_train,
            inputs.patients_validation,
            draws=increment_permutations,
            seed=seed,
        )

    # --- Honest verdict.
    qk_val = arms["quantum_kernel"]["validation_pr_auc"]
    rbf_val = arms["rbf_svm"]["validation_pr_auc"]
    c7_val = arms["c7"]["validation_pr_auc"]
    qk_survives = increments["quantum_kernel"].get("survives_gating_null")
    rbf_survives = increments["rbf_svm"].get("survives_gating_null")
    qk_beats_c7 = qk_val is not None and c7_val is not None and qk_val > c7_val
    qk_beats_rbf = qk_val is not None and rbf_val is not None and qk_val > rbf_val

    offdiag = arms["quantum_kernel"].get("kernel_offdiagonal_train", {})
    # A near-constant off-diagonal (sd collapsing toward 0) means the fidelity kernel has
    # concentrated and the SVM is degenerate; report it plainly.
    concentrated = bool(offdiag.get("sd", 1.0) < 1e-3)
    defensible_quantum_advantage = bool(qk_survives)
    quantum_specific = bool(qk_survives and not rbf_survives)

    if not anchor_ok:
        conclusion = "C7 anchor did not reproduce; experiment not trustworthy, no conclusion drawn."
    elif concentrated:
        conclusion = (
            "The fidelity quantum kernel has exponentially concentrated (train off-diagonal "
            f"sd {offdiag.get('sd'):.2e}): the SVM is degenerate and any score it produces is "
            "an artefact, not quantum signal. Reported as a measured negative (§29B)."
        )
    elif quantum_specific:
        conclusion = (
            "The C7 -> quantum-kernel increment survives the patient-blocked column-permutation "
            "null while the matched RBF-SVM increment does not: on this evidence the quantum "
            "kernel carries information beyond C7 that a classical RBF kernel on identical inputs "
            "does not. Freeze and re-examine before any test-set claim."
        )
    elif defensible_quantum_advantage:
        conclusion = (
            "The C7 -> quantum-kernel increment survives its gating null, but the matched RBF-SVM "
            "increment also survives: the added information is not quantum-specific. Not a quantum "
            "advantage."
        )
    else:
        conclusion = (
            "The C7 -> quantum-kernel increment does NOT survive the patient-blocked "
            "column-permutation null: on this evidence the fidelity quantum kernel adds no "
            "information beyond C7. A null result (§29B), reported as measured."
        )

    leaderboard = sorted(
        (
            {"arm": n, "validation_pr_auc": arms[n]["validation_pr_auc"],
             "validation_roc_auc": arms[n].get("validation_roc_auc"),
             "train_cv_pr_auc": arms[n]["train_cv_pr_auc"]}
            for n in arms
        ),
        key=lambda r: (r["validation_pr_auc"] is None, -(r["validation_pr_auc"] or 0.0)),
    )

    summary = {
        "primary_condition": PRIMARY,
        "c7_validation_pr_auc": c7_val,
        "quantum_kernel_validation_pr_auc": qk_val,
        "rbf_svm_validation_pr_auc": rbf_val,
        "quantum_kernel_beats_c7": qk_beats_c7,
        "quantum_kernel_beats_rbf_control": qk_beats_rbf,
        "quantum_kernel_concentrated": concentrated,
        "quantum_kernel_increment_survives_gating_null": qk_survives,
        "rbf_svm_increment_survives_gating_null": rbf_survives,
        "defensible_quantum_advantage_over_c7": defensible_quantum_advantage,
        "quantum_specific_advantage": quantum_specific,
        "leaderboard": leaderboard,
        "conclusion": conclusion,
    }

    duration = time.perf_counter() - started
    return {
        "e3_quantum_kernel_version": E3_QUANTUM_KERNEL_VERSION,
        "question": (
            "Does a fidelity quantum kernel over the 8-qubit E2 states add information beyond C7 "
            "(validation PR-AUC 0.913038), under identical patient-grouped folds, train-only "
            "encoding, and a matched classical RBF-SVM control?"
        ),
        "test_partition_used": False,
        "test_partition_note": (
            "Reads TRAIN and VALIDATION only via e1_failure_map.build_inputs, which iterates only "
            "('train', 'validation'). No code path selects the test partition."
        ),
        "primary_condition": PRIMARY,
        "cv_note": cv_note,
        "rows": inputs.shape_note,
        "c7_anchor": anchor,
        "quantum_feature_map_aer_check": aer_check,
        "arms": arms,
        "increment_vs_c7": increments,
        "summary": summary,
        "provenance": provenance,
        "protocol": {
            "reference_baseline": "C7 = hstack([C6 mobilenet(576), C5 descriptor+multiscale(619)]) = 1195, logistic",
            "quantum_kernel": "K(x,x')=|<psi(x)|psi(x')>|^2 over exact 8-qubit E2 statevectors; precomputed-kernel SVM",
            "matched_control": "RBF-SVM on the same L2-normalised PCA-8 input, median-heuristic bandwidth (DEC-033 analogue)",
            "folds": f"{N_FOLDS}-fold patient-grouped StratifiedGroupKFold on TRAIN, seed {seed}",
            "preprocessing": "train-only; PCA/angle encoding refit inside each fold-train for the OOF column",
            "scores": "SVM decision_function (rank-based PR/ROC-AUC; increment stack standardises)",
            "gating_test": "e1_failure_map._increment_null with C7 as reference",
            "concentration_witness": "off-diagonal spread of the train fidelity Gram (sd->0 means degenerate)",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "seed": seed,
            "n_folds": N_FOLDS,
            "bootstrap_draws": bootstrap_draws,
            "increment_permutations": increment_permutations,
            "pca_components": PCA_COMPONENTS,
        },
        "duration_seconds": round(duration, 3),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="E3 Round 2: quantum-kernel research experiment.")
    parser.add_argument("--out", type=str, default="reports_e3_quantum_kernel.json")
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--permutations", type=int, default=INCREMENT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    payload = run(seed=args.seed, bootstrap_draws=args.bootstrap_draws, increment_permutations=args.permutations)
    from pathlib import Path

    Path(args.out).write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    logger.info("Wrote %s", args.out)
    s = payload["summary"]
    logger.info(
        "C7=%s quantum_kernel=%s rbf=%s | QK beats C7: %s | QK increment survives: %s (rbf=%s) | concentrated: %s",
        s["c7_validation_pr_auc"], s["quantum_kernel_validation_pr_auc"], s["rbf_svm_validation_pr_auc"],
        s["quantum_kernel_beats_c7"], s["quantum_kernel_increment_survives_gating_null"],
        s["rbf_svm_increment_survives_gating_null"], s["quantum_kernel_concentrated"],
    )
    logger.info("%s", s["conclusion"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
