"""E3: does the E2 quantum-visual representation add information *beyond* the strongest
classical baseline (C7)?

The master-mission question (§30) is not "is the quantum model good" -- E2 already
answered that (validation PR-AUC ~0.879, a defensible secondary signal). It is the
harder one: **does a genuine quantum / hybrid configuration carry information the best
classical fusion baseline does not already have?** C7 -- the E1.1 fusion of MobileNetV3
(576-d) with the descriptor+multiscale family (619-d), 1195 features, logistic head --
scores validation PR-AUC 0.913038 on the primary leak-free condition. That is the bar.

Why this module is almost entirely reused code, by design
---------------------------------------------------------
E1.1 already gated C7 itself with a pre-registered test that answers exactly this shape
of question: :func:`backend.evaluation.e1_failure_map._increment_null`. It stacks a
*reference* prediction column with a *candidate* prediction column, fits a two-parameter
logistic on the leakage-free **out-of-fold TRAIN** predictions, scores VALIDATION, and
asks whether the real candidate column adds more than a **patient-blocked** shuffle of
that same column would (the shuffle preserves within-patient structure, so the null is
not artificially easy). E1 pointed it at ``C1 -> candidate``; E3 points the identical
machinery at ``C7 -> {quantum, rff, pca}``. Nothing about the statistical standard is
re-invented here: the folds (:func:`grouped_folds`), the standardiser
(:func:`_standardise`), the logistic arm (:func:`_logistic_arm`), the metric
(:func:`_metrics`), the patient-level bootstrap (:func:`bootstrap_pr_auc`) and the
increment null are E1/E0 verbatim, so a difference between E3 and E1 is a difference in
*representation*, never in harness.

The one genuinely new piece: fold-honest preprocessing
------------------------------------------------------
C7's features are raw cached descriptors -- no fitted preprocessing -- so E1's own
out-of-fold columns are leakage-free. The quantum / RFF / PCA representations are NOT:
each fits a PCA (and an angle scaler, or an RFF map) that the mission forbids fitting on
anything a fold is about to be scored on ("NEVER allow test-derived PCA", applied to
CV). So :func:`_score_arm` refits the whole preprocessing pipeline **inside each
fold-train only** before building the out-of-fold column. The 8-qubit feature map itself
is fold-independent (0 trainable parameters) and is built once. This makes the quantum
out-of-fold column *stricter* than C7's (its preprocessing never sees the holdout), which
is the conservative direction: it can only make a quantum "win" harder, never easier.

Anchors (correctness gates, not results)
-----------------------------------------
* **C7 must reproduce 0.913038.** The C7 matrix is rebuilt as E1's ``run()`` builds it on
  the primary condition -- ``hstack([C6 mobilenet(576), C5 descriptor+multiscale(619)])``
  -- and scored through the identical ``_standardise`` + ``_logistic_arm`` path. If the
  reproduced validation PR-AUC is not 0.913038 to ``1e-6`` the assembly is wrong and the
  run refuses to draw any conclusion.
* **The quantum feature map is validated against Aer** (``validate_against_aer``) before a
  single quantum feature is used, exactly as E2 gates its numbers.

Discipline (identical to E1/E2)
-------------------------------
TRAIN fits, VALIDATION scores once, **TEST is never read** -- ``build_inputs`` iterates
only ``("train", "validation")`` and there is no code path to the test partition. Every
scaler / PCA / RFF / logistic ``C`` is fitted on train (or fold-train) rows only. Matched
controls (RFF-16 at equal output dimension, PCA-8) are reported beside the quantum arm and
never omitted. No number is fabricated: a null result (quantum adds nothing C7 lacks) is a
valid, reportable outcome (mission §29B), and this module will report it plainly if that
is what the data say.

Usage::

    python -m backend.evaluation.e3_hybrid_fusion --out reports_e3_hybrid_fusion.json
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.core.config import Settings
from backend.evaluation.e0_controls import C_GRID, N_FOLDS, grouped_folds
from backend.evaluation.e0_readout import _metrics
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
from backend.ml.quantum_visual import (
    PCA_COMPONENTS,
    QuantumVisualPreprocessor,
    RandomFourierControl,
)
from quantum_ml.visual_circuit import (
    E2_N_FEATURES,
    QuantumVisualFeatureMap,
    validate_against_aer,
)

logger = logging.getLogger(__name__)

#: Bump when the E3 protocol semantics change so a stored report can never be reinterpreted.
E3_HYBRID_FUSION_VERSION = "v1-e3-hybrid-fusion-1"

#: The reproduction target: C7 (fusion of C6+C5) validation PR-AUC on the PRIMARY condition,
#: read from ``reports_e1_failure_map.json`` (condition ``A_lesion_polygon``, logistic arm).
#: This is a *correctness anchor*, not a number this module is trying to beat by tuning.
C7_REFERENCE_PR_AUC = 0.913038

#: How closely the rebuilt C7 must match the anchor. Same data, same seed, same code path,
#: so this is a strict floating-point agreement, not a tolerance for methodological drift.
C7_ANCHOR_TOLERANCE = 1e-6

#: Expected component/feature dimensions, asserted so a stale cache surfaces as an error
#: rather than a silently different experiment.
MOBILENET_DIM = 576
C5_DIM = 619  # descriptor(163) + multiscale(456)
C7_DIM = MOBILENET_DIM + C5_DIM  # 1195

#: Aer cross-check sample size for the quantum feature map (train rows only).
QUANTUM_VALIDATION_SAMPLE = 24


class HybridFusionError(RuntimeError):
    """Raised when an E3 arm cannot be measured on aligned, patient-safe rows."""


# --------------------------------------------------------------------- representations
#: A preprocessing factory takes a fold-train source matrix, fits every train-only object
#: on it (PCA / angle scaler / RFF), and returns a pure transform applicable to any rows.
Transform = Callable[[np.ndarray], np.ndarray]
Factory = Callable[[np.ndarray], Transform]


def _identity_factory(_fold_source: np.ndarray) -> Transform:
    """C7 carries no fitted preprocessing: its representation is the raw feature block."""
    return lambda x: np.asarray(x, dtype=np.float64)


def _make_quantum_factory(feature_map: QuantumVisualFeatureMap, *, seed: int) -> Factory:
    """MobileNet-576 -> (fold-train) PCA-8 + angle scaler -> fixed 8-qubit map -> 16-d.

    The preprocessor is refit on the fold-train source only; the feature map is
    fold-independent (0 trainable parameters) and passed in already built.
    """

    def factory(fold_source: np.ndarray) -> Transform:
        pre = QuantumVisualPreprocessor.fit(fold_source, seed=seed)
        return lambda x: feature_map.transform(pre.to_angles(x))

    return factory


def _make_pca_factory(*, seed: int) -> Factory:
    """MobileNet-576 -> (fold-train) PCA-8 scores. The raw-linear control at 8 dims."""

    def factory(fold_source: np.ndarray) -> Transform:
        pre = QuantumVisualPreprocessor.fit(fold_source, seed=seed)
        return lambda x: pre.pca_scores(x)

    return factory


def _make_rff_factory(*, seed: int, n_features: int) -> Factory:
    """MobileNet-576 -> (fold-train) PCA-8 -> RFF at equal (16) output dimension.

    DEC-033's honest analogue of a fixed quantum feature map: same PCA-8 input, same
    output width as the quantum vector, RBF bandwidth by the train median heuristic. Fit
    entirely on the fold-train source.
    """

    def factory(fold_source: np.ndarray) -> Transform:
        pre = QuantumVisualPreprocessor.fit(fold_source, seed=seed)
        rff = RandomFourierControl.fit(
            pre.pca_scores(fold_source), n_features=n_features, seed=seed
        )
        return lambda x: rff.transform(pre.pca_scores(x))

    return factory


# ------------------------------------------------------------------------- fold-honest arm
def _score_arm(
    source_train: np.ndarray,
    source_validation: np.ndarray,
    y_train: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    preprocess: Factory,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Fold-honest logistic scoring of one representation, mirroring ``_logistic_arm``.

    The difference from ``_logistic_arm`` is the ``preprocess`` hook: the representation
    is (re)built per fold from the fold-train source only, so a fitted PCA/RFF never sees
    a row it is about to score. For the identity factory (C7) this reduces to the E1
    logistic arm, differing only in that standardisation is fitted per fold-train rather
    than once on whole train -- the strictly leakage-free choice.

    Returns the leakage-free out-of-fold TRAIN column and the fit-on-all-train VALIDATION
    column (both aligned to ``y`` rows), plus the selected ``C`` and its CV trace -- exactly
    the fields ``_increment_null`` and ``bootstrap_pr_auc`` consume.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y_train, dtype=int)
    source_train = np.asarray(source_train, dtype=np.float64)
    source_validation = np.asarray(source_validation, dtype=np.float64)

    # Representations depend only on preprocessing, not on C, so build them once per fold.
    fold_reps: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in folds:
        transform = preprocess(source_train[train_idx])
        rep_train = transform(source_train[train_idx])
        rep_test = transform(source_train[test_idx])
        rep_train, rep_test = _standardise(rep_train, rep_test)
        fold_reps.append((train_idx, test_idx, rep_train, rep_test))

    def _oof_at(c: float) -> np.ndarray:
        oof = np.full(y.size, np.nan, dtype=np.float64)
        for train_idx, test_idx, rep_train, rep_test in fold_reps:
            if len(set(y[train_idx].tolist())) < 2:
                continue
            model = LogisticRegression(
                C=float(c), max_iter=5000, class_weight="balanced", random_state=seed
            )
            model.fit(rep_train, y[train_idx])
            oof[test_idx] = np.asarray(model.predict_proba(rep_test))[:, 1]
        return oof

    def _score(c: float) -> Optional[float]:
        oof = _oof_at(c)
        usable = np.isfinite(oof)
        if usable.sum() < 2:
            return None
        return _average_precision(y[usable], oof[usable])

    best_c, by_c = _cv_select(C_GRID, _score, label="e3 logistic C")
    oof = _oof_at(float(best_c))
    usable = np.isfinite(oof)
    oof_pr_auc = (
        None if usable.sum() < 2 else _average_precision(y[usable], oof[usable])
    )

    # VALIDATION: fit preprocessing + logistic on ALL train, score validation once.
    transform_all = preprocess(source_train)
    rep_train_all = transform_all(source_train)
    rep_validation = transform_all(source_validation)
    rep_train_all, rep_validation = _standardise(rep_train_all, rep_validation)
    final = LogisticRegression(
        C=float(best_c), max_iter=5000, class_weight="balanced", random_state=seed
    )
    final.fit(rep_train_all, y)
    validation_score = np.asarray(final.predict_proba(rep_validation))[:, 1]

    return {
        "representation_dim": int(rep_train_all.shape[1]),
        "selected_c": float(best_c),
        "cv_average_precision_by_setting": by_c,
        "train_out_of_fold": oof,
        "train_cv_pr_auc": None if oof_pr_auc is None else round(oof_pr_auc, 6),
        "validation_score": validation_score,
        "validation_probability": validation_score,
    }


def _arm_summary(
    name: str,
    fitted: Dict[str, Any],
    inputs: E1Inputs,
    *,
    bootstrap_draws: int,
    seed: int,
) -> Dict[str, Any]:
    """Attach the validation PR-AUC and a patient-level bootstrap CI to a fitted arm."""
    metrics = _metrics(inputs.y_validation, fitted["validation_score"])
    validation_pr_auc = None if not metrics else metrics.get("pr_auc")
    bootstrap = bootstrap_pr_auc(
        inputs.y_validation,
        fitted["validation_score"],
        inputs.patients_validation,
        draws=bootstrap_draws,
        seed=seed,
    )
    return {
        "arm": name,
        "representation_dim": fitted["representation_dim"],
        "selected_c": fitted["selected_c"],
        "cv_average_precision_by_setting": fitted["cv_average_precision_by_setting"],
        "train_cv_pr_auc": fitted["train_cv_pr_auc"],
        "validation_pr_auc": None if validation_pr_auc is None else round(validation_pr_auc, 6),
        "validation_metrics": metrics,
        "validation_bootstrap": bootstrap,
    }


# ------------------------------------------------------------------------------- driver
def run(
    *,
    config: Optional[Settings] = None,
    seed: int = SEED,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    increment_permutations: int = INCREMENT_PERMUTATIONS,
    quantum_validation_sample: int = QUANTUM_VALIDATION_SAMPLE,
) -> Dict[str, Any]:
    """Run the E3 hybrid-fusion experiment on the primary leak-free condition.

    Returns the full payload: the C7 anchor check, each standalone arm (C7 / quantum /
    RFF / PCA) with validation PR-AUC + patient bootstrap + fold-honest out-of-fold PR-AUC,
    the ``C7 -> candidate`` increment tests (the pre-registered gating null), and an honest
    leaderboard summary. Writes nothing; ``main`` persists it.
    """
    started = time.perf_counter()

    per_condition, provenance = build_inputs(config=config, conditions=(PRIMARY,))
    inputs = per_condition[PRIMARY]
    if set(inputs.patients_train) & set(inputs.patients_validation):
        raise HybridFusionError(
            "Patient overlap between train and validation; build_inputs should have "
            "refused. Refusing to measure."
        )

    folds, cv_note = grouped_folds(
        inputs.y_train, inputs.patients_train, seed=seed, n_splits=N_FOLDS
    )
    logger.info(
        "E3 primary %s: %d train / %d validation rows (%d / %d positive); %s",
        PRIMARY,
        inputs.y_train.size,
        inputs.y_validation.size,
        int(inputs.y_train.sum()),
        int(inputs.y_validation.sum()),
        cv_note,
    )

    # --- Rebuild C7 exactly as e1_failure_map.run() does on the primary condition:
    # the two strongest non-C1 families there are C6 (mobilenet, 576) then C5
    # (descriptor+multiscale, 619); C7 = hstack([C6, C5]) = 1195. Dimensions asserted so a
    # stale cache surfaces as an error, not a silently different experiment.
    specs = candidate_matrices(inputs)
    c6_train, c6_validation = specs["C6"]["train"], specs["C6"]["validation"]
    c5_train, c5_validation = specs["C5"]["train"], specs["C5"]["validation"]
    if c6_train.shape[1] != MOBILENET_DIM or c5_train.shape[1] != C5_DIM:
        raise HybridFusionError(
            f"Unexpected C6/C5 dimensions ({c6_train.shape[1]}/{c5_train.shape[1]}); "
            f"expected {MOBILENET_DIM}/{C5_DIM}. Cache is stale or misassembled."
        )
    c7_train = np.hstack([c6_train, c5_train]).astype(np.float64)
    c7_validation = np.hstack([c6_validation, c5_validation]).astype(np.float64)
    if c7_train.shape[1] != C7_DIM:
        raise HybridFusionError(f"C7 has {c7_train.shape[1]} columns, expected {C7_DIM}.")

    mobilenet_train = np.asarray(inputs.e1_train["mobilenet"], dtype=np.float64)
    mobilenet_validation = np.asarray(inputs.e1_validation["mobilenet"], dtype=np.float64)

    # --- ANCHOR: C7 must reproduce 0.913038 through E1's exact scoring path.
    c7_xt, c7_xv = _standardise(c7_train, c7_validation)
    c7_anchor_fit = _logistic_arm(c7_xt, inputs.y_train, c7_xv, folds, seed=seed)
    c7_anchor_metrics = _metrics(inputs.y_validation, c7_anchor_fit["validation_probability"])
    c7_anchor_pr_auc = None if not c7_anchor_metrics else c7_anchor_metrics.get("pr_auc")
    anchor_ok = (
        c7_anchor_pr_auc is not None
        and abs(float(c7_anchor_pr_auc) - C7_REFERENCE_PR_AUC) <= C7_ANCHOR_TOLERANCE
    )
    anchor = {
        "target_pr_auc": C7_REFERENCE_PR_AUC,
        "reproduced_pr_auc": None if c7_anchor_pr_auc is None else round(float(c7_anchor_pr_auc), 6),
        "selected_c": c7_anchor_fit["hyperparameters"].get("C"),
        "tolerance": C7_ANCHOR_TOLERANCE,
        "reproduced": bool(anchor_ok),
        "source": "reports_e1_failure_map.json :: A_lesion_polygon :: C7 :: logistic",
        "path": "e1_failure_map._standardise + _logistic_arm on hstack([C6, C5])",
    }
    if not anchor_ok:
        logger.warning(
            "C7 anchor NOT reproduced: got %s, expected %s. Conclusions suppressed.",
            anchor["reproduced_pr_auc"],
            C7_REFERENCE_PR_AUC,
        )

    # --- Quantum feature map, validated against Aer before any quantum number is used.
    feature_map = QuantumVisualFeatureMap()
    pre_all = QuantumVisualPreprocessor.fit(mobilenet_train, seed=seed)
    sample = min(int(quantum_validation_sample), mobilenet_train.shape[0])
    aer_check = validate_against_aer(
        feature_map, pre_all.to_angles(mobilenet_train[:sample])
    )

    # --- Standalone arms, all through the same fold-honest scorer.
    arm_factories: Dict[str, Tuple[np.ndarray, np.ndarray, Factory]] = {
        "c7": (c7_train, c7_validation, _identity_factory),
        "quantum": (mobilenet_train, mobilenet_validation, _make_quantum_factory(feature_map, seed=seed)),
        "rff": (mobilenet_train, mobilenet_validation, _make_rff_factory(seed=seed, n_features=E2_N_FEATURES)),
        "pca": (mobilenet_train, mobilenet_validation, _make_pca_factory(seed=seed)),
    }
    fitted: Dict[str, Dict[str, Any]] = {}
    arms: Dict[str, Dict[str, Any]] = {}
    for name, (src_train, src_validation, factory) in arm_factories.items():
        logger.info("  arm %s ...", name)
        fit = _score_arm(
            src_train, src_validation, inputs.y_train, folds, preprocess=factory, seed=seed
        )
        fitted[name] = fit
        arms[name] = _arm_summary(
            name, fit, inputs, bootstrap_draws=bootstrap_draws, seed=seed
        )

    # --- The pre-registered gating test, C7 -> candidate, for each non-C7 arm.
    reference = fitted["c7"]
    increments: Dict[str, Dict[str, Any]] = {}
    for name in ("quantum", "rff", "pca"):
        candidate = fitted[name]
        logger.info("  increment C7 -> %s ...", name)
        increments[name] = _increment_null(
            np.asarray(reference["validation_score"], dtype=np.float64),
            np.asarray(candidate["validation_score"], dtype=np.float64),
            np.asarray(reference["train_out_of_fold"], dtype=np.float64),
            np.asarray(candidate["train_out_of_fold"], dtype=np.float64),
            inputs.y_train,
            inputs.y_validation,
            inputs.patients_train,
            inputs.patients_validation,
            draws=increment_permutations,
            seed=seed,
        )

    # --- Honest leaderboard + verdict. No fabrication: a null is a valid outcome.
    def _pr(name: str, key: str) -> Optional[float]:
        return arms[name].get(key)

    quantum_beats_c7_validation = (
        _pr("quantum", "validation_pr_auc") is not None
        and _pr("c7", "validation_pr_auc") is not None
        and _pr("quantum", "validation_pr_auc") > _pr("c7", "validation_pr_auc")
    )
    quantum_beats_c7_oof = (
        arms["quantum"]["train_cv_pr_auc"] is not None
        and arms["c7"]["train_cv_pr_auc"] is not None
        and arms["quantum"]["train_cv_pr_auc"] > arms["c7"]["train_cv_pr_auc"]
    )
    quantum_increment_survives = increments["quantum"].get("survives_gating_null")
    rff_increment_survives = increments["rff"].get("survives_gating_null")
    pca_increment_survives = increments["pca"].get("survives_gating_null")

    # A defensible "quantum adds information beyond C7" requires the quantum increment to
    # survive the patient-blocked column-permutation null. That the matched controls do or
    # do not survive tells us whether any 16-/8-d appended column would -- i.e. whether the
    # quantum increment is quantum-specific or a generic dimensionality effect.
    defensible_quantum_advantage = bool(quantum_increment_survives)
    quantum_specific = bool(
        quantum_increment_survives
        and not rff_increment_survives
        and not pca_increment_survives
    )

    leaderboard = sorted(
        (
            {
                "arm": name,
                "validation_pr_auc": arms[name]["validation_pr_auc"],
                "train_cv_pr_auc": arms[name]["train_cv_pr_auc"],
                "validation_bootstrap_p2_5": arms[name]["validation_bootstrap"].get("p2_5"),
                "validation_bootstrap_p97_5": arms[name]["validation_bootstrap"].get("p97_5"),
            }
            for name in arms
        ),
        key=lambda row: (row["validation_pr_auc"] is None, -(row["validation_pr_auc"] or 0.0)),
    )

    if not anchor_ok:
        conclusion = (
            "C7 anchor did not reproduce 0.913038; the experiment is not trustworthy and "
            "no advantage/no-advantage conclusion is drawn. Rebuild the caches and rerun."
        )
    elif quantum_specific:
        conclusion = (
            "The C7 -> quantum increment survives the patient-blocked column-permutation "
            "null while the matched RFF and PCA increments do not: on this evidence the "
            "quantum representation carries information beyond C7 that a same-width classical "
            "control does not. Freeze and re-examine before any test-set claim."
        )
    elif defensible_quantum_advantage:
        conclusion = (
            "The C7 -> quantum increment survives the gating null, but a matched control "
            "increment also survives: the added information is not quantum-specific (a "
            "generic extra-dimension effect). Not a quantum advantage."
        )
    else:
        conclusion = (
            "The C7 -> quantum increment does NOT survive the patient-blocked "
            "column-permutation null: on this evidence the quantum representation adds no "
            "information beyond C7. A null result (mission §29B), reported as measured."
        )

    summary = {
        "primary_condition": PRIMARY,
        "c7_validation_pr_auc": arms["c7"]["validation_pr_auc"],
        "quantum_validation_pr_auc": arms["quantum"]["validation_pr_auc"],
        "quantum_beats_c7_on_validation": quantum_beats_c7_validation,
        "quantum_beats_c7_on_train_oof": quantum_beats_c7_oof,
        "quantum_increment_survives_gating_null": quantum_increment_survives,
        "rff_increment_survives_gating_null": rff_increment_survives,
        "pca_increment_survives_gating_null": pca_increment_survives,
        "defensible_quantum_advantage_over_c7": defensible_quantum_advantage,
        "quantum_specific_advantage": quantum_specific,
        "leaderboard": leaderboard,
        "conclusion": conclusion,
    }

    duration = time.perf_counter() - started
    return {
        "e3_hybrid_fusion_version": E3_HYBRID_FUSION_VERSION,
        "question": (
            "Does the E2 quantum-visual representation add information beyond the strongest "
            "classical baseline C7 (validation PR-AUC 0.913038) under identical "
            "patient-grouped folds, train-only fitting and matched controls?"
        ),
        "test_partition_used": False,
        "test_partition_note": (
            "E3 reads TRAIN and VALIDATION only via e1_failure_map.build_inputs, which "
            "iterates only ('train', 'validation'). No code path selects the test partition."
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
            "candidate_representations": {
                "quantum": "MobileNet-576 -> fold-train PCA-8 + angle scaler -> fixed 8-qubit Ry/CZ map -> 16 local-Z observables",
                "rff": "MobileNet-576 -> fold-train PCA-8 -> RFF-16 (matched control, DEC-033)",
                "pca": "MobileNet-576 -> fold-train PCA-8 (raw linear control)",
            },
            "folds": f"{N_FOLDS}-fold patient-grouped StratifiedGroupKFold on TRAIN, seed {seed}",
            "preprocessing": "train-only; PCA/angle/RFF refit inside each fold-train for the out-of-fold column",
            "gating_test": (
                "e1_failure_map._increment_null with C7 as reference: two-column logistic "
                "stack on leakage-free out-of-fold TRAIN predictions, scored on VALIDATION, "
                "gated by a patient-blocked column-permutation null on the candidate column"
            ),
            "reused_verbatim": [
                "grouped_folds", "_standardise", "_logistic_arm", "_cv_select", "_metrics",
                "bootstrap_pr_auc", "_increment_null", "build_inputs", "candidate_matrices",
            ],
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
            "quantum_features": E2_N_FEATURES,
        },
        "duration_seconds": round(duration, 3),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="E3 hybrid-fusion research experiment.")
    parser.add_argument("--out", type=str, default="reports_e3_hybrid_fusion.json")
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--permutations", type=int, default=INCREMENT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    payload = run(
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
        increment_permutations=args.permutations,
    )
    from pathlib import Path

    Path(args.out).write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    logger.info("Wrote %s", args.out)
    summary = payload["summary"]
    logger.info(
        "C7=%s quantum=%s | quantum increment survives null: %s (rff=%s, pca=%s)",
        summary["c7_validation_pr_auc"],
        summary["quantum_validation_pr_auc"],
        summary["quantum_increment_survives_gating_null"],
        summary["rff_increment_survives_gating_null"],
        summary["pca_increment_survives_gating_null"],
    )
    logger.info("%s", summary["conclusion"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
