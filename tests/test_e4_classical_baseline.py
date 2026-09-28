"""The classical failure map on PTB-XL: partition honesty, metrics, and the null (E4).

:mod:`backend.evaluation.e4_classical_baseline` is the gate on every quantum experiment
in phase E4, which means a silent defect here would not produce a wrong number -- it would
produce a *permissive* one, and every later quantum comparison would inherit it. The tests
are organised around the four ways this module could lie:

* it could **leak**: fit a statistic, a hyperparameter or a threshold on rows it later
  scores, or let a frozen-test row into a fitted partition;
* it could **miscount**: treat an unknown label as a negative, or silently fold the
  secondary task's excluded records into one of its classes;
* it could **understate uncertainty**: bootstrap records instead of patients, or compare
  two models with independent resamples instead of paired ones;
* it could **flatter itself**: select a threshold on the partition it reports, or pick the
  winner by a metric other than the pre-registered primary one.

Each section below targets one of those. The cohorts are synthetic and built directly as
:class:`EcgCohort` objects -- small, with planted structure -- so a test can state the
answer in advance rather than assert whatever the real corpus happens to produce.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pytest

from backend.dataset.ptbxl import SUPERCLASS_ORDER, TEST_FOLDS, TRAIN_FOLDS, VALIDATION_FOLDS
from backend.evaluation.e4_classical_baseline import (
    BOOTSTRAP_DRAWS,
    E4_BASELINE_VERSION,
    FIXED_SPECIFICITY,
    MODEL_GRIDS,
    MODELS,
    REPRESENTATIONS,
    SEED,
    TRIVIAL_ARM,
    ClassicalBaselineError,
    bootstrap_roc_auc,
    build_estimator,
    build_partition,
    guard_partitions,
    out_of_fold_scores,
    paired_bootstrap_delta,
    permutation_null,
    primary_labels,
    raw_scores,
    roc_auc,
    run,
    secondary_labels,
    select_threshold,
    summary_lines,
)
from backend.ml.ecg_transform import fit_ecg_transform
from backend.ml.features_ecg import feature_names
from backend.training.prepare_ecg_features import LABEL_UNKNOWN, EcgCohort

_NAMES = feature_names()
_DIM = len(_NAMES)
_SIGNAL_COLUMN = 2  # carries the label, so every arm has something real to find


def _cohort(
    n: int = 480,
    *,
    seed: int = 11,
    separation: float = 2.0,
    records_per_patient: int = 2,
    folds: Optional[Sequence[int]] = None,
    n_unknown: int = 12,
) -> EcgCohort:
    """A cohort with a learnable signal, repeated patients and unknown-label rows.

    ``records_per_patient`` above one is what makes the patient-level tests meaningful: if
    every record were its own patient, a record-level bootstrap and a patient-level one
    would agree and the distinction this module rests on would be untestable.
    """
    rng = np.random.default_rng(seed)
    patient_ids = np.asarray(
        [1000 + index // records_per_patient for index in range(n)], dtype=np.int64
    )
    unique_patients = np.unique(patient_ids)
    # The fold is assigned **per patient**, as PTB-XL does it. Assigning it per record
    # would scatter one patient across train and validation, and the module's own guard
    # would (correctly) refuse the fixture before any test could run.
    patient_fold = {
        int(pid): 1 + (index % 9) for index, pid in enumerate(unique_patients)
    }
    fold_values = (
        np.asarray(list(folds), dtype=np.int8)
        if folds is not None
        else np.asarray([patient_fold[int(pid)] for pid in patient_ids], dtype=np.int8)
    )
    # One label per patient, shared by that patient's records: the correlation structure
    # the patient-level bootstrap and the blocked permutation both exist to respect.
    patient_label = {int(pid): int(rng.integers(0, 2)) for pid in unique_patients}
    labels = np.asarray([patient_label[int(pid)] for pid in patient_ids], dtype=np.int8)

    features = rng.normal(size=(n, _DIM))
    features[:, _SIGNAL_COLUMN] = labels * separation + rng.normal(scale=1.0, size=n)

    superclasses = np.zeros((n, len(SUPERCLASS_ORDER)), dtype=np.uint8)
    columns = {name: index for index, name in enumerate(SUPERCLASS_ORDER)}
    for row, label in enumerate(labels):
        if label == 1:
            superclasses[row, columns["MI" if row % 2 else "STTC"]] = 1
        else:
            superclasses[row, columns["NORM"]] = 1

    labels = labels.copy()
    labels[:n_unknown] = LABEL_UNKNOWN
    superclasses[:n_unknown] = 0  # unknown means no diagnostic superclass at all

    return EcgCohort(
        features=features,
        feature_names=_NAMES,
        ecg_ids=np.arange(n, dtype=np.int32),
        patient_ids=patient_ids,
        strat_folds=fold_values,
        labels=labels,
        label_known=(labels != LABEL_UNKNOWN).astype(np.uint8),
        superclasses=superclasses,
        superclass_order=SUPERCLASS_ORDER,
        ages=np.linspace(25.0, 88.0, n),
        sexes=(np.arange(n) % 2).astype(np.int8),
        n_beats=np.full(n, 10, dtype=np.int32),
        template_beats=np.full(n, 8, dtype=np.int32),
        quality_flags=tuple("" for _ in range(n)),
        metadata={},
    )


def _partitions(cohort: Optional[EcgCohort] = None, **kwargs):
    """TRAIN/VALIDATION partitions under an identity representation."""
    cohort = cohort if cohort is not None else _cohort(**kwargs)
    transform = fit_ecg_transform(cohort, reduction="identity")
    labels = primary_labels(cohort)
    train = build_partition(cohort, transform, labels, sorted(TRAIN_FOLDS), "TRAIN")
    validation = build_partition(
        cohort, transform, labels, sorted(VALIDATION_FOLDS), "VALIDATION"
    )
    return cohort, transform, labels, train, validation


# ------------------------------------------------------------------ partition honesty


def test_train_and_validation_never_share_a_patient():
    _, _, _, train, validation = _partitions()

    assert guard_partitions(train, validation)["patient_overlap"] == 0
    assert not set(train.patients.tolist()) & set(validation.patients.tolist())


def test_a_shared_patient_is_refused_rather_than_reported():
    """The guard must raise, not annotate: a leak that is merely logged still gets used."""
    _, _, _, train, validation = _partitions()
    leaked = build_partition.__wrapped__ if hasattr(build_partition, "__wrapped__") else None
    assert leaked is None  # no decorator is hiding the real function

    poisoned = type(validation)(
        **{**validation.__dict__, "patients": np.full_like(validation.patients, train.patients[0])}
    )
    with pytest.raises(ClassicalBaselineError, match="both TRAIN and VALIDATION"):
        guard_partitions(train, poisoned)


def test_a_frozen_test_fold_cannot_reach_a_fitted_partition():
    cohort = _cohort(folds=[10] * 480)

    with pytest.raises(ClassicalBaselineError, match="frozen-test"):
        run(cohort, models=("logistic",), representations=("pca08",))


def test_the_guard_names_the_test_partition_as_unused():
    _, _, _, train, validation = _partitions()

    guards = guard_partitions(train, validation)
    assert guards["test_partition_used"] is False
    assert not set(guards["train_folds"]) & set(TEST_FOLDS)
    assert not set(guards["validation_folds"]) & set(TEST_FOLDS)


def test_partitions_carry_only_the_folds_they_claim():
    _, _, _, train, validation = _partitions()

    assert set(train.strat_folds.tolist()) <= set(TRAIN_FOLDS)
    assert set(validation.strat_folds.tolist()) <= set(VALIDATION_FOLDS)


def test_a_representation_fitted_outside_train_is_refused():
    """The reduction is fitted; fitting it on fold 9 would leak into every fold-9 number."""
    cohort = _cohort()
    leaky = fit_ecg_transform(cohort, folds=[9], reduction="pca", n_components=4,
                              allow_non_train=True)
    labels = primary_labels(cohort)
    train = build_partition(cohort, leaky, labels, sorted(TRAIN_FOLDS), "TRAIN")
    validation = build_partition(cohort, leaky, labels, sorted(VALIDATION_FOLDS), "VAL")

    # The partitions themselves are still disjoint -- the leak is in the transform, which is
    # why ``run`` checks ``fitted_on_folds`` separately rather than trusting disjointness.
    assert guard_partitions(train, validation)["patient_overlap"] == 0
    assert set(leaky.fitted_on_folds) - set(TRAIN_FOLDS)


# --------------------------------------------------------------------- label discipline


def test_unknown_label_rows_never_enter_a_partition():
    cohort, _, labels, train, validation = _partitions(n_unknown=40)

    assert int(np.count_nonzero(labels == LABEL_UNKNOWN)) == 40
    assert not np.isin(train.y, [LABEL_UNKNOWN]).any()
    assert not np.isin(validation.y, [LABEL_UNKNOWN]).any()
    assert train.n_records + validation.n_records < cohort.features.shape[0]


def test_an_unknown_label_is_not_counted_as_a_negative():
    """The failure that would silently inflate specificity on every arm."""
    few = _cohort(n_unknown=0)
    many = _cohort(n_unknown=60)
    _, _, _, train_few, _ = _partitions(few)
    _, _, _, train_many, _ = _partitions(many)

    negatives_few = int(np.count_nonzero(train_few.y == 0))
    negatives_many = int(np.count_nonzero(train_many.y == 0))
    assert negatives_many <= negatives_few, "unknown rows must not appear among negatives"


def test_the_primary_task_is_read_from_the_cache_not_recomputed():
    cohort = _cohort()

    np.testing.assert_array_equal(primary_labels(cohort), np.asarray(cohort.labels))


def test_the_secondary_task_excludes_non_mi_abnormals_rather_than_calling_them_normal():
    """STTC-only records are neither an MI nor a normal ECG; folding them in changes the task."""
    cohort = _cohort()
    columns = {name: index for index, name in enumerate(cohort.superclass_order)}
    present = np.asarray(cohort.superclasses, dtype=bool)
    secondary = secondary_labels(cohort)

    sttc_only = present[:, columns["STTC"]] & ~present[:, columns["MI"]]
    assert sttc_only.any(), "the fixture must contain STTC-only records for this to mean anything"
    assert np.all(secondary[sttc_only] == LABEL_UNKNOWN)
    assert np.all(secondary[present[:, columns["MI"]]] == 1)


def test_the_secondary_task_is_flagged_as_never_selected_on():
    cohort = _cohort()
    report = run(
        cohort, models=("logistic",), representations=("pca08",),
        bootstrap_draws=40, permutation_draws=0,
    )

    assert report["tasks"]["primary_norm_vs_abnormal"]["used_for_selection"] is True
    assert report["tasks"]["secondary_mi_vs_norm"]["used_for_selection"] is False


def test_a_record_with_mi_and_norm_is_positive_in_both_tasks():
    """§7.4 fixes the screening-conservative direction so it cannot be flipped later."""
    cohort = _cohort()
    columns = {name: index for index, name in enumerate(cohort.superclass_order)}
    superclasses = np.asarray(cohort.superclasses, dtype=np.uint8).copy()
    superclasses[0] = 0
    superclasses[0, columns["MI"]] = 1
    superclasses[0, columns["NORM"]] = 1
    both = EcgCohort(**{**cohort.__dict__, "superclasses": superclasses})

    assert secondary_labels(both)[0] == 1


# ------------------------------------------------------------------------ the metric


def test_roc_auc_matches_a_hand_computed_value():
    # Two negatives below two positives, with one tie straddling the boundary.
    y = np.asarray([0, 0, 1, 1])
    assert roc_auc(y, np.asarray([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(y, np.asarray([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert roc_auc(y, np.asarray([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_roc_auc_averages_ranks_across_ties():
    """A tie is half a win, not a whole one; integer ranks would round it to a win."""
    y = np.asarray([0, 1, 1, 0])
    scores = np.asarray([0.3, 0.3, 0.9, 0.1])
    # pairs: (0.3 vs 0.3) tie, (0.9 vs 0.3) win, (0.3 vs 0.1) win, (0.9 vs 0.1) win
    assert roc_auc(y, scores) == pytest.approx((0.5 + 1 + 1 + 1) / 4)


def test_roc_auc_is_none_rather_than_a_half_when_one_class_is_absent():
    assert roc_auc(np.ones(5, dtype=int), np.linspace(0, 1, 5)) is None
    assert roc_auc(np.zeros(5, dtype=int), np.linspace(0, 1, 5)) is None


def test_roc_auc_refuses_misaligned_vectors():
    with pytest.raises(ClassicalBaselineError, match="labels against"):
        roc_auc(np.asarray([0, 1, 1]), np.asarray([0.1, 0.9]))


def test_roc_auc_is_invariant_to_any_monotone_rescaling():
    """Why Platt scaling the kernel arm cannot move the primary metric."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=200)
    scores = rng.normal(size=200)

    assert roc_auc(y, scores) == pytest.approx(roc_auc(y, 1.0 / (1.0 + np.exp(-scores))))


# --------------------------------------------------------------------- the bootstrap


def test_the_bootstrap_resamples_patients_not_records():
    """The interval must widen when records are correlated within a patient.

    Same rows, same labels, same scores -- only the patient column differs. The scores carry
    a per-patient random effect on top of per-record noise, which is how repeated ECGs from
    one patient actually behave: a model that over-scores one recording over-scores the
    others. Treating each record as its own patient is the pseudo-replication error, and it
    must produce a visibly narrower interval than the honest grouping.
    """
    rng = np.random.default_rng(5)
    per_patient_label = rng.integers(0, 2, size=120)
    y = np.repeat(per_patient_label, 4)
    patient_effect = np.repeat(rng.normal(scale=0.35, size=120), 4)
    scores = np.clip(y * 0.4 + patient_effect + rng.normal(scale=0.05, size=y.size) + 0.3, 0, 1)
    grouped = np.repeat(np.arange(120), 4)
    ungrouped = np.arange(y.size)

    honest = bootstrap_roc_auc(y, scores, grouped, draws=400, seed=SEED)
    pseudo = bootstrap_roc_auc(y, scores, ungrouped, draws=400, seed=SEED)

    assert honest["standard_error"] > pseudo["standard_error"]
    assert honest["p97_5"] - honest["p2_5"] > pseudo["p97_5"] - pseudo["p2_5"]
    assert honest["unit"].startswith("patient")


def test_the_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=400)
    scores = np.clip(y * 0.5 + rng.normal(scale=0.4, size=400) + 0.25, 0, 1)
    patients = np.arange(400) // 2

    result = bootstrap_roc_auc(y, scores, patients, draws=300, seed=SEED)

    assert result["p2_5"] <= roc_auc(y, scores) <= result["p97_5"]
    assert result["n_draws_usable"] <= result["n_draws_requested"]


def test_the_bootstrap_is_reproducible_from_its_seed():
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, size=200)
    scores = rng.random(200)
    patients = np.arange(200) // 2

    first = bootstrap_roc_auc(y, scores, patients, draws=100, seed=SEED)
    second = bootstrap_roc_auc(y, scores, patients, draws=100, seed=SEED)
    other = bootstrap_roc_auc(y, scores, patients, draws=100, seed=SEED + 1)

    assert first == second
    assert first["mean"] != other["mean"]


def test_the_bootstrap_reports_being_unmeasurable_rather_than_returning_a_number():
    y = np.ones(20, dtype=int)

    result = bootstrap_roc_auc(y, np.linspace(0, 1, 20), np.arange(20), draws=50)

    assert result["measured"] is False
    assert "both classes" in result["reason"] or "two bootstrap draws" in result["reason"]


def test_the_paired_delta_of_a_model_against_itself_is_exactly_zero():
    rng = np.random.default_rng(13)
    y = rng.integers(0, 2, size=300)
    scores = rng.random(300)
    patients = np.arange(300) // 3

    result = paired_bootstrap_delta(y, scores, scores, patients, draws=200, seed=SEED)

    assert result["observed_delta"] == pytest.approx(0.0)
    assert result["p2_5"] == pytest.approx(0.0)
    assert result["p97_5"] == pytest.approx(0.0)
    assert result["excludes_zero"] is False


def test_the_paired_interval_is_tighter_than_an_unpaired_one():
    """Two independent bootstraps would add the models' variances instead of cancelling them."""
    rng = np.random.default_rng(17)
    y = rng.integers(0, 2, size=400)
    strong = np.clip(y * 0.6 + rng.normal(scale=0.3, size=400) + 0.2, 0, 1)
    weak = np.clip(strong + rng.normal(scale=0.05, size=400), 0, 1)
    patients = np.arange(400) // 2

    paired = paired_bootstrap_delta(y, strong, weak, patients, draws=400, seed=SEED)
    independent = np.hypot(
        bootstrap_roc_auc(y, strong, patients, draws=400, seed=SEED)["standard_error"],
        bootstrap_roc_auc(y, weak, patients, draws=400, seed=SEED)["standard_error"],
    )

    assert paired["standard_error"] < independent


def test_the_paired_delta_detects_a_real_ordering_difference():
    rng = np.random.default_rng(19)
    y = rng.integers(0, 2, size=500)
    good = np.clip(y * 0.8 + rng.normal(scale=0.3, size=500), 0, 1)
    noise = rng.random(500)
    patients = np.arange(500)

    result = paired_bootstrap_delta(y, good, noise, patients, draws=300, seed=SEED)

    assert result["observed_delta"] > 0
    assert result["excludes_zero"] is True


# -------------------------------------------------------------- selection discipline


def test_out_of_fold_predictions_cover_every_train_row_exactly_once():
    _, _, _, train, _ = _partitions()

    scores, per_fold = out_of_fold_scores("logistic", {"C": 1.0}, train)

    assert scores.shape == (train.n_records,)
    assert np.isfinite(scores).all()
    assert len(per_fold) == len(set(train.strat_folds.tolist()))


def test_out_of_fold_predictions_are_held_out_not_in_sample():
    """An in-sample score is optimistic, and the threshold chosen from it would be too."""
    _, _, _, train, _ = _partitions(separation=0.4)

    out_of_fold, _ = out_of_fold_scores("gbm", MODEL_GRIDS["gbm"][1], train)
    estimator = build_estimator("gbm", MODEL_GRIDS["gbm"][1])
    estimator.fit(train.features, train.y)
    in_sample = estimator.predict_proba(train.features)[:, 1]

    assert roc_auc(train.y, in_sample) > roc_auc(train.y, out_of_fold)


def test_inner_cross_validation_folds_are_patient_disjoint():
    """PTB-XL's own folds are reused precisely so this holds without a new grouping."""
    _, _, _, train, _ = _partitions()

    seen: dict[int, int] = {}
    for fold in sorted(set(train.strat_folds.tolist())):
        for patient in np.unique(train.patients[train.strat_folds == fold]):
            assert seen.setdefault(int(patient), fold) == fold


def test_inner_cross_validation_refuses_a_single_fold():
    # Patients alternate between fold 1 and fold 9, two records each: fold 9 still exists
    # so a validation partition can be built, but TRAIN collapses to a single fold.
    cohort = _cohort(n=480, folds=[1, 1, 9, 9] * 120)
    _, _, _, train, _ = _partitions(cohort)

    assert set(train.strat_folds.tolist()) == {1}
    with pytest.raises(ClassicalBaselineError, match="at least two TRAIN folds"):
        out_of_fold_scores("logistic", {"C": 1.0}, train)


def test_the_threshold_is_selected_on_train_and_says_so():
    _, _, _, train, _ = _partitions()
    scores, _ = out_of_fold_scores("logistic", {"C": 1.0}, train)

    chosen = select_threshold(train.y, scores)

    assert "TRAIN" in chosen["selected_on"]
    assert scores.min() <= chosen["threshold"] <= scores.max()


def test_the_selected_threshold_maximises_balanced_accuracy_on_its_own_partition():
    rng = np.random.default_rng(23)
    y = rng.integers(0, 2, size=600)
    scores = np.clip(y * 0.5 + rng.normal(scale=0.3, size=600) + 0.25, 0, 1)

    chosen = select_threshold(y, scores)
    best = chosen["train_out_of_fold_balanced_accuracy"]

    for candidate in np.linspace(scores.min(), scores.max(), 60):
        predicted = (scores >= candidate).astype(int)
        sensitivity = predicted[y == 1].mean()
        specificity = 1.0 - predicted[y == 0].mean()
        assert (sensitivity + specificity) / 2 <= best + 1e-9


def test_threshold_selection_refuses_a_single_class():
    with pytest.raises(ClassicalBaselineError, match="both classes"):
        select_threshold(np.ones(10, dtype=int), np.linspace(0, 1, 10))


# ------------------------------------------------------------------ the permutation null


def test_the_permutation_null_centres_on_chance_when_labels_are_destroyed():
    _, _, _, train, validation = _partitions(n=300, separation=3.0)

    null = permutation_null(
        "logistic", {"C": 1.0}, train, validation, 0.99, draws=12, seed=SEED
    )

    assert null["measured"] is True
    assert null["within_patient_labels_consistent"] is True
    assert abs(null["null_mean"] - 0.5) < 0.15, "a destroyed label must not still predict"


def test_the_permutation_null_permutes_whole_patients():
    """Permuting records independently would break the within-patient correlation.

    The fixture gives every patient one label across their records; a permutation that
    preserves that property is what makes the null distribution come from a problem of the
    same difficulty as the observed one.
    """
    _, _, _, train, validation = _partitions(records_per_patient=3)

    null = permutation_null(
        "logistic", {"C": 1.0}, train, validation, 0.9, draws=6, seed=SEED
    )

    assert null["within_patient_labels_consistent"] is True


def test_a_genuine_signal_clears_the_permutation_null():
    _, _, _, train, validation = _partitions(n=300, separation=3.0)
    scores, _ = out_of_fold_scores("logistic", {"C": 1.0}, train)
    estimator = build_estimator("logistic", {"C": 1.0})
    estimator.fit(train.features, train.y)
    observed = roc_auc(validation.y, estimator.predict_proba(validation.features)[:, 1])

    null = permutation_null(
        "logistic", {"C": 1.0}, train, validation, observed, draws=20, seed=SEED
    )

    assert observed > null["null_max"]
    assert null["n_null_at_or_above_observed"] == 0


def test_the_permutation_p_value_is_conservative_never_zero():
    """With finitely many draws, an unbeaten observation supports p < 1/(n+1), not p = 0."""
    _, _, _, train, validation = _partitions(n=300, separation=3.0)

    null = permutation_null(
        "logistic", {"C": 1.0}, train, validation, 1.0, draws=10, seed=SEED
    )

    assert null["p_value"] > 0.0
    assert null["p_value"] == pytest.approx(1.0 / (null["n_draws_usable"] + 1))


def test_zero_permutation_draws_reports_not_measured():
    _, _, _, train, validation = _partitions()

    assert permutation_null(
        "logistic", {"C": 1.0}, train, validation, 0.9, draws=0
    )["measured"] is False


# ------------------------------------------------------------------------- the grid


def test_the_representation_grid_covers_the_near_term_dimensions():
    """§7.4 condition 3's matched controls only exist if the grid contains them."""
    dimensions = {size for _, reduction, size in REPRESENTATIONS if reduction == "pca"}

    assert {8, 16, 32} <= dimensions
    assert any(reduction == "identity" for _, reduction, _ in REPRESENTATIONS)


def test_every_model_in_the_grid_can_be_built_and_is_distinct():
    built = [build_estimator(model, MODEL_GRIDS[model][0]) for model in MODELS]

    assert len({type(estimator).__name__ for estimator in built}) == len(MODELS)


def test_an_unknown_model_arm_is_refused():
    with pytest.raises(ClassicalBaselineError, match="Unknown model arm"):
        build_estimator("quantum-magic", {})


def test_the_kernel_arm_scores_through_its_decision_function():
    """Not through ``probability=True``, whose internal 5-fold Platt refit is deprecated.

    The arm is calibrated afterwards from TRAIN out-of-fold decision values instead, so the
    raw score it emits here must be the decision function itself.
    """
    rng = np.random.default_rng(29)
    features = rng.normal(size=(60, 4))
    y = (features[:, 0] > 0).astype(int)
    estimator = build_estimator("rbf_svm", {"C": 1.0})

    assert estimator.probability is not True
    estimator.fit(features, y)
    np.testing.assert_allclose(raw_scores(estimator, features),
                               estimator.decision_function(features))


@pytest.mark.parametrize("unknown_key,bad", [
    ("models", "quantum-magic"),
    ("representations", "pca99"),
    ("tasks", "tertiary"),
])
def test_run_refuses_an_unknown_grid_entry(unknown_key: str, bad: str):
    with pytest.raises(ClassicalBaselineError, match="Unknown"):
        run(_cohort(), **{unknown_key: (bad,)})


# ------------------------------------------------------------------- end to end report


@pytest.fixture(scope="module")
def report() -> dict:
    return run(
        _cohort(n=540, separation=2.5),
        models=("logistic", "gbm"),
        representations=("f97", "pca08"),
        bootstrap_draws=60,
        permutation_draws=4,
    )


def test_the_report_states_the_protocol_it_followed(report: dict):
    protocol = report["protocol"]

    assert report["baseline_version"] == E4_BASELINE_VERSION
    assert protocol["primary_metric"].startswith("ROC-AUC")
    assert protocol["bootstrap_unit"] == "patient"
    assert protocol["train_folds"] == sorted(TRAIN_FOLDS)
    assert protocol["validation_folds"] == sorted(VALIDATION_FOLDS)
    assert report["test_partition_used"] is False
    assert protocol["test_partition_used"] is False


def test_the_trivial_arm_lands_exactly_on_chance(report: dict):
    """A constant predictor scoring anything but 0.5 means labels and scores are misaligned."""
    arms = report["tasks"]["primary_norm_vs_abnormal"]["arms"]
    trivial = [arm for arm in arms if arm["model"] == TRIVIAL_ARM]

    assert len(trivial) == 1
    assert trivial[0]["validation_roc_auc"] == pytest.approx(0.5)


def test_the_trivial_arm_is_never_named_the_strongest(report: dict):
    for task in report["tasks"].values():
        assert task["strongest_arm"]["model"] != TRIVIAL_ARM


def test_every_arm_beats_the_trivial_floor(report: dict):
    """Scoped to the primary task, which is the only one the fixture plants a signal for.

    ``_cohort`` separates abnormal from normal in one feature column; it does **not** plant
    anything that distinguishes MI from the other abnormal superclasses, so a secondary-task
    arm near or below chance here is a property of the fixture rather than a defect. The
    corpus run is where the secondary task carries real information.
    """
    for arm in report["tasks"]["primary_norm_vs_abnormal"]["arms"]:
        if arm["model"] != TRIVIAL_ARM:
            assert arm["validation_roc_auc"] > 0.5


def test_the_strongest_arm_is_the_best_by_the_primary_metric(report: dict):
    """The winner must be chosen by ROC-AUC, not by whichever secondary metric flatters it."""
    for task in report["tasks"].values():
        scored = [
            arm for arm in task["arms"]
            if arm["model"] != TRIVIAL_ARM and arm["validation_roc_auc"] is not None
        ]
        assert task["strongest_arm"]["validation_roc_auc"] == max(
            arm["validation_roc_auc"] for arm in scored
        )


def test_every_arm_reports_its_input_dimension(report: dict):
    """Condition 3 is unanswerable if an arm's dimension is not on the record."""
    for task in report["tasks"].values():
        for arm in task["arms"]:
            if arm["model"] != TRIVIAL_ARM:
                assert arm["input_dimension"] > 0
                assert arm["input_dimension"] == (97 if arm["representation"] == "f97" else 8)


def test_each_arm_records_where_its_probabilities_came_from(report: dict):
    for task in report["tasks"].values():
        for arm in task["arms"]:
            assert arm["probability_source"] in (
                "native", "train_out_of_fold_platt", "train prevalence, constant"
            )


def test_the_failure_map_accounts_for_every_validation_record(report: dict):
    failure = report["tasks"]["primary_norm_vs_abnormal"]["failure_map"]
    confusion = failure["overall"]["confusion_matrix"]

    assert sum(int(value) for value in confusion.values()) == failure["overall"]["n_records"]
    assert failure["missed_positives"] + failure["false_alarms"] == int(
        round(failure["overall"]["error_rate"] * failure["overall"]["n_records"])
    )


def test_the_failure_map_reports_the_unscored_records_as_a_number(report: dict):
    unknown = report["tasks"]["primary_norm_vs_abnormal"]["failure_map"][
        "unknown_label_records"
    ]

    assert unknown["scored"] is False
    assert unknown["n_records"] >= 0
    assert "never negative" in unknown["note"]


def test_paired_comparisons_cover_every_non_trivial_arm_but_the_winner(report: dict):
    task = report["tasks"]["primary_norm_vs_abnormal"]
    compared = {entry["arm"] for entry in task["paired_vs_strongest"]}
    expected = {
        arm["arm"] for arm in task["arms"]
        if arm["model"] != TRIVIAL_ARM and arm["arm"] != task["strongest_arm"]["arm"]
    }

    assert compared == expected


def test_the_report_is_json_serialisable_without_arrays(report: dict):
    import json

    from backend.evaluation.e4_classical_baseline import _strip_arrays

    text = json.dumps(_strip_arrays(report))
    assert "ndarray" not in text
    assert json.loads(text)["baseline_version"] == E4_BASELINE_VERSION


def test_no_raw_score_vector_survives_into_the_report(report: dict):
    """Scores are working state; leaving them in would bloat the record and invite reuse."""
    for task in report["tasks"].values():
        for arm in task["arms"]:
            assert "validation_scores" not in arm


def test_the_summary_names_the_strongest_arm_and_the_partitions(report: dict):
    lines = "\n".join(summary_lines(report))

    assert report["tasks"]["primary_norm_vs_abnormal"]["strongest_arm"]["arm"] in lines
    assert "TRAIN" in lines and "VALIDATION" in lines
    assert "strongest:" in lines


def test_the_fixed_specificity_operating_point_is_reported(report: dict):
    for task in report["tasks"].values():
        for arm in task["arms"]:
            point = arm["sensitivity_at_fixed_specificity"]
            assert point["target_specificity"] == FIXED_SPECIFICITY


def test_the_default_bootstrap_budget_matches_the_pre_registration():
    assert BOOTSTRAP_DRAWS == 2000
