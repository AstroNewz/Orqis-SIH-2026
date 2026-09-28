"""The TRAIN-only fitted transform: leakage refusal, imputation, compaction (E4).

Every step in :mod:`backend.ml.ecg_transform` is fitted, which makes it the one
place in the PTB-XL pipeline where a validation statistic could leak into a
reported number. The tests are organised around the four guarantees the module
claims:

* it **refuses** to fit on a non-TRAIN fold unless an explicit override says so;
* it fits every statistic on TRAIN rows only, and the supervised step sees only
  the *labeled* TRAIN rows (an unknown label is never a negative);
* it **imputes rather than propagates** the extractor's ``NaN``, using TRAIN
  medians, and drops a column that is too sparse to impute honestly;
* the fitted artifact round-trips through plain arrays -- no pickle -- and still
  knows which patients it was fitted on, so disjointness stays checkable after a
  reload.

Cohorts here are built directly as :class:`EcgCohort` objects rather than
extracted from synthetic WFDB files: the extraction path is covered in
``tests/test_prepare_ecg_features.py``, and building the matrix explicitly is
what lets these tests plant a known constant column, a known all-NaN column and
a known separating feature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pytest

from backend.dataset.ptbxl import SUPERCLASS_ORDER, TRAIN_FOLDS, VALIDATION_FOLDS
from backend.ml.ecg_transform import (
    DEFAULT_COMPONENTS,
    ECG_TRANSFORM_VERSION,
    REDUCTION_METHODS,
    EcgFeatureTransform,
    EcgTransformError,
    assert_no_fit_eval_patient_overlap,
    fit_ecg_transform,
)
from backend.ml.features_ecg import FEATURE_SET_VERSION, feature_names
from backend.training.prepare_ecg_features import LABEL_UNKNOWN, EcgCohort

_NAMES = feature_names()
_DIM = len(_NAMES)

# Column roles planted in the synthetic matrix, by index into the 97-dim layout.
_CONSTANT_COLUMN = 3
_ALL_NAN_COLUMN = 5
_SPARSE_COLUMN = 7       # measurable in only 10% of records
_SEPARATING_COLUMN = 11  # carries the label almost perfectly


def _cohort(
    n: int = 360,
    *,
    seed: int = 7,
    plant: bool = True,
    n_unknown: int = 6,
    folds: Optional[Sequence[int]] = None,
) -> EcgCohort:
    """A cohort with known pathologies planted in known columns."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, _DIM))
    fold_values = (
        np.asarray(list(folds), dtype=np.int8)
        if folds is not None
        else np.asarray([1 + (index % 9) for index in range(n)], dtype=np.int8)
    )
    labels = rng.integers(0, 2, size=n).astype(np.int8)
    labels[:n_unknown] = LABEL_UNKNOWN

    if plant:
        features[:, _CONSTANT_COLUMN] = 4.25
        features[:, _ALL_NAN_COLUMN] = np.nan
        sparse = rng.random(n) < 0.10
        features[~sparse, _SPARSE_COLUMN] = np.nan
        # A feature that separates the classes, so supervised selection has
        # something unambiguous to find.
        features[:, _SEPARATING_COLUMN] = labels * 6.0 + rng.normal(scale=0.2, size=n)
        # Scattered missingness everywhere else, which must be imputed not dropped.
        holes = rng.random((n, _DIM)) < 0.01
        holes[:, [_CONSTANT_COLUMN, _SEPARATING_COLUMN]] = False
        features[holes] = np.nan

    return EcgCohort(
        features=features,
        feature_names=_NAMES,
        ecg_ids=np.arange(n, dtype=np.int32),
        patient_ids=np.arange(1000, 1000 + n, dtype=np.int64),
        strat_folds=fold_values,
        labels=labels,
        label_known=(labels != LABEL_UNKNOWN).astype(np.uint8),
        superclasses=np.zeros((n, len(SUPERCLASS_ORDER)), dtype=np.uint8),
        superclass_order=SUPERCLASS_ORDER,
        ages=np.full(n, 60.0),
        sexes=np.zeros(n, dtype=np.int8),
        n_beats=np.full(n, 10, dtype=np.int32),
        template_beats=np.full(n, 8, dtype=np.int32),
        quality_flags=tuple("" for _ in range(n)),
        metadata={},
    )


# ------------------------------------------------------------- leakage refusal


def test_default_fitting_cohort_is_the_train_folds_only():
    transform = fit_ecg_transform(_cohort())

    assert transform.fitted_on_folds == tuple(sorted(TRAIN_FOLDS))
    assert not set(transform.fitted_on_folds) & set(VALIDATION_FOLDS)


@pytest.mark.parametrize("folds", [[9], [10], [1, 9], [8, 9, 10]])
def test_fitting_on_a_non_train_fold_is_refused(folds: Sequence[int]):
    """A median fitted on fold 9 leaks into every fold-9 number afterwards."""
    with pytest.raises(EcgTransformError, match="non-TRAIN"):
        fit_ecg_transform(_cohort(), folds=folds)


def test_the_validation_fold_is_named_in_the_refusal():
    with pytest.raises(EcgTransformError, match="validation"):
        fit_ecg_transform(_cohort(), folds=[9])


def test_a_post_freeze_refit_is_possible_but_must_be_asked_for():
    transform = fit_ecg_transform(_cohort(), folds=[1, 9], allow_non_train=True)

    assert transform.fitted_on_folds == (1, 9)


def test_only_train_rows_enter_the_fitted_statistics():
    """Shift the validation rows hard; a TRAIN-only fit must not notice."""
    cohort = _cohort(plant=False)
    validation = cohort.fold_mask([9])
    shifted = cohort.features.copy()
    shifted[validation] += 500.0
    poisoned = EcgCohort(**{**cohort.__dict__, "features": shifted})

    clean = fit_ecg_transform(cohort)
    fitted_around_poison = fit_ecg_transform(poisoned)

    np.testing.assert_allclose(clean.means, fitted_around_poison.means)
    np.testing.assert_allclose(clean.scales, fitted_around_poison.scales)
    np.testing.assert_allclose(clean.medians, fitted_around_poison.medians)


def test_fit_records_the_patients_it_saw_and_can_prove_disjointness():
    cohort = _cohort()
    transform = fit_ecg_transform(cohort)
    validation_patients = np.asarray(cohort.patient_ids)[cohort.fold_mask([9])]

    assert transform.n_train_patients == int(transform.train_patient_ids.size)
    assert_no_fit_eval_patient_overlap(transform, validation_patients.tolist())

    with pytest.raises(EcgTransformError, match="both"):
        assert_no_fit_eval_patient_overlap(
            transform, [int(transform.train_patient_ids[0])]
        )


def test_disjointness_is_still_checkable_after_a_save_load_round_trip(tmp_path: Path):
    """The check must survive the process boundary, or it is only a convention."""
    cohort = _cohort()
    transform = fit_ecg_transform(cohort)
    reloaded = EcgFeatureTransform.load(transform.save(tmp_path / "t.npz"))

    with pytest.raises(EcgTransformError):
        assert_no_fit_eval_patient_overlap(
            reloaded, [int(transform.train_patient_ids[0])]
        )


# ------------------------------------------------------- imputation and dropping


def test_an_all_nan_feature_is_dropped_not_imputed():
    transform = fit_ecg_transform(_cohort())

    assert _NAMES[_ALL_NAN_COLUMN] in transform.dropped_features
    assert "coverage" in transform.drop_reasons[_NAMES[_ALL_NAN_COLUMN]]
    assert _ALL_NAN_COLUMN not in transform.kept_indices.tolist()


def test_a_mostly_missing_feature_is_dropped_rather_than_manufactured():
    """Imputing a 90%-missing column invents a constant and calls it a measurement."""
    transform = fit_ecg_transform(_cohort(), min_coverage=0.5)

    assert _NAMES[_SPARSE_COLUMN] in transform.dropped_features


def test_a_lower_coverage_floor_keeps_the_sparse_feature():
    transform = fit_ecg_transform(_cohort(), min_coverage=0.05)

    assert _NAMES[_SPARSE_COLUMN] not in transform.dropped_features
    assert _SPARSE_COLUMN in transform.kept_indices.tolist()


def test_a_constant_feature_is_dropped_for_zero_variance():
    transform = fit_ecg_transform(_cohort())

    assert _NAMES[_CONSTANT_COLUMN] in transform.dropped_features
    assert "variance" in transform.drop_reasons[_NAMES[_CONSTANT_COLUMN]]


def test_scattered_missingness_is_imputed_with_the_train_median():
    cohort = _cohort()
    transform = fit_ecg_transform(cohort)

    # Plant a NaN in a kept, well-covered column of a validation row.
    column = int(transform.kept_indices[0])
    position = int(np.flatnonzero(cohort.fold_mask([9]))[0])
    row = cohort.features[position].copy()
    row[column] = np.nan

    imputed = transform.transform(row[None, :])
    expected = transform.transform(
        np.where(np.arange(_DIM) == column, transform.medians[0], row)[None, :]
    )

    assert np.isfinite(imputed).all(), "a NaN must never reach the model"
    np.testing.assert_allclose(imputed, expected)


def test_an_infinity_is_treated_as_missing_rather_than_propagated():
    cohort = _cohort()
    transform = fit_ecg_transform(cohort)
    row = cohort.features[0].copy()
    row[int(transform.kept_indices[0])] = np.inf

    assert np.isfinite(transform.transform(row[None, :])).all()


def test_fitting_fails_loudly_when_nothing_survives_the_filters():
    cohort = _cohort(plant=False)
    empty = EcgCohort(
        **{**cohort.__dict__, "features": np.full_like(cohort.features, np.nan)}
    )

    with pytest.raises(EcgTransformError, match="coverage floor"):
        fit_ecg_transform(empty)


def test_fitting_fails_loudly_when_no_feature_varies():
    cohort = _cohort(plant=False)
    flat = EcgCohort(**{**cohort.__dict__, "features": np.ones_like(cohort.features)})

    with pytest.raises(EcgTransformError, match="varies"):
        fit_ecg_transform(flat)


# ------------------------------------------------------------------- compaction


def test_pca_reduces_to_the_requested_near_term_dimension():
    transform = fit_ecg_transform(_cohort(), reduction="pca", n_components=8)
    reduced = transform.transform(_cohort().features)

    assert transform.output_dimension == 8
    assert reduced.shape[1] == 8
    assert transform.output_feature_names() == tuple(f"pc{i:02d}" for i in range(1, 9))
    assert 0.0 < transform.total_explained_variance_ratio <= 1.0


def test_pca_components_are_orthonormal_and_ordered_by_variance():
    transform = fit_ecg_transform(_cohort(), n_components=6)
    components = transform.components

    np.testing.assert_allclose(components @ components.T, np.eye(6), atol=1e-9)
    ratios = transform.explained_variance_ratio
    assert np.all(np.diff(ratios) <= 1e-12), "components must be variance-ordered"


def test_pca_output_is_centred_on_the_train_fold_it_was_fitted_on():
    cohort = _cohort()
    transform = fit_ecg_transform(cohort)
    train = transform.transform(cohort.features[cohort.fold_mask(sorted(TRAIN_FOLDS))])

    assert np.abs(train.mean(axis=0)).max() < 1e-9


def test_default_component_count_sits_in_the_near_term_regime():
    assert 8 <= DEFAULT_COMPONENTS <= 32
    assert fit_ecg_transform(_cohort()).output_dimension == DEFAULT_COMPONENTS


def test_supervised_selection_finds_the_planted_separating_feature():
    transform = fit_ecg_transform(_cohort(), reduction="select", n_components=4)

    assert _NAMES[_SEPARATING_COLUMN] in transform.output_feature_names()
    assert transform.output_dimension == 4


def test_supervised_selection_ignores_the_unknown_label_rows():
    """An unknown label is not a negative, so it must not vote on feature choice."""
    cohort = _cohort(n_unknown=40)
    corrupted = cohort.features.copy()
    unknown = cohort.labels == LABEL_UNKNOWN
    # Make the unknown rows scream a *different* feature. If they were counted as
    # negatives, this column would outrank the genuinely separating one.
    corrupted[unknown, _SEPARATING_COLUMN + 1] = 500.0
    poisoned = EcgCohort(**{**cohort.__dict__, "features": corrupted})

    transform = fit_ecg_transform(poisoned, reduction="select", n_components=3)

    assert transform.n_train_labeled < transform.n_train_records
    assert _NAMES[_SEPARATING_COLUMN + 1] not in transform.output_feature_names()
    assert _NAMES[_SEPARATING_COLUMN] in transform.output_feature_names()


def test_supervised_selection_needs_both_classes_among_labeled_train_rows():
    cohort = _cohort()
    single = EcgCohort(
        **{
            **cohort.__dict__,
            "labels": np.zeros_like(cohort.labels),
            "label_known": np.ones_like(cohort.label_known),
        }
    )

    with pytest.raises(EcgTransformError, match="both classes"):
        fit_ecg_transform(single, reduction="select")


def test_identity_reduction_keeps_every_surviving_feature():
    transform = fit_ecg_transform(_cohort(), reduction="identity")

    assert transform.output_dimension == transform.kept_dimension
    assert transform.components is None
    assert transform.selected_indices is None
    assert set(transform.output_feature_names()) <= set(_NAMES)


def test_unknown_reduction_is_refused():
    with pytest.raises(EcgTransformError, match="Unknown reduction"):
        fit_ecg_transform(_cohort(), reduction="quantum-magic")


@pytest.mark.parametrize("reduction", REDUCTION_METHODS)
def test_every_reduction_method_round_trips_and_transforms(reduction: str, tmp_path: Path):
    cohort = _cohort()
    transform = fit_ecg_transform(cohort, reduction=reduction, n_components=8)
    reloaded = EcgFeatureTransform.load(transform.save(tmp_path / f"{reduction}.npz"))

    np.testing.assert_allclose(
        reloaded.transform(cohort.features), transform.transform(cohort.features)
    )
    assert reloaded.output_feature_names() == transform.output_feature_names()
    assert reloaded.summary() == transform.summary()


# ----------------------------------------------------------------- misuse guards


def test_transform_refuses_a_matrix_of_the_wrong_width():
    transform = fit_ecg_transform(_cohort())

    with pytest.raises(EcgTransformError, match="dimensional"):
        transform.transform(np.zeros((3, _DIM - 1)))


def test_transform_accepts_a_single_vector():
    cohort = _cohort()
    transform = fit_ecg_transform(cohort)

    assert transform.transform(cohort.features[0]).shape == (1, DEFAULT_COMPONENTS)


def test_fitting_refuses_a_cohort_with_a_different_feature_layout():
    cohort = _cohort()
    mismatched = EcgCohort(
        **{
            **cohort.__dict__,
            "features": cohort.features[:, :-1],
            "feature_names": _NAMES[:-1],
        }
    )

    with pytest.raises(EcgTransformError, match="Rebuild the cache"):
        fit_ecg_transform(mismatched)


def test_fitting_refuses_an_empty_fold_selection():
    with pytest.raises(EcgTransformError, match="No folds"):
        fit_ecg_transform(_cohort(), folds=[])


def test_fitting_refuses_folds_with_no_records():
    cohort = _cohort(n=20, folds=[1] * 20)

    with pytest.raises(EcgTransformError, match="No records in folds"):
        fit_ecg_transform(cohort, folds=[2, 3])


def test_load_refuses_an_artifact_from_another_transform_version(tmp_path: Path):
    import json

    transform = fit_ecg_transform(_cohort())
    path = transform.save(tmp_path / "t.npz")
    with np.load(path, allow_pickle=True) as archive:
        payload = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(payload.pop("metadata")))
    metadata["transform_version"] = "ecg-fit-0"
    np.savez_compressed(path, metadata=json.dumps(metadata), **payload)

    with pytest.raises(EcgTransformError, match="ecg-fit"):
        EcgFeatureTransform.load(path)


def test_load_refuses_an_artifact_fitted_on_another_feature_set(tmp_path: Path):
    import json

    transform = fit_ecg_transform(_cohort())
    path = transform.save(tmp_path / "t.npz")
    with np.load(path, allow_pickle=True) as archive:
        payload = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(payload.pop("metadata")))
    metadata["feature_set_version"] = "ecg-v0"
    np.savez_compressed(path, metadata=json.dumps(metadata), **payload)

    with pytest.raises(EcgTransformError, match="feature set"):
        EcgFeatureTransform.load(path)


def test_load_reports_a_missing_artifact(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        EcgFeatureTransform.load(tmp_path / "absent.npz")


def test_saved_artifact_is_arrays_and_json_not_a_pickled_object(tmp_path: Path):
    """An artifact that reproduces a reported number must load without sklearn."""
    transform = fit_ecg_transform(_cohort())
    path = transform.save(tmp_path / "t.npz")

    with np.load(path, allow_pickle=False) as archive:  # would raise on a pickle
        assert "kept_indices" in archive.files
        assert "medians" in archive.files
        assert "train_patient_ids" in archive.files


def test_summary_states_the_cohort_and_the_versions():
    cohort = _cohort()
    summary = fit_ecg_transform(cohort).summary()

    assert summary["transform_version"] == ECG_TRANSFORM_VERSION
    assert summary["feature_set_version"] == FEATURE_SET_VERSION
    assert summary["fitted_on_folds"] == sorted(TRAIN_FOLDS)
    assert summary["n_train_records"] == int(cohort.fold_mask(sorted(TRAIN_FOLDS)).sum())
    assert summary["n_dropped_features"] == len(summary["dropped_features"])
    assert summary["input_dimension"] == _DIM
