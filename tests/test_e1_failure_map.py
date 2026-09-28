"""Tests for the E1.1 classical failure map.

The failure map exists to answer one question honestly -- *does the 181-feature
descriptor's known structural incompleteness cost measurable accuracy?* -- and the
danger it must be defended against is the opposite of a bug: a harness that quietly
**manufactures** a bottleneck. So the invariants worth pinning down here are the ones
that make a verdict-B (quantum-motivating) result hard to produce by accident:

* the test partition is never read and no fitted quantity ever sees a validation row;
* every candidate is fitted with both a linear and a nonlinear arm, because §3's guard
  rail makes "both fell short" the precondition for calling a family a failure -- a
  single weak classifier must not be reportable as a bottleneck;
* the pre-registered grid (candidates, arms, draws, seed, conditions) is a frozen
  constant, not something assembled after the numbers are in;
* ``assign_verdict`` -- the single place a quantum motivation can be created -- only
  creates one for a *nonlinear-only* gain that clears the bootstrap margin **and**
  survives the patient-blocked column-permutation null, and a gain a linear model also
  captures is routed to the classical fix instead;
* the numeric helpers (train-only standardisation, mean-pooling, BH correction,
  patient-level bootstrap) do what their docstrings claim on constructed data.

The real numbers come from the driver against the frozen caches; these tests are what
make those numbers worth reading.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

import backend.evaluation.e1_failure_map as fmap
from backend.evaluation.e1_failure_map import (
    ARM_SPECS,
    BOOTSTRAP_DRAWS,
    CONDITIONS,
    FIXED_THRESHOLD,
    INCREMENT_PERMUTATIONS,
    PRIMARY,
    SEED,
    FailureMapError,
    _bh_adjust,
    _pooled_pixels,
    _standardise,
    assign_verdict,
    bootstrap_pr_auc,
)
from backend.training.pixel_data import CONDITION_ORACLE_ALL, CONDITION_ORACLE_LESION

SOURCE = Path(fmap.__file__).read_text(encoding="utf-8")

#: ``SOURCE`` with adjacent implicitly-concatenated string literals joined, so a
#: source-inspection assertion tests the *statement* and not where the author happened to
#: wrap it at 100 columns.
FLAT_SOURCE = re.sub(r'"\s*\n\s*"', "", SOURCE)

#: The classifier arms that count as linear vs nonlinear for §3's guard rail.
LINEAR_ARMS = {"logistic", "linear_svm"}
NONLINEAR_ARMS = {"rbf_svm", "random_forest", "gradient_boosting"}

#: Where the driver writes its payload. Present only after a full run has been executed,
#: so the payload-level assertions skip rather than fail when it has not been.
REPORT_PATH = Path("reports_e1_failure_map.json")


@pytest.fixture
def payload() -> dict:
    """The driver's emitted payload, or a skip when no run has produced one yet."""
    if not REPORT_PATH.exists():
        pytest.skip("reports_e1_failure_map.json not present; run the driver first")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------ helpers
def _candidate(
    name: str,
    *,
    best: float | None,
    linear: float | None,
    nonlinear: float | None,
    train_cv: float | None,
) -> dict:
    """A minimal candidate payload shaped exactly as :func:`assign_verdict` reads it."""
    arms = {}
    if train_cv is not None:
        arms["arm"] = {"train_cv_pr_auc": train_cv}
    return {
        "candidate": name,
        "best_validation_pr_auc": best,
        "best_linear_validation_pr_auc": linear,
        "best_nonlinear_validation_pr_auc": nonlinear,
        "arms": arms,
    }


# ----------------------------------------------------------- the test partition is sacred
class TestTestPartitionIsNeverRead:
    """The one claim the whole phase rests on: TRAIN + VALIDATION only."""

    def test_the_module_never_selects_the_test_partition(self):
        # The same source-inspection guard the E0 driver carries, for the same reason:
        # a single ``.test`` or ``partitions["test"]`` would silently invalidate the run.
        assert not re.search(r"\.test\b", SOURCE), (
            "the failure map must not touch the .test partition attribute"
        )
        assert not re.search(r"partitions\s*\[\s*[\"']test[\"']\s*\]", SOURCE)
        assert not re.search(r"roi_mode\s*=\s*[\"']test[\"']", SOURCE)

    def test_the_only_partitions_iterated_are_train_and_validation(self):
        # build_inputs loops explicitly over ("train", "validation"); no third partition
        # name is ever introduced into that loop.
        assert re.search(r'for partition_name in \("train", "validation"\)', SOURCE)
        assert 'partitions["test"]' not in SOURCE

    def test_the_payload_declares_the_test_partition_unused(self):
        assert '"test_partition_used": False' in SOURCE
        assert "test_partition_note" in SOURCE

    def test_conditions_are_oracle_only_no_predicted_or_test_condition(self):
        # §2 fixes the two oracle conditions. A predicted or test condition would be a
        # different experiment; there is deliberately no way to name one here.
        assert PRIMARY == CONDITION_ORACLE_LESION
        assert CONDITIONS == (CONDITION_ORACLE_LESION, CONDITION_ORACLE_ALL)


# ----------------------------------------------------------------- no validation leakage
class TestFittedQuantitiesSeeTrainOnly:
    def test_standardiser_uses_train_mean_and_scale_only(self):
        train = np.array([[0.0], [2.0]])
        validation = np.array([[4.0]])
        std_train, std_validation = _standardise(train, validation)
        # train -> exact zero mean, unit sd; validation transformed by TRAIN stats (mean
        # 1, sd 1), so 4 -> 3. A validation-fitted scaler could never produce 3 here.
        assert np.allclose(std_train, [[-1.0], [1.0]])
        assert np.allclose(std_validation, [[3.0]])

    def test_constant_train_column_is_passed_through_not_divided_by_zero(self):
        train = np.array([[5.0], [5.0]])
        validation = np.array([[8.0]])
        std_train, std_validation = _standardise(train, validation)
        assert np.all(np.isfinite(std_train))
        assert np.allclose(std_train, [[0.0], [0.0]])
        assert np.allclose(std_validation, [[3.0]])  # (8 - 5) / scale-of-1

    def test_arms_fit_on_train_indices_only(self):
        # The three fitting helpers only ever call .fit on x_train / fold slices of it;
        # validation appears solely inside decision_function / predict_proba calls.
        for call in re.findall(r"\.fit\(([^)]*)\)", FLAT_SOURCE):
            assert "x_validation" not in call and "validation_" not in call, (
                f".fit() must never receive a validation argument: {call!r}"
            )


# ------------------------------------------------------------------ the §3 guard rail
class TestEveryFamilyHasLinearAndNonlinearArm:
    """A single weak classifier is not a bottleneck (§3). Both must fall short."""

    def test_each_candidate_has_at_least_one_linear_and_one_nonlinear_arm(self):
        for name, arms in ARM_SPECS.items():
            arm_set = set(arms)
            assert arm_set & LINEAR_ARMS, f"{name} has no linear arm"
            assert arm_set & NONLINEAR_ARMS, f"{name} has no nonlinear arm"

    def test_every_arm_is_a_known_classifier(self):
        for arms in ARM_SPECS.values():
            for arm in arms:
                assert arm in (LINEAR_ARMS | NONLINEAR_ARMS)


# --------------------------------------------------------------- the grid is pre-frozen
class TestFrozenPreRegistration:
    def test_the_eight_candidates_are_the_pre_registered_set(self):
        assert set(ARM_SPECS) == {"C1", "C2", "C2b", "C3", "C4", "C5", "C6", "C7"}

    def test_bootstrap_and_permutation_budgets_match_the_spec(self):
        assert BOOTSTRAP_DRAWS >= 2000  # §9.4
        assert INCREMENT_PERMUTATIONS == 200  # §9.2

    def test_seed_and_threshold_are_fixed_constants(self):
        assert SEED == 42
        assert FIXED_THRESHOLD == 0.50

    def test_the_grid_is_frozen_before_the_first_fit(self):
        # The multiplicity count must be declared in advance, not counted from whatever
        # happened to run -- E0's 17-of-136 artefact is why.
        assert '"frozen_before_first_fit": True' in SOURCE


# ----------------------------------------------------------------------- numeric helpers
class TestPooledPixels:
    def test_mean_pools_a_known_grid(self):
        row = np.arange(16, dtype=float).reshape(1, 16)  # a single 4x4 image, row-major
        pooled = _pooled_pixels(row, 2)
        # 2x2 block means of 0..15 laid out 4x4.
        assert np.allclose(pooled, [[2.5, 4.5, 10.5, 12.5]])

    def test_pooling_is_an_average_not_a_subsample(self):
        # A stride subsample of a ramp would return corner pixels (0, 2, 8, 10); the mean
        # pool returns block averages, which is the whole point of _pooled_pixels.
        row = np.arange(16, dtype=float).reshape(1, 16)
        pooled = _pooled_pixels(row, 2)
        assert not np.allclose(pooled, [[0.0, 2.0, 8.0, 10.0]])

    def test_non_square_rows_are_refused(self):
        with pytest.raises(FailureMapError):
            _pooled_pixels(np.zeros((1, 15)), 2)

    def test_indivisible_edge_is_refused(self):
        with pytest.raises(FailureMapError):
            _pooled_pixels(np.zeros((1, 16)), 3)  # side 4 not divisible by 3


class TestBHAdjust:
    def test_none_is_passed_through_in_place(self):
        assert _bh_adjust([None, 0.02]) == [None, 0.02]

    def test_two_value_step_up_matches_the_hand_computation(self):
        # positions [0.9, 0.01]: sorted the small one gets 0.01*2/1 = 0.02, the large one
        # 0.9*2/2 = 0.9, and order is preserved.
        assert _bh_adjust([0.9, 0.01]) == [0.9, 0.02]

    def test_adjusted_values_never_exceed_one(self):
        out = _bh_adjust([0.6, 0.7, 0.8, 0.9])
        assert all(p is not None and p <= 1.0 for p in out)

    def test_empty_and_all_none_are_stable(self):
        assert _bh_adjust([]) == []
        assert _bh_adjust([None, None]) == [None, None]


class TestPatientLevelBootstrap:
    def test_resamples_patients_not_images(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.7])
        patients = ["p1", "p1", "p2", "p2", "p3", "p3"]
        result = bootstrap_pr_auc(y, scores, patients, draws=200, seed=SEED)
        assert result["measured"] is True
        assert result["unit"] == "patient (resampled with replacement)"
        # A sane interval brackets its own mean.
        assert result["p2_5"] <= result["mean"] <= result["p97_5"]

    def test_is_deterministic_at_a_fixed_seed(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        scores = np.array([0.2, 0.7, 0.3, 0.9, 0.8, 0.1])
        patients = ["a", "a", "b", "b", "c", "c"]
        first = bootstrap_pr_auc(y, scores, patients, draws=200, seed=7)
        second = bootstrap_pr_auc(y, scores, patients, draws=200, seed=7)
        assert first == second


# ------------------------------- the single place a quantum motivation can be created
class TestAssignVerdict:
    C1 = 0.80
    DELTA = 0.05

    def test_c1_is_always_the_control(self):
        verdict = assign_verdict(
            _candidate("C1", best=0.80, linear=0.80, nonlinear=0.80, train_cv=0.80),
            c1_pr_auc=self.C1,
            delta=self.DELTA,
            increment=None,
        )
        assert verdict["verdict"] == "control"
        assert verdict["motivates_quantum"] is False

    def test_nonlinear_only_gain_that_survives_the_null_is_the_only_verdict_B(self):
        verdict = assign_verdict(
            # nonlinear clears (0.90 - 0.80 = 0.10 >= delta), linear does not (0.02).
            _candidate("C4", best=0.90, linear=0.82, nonlinear=0.90, train_cv=0.88),
            c1_pr_auc=self.C1,
            delta=self.DELTA,
            increment={"survives_gating_null": True},
        )
        assert verdict["verdict"] == "B"
        assert verdict["motivates_quantum"] is True

    def test_a_gain_a_linear_model_also_captures_is_verdict_A_not_B(self):
        # §3 guard rail: if the information is linearly accessible, the fix is the
        # feature, not a quantum transformation.
        verdict = assign_verdict(
            _candidate("C5", best=0.90, linear=0.90, nonlinear=0.90, train_cv=0.90),
            c1_pr_auc=self.C1,
            delta=self.DELTA,
            increment={"survives_gating_null": True},
        )
        assert verdict["verdict"] == "A"
        assert verdict["motivates_quantum"] is False

    def test_a_nonlinear_gain_that_fails_the_null_is_verdict_C(self):
        verdict = assign_verdict(
            _candidate("C6", best=0.90, linear=0.70, nonlinear=0.90, train_cv=0.90),
            c1_pr_auc=self.C1,
            delta=self.DELTA,
            increment={"survives_gating_null": False},
        )
        assert verdict["verdict"] == "C"
        assert verdict["motivates_quantum"] is False

    def test_validation_gain_without_a_train_cv_gain_is_the_E0_signature_C(self):
        # Clears on validation but no within-train CV gain, and no increment measured:
        # E0's exact failure mode, reported as weak / non-generalizable.
        verdict = assign_verdict(
            _candidate("C3", best=0.90, linear=0.70, nonlinear=0.90, train_cv=0.70),
            c1_pr_auc=self.C1,
            delta=self.DELTA,
            increment=None,
        )
        assert verdict["verdict"] == "C"
        assert verdict["motivates_quantum"] is False

    def test_within_margin_is_never_quantum_motivating(self):
        for increment in ({"survives_gating_null": False}, None):
            verdict = assign_verdict(
                _candidate("C4", best=0.82, linear=0.82, nonlinear=0.82, train_cv=0.82),
                c1_pr_auc=self.C1,
                delta=self.DELTA,
                increment=increment,
            )
            assert verdict["verdict"] in ("A", "D")
            assert verdict["motivates_quantum"] is False

    def test_a_family_that_falls_short_of_c1_is_verdict_C(self):
        verdict = assign_verdict(
            _candidate("C2", best=0.70, linear=0.70, nonlinear=0.70, train_cv=0.70),
            c1_pr_auc=self.C1,
            delta=self.DELTA,
            increment={"survives_gating_null": False},
        )
        assert verdict["verdict"] == "C"
        assert verdict["motivates_quantum"] is False

    def test_missing_margin_is_not_measurable_and_never_motivates_quantum(self):
        verdict = assign_verdict(
            _candidate("C4", best=None, linear=None, nonlinear=None, train_cv=None),
            c1_pr_auc=None,
            delta=None,
            increment=None,
        )
        assert verdict["verdict"] is None
        assert verdict["motivates_quantum"] is False


# --------------------------------------------------- the emitted payload, when it exists
class TestEmittedPayload:
    """Assertions against a real run's payload. Skipped when no report is on disk."""

    def test_payload_declares_the_test_partition_unused(self, payload: dict):
        assert payload["test_partition_used"] is False

    def test_no_patient_appears_in_both_train_and_validation(self, payload: dict):
        for condition in payload["conditions"].values():
            assert condition["rows"]["patient_overlap_train_validation"] == 0

    def test_the_candidate_grid_was_frozen_before_the_first_fit(self, payload: dict):
        assert payload["frozen_candidate_grid"]["frozen_before_first_fit"] is True

    def test_a_null_result_recommends_no_quantum_implementation(self, payload: dict):
        summary = payload["bottleneck_summary"]
        if not summary["classical_bottleneck_measured"]:
            assert "NO QUANTUM IMPLEMENTATION" in summary["conclusion"]
            assert summary["families_motivating_quantum"] == []

    def test_a_measured_bottleneck_names_a_verdict_B_family(self, payload: dict):
        summary = payload["bottleneck_summary"]
        if summary["classical_bottleneck_measured"]:
            assert summary["families_motivating_quantum"]
            primary = payload["conditions"][payload["primary_condition"]]
            for name in summary["families_motivating_quantum"]:
                assert primary["candidates"][name]["classification"]["verdict"] == "B"
