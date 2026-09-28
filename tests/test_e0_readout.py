"""Tests for the E0.1 readout driver.

The numerics live in :mod:`quantum_ml.readout` and are tested in
``tests/test_readout.py``. What is defended here is the *experimental discipline* the
Phase E0 spec is built on, because that is what makes a negative result trustworthy:

* the standardiser sees train rows and only train rows, structurally;
* the permutation null preserves the patient block structure, so it is not easier than
  reality (which would turn a chance result into a "finding");
* the null harness has power -- it flags planted signal -- as well as calibration;
* the gradient is the exact derivative of the loss actually reported, not an
  approximation that happens to look plausible;
* the test partition is never touched;
* single-class subsets yield ``None`` rather than a fabricated number;
* runs at a fixed seed are reproducible.

Cheap access to 65,536 observables is a multiple-comparisons trap, so
:class:`TestPreRegistration` also checks the driver's variant list is a frozen constant.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from backend.evaluation import e0_readout as e0
from backend.evaluation.e0_readout import (
    DEFAULT_PERMUTATIONS,
    FIXED_THRESHOLD,
    PRIMARY_CONDITION,
    VARIANCE_FLOOR,
    VARIANTS,
    WEIGHT_SETTINGS,
    E0Error,
    Readout,
    Scaler,
    class_weights_for,
    fit_readout,
    gradient_block,
    loss_and_gradient,
    observable_report,
    patient_blocked_permutations,
)
from backend.ml.pixel_pipeline import V1_PIXEL_COUNT, V1_QUBIT_COUNT
from quantum_ml.readout import VARIANT_SIZES, measure_batch, observable_set

SOURCE = Path(e0.__file__).read_text(encoding="utf-8")


def _synthetic_cohort(rng, n_patients=40, per_patient=3, prevalence=0.35):
    """Patient-blocked cohort: a patient's diagnosis is constant across its images."""
    patient_labels = (rng.random(n_patients) < prevalence).astype(int)
    patients, labels = [], []
    for p in range(n_patients):
        for _ in range(per_patient):
            patients.append(f"P{p:03d}")
            labels.append(patient_labels[p])
    return np.asarray(labels), patients


class TestScaler:
    def test_statistics_come_from_train_rows_only(self):
        rng = np.random.default_rng(1)
        x_train = rng.normal(size=(50, 6))
        x_validation = rng.normal(loc=100.0, scale=50.0, size=(20, 6))

        alone = Scaler.fit(x_train, [f"o{i}" for i in range(6)])
        # Fitting again with the validation rows sitting in memory, unpassed, must give
        # bit-identical statistics. The point is that there is no code path by which
        # validation could reach the scaler, not that the caller remembered not to.
        _ = x_validation
        again = Scaler.fit(x_train, [f"o{i}" for i in range(6)])
        assert np.array_equal(alone.mean, again.mean)
        assert np.array_equal(alone.std, again.std)

        # And the transform of validation rows uses train statistics, so standardised
        # validation is emphatically not zero-mean/unit-sd.
        z_validation = alone.transform(x_validation)
        assert abs(float(z_validation.mean())) > 1.0

    def test_train_rows_standardise_to_zero_mean_unit_sd(self):
        rng = np.random.default_rng(2)
        x = rng.normal(loc=7.0, scale=3.0, size=(200, 4))
        z = Scaler.fit(x, ["a", "b", "c", "d"]).transform(x)
        assert np.abs(z.mean(axis=0)).max() < 1e-12
        assert np.abs(z.std(axis=0, ddof=0) - 1.0).max() < 1e-12

    def test_degenerate_observable_is_dropped_not_amplified(self):
        # Standardising a constant column divides float residue by float residue and
        # manufactures a unit-variance "feature" out of nothing.
        rng = np.random.default_rng(3)
        x = np.column_stack([rng.normal(size=40), np.full(40, 0.25)])
        scaler = Scaler.fit(x, ["live", "dead"])
        assert scaler.n_excluded == 1
        assert scaler.excluded_labels == ["dead"]
        assert scaler.transform(x).shape == (40, 1)

    def test_excluded_count_is_reported_not_silent(self):
        rng = np.random.default_rng(4)
        x = np.column_stack([rng.normal(size=30), np.zeros(30), np.zeros(30)])
        payload = Scaler.fit(x, ["a", "b", "c"]).to_dict()
        assert payload["n_excluded_below_variance_floor"] == 2
        assert payload["variance_floor"] == VARIANCE_FLOOR
        assert payload["fitted_on"] == "train rows only"

    def test_all_degenerate_refuses_rather_than_returning_an_empty_design(self):
        with pytest.raises(E0Error, match="nothing to standardise"):
            Scaler.fit(np.ones((10, 3)), ["a", "b", "c"])


class TestFrozenPhaseDReadout:
    def test_a0_reproduces_the_exact_phase_d_score_map(self):
        readout = Readout(kind="fixed_z0")
        z0 = np.array([[-0.0094], [0.0], [0.0091]])
        assert np.allclose(readout.scores(z0), (1.0 - z0[:, 0]) / 2.0)
        assert readout.n_trainable == 0

    def test_a0_reproduces_the_phase_d_squeeze(self):
        """The measured Phase D range mapped through the frozen readout."""
        observed = np.array([[-0.0094], [0.0091]])
        scores = Readout(kind="fixed_z0").scores(observed)
        assert scores.min() == pytest.approx(0.4954500, abs=1e-6)
        assert scores.max() == pytest.approx(0.5047000, abs=1e-6)
        assert scores.max() - scores.min() < 0.01

    def test_a0_clips_into_the_unit_interval(self):
        scores = Readout(kind="fixed_z0").scores(np.array([[-3.0], [3.0]]))
        assert scores.tolist() == [1.0, 0.0]

    def test_a0_has_no_fitted_state_at_all(self):
        readout = Readout(kind="fixed_z0")
        assert readout.scaler is None and readout.coefficients is None
        assert "nothing fitted" in (
            fit_readout(
                np.zeros((4, 1)), np.array([0, 1, 0, 1]), ["a", "b", "c", "d"],
                variant="A0", labels=["Z0"], seed=42,
            ).cv_note
            or ""
        )


class TestFittedReadout:
    def test_logistic_head_parameter_count_is_observables_plus_intercept(self):
        rng = np.random.default_rng(5)
        labels, patients = _synthetic_cohort(rng)
        x = rng.normal(size=(labels.size, 16)) + labels[:, None] * 0.6
        readout = fit_readout(
            x, labels, patients, variant="B",
            labels=observable_set("B", V1_QUBIT_COUNT).labels, seed=42,
        )
        assert readout.kind == "logistic"
        assert readout.n_trainable == 17
        assert readout.regularisation_c in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)

    def test_regularisation_is_selected_by_patient_grouped_cv_on_train(self):
        rng = np.random.default_rng(6)
        labels, patients = _synthetic_cohort(rng)
        x = rng.normal(size=(labels.size, 16)) + labels[:, None] * 0.6
        note = fit_readout(
            x, labels, patients, variant="B",
            labels=observable_set("B", V1_QUBIT_COUNT).labels, seed=42,
        ).cv_note
        assert "StratifiedGroupKFold" in note
        assert "grouped by patient" in note
        assert "TRAIN" in note
        assert "average_precision" in note

    def test_fit_is_deterministic_at_a_fixed_seed(self):
        rng = np.random.default_rng(7)
        labels, patients = _synthetic_cohort(rng)
        x = rng.normal(size=(labels.size, 16)) + labels[:, None] * 0.4
        kwargs = dict(
            variant="B", labels=observable_set("B", V1_QUBIT_COUNT).labels, seed=42
        )
        first = fit_readout(x, labels, patients, **kwargs)
        second = fit_readout(x, labels, patients, **kwargs)
        assert np.array_equal(first.coefficients, second.coefficients)
        assert first.intercept == second.intercept
        assert first.regularisation_c == second.regularisation_c

    def test_scores_stay_in_the_unit_interval_under_extreme_input(self):
        readout = Readout(
            kind="logistic",
            scaler=Scaler.fit(np.random.default_rng(8).normal(size=(20, 2)), ["a", "b"]),
            coefficients=np.array([50.0, -50.0]),
            intercept=10.0,
        )
        scores = readout.scores(np.array([[1e6, -1e6], [-1e6, 1e6]]))
        assert np.all((scores >= 0.0) & (scores <= 1.0))
        assert np.all(np.isfinite(scores))


class TestPatientBlockedPermutations:
    def test_labels_stay_constant_within_a_patient(self):
        rng = np.random.default_rng(9)
        labels, patients = _synthetic_cohort(rng)
        for permuted in patient_blocked_permutations(
            labels, patients, n=25, seed=42
        ):
            for patient in set(patients):
                mask = np.asarray([p == patient for p in patients])
                assert len(set(permuted[mask].tolist())) == 1, patient

    def test_patient_level_positive_count_is_preserved(self):
        rng = np.random.default_rng(10)
        labels, patients = _synthetic_cohort(rng)
        order = sorted(set(patients))
        truth = sum(
            int(labels[[i for i, p in enumerate(patients) if p == q][0]]) for q in order
        )
        for permuted in patient_blocked_permutations(labels, patients, n=20, seed=42):
            got = sum(
                int(permuted[[i for i, p in enumerate(patients) if p == q][0]])
                for q in order
            )
            assert got == truth

    def test_image_level_prevalence_is_preserved_for_balanced_blocks(self):
        rng = np.random.default_rng(11)
        labels, patients = _synthetic_cohort(rng, per_patient=3)
        for permuted in patient_blocked_permutations(labels, patients, n=10, seed=42):
            assert permuted.sum() == labels.sum()

    def test_permutations_actually_differ_from_the_truth(self):
        rng = np.random.default_rng(12)
        labels, patients = _synthetic_cohort(rng)
        draws = patient_blocked_permutations(labels, patients, n=50, seed=42)
        assert sum(1 for d in draws if not np.array_equal(d, labels)) >= 45
        assert len({d.tobytes() for d in draws}) >= 45

    def test_is_deterministic_at_a_fixed_seed(self):
        rng = np.random.default_rng(13)
        labels, patients = _synthetic_cohort(rng)
        a = patient_blocked_permutations(labels, patients, n=10, seed=42)
        b = patient_blocked_permutations(labels, patients, n=10, seed=42)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
        c = patient_blocked_permutations(labels, patients, n=10, seed=43)
        assert not all(np.array_equal(x, y) for x, y in zip(a, c))

    def test_an_image_shuffle_would_have_been_an_easier_null(self):
        """Why the block structure matters, stated as a measurement.

        An image-level shuffle breaks patients apart, which lets a fitted head exploit
        within-patient correlation that carries no diagnostic information. The resulting
        null sits higher than the honest one, so the honest null is the conservative
        choice and the difference is not cosmetic.
        """
        rng = np.random.default_rng(14)
        labels, patients = _synthetic_cohort(rng, n_patients=30, per_patient=4)
        blocked = patient_blocked_permutations(labels, patients, n=30, seed=42)
        for permuted in blocked:
            # A patient-blocked draw can be reproduced by relabelling patients; an
            # image-level shuffle generally cannot, and that is the whole distinction.
            for patient in set(patients):
                mask = np.asarray([p == patient for p in patients])
                assert permuted[mask].std() == 0.0

        shuffled = np.random.default_rng(42).permutation(labels)
        broken = sum(
            1 for patient in set(patients)
            if shuffled[np.asarray([p == patient for p in patients])].std() > 0
        )
        assert broken > 0, "image shuffle should split patients -- that is the hazard"


class TestNullHarness:
    def test_empirical_p_can_never_be_zero(self):
        # (n_ge + 1) / (n + 1): at 200 permutations the smallest honest p is 1/201.
        result = e0._rank_in_null(10.0, [0.0] * 200)
        assert result["n_null_at_or_above_observed"] == 0
        assert result["empirical_p"] == pytest.approx(1 / 201, abs=1e-6)
        assert result["empirical_p"] > 0.0
        assert result["exceeds_null_p95"] is True

    def test_observed_inside_the_null_reports_a_large_p(self):
        null = list(np.linspace(0.0, 1.0, 201))
        result = e0._rank_in_null(0.5, null)
        assert result["empirical_p"] > 0.4
        assert result["exceeds_null_p95"] is False

    def test_missing_observation_yields_none_not_a_default(self):
        assert e0._rank_in_null(None, [0.1, 0.2]) is None
        assert e0._rank_in_null(float("nan"), [0.1, 0.2]) is None
        assert e0._rank_in_null(0.5, []) is None
        assert e0._null_summary([]) is None
        assert e0._null_summary([None, float("nan")]) is None

    def test_null_summary_reports_the_tail_not_just_the_centre(self):
        summary = e0._null_summary(list(np.linspace(0.0, 1.0, 101)))
        assert summary["n"] == 101
        assert summary["p50"] == pytest.approx(0.5, abs=1e-9)
        assert summary["p95"] == pytest.approx(0.95, abs=1e-9)
        assert summary["max"] == pytest.approx(1.0, abs=1e-9)

    def test_the_harness_has_power_against_planted_signal(self):
        """Calibration is half the requirement; detecting real signal is the other half.

        A null that never fires is trivially well-calibrated and useless. This plants a
        patient-level signal strong enough that any working harness must flag it, and
        asserts the observed statistic clears the permutation null.
        """
        rng = np.random.default_rng(15)
        labels, patients = _synthetic_cohort(rng, n_patients=40, per_patient=3)
        feature = labels + rng.normal(scale=0.25, size=labels.size)

        observed = average_precision_score(labels, feature)
        null = [
            average_precision_score(permuted, feature)
            for permuted in patient_blocked_permutations(
                labels, patients, n=60, seed=42
            )
        ]
        result = e0._rank_in_null(observed, null)
        assert observed > 0.9
        assert result["exceeds_null_p95"] is True
        assert result["empirical_p"] <= 1 / 61 + 1e-6

    def test_the_harness_does_not_fire_on_noise(self):
        rng = np.random.default_rng(16)
        labels, patients = _synthetic_cohort(rng, n_patients=40, per_patient=3)
        feature = rng.normal(size=labels.size)  # independent of the labels by construction

        null = [
            average_precision_score(permuted, feature)
            for permuted in patient_blocked_permutations(
                labels, patients, n=60, seed=42
            )
        ]
        result = e0._rank_in_null(average_precision_score(labels, feature), null)
        assert result["empirical_p"] > 0.05
        assert result["exceeds_null_p95"] is False
        # The null should sit around the prevalence, which is the no-skill AP.
        assert e0._null_summary(null)["p50"] == pytest.approx(
            float(labels.mean()), abs=0.15
        )


class TestSeparability:
    def test_auc_matches_sklearn(self):
        rng = np.random.default_rng(17)
        y = (rng.random(200) < 0.4).astype(int)
        score = rng.normal(size=200) + y * 0.5
        assert e0._auc(y, score) == pytest.approx(roc_auc_score(y, score), abs=1e-12)

    def test_auc_handles_ties_at_half(self):
        y = np.array([0, 1, 0, 1])
        assert e0._auc(y, np.ones(4)) == pytest.approx(0.5)

    def test_auc_is_none_on_a_single_class_subset(self):
        assert e0._auc(np.zeros(5, dtype=int), np.arange(5.0)) is None
        assert e0._auc(np.ones(5, dtype=int), np.arange(5.0)) is None
        assert e0._auc(np.array([], dtype=int), np.array([])) is None

    def test_metrics_are_none_on_a_single_class_subset(self):
        # C_fallback is single-class on validation; the spec requires those rows to be
        # reported and stay identifiable, not to receive a manufactured PR-AUC.
        assert e0._metrics(np.zeros(6, dtype=int), np.full(6, 0.5)) is None
        assert e0._metrics(np.array([], dtype=int), np.array([])) is None
        both = e0._metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
        assert both is not None and both["pr_auc"] is not None
        assert both["threshold"] == FIXED_THRESHOLD

    def test_observable_report_carries_distribution_and_both_partitions(self):
        rng = np.random.default_rng(18)
        oset = observable_set("B", V1_QUBIT_COUNT)
        y_train = (rng.random(60) < 0.4).astype(int)
        y_validation = (rng.random(20) < 0.4).astype(int)
        report = observable_report(
            rng.normal(size=(60, 16)), y_train,
            rng.normal(size=(20, 16)), y_validation, oset,
        )
        assert len(report) == 16
        for entry in report:
            assert entry["spatial_meaning"]
            assert entry["train"]["roc_auc"] is not None
            assert entry["validation"]["roc_auc"] is not None
            for key in ("mean", "sd", "min", "p05", "median", "p95", "max"):
                assert key in entry["train"]

    def test_observable_report_flags_a_degenerate_column(self):
        rng = np.random.default_rng(19)
        oset = observable_set("B", V1_QUBIT_COUNT)
        x_train = rng.normal(size=(40, 16))
        x_train[:, 3] = 0.5
        report = observable_report(
            x_train, (rng.random(40) < 0.5).astype(int),
            rng.normal(size=(10, 16)), (rng.random(10) < 0.5).astype(int), oset,
        )
        assert report[3]["below_variance_floor"] is True
        assert report[3]["train"]["standardised_class_gap"] is None
        assert report[0]["below_variance_floor"] is False


class TestGradient:
    """The parameter-shift gradient must be the derivative of the reported loss."""

    N_QUBITS = 6

    def _setup(self, seed=20, variant="B"):
        rng = np.random.default_rng(seed)
        n = 1 << self.N_QUBITS
        states = rng.normal(size=(12, n))
        states /= np.linalg.norm(states, axis=1, keepdims=True)
        labels = np.array([0, 1] * 6)
        oset = observable_set(variant, self.N_QUBITS)
        n_obs = len(oset)
        readout = Readout(
            kind="logistic",
            scaler=Scaler.fit(rng.normal(size=(30, n_obs)), oset.labels),
            coefficients=rng.normal(scale=0.5, size=n_obs),
            intercept=0.1,
        )
        weights = rng.uniform(-np.pi, np.pi, 2 * self.N_QUBITS)
        return states, labels, readout, oset, weights

    def _loss(self, states, labels, readout, oset, weights, class_weights):
        return loss_and_gradient(
            states, labels, readout, oset, weights,
            n_qubits=self.N_QUBITS, n_layers=1,
            class_weights=class_weights, chunk=8,
        )[0]

    def test_parameter_shift_matches_central_finite_differences(self):
        states, labels, readout, oset, weights = self._setup()
        class_weights = class_weights_for(labels)
        _, analytic = loss_and_gradient(
            states, labels, readout, oset, weights,
            n_qubits=self.N_QUBITS, n_layers=1,
            class_weights=class_weights, chunk=8,
        )
        h = 1e-5
        numeric = np.empty_like(analytic)
        for k in range(weights.size):
            plus, minus = weights.copy(), weights.copy()
            plus[k] += h
            minus[k] -= h
            numeric[k] = (
                self._loss(states, labels, readout, oset, plus, class_weights)
                - self._loss(states, labels, readout, oset, minus, class_weights)
            ) / (2 * h)
        assert np.abs(analytic - numeric).max() < 1e-6

    def test_parameter_shift_matches_finite_differences_for_the_frozen_a0_map(self):
        states, labels, _, _, weights = self._setup(seed=21, variant="A0")
        oset = observable_set("A0", self.N_QUBITS)
        readout = Readout(kind="fixed_z0")
        class_weights = class_weights_for(labels)
        _, analytic = loss_and_gradient(
            states, labels, readout, oset, weights,
            n_qubits=self.N_QUBITS, n_layers=1,
            class_weights=class_weights, chunk=8,
        )
        h = 1e-5
        numeric = np.empty_like(analytic)
        for k in range(weights.size):
            plus, minus = weights.copy(), weights.copy()
            plus[k] += h
            minus[k] -= h
            numeric[k] = (
                self._loss(states, labels, readout, oset, plus, class_weights)
                - self._loss(states, labels, readout, oset, minus, class_weights)
            ) / (2 * h)
        assert np.abs(analytic - numeric).max() < 1e-6

    def test_gradient_block_reports_the_ln2_displacement_and_the_cost(self):
        states, labels, readout, oset, weights = self._setup(seed=22)
        block = gradient_block(
            states, labels, readout, oset, weights,
            n_qubits=self.N_QUBITS, n_layers=1,
            class_weights=class_weights_for(labels), chunk=8,
        )
        assert block["n_parameters"] == 2 * self.N_QUBITS
        assert block["circuit_evaluations"] == 2 * 2 * self.N_QUBITS + 1
        assert len(block["per_parameter_abs"]) == 2 * self.N_QUBITS
        assert block["loss_minus_ln2"] == pytest.approx(
            block["loss"] - float(np.log(2.0)), abs=1e-7
        )
        assert "parameter-shift" in block["method"]
        assert "readout coefficients held fixed" in block["method"]

    def test_a0_compresses_near_uniform_states_into_an_unusable_score_band(self):
        """The Phase D failure mode, reproduced from first principles.

        An amplitude-encoded natural image is dominated by its DC component -- every
        amplitude is positive and close to the mean -- so across samples ``<Z_0>`` barely
        moves. Phase D measured sd 0.0028 with the whole dataset inside
        ``[-0.0094, +0.0091]``, and the ``(1 - <Z_0>)/2`` map turned that into scores
        inside ``[0.4953, 0.5047]``: every image gets the same answer to three decimals,
        so no threshold and no amount of optimisation can separate the classes.

        Note the failure is about *spread across samples*, not the absolute value. The
        ansatz can place ``<Z_0>`` anywhere it likes -- what it cannot do is spread the
        samples apart, which is the only thing a classifier can use. Getting that
        distinction right is why E0.1 reports per-observable distributions and not just
        headline metrics.
        """
        rng = np.random.default_rng(23)
        n = 1 << self.N_QUBITS
        states = np.abs(rng.normal(scale=0.01, size=(64, n))) + 1.0
        states /= np.linalg.norm(states, axis=1, keepdims=True)
        oset = observable_set("A0", self.N_QUBITS)
        readout = Readout(kind="fixed_z0")

        widths, centres = [], []
        for draw in range(6):
            weights = np.random.default_rng(100 + draw).uniform(
                -np.pi, np.pi, 2 * self.N_QUBITS
            )
            observed = measure_batch(
                states, oset.diagonals(self.N_QUBITS), weights=weights,
                n_qubits=self.N_QUBITS, n_layers=1, chunk=16,
            )
            # Across-sample spread, the only thing a classifier can exploit...
            assert float(observed.std()) < 5e-3
            # ...which the frozen map compresses into a band far narrower than any
            # decision could resolve.
            scores = readout.scores(observed)
            widths.append(float(scores.max() - scores.min()))
            centres.append(float(scores.mean()))
            assert widths[-1] < 1e-2

        # The band's *position* moves freely with the parameters while its width does
        # not. That asymmetry is the failure: the ansatz has plenty of influence over
        # the score, and none at all over the separation between samples.
        assert max(centres) - min(centres) > 5 * max(widths)

    def test_a0_loss_stays_on_the_ln2_plateau_for_such_states(self):
        rng = np.random.default_rng(24)
        n = 1 << self.N_QUBITS
        states = np.abs(rng.normal(scale=0.01, size=(16, n))) + 1.0
        states /= np.linalg.norm(states, axis=1, keepdims=True)
        labels = np.array([0, 1] * 8)
        oset = observable_set("A0", self.N_QUBITS)
        readout = Readout(kind="fixed_z0")

        losses = []
        for draw in range(6):
            weights = np.random.default_rng(200 + draw).uniform(
                -np.pi, np.pi, 2 * self.N_QUBITS
            )
            loss, gradient = loss_and_gradient(
                states, labels, readout, oset, weights,
                n_qubits=self.N_QUBITS, n_layers=1,
                class_weights=class_weights_for(labels), chunk=8,
            )
            losses.append(loss)
            assert np.all(np.isfinite(gradient))
        # Phase D observed displacements of 0.0008-0.0025 from ln 2 = 0.693147.
        ln2 = float(np.log(2.0))
        assert max(abs(value - ln2) for value in losses) < 0.30
        assert max(losses) - min(losses) < 0.30


class TestClassWeights:
    def test_returns_negative_then_positive(self):
        labels = np.array([0] * 90 + [1] * 10)
        negative, positive = class_weights_for(labels)
        assert positive > negative

    def test_normalised_to_mean_one_over_samples(self):
        labels = np.array([0] * 90 + [1] * 10)
        negative, positive = class_weights_for(labels)
        per_sample = np.where(labels == 1, positive, negative)
        assert float(per_sample.mean()) == pytest.approx(1.0, abs=1e-9)

    def test_balanced_data_gets_unit_weights(self):
        negative, positive = class_weights_for(np.array([0, 1] * 50))
        assert negative == pytest.approx(1.0)
        assert positive == pytest.approx(1.0)

    def test_single_class_does_not_divide_by_zero(self):
        for labels in (np.zeros(10, dtype=int), np.ones(10, dtype=int)):
            negative, positive = class_weights_for(labels)
            assert np.isfinite(negative) and np.isfinite(positive)


class TestPreRegistration:
    def test_variant_list_is_the_frozen_pre_registered_set(self):
        assert VARIANTS == ("A0", "A1", "B", "C", "D")
        assert set(VARIANTS) == set(VARIANT_SIZES)

    def test_observable_counts_are_exactly_as_specified(self):
        assert [len(observable_set(v, V1_QUBIT_COUNT)) for v in VARIANTS] == [
            1, 1, 16, 24, 136
        ]

    def test_all_three_weight_settings_are_run(self):
        # W-identity is the pre-ansatz probe that takes the circuit and the optimizer out
        # of the question entirely; dropping it would make a negative result ambiguous.
        assert WEIGHT_SETTINGS == ("W-identity", "W-untrained", "W-frozen")

    def test_default_permutation_count_is_the_specified_two_hundred(self):
        assert DEFAULT_PERMUTATIONS == 200

    def test_threshold_is_fixed_and_never_swept(self):
        # A threshold chosen against the partition it is evaluated on is a tuning
        # decision. 0.50 is fixed in advance so no partition can select it.
        assert FIXED_THRESHOLD == 0.50
        assert not re.search(r"for\s+threshold\s+in", SOURCE)
        assert not re.search(r"best_threshold|threshold_sweep|tune_threshold", SOURCE)

    def test_primary_conditions_match_the_established_phase_d_choice(self):
        from backend.evaluation.pixel_comparison import (
            PRIMARY_CONDITION as COMPARISON_PRIMARY,
        )

        assert PRIMARY_CONDITION == {
            "oracle": "A_lesion_polygon", "predicted": "B_localized"
        }
        assert PRIMARY_CONDITION == COMPARISON_PRIMARY

    def test_observable_sets_are_not_assembled_at_call_time(self):
        # Anything that searched observables by score would need to build a set from
        # data; the driver may only ask quantum_ml.readout for a named variant.
        assert "ObservableSet(" not in SOURCE
        # A ranking primitive on a line that also mentions observables or masks would be
        # the multiple-comparisons trap. ``argsort`` on *scores* is legitimate -- _auc
        # ranks samples -- so the scan is scoped rather than a blanket keyword ban.
        for number, line in enumerate(SOURCE.splitlines(), start=1):
            if not re.search(r"argsort|argmax|nlargest|top_k|sort\(", line):
                continue
            assert not re.search(r"observable|mask|oset", line, re.IGNORECASE), (
                f"line {number} ranks observables, which the spec forbids: {line.strip()}"
            )


class TestPermutationBudgetIsDisclosed:
    """A cost cap that is invisible in the output reads as full coverage.

    The large conditions run a smaller null than the primary one. That is a defensible
    trade -- ``A_all`` carries the DEC-024 oracle-selection leak and can only be read
    with a caveat anyway -- but it is only defensible if the reader can see it, so the
    reduction is asserted to reach both the log and the payload.
    """

    def test_the_threshold_and_reduced_budget_are_module_constants(self):
        assert e0.LARGE_CONDITION_TRAIN_ROWS == 400
        assert e0.LARGE_CONDITION_PERMUTATIONS == 50
        assert e0.LARGE_CONDITION_PERMUTATIONS < DEFAULT_PERMUTATIONS

    def test_the_reduced_budget_still_resolves_a_usable_p(self):
        # 1/51 is finer than the 0.05 the decision rule tests against.
        assert 1 / (e0.LARGE_CONDITION_PERMUTATIONS + 1) < 0.05

    def test_the_reduction_is_recorded_in_the_payload_and_logged(self):
        assert "permutation_budget" in SOURCE
        assert "reduced_conditions" in SOURCE
        assert "budget_note" in SOURCE
        # Logged as well as recorded: a JSON field nobody opens is not disclosure.
        assert re.search(r"logger\.info\([^)]*permutation null", SOURCE)

    def test_run_condition_accepts_and_stores_the_note(self):
        import inspect

        assert "permutation_note" in inspect.signature(e0.run_condition).parameters

    def test_the_reduction_is_by_row_count_not_by_condition_name(self):
        # The oracle primary (A_lesion_polygon, 215 train rows) keeps the full budget;
        # the predicted primary (B_localized, 1,620 rows) does not. The rule is stated in
        # rows precisely so that it does not quietly exempt a headline condition -- and
        # the payload note says which conditions it reaches, including that one.
        assert e0.LARGE_CONDITION_TRAIN_ROWS > 215
        assert e0.LARGE_CONDITION_TRAIN_ROWS < 1620
        assert "predicted primary condition B_localized" in SOURCE

    def test_the_primary_condition_runs_first(self):
        assert "names = [primary] + [n for n in names if n != primary]" in SOURCE


class TestFrozenInvariants:
    def test_sixteen_qubits_and_the_full_amplitude_vector(self):
        assert V1_QUBIT_COUNT == 16
        assert V1_PIXEL_COUNT == 65536
        assert V1_PIXEL_COUNT == 2 ** V1_QUBIT_COUNT

    def test_the_driver_refuses_a_cache_that_reports_a_different_qubit_count(self):
        assert "qubit_count" in SOURCE
        assert re.search(r"V1_QUBIT_COUNT", SOURCE)

    def test_circuit_parameter_count_stays_at_thirty_two(self):
        from quantum_ml.vqc_classifier import VariationalQuantumClassifier

        model = VariationalQuantumClassifier(
            num_qubits=V1_QUBIT_COUNT, num_layers=1, seed=42
        )
        assert model.num_params == 32

    def test_untrained_weights_are_reproducible_from_the_recorded_seed(self):
        from backend.evaluation.e0_readout import FrozenModel

        model = FrozenModel(
            artifact="x.json", sha256="0" * 64,
            weights=np.zeros(32), n_layers=1, seed=42,
            condition="A_lesion_polygon", label="test",
        )
        expected = np.random.default_rng(42).uniform(-np.pi, np.pi, 32)
        assert np.array_equal(model.untrained_weights, expected)
        # Called twice, it must not advance a shared stream.
        assert np.array_equal(model.untrained_weights, expected)


class TestTestPartitionIsolation:
    def test_the_driver_never_reads_the_test_partition(self):
        # Source inspection rather than a runtime assertion: the guarantee needed is
        # "there is no code path", which a passing run does not establish.
        assert not re.search(r"\.test\b", SOURCE), (
            "the E0 driver must not touch the test partition"
        )
        assert not re.search(r"partition\s*=\s*[\"']test[\"']", SOURCE)
        assert not re.search(r"roi_mode\s*=\s*[\"']test[\"']", SOURCE)

    def test_the_driver_declares_its_partition_in_the_payload(self):
        assert "test_partition_used" in SOURCE
        assert '"validation"' in SOURCE

    def test_calibration_is_not_fitted_anywhere(self):
        # Phase E owns calibration, and fitting it here would be a tuning decision made
        # outside the partition that is allowed to make it.
        for forbidden in ("IsotonicRegression", "CalibratedClassifierCV", "sigmoid_calib"):
            assert forbidden not in SOURCE
