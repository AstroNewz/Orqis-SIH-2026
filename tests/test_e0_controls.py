"""Tests for the E0 §9 classical controls and the complementarity test.

The controls exist to set the bar the quantum readout has to clear, so the thing worth
defending here is not that they run -- it is that they are **not allowed to cheat, and not
crippled either**. Specifically:

* no control's ``C``, ``gamma`` or calibration map may be chosen with a validation row;
* the RBF width comes from train pairwise distances only, and from the *distinct* pairs;
* the stack's features are out of fold, because in-sample forest scores would guarantee
  the quantum feature never earns a coefficient;
* the stack's null is on the **increment**, which :class:`TestIncrementNullDiscriminates`
  checks actually discriminates -- a null that fires for a noise feature would certify
  complementarity for nothing at all, and that is the specific failure mode the earlier
  absolute-PR-AUC null had.

Everything here runs on constructed data with known structure. The real numbers come from
the driver; these tests are what make those numbers worth reading.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

import backend.evaluation.e0_controls as controls
from backend.evaluation.e0_controls import (
    C_GRID,
    CONTROLS_VERSION,
    MAX_GRAM_ROWS,
    N_FOLDS,
    ControlsError,
    _fit_precomputed_svm,
    amplitude_controls,
    best_e0_setting,
    complementarity,
    grouped_folds,
    median_heuristic_gamma,
    verdict,
)

SOURCE = Path(controls.__file__).read_text(encoding="utf-8")

#: ``SOURCE`` with adjacent implicitly-concatenated string literals joined, so a
#: source-inspection assertion tests the *message* rather than where the author happened
#: to wrap it at 100 columns.
FLAT_SOURCE = re.sub(r'"\s*\n\s*"', "", SOURCE)


# ----------------------------------------------------------------------------- helpers
def _patient_dataset(
    n_patients: int, per_patient: int, *, prevalence: float, seed: int
) -> Tuple[np.ndarray, List[str]]:
    """Patient-blocked labels: the diagnosis is constant within a patient, as in reality."""
    rng = np.random.default_rng(seed)
    patient_labels = (rng.random(n_patients) < prevalence).astype(int)
    # Guarantee both classes so the tests are about the statistics, not about a
    # degenerate draw.
    if patient_labels.sum() == 0:
        patient_labels[0] = 1
    if patient_labels.sum() == n_patients:
        patient_labels[0] = 0
    labels: List[int] = []
    patients: List[str] = []
    for index, label in enumerate(patient_labels):
        labels.extend([int(label)] * per_patient)
        patients.extend([f"P{index:03d}"] * per_patient)
    return np.asarray(labels, dtype=int), patients


def _unit_rows(rng, n: int, d: int) -> np.ndarray:
    x = np.abs(rng.normal(size=(n, d))) + 0.1
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _complementary_cohort(
    n_patients: int,
    per_patient: int,
    *,
    seed: int,
    threshold: float = 0.25,
    noise: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """A cohort whose label needs *both* features, with neither able to substitute.

    Two independent latent factors ``a`` and ``b`` are drawn per patient and the
    diagnosis is ``a + b > threshold``. The RF column sees only ``a`` and the quantum
    column only ``b``, each through image-level noise. So RF alone is genuinely
    informative but capped, and a linear stack that adds the second column recovers the
    missing half of the decision boundary.

    This is the right shape to test power with. A construction where each feature is
    informative on a *disjoint subset of patients* looks complementary but is not
    learnable by a linear stack -- the model cannot condition on which subset a row is
    in, so it carries both signals and both noises everywhere and can come out behind.
    Failing to detect that would be correct behaviour, not a missed detection.

    Returns ``(rf, quantum, y, patients)``.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n_patients)
    b = rng.normal(size=n_patients)
    patient_label = ((a + b) > threshold).astype(int)
    if patient_label.sum() == 0:
        patient_label[0] = 1
    if patient_label.sum() == n_patients:
        patient_label[0] = 0
    index = np.repeat(np.arange(n_patients), per_patient)
    y = patient_label[index]
    patients = [f"P{i:03d}" for i in index]
    rf = a[index] + noise * rng.normal(size=y.size)
    quantum = b[index] + noise * rng.normal(size=y.size)
    return rf, quantum, y, patients


class TestGroupedFolds:
    def test_no_patient_appears_on_both_sides_of_a_split(self):
        y, patients = _patient_dataset(30, 4, prevalence=0.4, seed=1)
        folds, note = grouped_folds(y, patients, seed=42)
        assert "StratifiedGroupKFold" in note and "grouped by patient" in note
        array = np.asarray(patients)
        for train_idx, test_idx in folds:
            assert not (set(array[train_idx]) & set(array[test_idx])), (
                "a patient on both sides of a fold would leak the label into the "
                "hyperparameter that fold selects"
            )

    def test_every_row_is_tested_exactly_once(self):
        y, patients = _patient_dataset(25, 3, prevalence=0.4, seed=2)
        folds, _ = grouped_folds(y, patients, seed=42)
        tested = np.concatenate([test for _, test in folds])
        assert np.array_equal(np.sort(tested), np.arange(y.size))

    def test_names_its_fallback_instead_of_substituting_silently(self):
        # One patient contributing every row: there is no second group to hold out, so
        # grouped stratification cannot be built. The fallback must then be used *and
        # said out loud* in the note the payload carries, because a reader who sees
        # "grouped by patient" in the report has to be able to trust it.
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        patients = ["P000"] * 8
        _, note = grouped_folds(y, patients, seed=42)
        assert "StratifiedKFold" in note
        assert "StratifiedGroupKFold" not in note
        assert "patient grouping unavailable" in note

    def test_the_fold_count_degrades_to_the_minority_class_size(self):
        """Fewer folds is a stated reduction, not a fallback to ungrouped splitting."""
        y = np.array([0] * 12 + [1, 1, 1])
        patients = [f"P{i:02d}" for i in range(15)]
        folds, note = grouped_folds(y, patients, seed=42)
        assert len(folds) == 3
        assert note.startswith("3-fold StratifiedGroupKFold")

    def test_is_deterministic_at_a_seed(self):
        y, patients = _patient_dataset(20, 3, prevalence=0.4, seed=3)
        first, _ = grouped_folds(y, patients, seed=7)
        second, _ = grouped_folds(y, patients, seed=7)
        for (a_tr, a_te), (b_tr, b_te) in zip(first, second):
            assert np.array_equal(a_tr, b_tr) and np.array_equal(a_te, b_te)

    def test_default_fold_count_is_the_declared_constant(self):
        y, patients = _patient_dataset(40, 4, prevalence=0.4, seed=4)
        folds, _ = grouped_folds(y, patients, seed=42)
        assert len(folds) == N_FOLDS


class TestMedianHeuristic:
    def test_matches_an_explicit_distance_median(self):
        rng = np.random.default_rng(10)
        x = _unit_rows(rng, 40, 64)
        gram = x @ x.T
        gamma, median = median_heuristic_gamma(gram)
        explicit = []
        for i in range(40):
            for j in range(i + 1, 40):
                explicit.append(float(np.sum((x[i] - x[j]) ** 2)))
        assert median == pytest.approx(float(np.median(explicit)), rel=1e-9)
        assert gamma == pytest.approx(1.0 / median, rel=1e-12)

    def test_excludes_the_zero_diagonal(self):
        """Including ``d(x, x) = 0`` would pull the median towards a scale no pair has.

        With 4 rows there are 6 distinct pairs and 4 diagonal zeros, so a version that
        included the diagonal would return a visibly smaller median. The check is that
        the returned value is the distinct-pair median, not merely positive.
        """
        rng = np.random.default_rng(11)
        x = _unit_rows(rng, 4, 32)
        gram = x @ x.T
        _, median = median_heuristic_gamma(gram)
        distinct = [
            float(np.sum((x[i] - x[j]) ** 2)) for i in range(4) for j in range(i + 1, 4)
        ]
        with_diagonal = distinct + [0.0] * 4
        assert median == pytest.approx(float(np.median(distinct)), rel=1e-9)
        assert median != pytest.approx(float(np.median(with_diagonal)), rel=1e-6)

    def test_identical_rows_are_refused_not_given_an_arbitrary_width(self):
        gram = np.ones((5, 5))
        with pytest.raises(ControlsError, match="degenerate"):
            median_heuristic_gamma(gram)

    def test_single_row_is_refused(self):
        with pytest.raises(ControlsError, match="at least two"):
            median_heuristic_gamma(np.ones((1, 1)))

    def test_uses_only_the_matrix_it_is_given(self):
        """The train Gram is the only argument, so a validation row cannot reach gamma.

        This is a structural property rather than a claim about call order: there is no
        parameter through which validation data could be passed.
        """
        import inspect

        signature = inspect.signature(median_heuristic_gamma)
        assert list(signature.parameters) == ["k_tt"]


class TestPrecomputedSVM:
    def _problem(self, seed: int = 20, n: int = 60, informative: bool = True):
        rng = np.random.default_rng(seed)
        y, patients = _patient_dataset(n // 3, 3, prevalence=0.4, seed=seed)
        d = 32
        base = _unit_rows(rng, y.size, d)
        if informative:
            base[y == 1, 0] += 0.8
            base /= np.linalg.norm(base, axis=1, keepdims=True)
        split = y.size // 2
        x_train, x_validation = base[:split], base[split:]
        y_train, y_validation = y[:split], y[split:]
        folds, _ = grouped_folds(y_train, patients[:split], seed=seed)
        return (
            x_train @ x_train.T, y_train, x_validation @ x_train.T, y_validation, folds
        )

    def test_selected_c_comes_from_the_declared_grid(self):
        k_tt, y_train, k_vt, _, folds = self._problem()
        result = _fit_precomputed_svm(k_tt, y_train, k_vt, folds, seed=42)
        assert result["selected_C"] in C_GRID
        assert set(result["cv_average_precision_by_C"]) == {str(c) for c in C_GRID}

    def test_decision_functions_have_the_right_shapes(self):
        k_tt, y_train, k_vt, y_validation, folds = self._problem()
        result = _fit_precomputed_svm(k_tt, y_train, k_vt, folds, seed=42)
        assert result["train_decision"].shape == (y_train.size,)
        assert result["validation_decision"].shape == (y_validation.size,)
        assert result["train_decision_out_of_fold"].shape == (y_train.size,)

    def test_out_of_fold_covers_every_train_row(self):
        k_tt, y_train, k_vt, _, folds = self._problem()
        result = _fit_precomputed_svm(k_tt, y_train, k_vt, folds, seed=42)
        assert np.isfinite(result["train_decision_out_of_fold"]).all()

    def test_platt_map_is_monotone_so_rank_metrics_are_unchanged(self):
        """The probability map may not reorder anything.

        Brier and ECE need a probability; PR-AUC and ROC-AUC are rank statistics. If the
        map were not monotone the two metric blocks would disagree about ranking and the
        report would contain two different answers to the same question.
        """
        k_tt, y_train, k_vt, _, folds = self._problem()
        result = _fit_precomputed_svm(k_tt, y_train, k_vt, folds, seed=42)
        probability = result["validation_probability"]
        assert probability is not None
        order_decision = np.argsort(np.argsort(result["validation_decision"]))
        order_probability = np.argsort(np.argsort(probability))
        assert np.array_equal(order_decision, order_probability)
        assert probability.min() >= 0.0 and probability.max() <= 1.0

    def test_platt_map_is_fitted_on_out_of_fold_values_only(self):
        k_tt, y_train, k_vt, _, folds = self._problem()
        result = _fit_precomputed_svm(k_tt, y_train, k_vt, folds, seed=42)
        assert "out-of-fold" in result["platt_note"]
        assert "no validation row" in result["platt_note"]

    def test_separable_problem_is_solved(self):
        # A control that could not solve an easy problem would set an artificially low
        # bar, which flatters the quantum model exactly as much as a leaky control
        # penalises it.
        rng = np.random.default_rng(21)
        y = np.array([0] * 20 + [1] * 20)
        patients = [f"P{i:02d}" for i in range(40)]
        x = _unit_rows(rng, 40, 16) * 0.05
        x[y == 1, 0] += 1.0
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        folds, _ = grouped_folds(y, patients, seed=21)
        result = _fit_precomputed_svm(x @ x.T, y, x @ x.T, folds, seed=42)
        from sklearn.metrics import average_precision_score

        assert average_precision_score(y, result["validation_decision"]) > 0.95

    def test_ties_break_towards_the_stronger_penalty(self):
        """A tie means the data did not distinguish the settings; take the simpler model.

        Constructed by making every ``C`` score identically, which happens when the CV
        folds cannot separate at all.
        """
        n = 24
        y, patients = _patient_dataset(8, 3, prevalence=0.5, seed=22)
        gram = np.eye(n)
        folds, _ = grouped_folds(y, patients, seed=22)
        result = _fit_precomputed_svm(gram, y, gram, folds, seed=42)
        scores = [v for v in result["cv_average_precision_by_C"].values() if v is not None]
        if len(set(scores)) == 1:
            assert result["selected_C"] == min(C_GRID)

    def test_degenerate_folds_are_refused_with_the_reason(self):
        y = np.array([0, 0, 1, 1])
        gram = np.eye(4)
        # A fold whose training side is single-class is skipped, so no C scores at all.
        folds = [(np.array([0, 1]), np.array([2, 3]))]
        with pytest.raises(ControlsError, match="degenerate"):
            _fit_precomputed_svm(gram, y, gram, folds, seed=42)


class TestAmplitudeControls:
    def _data(self, seed: int = 30, d: int = 64, informative: bool = True):
        rng = np.random.default_rng(seed)
        y, patients = _patient_dataset(30, 3, prevalence=0.4, seed=seed)
        x = _unit_rows(rng, y.size, d)
        if informative:
            x[y == 1, 0] += 0.6
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        split = y.size // 2
        return (
            x[:split], y[:split], x[split:], y[split:], patients[:split]
        )

    def test_reports_both_kernels_with_their_selection_notes(self):
        x_t, y_t, x_v, y_v, patients = self._data()
        out = amplitude_controls(x_t, y_t, x_v, y_v, patients, seed=42)
        for name in ("linear_svm_on_amplitudes", "rbf_svm_on_amplitudes"):
            assert out[name]["measured"] is True
            assert "TRAIN" in out[name]["selection"]
            assert out[name]["input_dimension"] == x_t.shape[1]

    def test_the_linear_control_records_that_it_was_redone(self):
        x_t, y_t, x_v, y_v, patients = self._data()
        out = amplitude_controls(x_t, y_t, x_v, y_v, patients, seed=42)
        note = out["linear_svm_on_amplitudes"]["redone_note"]
        assert "TRAIN only" in note
        assert "optimistic for the control" in note

    def test_rbf_gamma_note_names_the_median_heuristic_and_the_partition(self):
        x_t, y_t, x_v, y_v, patients = self._data()
        out = amplitude_controls(x_t, y_t, x_v, y_v, patients, seed=42)
        block = out["rbf_svm_on_amplitudes"]
        assert "median heuristic" in block["gamma_selection"]
        assert "No validation row" in block["gamma_selection"]
        assert block["gamma"] > 0.0

    def test_the_distance_identity_the_shared_gram_relies_on(self):
        """``||x - y||^2 = 2 - 2<x, y>`` holds only for unit-norm rows.

        The module derives every RBF distance from the linear Gram on that basis. If the
        amplitude vectors were not unit-norm the RBF control would be silently measuring
        a different kernel, so the identity is checked against explicit distances.
        """
        rng = np.random.default_rng(31)
        x = _unit_rows(rng, 12, 32)
        gram = x @ x.T
        for i in range(12):
            for j in range(12):
                explicit = float(np.sum((x[i] - x[j]) ** 2))
                assert 2.0 - 2.0 * gram[i, j] == pytest.approx(explicit, abs=1e-12)

    def test_an_oversized_condition_is_reported_not_substituted(self, monkeypatch):
        monkeypatch.setattr(controls, "MAX_GRAM_ROWS", 4)
        x_t, y_t, x_v, y_v, patients = self._data()
        out = controls.amplitude_controls(x_t, y_t, x_v, y_v, patients, seed=42)
        assert out["linear_svm_on_amplitudes"]["measured"] is False
        assert "not measured rather than substituted" in (
            out["linear_svm_on_amplitudes"]["reason"]
        )
        assert MAX_GRAM_ROWS == 6000  # the real threshold is untouched by the patch

    def test_is_deterministic_at_a_seed(self):
        x_t, y_t, x_v, y_v, patients = self._data()
        first = amplitude_controls(x_t, y_t, x_v, y_v, patients, seed=42)
        second = amplitude_controls(x_t, y_t, x_v, y_v, patients, seed=42)
        a = first.pop("_arrays")
        b = second.pop("_arrays")
        assert first == second
        for key in ("linear", "rbf"):
            assert np.allclose(
                a[key]["validation_decision"], b[key]["validation_decision"]
            )


class TestComplementarityPower:
    """The test must fire when the second feature genuinely adds ranking information."""

    def _setup(self, seed: int = 40):
        rf_t, quantum_t, y_train, patients_train = _complementary_cohort(
            80, 4, seed=seed
        )
        rf_v, quantum_v, y_validation, _ = _complementary_cohort(40, 4, seed=seed + 1)
        return (
            rf_t, quantum_t, y_train, rf_v, quantum_v, y_validation, patients_train
        )

    def test_a_genuinely_complementary_feature_is_detected(self):
        args = self._setup()
        result = complementarity(*args, seed=42, n_permutations=60)
        assert result["measured"] is True
        assert result["stack_beats_rf_alone"] is True
        assert result["increment"]["stack_minus_rf_only"] > 0.0
        assert result["gain_survives_its_own_null"] is True
        assert result["complementarity_claimed"] is True

    def test_the_label_permutation_null_is_the_one_that_lacks_power(self):
        """Documents the reason the gate does not use it, on a case known to be real.

        The label null returns "does not survive" on a construction the column null
        detects. That is the measured basis for the docstring's claim, not an assertion
        about it -- and it is why ``gates_the_claim`` is False on that block.
        """
        args = self._setup()
        result = complementarity(*args, seed=42, n_permutations=60)
        assert result["gain_survives_label_permutation_null"] is False
        assert result["permutation_null"]["gates_the_claim"] is False
        assert result["column_permutation_null"]["gates_the_claim"] is True
        label_p = result["permutation_null"]["observed_increment_vs_null"]["empirical_p"]
        column_p = (
            result["column_permutation_null"]["observed_increment_vs_null"]["empirical_p"]
        )
        assert column_p < label_p

    def test_the_added_feature_earns_a_coefficient(self):
        args = self._setup()
        result = complementarity(*args, seed=42, n_permutations=0)
        assert abs(result["coefficients"]["quantum_readout_score"]) > 0.1

    def test_the_rf_only_logistic_reproduces_the_raw_score_ranking(self):
        """A one-feature logistic is monotone, so its PR-AUC must equal the raw score's.

        This is what makes ``stack - rf_only`` the same quantity as "the stack beats RF
        alone" rather than an artefact of wrapping RF in a model.
        """
        args = self._setup()
        result = complementarity(*args, seed=42, n_permutations=0)
        assert result["increment"]["rf_only_logistic_pr_auc"] == pytest.approx(
            result["raw_rf_score_pr_auc"], abs=1e-9
        )


class TestIncrementNullDiscriminates:
    """The null must *not* fire for a feature that adds nothing.

    This is the property the earlier absolute-PR-AUC null lacked. With a real RF feature
    and a pure-noise second column, a null over the stack's own PR-AUC is exceeded easily
    -- the forest carries the signal -- so complementarity would be certified for noise.
    The increment null has to come back negative.
    """

    def _setup(self, seed: int = 50):
        rng = np.random.default_rng(seed)
        y_train, patients_train = _patient_dataset(60, 4, prevalence=0.4, seed=seed)
        y_validation, _ = _patient_dataset(30, 4, prevalence=0.4, seed=seed + 1)
        rf_t = 1.5 * y_train + rng.normal(size=y_train.size)
        rf_v = 1.5 * y_validation + rng.normal(size=y_validation.size)
        noise_t = rng.normal(size=y_train.size)
        noise_v = rng.normal(size=y_validation.size)
        return rf_t, noise_t, y_train, rf_v, noise_v, y_validation, patients_train

    def test_a_noise_feature_is_not_certified_as_complementary(self):
        args = self._setup()
        result = complementarity(*args, seed=42, n_permutations=80)
        assert result["measured"] is True
        assert result["complementarity_claimed"] is not True
        assert result["permutation_null"]["n_usable"] > 40

    def test_the_null_quantity_is_the_increment_not_the_stack(self):
        args = self._setup()
        result = complementarity(*args, seed=42, n_permutations=20)
        assert result["permutation_null"]["quantity"] == (
            "stack PR-AUC minus RF-only PR-AUC, both refitted per draw"
        )
        assert "permuted quantum column" in (
            result["column_permutation_null"]["quantity"]
        )
        # An increment null straddles zero. A null over the stack's absolute PR-AUC
        # could not, because PR-AUC is non-negative -- so this is a direct check that
        # the reported null is the one the spec's wording asks for.
        for block in ("permutation_null", "column_permutation_null"):
            summary = result[block]["increment"]
            assert summary is not None
            assert summary["p50"] < 0.5

    def test_the_column_null_holds_the_rf_feature_fixed(self):
        """Its spread must come from the shuffled column alone.

        Since the RF-only score is constant across draws, every draw differs only in the
        quantum column -- which is what makes the null a statement about that column
        rather than about the stack as a whole.
        """
        args = self._setup()
        result = complementarity(*args, seed=42, n_permutations=40)
        assert result["column_permutation_null"]["n_usable"] == 40
        assert "labels and the RF column" in result["column_permutation_null"]["blocking"]
        assert "untouched" in result["column_permutation_null"]["blocking"]

    def test_the_null_is_patient_blocked(self):
        args = self._setup()
        result = complementarity(*args, seed=42, n_permutations=10)
        assert result["permutation_null"]["blocking"] == (
            "patient-level label permutation within TRAIN"
        )

    def test_empirical_p_can_never_be_zero(self):
        args = self._setup()
        result = complementarity(*args, seed=42, n_permutations=30)
        rank = result["permutation_null"]["observed_increment_vs_null"]
        assert rank is not None
        assert rank["empirical_p"] > 0.0


class TestPatientBlockedColumnShuffle:
    def test_it_is_a_permutation_of_the_input(self):
        rng = np.random.default_rng(70)
        values = np.arange(12, dtype=np.float64)
        patients = [f"P{i // 3}" for i in range(12)]
        out = controls._patient_blocked_column_shuffle(values, patients, rng)
        assert np.array_equal(np.sort(out), np.sort(values))

    def test_it_moves_whole_patient_blocks_not_individual_rows(self):
        """Each patient's set of values must survive intact somewhere in the output.

        With equal-sized blocks the shuffle is exactly a block permutation, so every
        original block still appears as a contiguous group of the sorted output. A row
        shuffle would break the values apart and make the shuffled column less
        autocorrelated than any real feature -- an easier null than reality.
        """
        rng = np.random.default_rng(71)
        values = np.array([0.0, 0.1, 0.2, 10.0, 10.1, 10.2, 20.0, 20.1, 20.2])
        patients = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
        out = controls._patient_blocked_column_shuffle(values, patients, rng)
        recovered = {tuple(np.sort(out[i:i + 3])) for i in (0, 3, 6)}
        original = {
            tuple(np.sort(values[i:i + 3])) for i in (0, 3, 6)
        }
        assert recovered == original

    def test_it_changes_the_order_for_at_least_some_seeds(self):
        values = np.arange(24, dtype=np.float64)
        patients = [f"P{i // 4}" for i in range(24)]
        moved = 0
        for seed in range(8):
            rng = np.random.default_rng(seed)
            out = controls._patient_blocked_column_shuffle(values, patients, rng)
            moved += int(not np.array_equal(out, values))
        assert moved >= 6

    def test_a_single_patient_cannot_be_shuffled_and_is_returned_unchanged(self):
        rng = np.random.default_rng(72)
        values = np.array([1.0, 2.0, 3.0])
        out = controls._patient_blocked_column_shuffle(values, ["P0"] * 3, rng)
        assert np.array_equal(out, values)

    def test_it_handles_unequal_block_sizes(self):
        rng = np.random.default_rng(73)
        values = np.arange(7, dtype=np.float64)
        patients = ["A", "A", "A", "A", "B", "C", "C"]
        out = controls._patient_blocked_column_shuffle(values, patients, rng)
        assert out.shape == values.shape
        assert np.array_equal(np.sort(out), np.sort(values))


class TestValidationPatientsAreUsedWhenGiven:
    def test_supplying_them_removes_the_iid_caveat_from_the_note(self):
        rf_t, q_t, y_t, p_t = _complementary_cohort(40, 4, seed=80)
        rf_v, q_v, y_v, p_v = _complementary_cohort(20, 4, seed=81)
        with_patients = complementarity(
            rf_t, q_t, y_t, rf_v, q_v, y_v, p_t, seed=42, n_permutations=5,
            patients_validation=p_v,
        )
        without = complementarity(
            rf_t, q_t, y_t, rf_v, q_v, y_v, p_t, seed=42, n_permutations=5,
        )
        assert "i.i.d." not in with_patients["column_permutation_null"]["blocking"]
        assert "i.i.d." in without["column_permutation_null"]["blocking"]

    def test_a_mismatched_length_falls_back_and_says_so(self):
        rf_t, q_t, y_t, p_t = _complementary_cohort(40, 4, seed=82)
        rf_v, q_v, y_v, _ = _complementary_cohort(20, 4, seed=83)
        result = complementarity(
            rf_t, q_t, y_t, rf_v, q_v, y_v, p_t, seed=42, n_permutations=5,
            patients_validation=["P000"],  # wrong length: cannot be trusted
        )
        assert "i.i.d." in result["column_permutation_null"]["blocking"]

    def test_the_driver_passes_validation_patients(self):
        assert "patients_validation=patients_validation" in SOURCE
        assert "validation.patient_ids[int(r)] for r in validation_rows" in SOURCE


class TestComplementarityDiscipline:
    def test_non_finite_out_of_fold_rows_are_dropped_not_imputed(self):
        y_train, patients_train = _patient_dataset(20, 3, prevalence=0.4, seed=60)
        rng = np.random.default_rng(60)
        rf_t = rng.normal(size=y_train.size)
        quantum_t = rng.normal(size=y_train.size)
        quantum_t[:10] = np.nan  # a fold that could not be fitted
        y_validation, _ = _patient_dataset(10, 3, prevalence=0.4, seed=61)
        result = complementarity(
            rf_t, quantum_t, y_train,
            rng.normal(size=y_validation.size), rng.normal(size=y_validation.size),
            y_validation, patients_train, seed=42, n_permutations=0,
        )
        assert result["n_train_rows_used"] == y_train.size - 10

    def test_too_few_usable_rows_refuses_with_the_count(self):
        y_train = np.array([0, 1, 0, 1, 0, 1])
        patients = [f"P{i}" for i in range(6)]
        rf = np.array([np.nan, np.nan, np.nan, np.nan, 0.1, 0.2])
        quantum = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result = complementarity(
            rf, quantum, y_train, np.array([0.1, 0.9]), np.array([0.2, 0.8]),
            np.array([0, 1]), patients, seed=42, n_permutations=0,
        )
        assert result["measured"] is False
        assert "only 2 train rows" in result["reason"]

    def test_it_says_the_features_were_fitted_on_train_only(self):
        y_train, patients_train = _patient_dataset(30, 3, prevalence=0.4, seed=62)
        y_validation, _ = _patient_dataset(15, 3, prevalence=0.4, seed=63)
        rng = np.random.default_rng(62)
        result = complementarity(
            rng.normal(size=y_train.size), rng.normal(size=y_train.size), y_train,
            rng.normal(size=y_validation.size), rng.normal(size=y_validation.size),
            y_validation, patients_train, seed=42, n_permutations=0,
        )
        assert "TRAIN only" in result["fitted_on"]
        assert "out-of-fold" in result["fitted_on"]

    def test_score_correlation_is_reported_both_ways(self):
        y_train, patients_train = _patient_dataset(30, 3, prevalence=0.4, seed=64)
        y_validation, _ = _patient_dataset(15, 3, prevalence=0.4, seed=65)
        rng = np.random.default_rng(64)
        shared = rng.normal(size=y_validation.size)
        result = complementarity(
            rng.normal(size=y_train.size), rng.normal(size=y_train.size), y_train,
            shared, shared * 3.0 + 1.0,  # perfectly rank-identical
            y_validation, patients_train, seed=42, n_permutations=0,
        )
        assert result["score_correlation"]["spearman"] == pytest.approx(1.0, abs=1e-9)
        assert result["score_correlation"]["pearson"] == pytest.approx(1.0, abs=1e-9)
        assert "re-deriving the forest's ranking" in result["score_correlation"]["note"]

    def test_a_degenerate_feature_column_is_named(self):
        y_train, patients_train = _patient_dataset(30, 3, prevalence=0.4, seed=66)
        y_validation, _ = _patient_dataset(15, 3, prevalence=0.4, seed=67)
        rng = np.random.default_rng(66)
        result = complementarity(
            rng.normal(size=y_train.size), np.full(y_train.size, 0.25), y_train,
            rng.normal(size=y_validation.size), np.full(y_validation.size, 0.25),
            y_validation, patients_train, seed=42, n_permutations=0,
        )
        assert result["degenerate_feature_columns"] == ["quantum_readout_score"]

    def test_is_deterministic_at_a_seed(self):
        y_train, patients_train = _patient_dataset(40, 3, prevalence=0.4, seed=68)
        y_validation, _ = _patient_dataset(20, 3, prevalence=0.4, seed=69)
        rng = np.random.default_rng(68)
        args = (
            rng.normal(size=y_train.size), rng.normal(size=y_train.size), y_train,
            rng.normal(size=y_validation.size), rng.normal(size=y_validation.size),
            y_validation, patients_train,
        )
        first = complementarity(*args, seed=42, n_permutations=15)
        second = complementarity(*args, seed=42, n_permutations=15)
        assert first == second


class TestBestE0Setting:
    PAYLOAD = {
        "conditions": [
            {
                "condition": "A_lesion_polygon",
                "results": [
                    {"variant": "A0", "weight_setting": "W-frozen",
                     "validation_metrics": {"pr_auc": 0.52}},
                    {"variant": "C", "weight_setting": "W-identity",
                     "validation_metrics": {"pr_auc": 0.61}},
                    {"variant": "D", "weight_setting": "W-frozen",
                     "validation_metrics": {"pr_auc": None}},
                    {"variant": "B", "weight_setting": "W-untrained",
                     "validation_metrics": None},
                ],
            }
        ]
    }

    def test_picks_the_highest_validation_pr_auc(self):
        best = best_e0_setting(self.PAYLOAD, "A_lesion_polygon")
        assert best == {"variant": "C", "weight_setting": "W-identity", "pr_auc": 0.61}

    def test_no_payload_means_no_selection_rather_than_a_default(self):
        assert best_e0_setting(None, "A_lesion_polygon") is None
        assert best_e0_setting({}, "A_lesion_polygon") is None

    def test_a_condition_not_in_the_payload_returns_none(self):
        assert best_e0_setting(self.PAYLOAD, "B_localized") is None

    def test_unmeasurable_variants_are_skipped_not_ranked_as_zero(self):
        payload = {
            "conditions": [{
                "condition": "X",
                "results": [
                    {"variant": "A0", "weight_setting": "W-frozen",
                     "validation_metrics": {"pr_auc": None}},
                ],
            }]
        }
        assert best_e0_setting(payload, "X") is None


class TestVerdict:
    def _payload(self, **overrides):
        entry = {
            "condition": "A_lesion_polygon",
            "validation": {"n_samples": 52, "prevalence": 0.403846},
            "controls": {
                "random_forest": {"metrics": {"pr_auc": 0.79}},
                "linear_svm_on_amplitudes": {"metrics_rank": {"pr_auc": 0.46}},
                "rbf_svm_on_amplitudes": {"metrics_rank": {"pr_auc": 0.55}},
                "phase_d_vqc": {"metrics": {"pr_auc": 0.53}},
            },
            "selected_quantum_readout": {
                "variant": "C", "weight_setting": "W-frozen",
                "reproduced_validation_pr_auc": 0.49,
            },
            "complementarity": {
                "complementarity_claimed": False,
                "increment": {"stack_minus_rf_only": -0.01},
                "permutation_null": {
                    "observed_increment_vs_null": {"empirical_p": 0.7},
                    "gates_the_claim": False,
                },
                "column_permutation_null": {
                    "observed_increment_vs_null": {"empirical_p": 0.42},
                    "gates_the_claim": True,
                },
                "score_correlation": {"spearman": 0.05},
            },
        }
        entry.update(overrides)
        return {"conditions": [entry]}

    def test_names_the_strongest_control_and_the_margin(self):
        row = verdict(self._payload())["per_condition"][0]
        assert row["best_control"] == "random_forest"
        assert row["best_control_pr_auc"] == 0.79
        assert row["quantum_beats_best_control"] is False
        assert row["margin_vs_best_control"] == pytest.approx(-0.30, abs=1e-9)

    def test_reports_every_control_not_only_the_winner(self):
        row = verdict(self._payload())["per_condition"][0]
        assert set(row["control_pr_auc"]) == {
            "random_forest", "linear_svm_on_amplitudes",
            "rbf_svm_on_amplitudes", "phase_d_vqc",
        }

    def test_surfaces_the_increment_and_both_p_values(self):
        """The gating p is the column null's; the label null's is carried beside it.

        A summary that showed only one of them would ask the reader to accept the gate's
        choice of null without being able to see the other reading.
        """
        row = verdict(self._payload())["per_condition"][0]
        assert row["stack_minus_rf_only_pr_auc"] == -0.01
        assert row["increment_empirical_p"] == 0.42
        assert row["increment_empirical_p_label_null"] == 0.7

    def test_an_unmeasured_quantum_score_is_not_scored_as_a_loss(self):
        payload = self._payload(selected_quantum_readout={})
        row = verdict(payload)["per_condition"][0]
        assert row["quantum_beats_best_control"] is None
        assert row["margin_vs_best_control"] is None

    def test_a_skipped_condition_is_passed_through(self):
        payload = {"conditions": [{"condition": "C_fallback", "skipped": "single-class"}]}
        row = verdict(payload)["per_condition"][0]
        assert row == {"condition": "C_fallback", "skipped": "single-class"}

    def test_the_gate_rule_states_both_halves_of_the_spec_requirement(self):
        rule = verdict(self._payload())["gate_rule"]
        assert "beats the best control" in rule
        assert "increment" in rule
        assert "neither is inferred from the other" in rule

    def test_the_gate_rule_says_which_null_gates_and_why(self):
        rule = verdict(self._payload())["gate_rule"]
        assert "permutes the quantum column" in rule
        assert "does not gate" in rule
        assert "0.25" in rule


class TestSharedHyperparameterFreedom:
    def test_the_c_grid_matches_the_quantum_readouts_grid(self):
        """Otherwise "the control had less freedom" explains any difference found."""
        import backend.evaluation.e0_readout as e0

        readout_source = Path(e0.__file__).read_text(encoding="utf-8")
        match = re.search(r'grid = \{"C": \[([^\]]+)\]\}', readout_source)
        assert match, "the E0.1 readout's C grid moved; this comparison needs updating"
        readout_grid = tuple(float(v) for v in match.group(1).split(","))
        assert readout_grid == C_GRID


class TestTestPartitionIsolation:
    """§10: the test partition is not read. Not once, not for a sanity check."""

    def test_no_test_partition_attribute_access(self):
        assert not re.search(r"\.test\b", SOURCE)

    def test_no_test_partition_keyword(self):
        assert 'partition="test"' not in SOURCE
        assert "partition='test'" not in SOURCE
        assert 'roi_mode="test"' not in SOURCE

    def test_it_declares_which_partitions_it_reads(self):
        assert '"partitions_read": ["train", "validation"]' in SOURCE
        assert '"test_partition_used": False' in SOURCE

    def test_a_payload_that_read_test_is_refused_as_an_e0_input(self):
        assert (
            "refusing to build the decision gate on a payload that read the "
            "held-out set."
        ) in FLAT_SOURCE


class TestNoTuningAgainstTheEvaluationPartition:
    def test_no_threshold_sweep(self):
        for pattern in (r"for threshold in", r"best_threshold", r"threshold_sweep",
                        r"tune_threshold"):
            assert not re.search(pattern, SOURCE), pattern

    def test_no_post_hoc_calibrator_fitted_outside_train(self):
        # A Platt map *is* fitted, on out-of-fold TRAIN decisions. The banned thing is
        # sklearn's wrappers, which would fit on whatever partition they were handed.
        assert "CalibratedClassifierCV" not in SOURCE
        assert "IsotonicRegression" not in SOURCE

    def test_every_selection_is_declared_as_within_train(self):
        assert "within-TRAIN patient-grouped CV" in SOURCE
        assert "no_validation_selection" in SOURCE

    def test_the_only_validation_selection_is_which_readout_to_stack(self):
        assert (
            "no control's C, gamma, threshold or calibration map was chosen using a "
            "validation row."
        ) in FLAT_SOURCE
        assert (
            "The only validation-based selection anywhere in this module is which "
            "E0.1 variant to stack, which the spec permits."
        ) in FLAT_SOURCE


class TestFrozenInvariants:
    def test_the_encoding_width_is_not_changed_by_the_controls(self):
        from backend.ml.pixel_pipeline import V1_PIXEL_COUNT, V1_QUBIT_COUNT

        assert V1_PIXEL_COUNT == 65536
        assert V1_QUBIT_COUNT == 16
        assert "V1_PIXEL_COUNT" in SOURCE and "V1_QUBIT_COUNT" in SOURCE

    def test_the_controls_read_the_same_amplitude_vectors_as_the_quantum_model(self):
        # ``partition.amplitudes(rows)`` is the single derivation path. A control that
        # built its own features would not be a control on the same representation.
        assert "train.amplitudes(train_rows)" in SOURCE
        assert "validation.amplitudes(validation_rows)" in SOURCE

    def test_version_is_declared(self):
        assert CONTROLS_VERSION == "v1-e0-controls-1"
