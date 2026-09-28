"""The quantum rung on PTB-XL: the same-shape contract and the guards that keep it (E4).

:mod:`backend.evaluation.e4_quantum` exists to answer one question -- does the quantum
feature map contribute anything a *classical* map of the same shape does not? -- and the
answer is only worth having if the comparison is fair in both directions. A defect here
would not usually produce a number that is obviously wrong; it would produce one that is
quietly unearned. The tests are organised around the five ways this module could lie:

* it could **score a constant as a model**: accept a map whose every output column is
  identical for every row, so the head learns an intercept and the reported AUROC is
  whatever that degenerate fit happens to give. This is not hypothetical -- the first
  version of the guard read the *clipped* standard deviation, which is floored at 1.0
  precisely so that scaling a constant column stays finite, and so could never fire;
* it could **hobble the control**: give ``poly2`` a different width, a different index
  set, or pair columns in a different order than the quantum observables, so that "the
  same shape" is a claim in the docstring rather than a fact about the arrays;
* it could **leak**: let a frozen-test row into either cache, or fit a map's statistics on
  the rows it later scores instead of on fold-train alone;
* it could **flatter the quantum arm**: give it more grid cells than its controls, or
  select on the partition it reports;
* it could **fabricate the baseline**: accept a recovered fusion member that does not
  reproduce the recorded score, so the quantum fusion is compared against a baseline that
  is not the one on record.

Each section below targets one of those. The inputs are synthetic and small -- eight
components is the circuit's real width, so a fixture can use the true width and still run
in milliseconds -- which lets a test state the answer in advance instead of asserting
whatever the real corpus happens to produce.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pytest

from backend.dataset.ptbxl import (
    SUPERCLASS_ORDER,
    TEST_FOLDS,
    TRAIN_FOLDS,
    VALIDATION_FOLDS,
)
from backend.evaluation import e4_quantum
from backend.evaluation.e3_feature_map import N_OBSERVABLES, REPS_GRID
from backend.evaluation.e4_quantum import (
    ANGLE_MAX,
    DEGENERATE_MAP_FLOOR,
    E4_QUANTUM_VERSION,
    HEAD_C_GRID,
    QUANTUM_DIM,
    QUANTUM_REPS_GRID,
    RFF_FEATURES,
    E4QuantumError,
    RandomFourierMap,
    _assert_reproduces,
    _pair_indices,
    _standardiser,
    blend,
    entangling_witness,
    fit_map,
    poly2_expand,
    score_map_arm,
)
from backend.ml.features_ecg import feature_names
from backend.training.prepare_ecg_features import (
    ECG_CACHE_VERSION,
    FEATURE_SET_VERSION,
    EcgCohort,
)
from quantum_ml.readout import all_pair_masks

FAMILIES = ("zz", "poly2", "rff36")

# Every ``zz`` row costs one Aer statevector evaluation, so the fixtures are sized to the
# smallest count that still exercises the logic under test rather than to a round number.
FIT_ROWS = 32
EVAL_ROWS = 8
ARM_ROWS, ARM_INNER_TRAIN, ARM_EVAL = 60, 40, 12


def _components(n: int = FIT_ROWS, *, seed: int = 5) -> np.ndarray:
    """``(n, 8)`` stand-ins for PCA scores, on a deliberately non-unit scale.

    The scale matters: a map that silently assumed standardised input would pass on
    ``N(0, 1)`` draws and fail on real PCA scores, whose columns have decreasing variance.
    """
    rng = np.random.default_rng(seed)
    scales = np.linspace(4.0, 0.25, QUANTUM_DIM)
    return rng.normal(size=(n, QUANTUM_DIM)) * scales


def _reps_for(family: str) -> int | None:
    return 2 if family == "zz" else None


# --------------------------------------------------------------- it could score a constant
class TestDegeneracyGuard:
    """``reps=1`` is a provable product state, and the guard must say so out loud."""

    def test_reps_one_is_refused_because_its_output_is_constant(self) -> None:
        with pytest.raises(E4QuantumError, match="degenerate"):
            fit_map("zz", _components(), reps=1)

    def test_the_refusal_quotes_a_spread_at_or_below_the_floor(self) -> None:
        """Regression: the guard once read a std that is floored at 1.0 and never fired.

        Asserting on the *number in the message* is what distinguishes a guard that fires
        for the right reason from one that happens to raise.
        """
        with pytest.raises(E4QuantumError) as excinfo:
            fit_map("zz", _components(), reps=1)
        message = str(excinfo.value)
        quoted = float(message.split("standard deviation is ")[1].split(",")[0])
        assert quoted <= DEGENERATE_MAP_FLOOR
        assert quoted < 1.0, "a clipped std was read instead of the raw one"

    def test_the_measured_spread_at_reps_one_really_is_zero(self) -> None:
        """The guard's premise, measured directly rather than trusted."""
        witness = entangling_witness(_components(64), reps=1, sample=16)
        assert witness["mean_abs_connected_correlation"] == pytest.approx(0.0, abs=1e-12)
        assert witness["is_entangling"] is False

    def test_reps_two_is_entangling_and_accepted(self) -> None:
        witness = entangling_witness(_components(64), reps=2, sample=16)
        assert witness["is_entangling"] is True
        assert witness["mean_abs_connected_correlation"] > witness["floor"]
        fitted = fit_map("zz", _components(), reps=2)
        assert fitted.reps == 2

    def test_the_scored_grid_excludes_reps_one_structurally(self) -> None:
        """Belt and braces: the guard is the proof, the grid is the policy."""
        assert 1 not in QUANTUM_REPS_GRID
        assert QUANTUM_REPS_GRID == (2,)
        assert 1 in REPS_GRID, "E3's grid is unchanged; the exclusion is E4's, not a rewrite"

    @pytest.mark.parametrize("family", FAMILIES)
    def test_a_constant_input_is_refused_for_every_family(self, family: str) -> None:
        """No family may report a score on an input that carries no information at all."""
        constant = np.ones((32, QUANTUM_DIM), dtype=np.float64)
        reps = 2 if family == "zz" else None
        with pytest.raises(E4QuantumError, match="degenerate"):
            fit_map(family, constant, reps=reps)


# ------------------------------------------------------------------ it could hobble control
class TestSameShapeContract:
    """The control is only a control if it is the same shape, index for index."""

    @pytest.mark.parametrize("family", FAMILIES)
    def test_every_family_emits_the_same_width(self, family: str) -> None:
        fitted = fit_map(family, _components(), reps=_reps_for(family))
        out = fitted.transform(_components(EVAL_ROWS, seed=9))
        assert out.shape == (EVAL_ROWS, N_OBSERVABLES)
        assert RFF_FEATURES == N_OBSERVABLES

    @pytest.mark.parametrize("family", FAMILIES)
    def test_every_family_adds_zero_trainable_parameters(self, family: str) -> None:
        """A map with free parameters would be a different experiment, not a control."""
        fitted = fit_map(family, _components(), reps=_reps_for(family))
        assert fitted.telemetry["n_trainable_parameters"] == 0

    def test_poly2_pairs_are_in_all_pair_masks_order(self) -> None:
        """"Same index set" must be true column-for-column, not merely as a count."""
        masks = all_pair_masks(QUANTUM_DIM)
        recovered = []
        for mask in masks:
            bits = [i for i in range(QUANTUM_DIM) if mask & (1 << i)]
            assert len(bits) == 2
            recovered.append((bits[0], bits[1]))
        assert recovered == _pair_indices(QUANTUM_DIM)

    def test_poly2_columns_are_the_singletons_then_those_products(self) -> None:
        values = _components(7, seed=3)
        out = poly2_expand(values)
        assert out.shape == (7, N_OBSERVABLES)
        np.testing.assert_allclose(out[:, :QUANTUM_DIM], values)
        for offset, (i, j) in enumerate(_pair_indices()):
            np.testing.assert_allclose(out[:, QUANTUM_DIM + offset], values[:, i] * values[:, j])

    def test_poly2_refuses_a_width_other_than_the_circuit_width(self) -> None:
        with pytest.raises(E4QuantumError, match="8 columns"):
            poly2_expand(np.ones((4, 7)))

    def test_fit_map_refuses_a_width_other_than_the_circuit_width(self) -> None:
        with pytest.raises(E4QuantumError, match="components"):
            fit_map("poly2", np.ones((16, 7)))

    def test_an_unknown_family_is_refused(self) -> None:
        with pytest.raises(E4QuantumError, match="Unknown map family"):
            fit_map("quantum_inspired", _components())

    def test_zz_refuses_a_reps_outside_the_grid(self) -> None:
        with pytest.raises(E4QuantumError, match="reps must be in"):
            fit_map("zz", _components(), reps=7)


# -------------------------------------------------------------------------- it could leak
class TestTrainOnlyFitting:
    """Every statistic around every map comes from fold-train rows and nowhere else."""

    @pytest.mark.parametrize("family", FAMILIES)
    def test_transform_is_a_pure_function_of_the_fitted_statistics(self, family: str) -> None:
        """Scoring rows one at a time must equal scoring them together.

        If any statistic were computed from the rows being transformed, a single row would
        standardise to zero and the two calls would disagree. This is the cheapest test
        that actually detects transform-time leakage.
        """
        fitted = fit_map(family, _components(), reps=_reps_for(family))
        held_out = _components(4, seed=21)
        together = fitted.transform(held_out)
        apart = np.vstack([fitted.transform(held_out[i : i + 1]) for i in range(held_out.shape[0])])
        np.testing.assert_allclose(together, apart, atol=1e-12)

    @pytest.mark.parametrize("family", FAMILIES)
    def test_held_out_rows_do_not_standardise_to_zero_mean(self, family: str) -> None:
        """A map refitted on the scored rows would centre them exactly; a fitted one does not."""
        train = _components(FIT_ROWS, seed=1)
        fitted = fit_map(family, train, reps=_reps_for(family))
        shifted = _components(EVAL_ROWS, seed=2) + 3.0
        out = fitted.transform(shifted)
        assert np.max(np.abs(out.mean(axis=0))) > 1e-6

    @pytest.mark.parametrize("family", FAMILIES)
    def test_fold_train_output_is_standardised_by_construction(self, family: str) -> None:
        train = _components(FIT_ROWS, seed=1)
        fitted = fit_map(family, train, reps=_reps_for(family))
        out = fitted.transform(train)
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-9)
        np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-9)

    def test_standardiser_floors_a_constant_column_rather_than_dividing_by_zero(self) -> None:
        """The behaviour the degeneracy guard must not rely on, pinned so it stays deliberate."""
        values = np.column_stack([np.arange(10.0), np.ones(10)])
        mean, std = _standardiser(values)
        assert std[1] == 1.0
        assert np.isfinite((values - mean) / std).all()

    def test_the_angle_encoding_stays_inside_the_declared_range_on_train(self) -> None:
        """A phase outside ``[0, pi]`` would wrap and silently alias two different inputs."""
        train = _components(FIT_ROWS, seed=4)
        fitted = fit_map("zz", train, reps=2)
        assert fitted.telemetry["angle_scaler"]["max_angle"] == ANGLE_MAX
        assert fitted.telemetry["angle_scaler"]["fitted_on"] == "fold-train rows only"

    def test_rff_bandwidth_is_deterministic_for_a_seed(self) -> None:
        train = _components(64, seed=8)
        train = train - train.mean(axis=0)
        first = RandomFourierMap.fit(train, seed=42)
        second = RandomFourierMap.fit(train, seed=42)
        assert first.gamma == second.gamma
        np.testing.assert_array_equal(first.weights, second.weights)
        np.testing.assert_array_equal(first.offsets, second.offsets)


# --------------------------------------------------------- it could flatter the quantum arm
class TestSelectionBudget:
    """The quantum arm gets exactly the search every control gets -- no more."""

    def _arm(self, family: str) -> dict:
        components = _components(ARM_ROWS, seed=6)
        rng = np.random.default_rng(6)
        y = (components[:, 0] + rng.normal(scale=0.5, size=ARM_ROWS) > 0).astype(int)
        inner_train = np.zeros(ARM_ROWS, dtype=bool)
        inner_train[:ARM_INNER_TRAIN] = True
        inner_validation = ~inner_train
        return score_map_arm(
            family,
            components,
            y,
            inner_train,
            inner_validation,
            _components(ARM_EVAL, seed=7),
            reps_grid=QUANTUM_REPS_GRID if family == "zz" else (None,),
        )

    @pytest.mark.parametrize("family", ("poly2", "rff36"))
    def test_each_classical_family_searches_the_same_number_of_cells(self, family: str) -> None:
        assert self._arm(family)["n_grid_cells"] == len(HEAD_C_GRID)

    def test_the_quantum_family_searches_the_same_number_of_cells(self) -> None:
        arm = self._arm("zz")
        assert arm["n_grid_cells"] == len(HEAD_C_GRID) * len(QUANTUM_REPS_GRID)
        assert arm["n_grid_cells"] == len(HEAD_C_GRID), "budget parity with the controls"

    @pytest.mark.parametrize("family", FAMILIES)
    def test_selection_reports_the_inner_partition_not_the_scored_one(self, family: str) -> None:
        arm = self._arm(family)
        assert "fold 8" in arm["selected_on"]
        assert arm["selected"]["C"] in HEAD_C_GRID
        assert len(arm["inner_scores"]) == ARM_ROWS - ARM_INNER_TRAIN
        assert len(arm["validation_scores"]) == ARM_EVAL

    @pytest.mark.parametrize("family", FAMILIES)
    def test_the_selected_cell_is_the_grid_argmax(self, family: str) -> None:
        """The recorded winner must be the best cell, not the last one fitted."""
        arm = self._arm(family)
        best = max(arm["grid"], key=lambda cell: cell["inner_roc_auc"])
        assert arm["selected_inner_roc_auc"] == best["inner_roc_auc"]


class TestBlender:
    """Every fusion differs in one column and in nothing else."""

    def test_the_blender_refuses_mismatched_column_and_name_counts(self) -> None:
        y = np.array([0, 1, 0, 1])
        with pytest.raises(E4QuantumError, match="disagree"):
            blend([np.zeros(4), np.ones(4)], y, [np.zeros(2), np.ones(2)], names=["only_one"])

    def test_the_blender_records_one_coefficient_per_named_member(self) -> None:
        rng = np.random.default_rng(0)
        y = (rng.random(80) > 0.5).astype(int)
        columns = [y + rng.normal(scale=0.5, size=80) for _ in range(3)]
        evaluation = [rng.normal(size=20) for _ in range(3)]
        names = ["cnn", "gbm", "zz"]
        out = blend(columns, y, evaluation, names=names)
        assert out["members"] == names
        assert sorted(out["coefficients"]) == sorted(names)


# ------------------------------------------------------- it could fabricate the baseline
class TestReproductionAssertions:
    """A member that does not reproduce the record is refused, not rounded away."""

    def test_a_drifted_member_is_refused(self) -> None:
        with pytest.raises(E4QuantumError, match="refusing to continue"):
            _assert_reproduces("cnn@resnet_small", 0.9404, 0.9404758944556537, 1e-9)

    def test_a_missing_member_is_refused_rather_than_skipped(self) -> None:
        with pytest.raises(E4QuantumError, match="refusing to continue"):
            _assert_reproduces("fusion@cnn+gbm", None, 0.9462926980022166, 1e-9)

    def test_an_exact_member_passes(self) -> None:
        _assert_reproduces("fusion@cnn+gbm", 0.9462926980022166, 0.9462926980022166, 1e-9)

    def test_the_tolerance_is_honoured_rather_than_ignored(self) -> None:
        _assert_reproduces("cnn@resnet_small", 0.946739 + 4e-7, 0.946739, 5e-7)
        with pytest.raises(E4QuantumError):
            _assert_reproduces("cnn@resnet_small", 0.946739 + 6e-7, 0.946739, 5e-7)


class TestFrozenTestRefusal:
    """Fold 10 must stop the run at the door, before a single map is fitted.

    The frozen test partition is protected by policy everywhere else in E4; here it is
    protected by a refusal that runs before any expensive work, so a cache that should not
    exist cannot be silently consumed by a 45-minute job.
    """

    def _cohort(self, folds: Sequence[int]) -> EcgCohort:
        n = len(folds)
        names = feature_names()
        labels = np.tile([0, 1], n // 2 + 1)[:n].astype(np.int8)
        superclasses = np.zeros((n, len(SUPERCLASS_ORDER)), dtype=np.uint8)
        superclasses[:, 0] = 1
        return EcgCohort(
            features=np.zeros((n, len(names)), dtype=np.float64),
            feature_names=names,
            ecg_ids=np.arange(n, dtype=np.int32),
            patient_ids=np.arange(1000, 1000 + n, dtype=np.int64),
            strat_folds=np.asarray(folds, dtype=np.int8),
            labels=labels,
            label_known=np.ones(n, dtype=np.uint8),
            superclasses=superclasses,
            superclass_order=SUPERCLASS_ORDER,
            ages=np.full(n, 60.0, dtype=np.float32),
            sexes=np.zeros(n, dtype=np.int8),
            n_beats=np.full(n, 10, dtype=np.int32),
            template_beats=np.zeros((n, 1, 1), dtype=np.float32),
            quality_flags=(),
            metadata={
                "feature_version": FEATURE_SET_VERSION,
                "cache_version": ECG_CACHE_VERSION,
            },
        )

    def test_a_feature_cache_containing_fold_ten_is_refused(self, monkeypatch) -> None:
        cohort = self._cohort([1, 2, 9, 10])
        monkeypatch.setattr(e4_quantum, "load_cohort", lambda *_a, **_k: cohort)
        monkeypatch.setattr(
            e4_quantum,
            "load_signal_cache",
            lambda *_a, **_k: pytest.fail("refused too late: the signal cache was loaded"),
        )
        with pytest.raises(E4QuantumError, match="contains fold 10"):
            e4_quantum.run()

    def test_a_signal_cache_containing_fold_ten_is_refused(self, monkeypatch) -> None:
        cohort = self._cohort([1, 2, 9, 9])
        signals = SimpleNamespace(metadata={"folds": [1, 2, 9, 10]})
        monkeypatch.setattr(e4_quantum, "load_cohort", lambda *_a, **_k: cohort)
        monkeypatch.setattr(e4_quantum, "load_signal_cache", lambda *_a, **_k: signals)
        monkeypatch.setattr(
            e4_quantum,
            "fit_ecg_transform",
            lambda *_a, **_k: pytest.fail("refused too late: a transform was fitted"),
        )
        with pytest.raises(E4QuantumError, match="contains fold 10"):
            e4_quantum.run()

    def test_the_test_folds_constant_is_what_is_being_guarded(self) -> None:
        assert TEST_FOLDS == frozenset({10})
        assert not (TRAIN_FOLDS | VALIDATION_FOLDS) & TEST_FOLDS


def test_the_version_is_stamped() -> None:
    assert E4_QUANTUM_VERSION == "v1-e4-quantum-1"
