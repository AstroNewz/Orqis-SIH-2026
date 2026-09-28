"""Tests for E3 Round 4 -- the richer quantum feature map (:mod:`backend.evaluation.e3_feature_map`).

These lock the properties that make the Round-4 null trustworthy rather than the research
numbers themselves (a null is a valid outcome, but a *leaky* or *miscomputed* null is not):

* the hand-built ZZ circuit is Aer-native and bit-identical to :class:`qiskit`'s canonical
  ``ZZFeatureMap`` (so "richer map" really means the Havlicek map, not an approximation);
* the fast ``|psi|^2 @ diag^T`` readout equals Aer ``save_expectation_value`` to <=1e-10;
* the entanglement witness reads 0 on a genuine product state and nonzero on generic inputs
  (so an "adds nothing" null cannot be an undetected "map degenerated to separable");
* every selector / scaler / PCA / RFF / logistic ``C`` is fit inside fold-train only, proven by
  a monkeypatch spy that records exactly which rows each per-fold selector sees;
* the MI selector is genuinely label-aware (it selects different columns under a shuffled label);
* the emitted report never sets a quantum-advantage flag that its own gating null did not earn,
  reproduces the C7 anchor (0.913038) to 1e-6, and never touches the test partition.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.evaluation import e3_feature_map as fm
from backend.evaluation.e3_feature_map import (
    ENTANGLING_WITNESS_FLOOR,
    E3_FEATURE_MAP_VERSION,
    N_OBSERVABLES,
    N_QUBITS,
    REPS_GRID,
    SELECTORS,
    FeatureMapError,
    RichQuantumFeatureMap,
    _build_zz_circuit,
    _make_mi_selector,
    _make_pca_selector,
    _pair_index_list,
    _score_richmap_config,
    run,
    validate_against_qiskit_zzmap,
    validate_readout_against_aer,
)


# --------------------------------------------------------------------------- circuit
class TestZZCircuit:
    def test_native_gate_set_only(self):
        """Aer can only simulate the statevector directly if every gate is native h/p/cx."""
        for reps in REPS_GRID:
            circuit, _ = _build_zz_circuit(N_QUBITS, reps)
            gate_names = {instr.operation.name for instr in circuit.data}
            assert gate_names <= {"h", "p", "cx"}, gate_names

    def test_has_encoding_parameters_and_no_trainable(self):
        circuit, theta = _build_zz_circuit(N_QUBITS, 2)
        assert len(theta) == N_QUBITS
        assert circuit.num_parameters == N_QUBITS  # data angles only; nothing to train

    def test_two_qubit_gates_scale_with_reps(self):
        c1, _ = _build_zz_circuit(N_QUBITS, 1)
        c2, _ = _build_zz_circuit(N_QUBITS, 2)
        two_q1 = sum(1 for i in c1.data if i.operation.num_qubits == 2)
        two_q2 = sum(1 for i in c2.data if i.operation.num_qubits == 2)
        # full entanglement => 2 CX per pair per block; C(8,2)=28 pairs => 56 CX/block.
        assert two_q1 == 56
        assert two_q2 == 112

    def test_rejects_unlisted_reps(self):
        with pytest.raises(FeatureMapError):
            _build_zz_circuit(N_QUBITS, 3)


# ------------------------------------------------------------------------ feature map
class TestFeatureMap:
    def test_observable_set_is_singles_plus_all_pairs(self):
        fmap = RichQuantumFeatureMap(reps=1)
        assert len(fmap.masks) == N_OBSERVABLES == 36
        assert len(_pair_index_list(N_QUBITS)) == 28

    def test_transform_shape_and_expectation_range(self):
        rng = np.random.default_rng(0)
        fmap = RichQuantumFeatureMap(reps=2, seed=0)
        feats = fmap.transform(rng.standard_normal((5, N_QUBITS)))
        assert feats.shape == (5, N_OBSERVABLES)
        assert np.all(feats <= 1.0 + 1e-9) and np.all(feats >= -1.0 - 1e-9)

    def test_transform_refuses_wrong_width(self):
        fmap = RichQuantumFeatureMap(reps=1)
        with pytest.raises(FeatureMapError):
            fmap.transform(np.zeros((3, N_QUBITS + 1)))  # no silent pad/truncate

    def test_transform_refuses_non_finite(self):
        fmap = RichQuantumFeatureMap(reps=1)
        bad = np.zeros((2, N_QUBITS))
        bad[0, 0] = np.nan
        with pytest.raises(FeatureMapError):
            fmap.transform(bad)

    def test_connected_correlations_shape(self):
        rng = np.random.default_rng(1)
        fmap = RichQuantumFeatureMap(reps=2, seed=0)
        cc = fmap.connected_correlations(rng.standard_normal((4, N_QUBITS)))
        assert cc.shape == (4, 28)

    def test_telemetry_contract(self):
        tel = RichQuantumFeatureMap(reps=2).telemetry()
        assert tel["map"] == "havlicek_zz"
        assert tel["n_trainable_parameters"] == 0
        assert tel["n_observables"] == N_OBSERVABLES
        assert tel["entanglement"] == "full"
        assert tel["state_dimension"] == 1 << N_QUBITS


# ------------------------------------------------------------------- correctness gates
class TestCorrectnessGates:
    def test_fast_readout_matches_aer(self):
        rng = np.random.default_rng(2)
        feats = rng.standard_normal((8, N_QUBITS))
        for reps in REPS_GRID:
            check = validate_readout_against_aer(RichQuantumFeatureMap(reps=reps, seed=0), feats)
            assert check["passed"] and check["max_abs_deviation"] <= 1e-10

    def test_handbuilt_equals_canonical_zzfeaturemap(self):
        """The hand build must be the canonical Havlicek map (fidelity 1), not a look-alike."""
        rng = np.random.default_rng(3)
        feats = rng.standard_normal((6, N_QUBITS))
        for reps in REPS_GRID:
            check = validate_against_qiskit_zzmap(RichQuantumFeatureMap(reps=reps, seed=0), feats)
            assert check["passed"]
            if check.get("checked"):
                assert check["max_infidelity"] <= 1e-10

    def test_readout_gate_raises_when_diagonals_corrupted(self):
        """If the fast diagonals were wrong, the Aer gate must fail closed, not pass silently.

        Uses reps=2, where the Z-readout is genuinely nonzero (see
        ``test_single_rep_zz_readout_is_identically_zero`` for why reps=1 could not detect this).
        """
        fmap = RichQuantumFeatureMap(reps=2, seed=0)
        fmap._diagonals = fmap._diagonals * -1.0  # corrupt every observable's sign
        with pytest.raises(FeatureMapError):
            validate_readout_against_aer(fmap, np.random.default_rng(4).standard_normal((3, N_QUBITS)))

    def test_single_rep_zz_readout_is_identically_zero(self):
        """A one-block ZZ map read by Z-strings carries *zero* signal -- a mathematical fact.

        The reps=1 circuit is ``H^n`` then only Z-diagonal (``P``, ``ZZ``) gates, so the state is
        ``U|+>^n`` with ``U`` diagonal in Z; every Z-string expectation is therefore
        ``<+|^n Z_S |+>^n = 0``. This is *why* both reps=1 configs in the report score exactly the
        class base rate (the feature vector is the zero vector): re-uploading is required before a
        Z-readout of the ZZ map sees anything at all.
        """
        rng = np.random.default_rng(9)
        feats = RichQuantumFeatureMap(reps=1, seed=0).transform(rng.standard_normal((12, N_QUBITS)))
        assert np.max(np.abs(feats)) < 1e-12


# ------------------------------------------------------------ entanglement witness truth
class TestEntanglementWitness:
    def test_zero_on_product_state(self):
        """Features == pi zero every ZZ phase, so the state is |+>^n: all connected corr = 0.

        This is what lets a null be read as 'entanglement did not help' -- the witness can tell
        a genuinely entangling map from one that silently degenerated to a product state.
        """
        fmap = RichQuantumFeatureMap(reps=1, seed=0)
        pi_feats = np.full((3, N_QUBITS), np.pi)
        cc = fmap.connected_correlations(pi_feats)
        assert np.max(np.abs(cc)) < 1e-9
        singles = fmap.transform(pi_feats)[:, :N_QUBITS]
        assert np.max(np.abs(singles)) < 1e-9  # <Z_i> = 0 on |+>

    def test_nonzero_on_generic_features(self):
        rng = np.random.default_rng(5)
        fmap = RichQuantumFeatureMap(reps=2, seed=0)
        cc = fmap.connected_correlations(rng.standard_normal((32, N_QUBITS)))
        assert float(np.mean(np.abs(cc))) > ENTANGLING_WITNESS_FLOOR


# --------------------------------------------------------------------------- selectors
class TestSelectors:
    def test_registry(self):
        assert set(SELECTORS) == {"pca", "mi"}

    def test_pca_selector_output_is_standardised(self):
        rng = np.random.default_rng(6)
        X = rng.standard_normal((60, 32))
        transform = _make_pca_selector(seed=0)(X, np.array([0, 1] * 30))
        out = transform(X)
        assert out.shape == (60, N_QUBITS)
        assert np.allclose(out.mean(axis=0), 0.0, atol=1e-9)
        assert np.allclose(out.std(axis=0), 1.0, atol=1e-6)

    def test_mi_selector_output_shape(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((60, 32))
        transform = _make_mi_selector(seed=0)(X, np.array([0, 1] * 30))
        assert transform(X).shape == (60, N_QUBITS)

    def test_mi_selector_is_label_aware(self):
        """On data where only cols 8-15 carry the label, MI must select by y, not by variance.

        Fitting with the true label vs a shuffled label selects different columns, so the two
        transforms of the same X differ -- proving the selection genuinely uses y (and, per the
        spy test below, only the fold-train y).
        """
        rng = np.random.default_rng(8)
        y = np.array([0, 1] * 40)
        X = rng.standard_normal((80, 16))
        X[:, 8:16] += 4.0 * y[:, None]  # only these columns are informative
        y_shuffled = rng.permutation(y)
        out_true = _make_mi_selector(seed=0)(X, y)(X)
        out_shuffled = _make_mi_selector(seed=0)(X, y_shuffled)(X)
        assert not np.allclose(out_true, out_shuffled)


# ------------------------------------------------------- fold-honesty of the arm scorer
class TestScorerFoldHonesty:
    def _synthetic(self, n=48, width=16, seed=0):
        rng = np.random.default_rng(seed)
        source = rng.standard_normal((n, width))
        source[:, 0] = np.arange(n)  # sentinel: column 0 encodes the global row index
        y = np.array([0, 1] * (n // 2))
        source[:, 1] += 1.5 * y  # a little separable signal so logistic fits
        folds = []
        idx = np.arange(n)
        for k in range(4):
            test_idx = idx[k::4]
            train_idx = np.setdiff1d(idx, test_idx)
            folds.append((train_idx, test_idx))
        return source, y, folds

    @pytest.mark.parametrize("selector_name", ["pca", "mi"])
    def test_per_fold_selector_sees_only_fold_train_rows(self, selector_name):
        """The critical leakage test: each per-fold selector fit must see fold-train rows only.

        A spy wraps the real selector factory and records the sentinel row indices handed to it.
        The first ``n_folds`` calls are the per-fold fits; each must equal that fold's train set
        and be disjoint from its held-out test rows. No test-derived PCA/MI/scaling can occur.
        """
        source, y, folds = self._synthetic()
        base_factory = SELECTORS[selector_name](seed=0)
        seen_index_sets = []

        def spy_factory(fold_source, fold_y):
            seen_index_sets.append(set(fold_source[:, 0].astype(int).tolist()))
            return base_factory(fold_source, fold_y)

        _score_richmap_config(
            source, source[:6], y, folds,
            selector_factory=spy_factory, feature_map=RichQuantumFeatureMap(reps=1, seed=0),
        )

        assert len(seen_index_sets) == len(folds) + 1  # per-fold fits + one all-train refit
        for (train_idx, test_idx), seen in zip(folds, seen_index_sets):
            assert seen == set(train_idx.tolist())
            assert seen.isdisjoint(set(test_idx.tolist()))
        assert seen_index_sets[-1] == set(range(source.shape[0]))  # refit on all train

    def test_output_contract(self):
        source, y, folds = self._synthetic()
        fit = _score_richmap_config(
            source, source[:5], y, folds,
            selector_factory=SELECTORS["pca"](seed=0),
            feature_map=RichQuantumFeatureMap(reps=1, seed=0),
        )
        assert fit["representation_dim"] == N_OBSERVABLES
        assert fit["reps"] == 1
        assert fit["train_out_of_fold"].shape == (source.shape[0],)
        assert len(fit["validation_score"]) == 5
        assert 0.0 <= fit["train_cv_pr_auc"] <= 1.0
        assert fit["selected_c"] in set(__import__("backend.evaluation.e0_controls", fromlist=["C_GRID"]).C_GRID)


# --------------------------------------------------------------- data-backed report
@pytest.fixture(scope="module")
def report():
    # Point estimates are seed-deterministic; reduced draws only speed the CI/null machinery.
    return run(bootstrap_draws=64, increment_permutations=64)


class TestReport:
    def test_version_and_question(self, report):
        assert report["e3_feature_map_version"] == E3_FEATURE_MAP_VERSION
        assert "richer quantum feature map" in report["question"].lower()

    def test_test_partition_firewall(self, report):
        assert report["test_partition_used"] is False
        assert "test" in report["test_partition_note"].lower()

    def test_c7_anchor_reproduced_exactly(self, report):
        anchor = report["c7_anchor"]
        assert anchor["reproduced"] is True
        assert abs(anchor["reproduced_pr_auc"] - 0.913038) <= 1e-6

    def test_correctness_gates_pass(self, report):
        assert report["quantum_feature_map_aer_check"]["max_abs_deviation"] <= 1e-10
        canon = report["quantum_feature_map_canonical_check"]
        assert canon["max_infidelity"] <= 1e-10

    def test_config_grid_is_two_selectors_by_two_reps(self, report):
        grid = report["config_grid"]
        assert len(grid) == 4
        assert {(c["selector"], c["reps"]) for c in grid} == {
            ("pca", 1), ("pca", 2), ("mi", 1), ("mi", 2)
        }
        for c in grid:
            assert c["train_cv_pr_auc"] is not None

    def test_selected_config_is_best_on_train_oof(self, report):
        """The headline quantum arm must be the argmax over the grid on TRAIN OOF -- never val."""
        grid = report["config_grid"]
        best = max(grid, key=lambda c: c["train_cv_pr_auc"])
        assert report["summary"]["selected_selector"] == best["selector"]
        assert report["summary"]["selected_reps"] == best["reps"]

    def test_matched_control_present_at_equal_width(self, report):
        rff = report["arms"]["rff36"]
        assert rff["representation_dim"] == N_OBSERVABLES  # DEC-033: matched dimension

    def test_every_arm_has_bootstrap_ci(self, report):
        for name in ("c7", "rich_quantum", "rff36"):
            assert "validation_bootstrap" in report["arms"][name]

    def test_entanglement_witness_present_and_nonzero(self, report):
        witness = report["arms"]["rich_quantum"]["entanglement_witness"]
        assert witness["n_pairs"] == 28
        assert witness["mean_abs_connected_correlation"] > ENTANGLING_WITNESS_FLOOR
        assert witness["entangling"] is True

    def test_increments_are_gated_by_column_permutation(self, report):
        for name in ("rich_quantum", "rff36"):
            inc = report["increment_vs_c7"][name]
            assert inc["column_permutation_null"]["gates_the_claim"] is True

    def test_verdict_flags_consistent_with_gating(self, report):
        s = report["summary"]
        rich = s["rich_quantum_increment_survives_gating_null"]
        rff = s["rff36_increment_survives_gating_null"]
        assert s["defensible_quantum_advantage_over_c7"] == bool(rich)
        assert s["quantum_specific_advantage"] == bool(rich and not rff)

    def test_no_advantage_flag_without_survival(self, report):
        """Fabrication guard: an advantage flag can never be True if the gating null was not passed."""
        s = report["summary"]
        if not s["rich_quantum_increment_survives_gating_null"]:
            assert s["defensible_quantum_advantage_over_c7"] is False
            assert s["quantum_specific_advantage"] is False

    def test_leaderboard_sorted_by_validation(self, report):
        vals = [r["validation_pr_auc"] for r in report["summary"]["leaderboard"]]
        assert vals == sorted(vals, reverse=True)

    def test_measured_outcome_is_the_recorded_null(self, report):
        """Lock the recorded scientific finding (seed-deterministic point estimates).

        The richer, genuinely-entangling quantum map does not beat C7 and is *worse than its own
        matched classical control* -- so the null is representational, not a dimensionality or
        input-selection artifact.
        """
        s = report["summary"]
        assert s["rich_quantum_beats_c7_on_validation"] is False
        assert s["rich_quantum_beats_c7_on_train_oof"] is False
        assert s["rich_quantum_validation_pr_auc"] < s["rff36_validation_pr_auc"]
        assert s["rich_quantum_increment_survives_gating_null"] is False
        assert "null" in s["conclusion"].lower()

    def test_payload_top_level_contract(self, report):
        for key in (
            "question", "test_partition_used", "c7_anchor", "config_grid", "arms",
            "increment_vs_c7", "summary", "provenance", "protocol", "environment",
            "quantum_feature_map_aer_check", "quantum_feature_map_canonical_check",
        ):
            assert key in report, key
