"""Tests for E3 Round 2 -- the quantum-kernel (QSVM) advantage experiment (DEC-038).

Round 2 asks the same mission-§30 question as Round 1 but with a *kernel* rather than a
linear head: does a fidelity quantum kernel ``K(x,x') = |<psi(x)|psi(x')>|^2`` over the
8-qubit E2 states add information beyond the best classical baseline C7? The dishonesty
modes these tests pin down are Round 1's, plus two specific to a kernel method:

* the test partition is never read -- the driver reads only TRAIN and VALIDATION via
  ``build_inputs(conditions=(PRIMARY,))``;
* the comparison is against the *real* C7 -- the rebuilt baseline reproduces 0.913038;
* neither kernel scorer fits its per-fold PCA/angle encoding on a row it is about to
  score -- proven with a spy that records exactly which rows each per-fold ``fit`` sees;
* the fidelity Gram is a *correct* kernel -- symmetric, unit diagonal, entries in [0,1] --
  so "quantum kernel" is not a mislabelled classical matrix;
* the concentration witness is real -- ``_offdiagonal_summary`` reports the spread that
  distinguishes a healthy kernel from an exponentially-concentrated (degenerate) one, so a
  degenerate kernel cannot be silently reported as signal;
* the matched classical RBF-SVM control is present beside the quantum kernel at every turn.

The kernel-math and scorer tests run cache-free on tiny synthetic patient-grouped data
(real Aer statevectors); the payload tests read the driver's report and skip when absent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

import backend.evaluation.e3_quantum_kernel as e3qk
from backend.evaluation.e0_controls import N_FOLDS, grouped_folds
from backend.evaluation.e1_failure_map import PRIMARY, SEED
from backend.evaluation.e3_hybrid_fusion import C7_DIM, C7_REFERENCE_PR_AUC, MOBILENET_DIM
from backend.evaluation.e3_quantum_kernel import (
    E3_QUANTUM_KERNEL_VERSION,
    _fidelity_gram,
    _offdiagonal_summary,
    _score_quantum_kernel_arm,
    _score_rbf_arm,
)
from backend.ml.quantum_visual import PCA_COMPONENTS, QuantumVisualPreprocessor
from quantum_ml.visual_circuit import QuantumVisualFeatureMap

DRIVER_SOURCE = Path(e3qk.__file__).read_text(encoding="utf-8")

#: The driver's payload. Present only after a full run, so payload assertions skip.
REPORT_PATH = Path("reports_e3_quantum_kernel.json")


@pytest.fixture
def payload() -> dict:
    if not REPORT_PATH.exists():
        pytest.skip("reports_e3_quantum_kernel.json not present; run the driver first")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _synthetic(n_patients: int = 24, per_patient: int = 2, dim: int = MOBILENET_DIM, seed: int = 0):
    """Patient-grouped synthetic MobileNet-like data; rows unique for exact-row comparison."""
    rng = np.random.default_rng(seed)
    patients, y_rows, feats = [], [], []
    for p in range(n_patients):
        label = int(p % 2)
        centre = rng.normal(label * 3.0, 1.0, size=dim)
        for _ in range(per_patient):
            feats.append(centre + rng.normal(0.0, 0.1, size=dim))
            y_rows.append(label)
            patients.append(f"P{p:03d}")
    return np.asarray(feats, dtype=np.float64), np.asarray(y_rows, dtype=int), patients


def _random_states(n: int, n_qubits: int = 8, seed: int = 0) -> np.ndarray:
    """A batch of unit-norm complex statevectors (not from the circuit -- for kernel math)."""
    rng = np.random.default_rng(seed)
    dim = 1 << n_qubits
    raw = rng.normal(size=(n, dim)) + 1j * rng.normal(size=(n, dim))
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


# ----------------------------------------------------------------- test firewall
class TestDriverNeverTouchesTheTestPartition:
    def test_no_test_partition_reference(self):
        assert not re.search(r"\.test\b", DRIVER_SOURCE)
        assert not re.search(r"partitions\s*\[\s*[\"']test[\"']\s*\]", DRIVER_SOURCE)
        assert not re.search(r"roi_mode\s*=\s*[\"']test[\"']", DRIVER_SOURCE)

    def test_only_the_primary_condition_is_built(self):
        assert "conditions=(PRIMARY,)" in DRIVER_SOURCE

    def test_the_payload_declares_the_test_partition_unused(self):
        assert '"test_partition_used": False' in DRIVER_SOURCE
        assert "test_partition_note" in DRIVER_SOURCE

    def test_no_advantage_is_asserted_unconditionally(self):
        assert "defensible_quantum_advantage_over_c7" in DRIVER_SOURCE
        assert "quantum_kernel_concentrated" in DRIVER_SOURCE


# --------------------------------------------------------------- frozen wiring
class TestFrozenWiring:
    def test_version_and_shared_constants(self):
        assert E3_QUANTUM_KERNEL_VERSION == "v1-e3-quantum-kernel-1"
        assert MOBILENET_DIM == 576
        assert C7_DIM == 1195
        assert PCA_COMPONENTS == 8
        assert C7_REFERENCE_PR_AUC == 0.913038


# ------------------------------------------------------- fidelity kernel is a kernel
class TestFidelityKernelMath:
    def test_self_gram_is_symmetric_unit_diagonal_and_bounded(self):
        states = _random_states(9, seed=1)
        gram = _fidelity_gram(states, states)
        assert gram.shape == (9, 9)
        assert np.allclose(np.diag(gram), 1.0, atol=1e-9)  # |<psi|psi>|^2 = 1
        assert np.allclose(gram, gram.T, atol=1e-9)  # symmetric
        assert np.all(gram >= -1e-9) and np.all(gram <= 1.0 + 1e-9)  # a fidelity in [0,1]

    def test_cross_gram_has_expected_shape_and_bounds(self):
        rows = _random_states(4, seed=2)
        cols = _random_states(7, seed=3)
        cross = _fidelity_gram(rows, cols)
        assert cross.shape == (4, 7)
        assert np.all(cross >= -1e-9) and np.all(cross <= 1.0 + 1e-9)

    def test_identical_states_have_fidelity_one_orthogonal_states_zero(self):
        e0 = np.zeros((1, 4), dtype=complex); e0[0, 0] = 1.0
        e1 = np.zeros((1, 4), dtype=complex); e1[0, 1] = 1.0
        assert _fidelity_gram(e0, e0)[0, 0] == pytest.approx(1.0)
        assert _fidelity_gram(e0, e1)[0, 0] == pytest.approx(0.0, abs=1e-12)

    def test_offdiagonal_summary_flags_a_concentrated_kernel(self):
        # A near-constant off-diagonal is the degenerate case the witness must catch.
        concentrated = np.full((5, 5), 0.5); np.fill_diagonal(concentrated, 1.0)
        summ = _offdiagonal_summary(concentrated)
        assert summ["n_offdiagonal"] == 20
        assert summ["sd"] == pytest.approx(0.0, abs=1e-12)
        assert summ["mean"] == pytest.approx(0.5)
        # A spread-out kernel reports a non-zero sd.
        spread = _fidelity_gram(_random_states(12, seed=4), _random_states(12, seed=4))
        assert _offdiagonal_summary(spread)["sd"] > 1e-3


# ------------------------------------------------------- fold-honest, no leakage
class TestFoldHonestKernelScorers:
    def test_quantum_kernel_scorer_never_fits_on_a_holdout_row(self, monkeypatch):
        """Each per-fold ``QuantumVisualPreprocessor.fit`` sees only its fold-train rows.

        A spy replaces the module-level preprocessor with one that records the exact rows
        every ``fit`` is given, then delegates to the real class. ``_score_quantum_kernel_arm``
        fits once per fold (in fold order) to build the fold kernels, then once on all TRAIN
        for the validation column -- so the recorded fit-sets must be each fold's train rows
        (disjoint from its holdout) then all TRAIN. A holdout row in any fold's fit-set would
        be the CV leakage the mission forbids -- and a kernel entry built from it would be
        doubly wrong, since it enters every SVM decision through the Gram.
        """
        x, y, patients = _synthetic(seed=1)
        folds, _ = grouped_folds(y, patients, seed=SEED, n_splits=N_FOLDS)
        x_val, _, _ = _synthetic(n_patients=6, seed=99)
        fmap = QuantumVisualFeatureMap()

        captured: list[np.ndarray] = []
        real_cls = QuantumVisualPreprocessor

        class SpyPre:
            @staticmethod
            def fit(x_train, *, seed):
                captured.append(np.asarray(x_train, dtype=np.float64).copy())
                return real_cls.fit(x_train, seed=seed)

        monkeypatch.setattr(e3qk, "QuantumVisualPreprocessor", SpyPre)
        _score_quantum_kernel_arm(x, x_val, y, folds, feature_map=fmap, seed=SEED)

        assert len(captured) == len(folds) + 1, "one fit per fold, plus one on all-train"

        def rowset(a: np.ndarray) -> set:
            return {row.tobytes() for row in np.asarray(a, dtype=np.float64)}

        for k, (train_idx, test_idx) in enumerate(folds):
            fit_rows = rowset(captured[k])
            assert fit_rows == rowset(x[train_idx])
            assert fit_rows.isdisjoint(rowset(x[test_idx]))
        assert rowset(captured[-1]) == rowset(x)

    def test_quantum_kernel_scorer_returns_aligned_columns_and_hilbert_dim(self):
        x, y, patients = _synthetic(seed=2)
        x_val, _, _ = _synthetic(n_patients=6, seed=7)
        folds, _ = grouped_folds(y, patients, seed=SEED, n_splits=N_FOLDS)
        fmap = QuantumVisualFeatureMap()
        fit = _score_quantum_kernel_arm(x, x_val, y, folds, feature_map=fmap, seed=SEED)
        assert fit["train_out_of_fold"].shape == (x.shape[0],)
        assert fit["validation_score"].shape == (x_val.shape[0],)
        assert fit["representation_dim"] == 256  # 2**8 Hilbert-space dimension
        assert np.all(np.isfinite(fit["validation_score"]))
        off = fit["kernel_offdiagonal_train"]
        assert off["n_offdiagonal"] > 0 and 0.0 <= off["mean"] <= 1.0

    def test_rbf_control_scorer_returns_pca8_columns_and_a_bandwidth(self):
        x, y, patients = _synthetic(seed=3)
        x_val, _, _ = _synthetic(n_patients=6, seed=8)
        folds, _ = grouped_folds(y, patients, seed=SEED, n_splits=N_FOLDS)
        fit = _score_rbf_arm(x, x_val, y, folds, seed=SEED)
        assert fit["train_out_of_fold"].shape == (x.shape[0],)
        assert fit["validation_score"].shape == (x_val.shape[0],)
        assert fit["representation_dim"] == PCA_COMPONENTS  # matched control on PCA-8
        assert fit["selected_gamma"] > 0.0


# --------------------------------------------------------------- emitted payload
class TestEmittedPayload:
    def test_c7_anchor_reproduces_the_baseline(self, payload: dict):
        anchor = payload["c7_anchor"]
        assert anchor["reproduced"] is True
        assert anchor["reproduced_pr_auc"] == C7_REFERENCE_PR_AUC

    def test_test_partition_is_declared_unused(self, payload: dict):
        assert payload["test_partition_used"] is False
        assert payload["primary_condition"] == PRIMARY

    def test_aer_gate_passed(self, payload: dict):
        aer = payload["quantum_feature_map_aer_check"]
        assert aer["passed"] is True
        assert aer["max_abs_deviation"] <= aer["tolerance"]

    def test_three_arms_present_at_expected_dimensions(self, payload: dict):
        arms = payload["arms"]
        assert set(arms) == {"c7", "quantum_kernel", "rbf_svm"}
        assert arms["c7"]["representation_dim"] == C7_DIM
        assert arms["quantum_kernel"]["representation_dim"] == 256
        assert arms["rbf_svm"]["representation_dim"] == PCA_COMPONENTS
        for arm in arms.values():
            boot = arm["validation_bootstrap"]
            assert boot["p2_5"] <= boot["p97_5"]

    def test_quantum_kernel_carries_a_concentration_witness(self, payload: dict):
        off = payload["arms"]["quantum_kernel"]["kernel_offdiagonal_train"]
        assert {"mean", "sd", "min", "max", "n_offdiagonal"} <= set(off)
        # The recorded 'concentrated' verdict must agree with the witness it is derived from.
        concentrated = payload["summary"]["quantum_kernel_concentrated"]
        assert concentrated == bool(off["sd"] < 1e-3)

    def test_increment_tests_present_for_both_candidates(self, payload: dict):
        inc = payload["increment_vs_c7"]
        assert set(inc) == {"quantum_kernel", "rbf_svm"}
        for block in inc.values():
            assert block["reference_only_pr_auc"] == C7_REFERENCE_PR_AUC
            assert isinstance(block["survives_gating_null"], bool)

    def test_verdict_is_internally_consistent_with_the_gating_null(self, payload: dict):
        summary = payload["summary"]
        qk_survives = payload["increment_vs_c7"]["quantum_kernel"]["survives_gating_null"]
        rbf_survives = payload["increment_vs_c7"]["rbf_svm"]["survives_gating_null"]
        assert summary["quantum_kernel_increment_survives_gating_null"] == qk_survives
        assert summary["defensible_quantum_advantage_over_c7"] == qk_survives
        if summary["quantum_specific_advantage"]:
            assert qk_survives and not rbf_survives

    def test_matched_rbf_control_reported_beside_quantum(self, payload: dict):
        assert "rbf_svm" in payload["arms"]
        assert "matched_control" in payload["protocol"]

    def test_leaderboard_is_sorted_by_validation_pr_auc(self, payload: dict):
        board = payload["summary"]["leaderboard"]
        scores = [r["validation_pr_auc"] for r in board if r["validation_pr_auc"] is not None]
        assert scores == sorted(scores, reverse=True)
