"""Tests for E3 Round 3 -- the *trainable* shallow-VQC advantage experiment (DEC-039).

Round 3 asks the same mission-§30 question as Rounds 1-2, but now the circuit *learns*: on
the fixed 8-qubit E2 state we apply ``L`` trainable ``{Ry ring; CZ ring}`` layers, read the
eight single-qubit ``<Z_i>`` into a logistic head, and train end-to-end. The dishonesty
modes these tests pin down are the shared ones (test partition never read; the comparison is
the *real* C7; per-fold encoding never fits a holdout row) plus the ones specific to a
trainable quantum model:

* the torch forward really *is* the quantum circuit -- its eight ``<Z_i>`` match Qiskit's
  ``Statevector.evolve`` to ``<=1e-10`` (this is what makes "quantum" not a mislabelled
  neural net), across every depth and both entangling modes actually used;
* the E2 state is genuinely real -- ``_real_states`` accepts the (real) E2 statevector and
  *rejects* a complex one, so the real-arithmetic shortcut can never silently corrupt a state;
* the barren-plateau witness is real and head-independent -- it reports the variance of the
  bare ``<Z_0>`` gradient at random init, so an untrainable circuit cannot be dressed up as
  signal, and a zero-head artefact cannot masquerade as a plateau;
* training actually trains -- the loss decreases and the whole pipeline is deterministic;
* both matched controls (a capacity-matched classical MLP and a no-entangler ablation) are
  present beside the VQC at every turn.

The circuit-math, witness, training and scorer tests run cache-free on tiny synthetic
patient-grouped data (real Aer statevectors); the payload tests read the driver's report and
skip when it is absent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

import backend.evaluation.e3_trainable_vqc as e3vqc
from backend.evaluation.e0_controls import N_FOLDS, grouped_folds
from backend.evaluation.e1_failure_map import PRIMARY, SEED
from backend.evaluation.e3_hybrid_fusion import C7_DIM, C7_REFERENCE_PR_AUC, MOBILENET_DIM
from backend.evaluation.e3_trainable_vqc import (
    E3_TRAINABLE_VQC_VERSION,
    TrainableVQCError,
    VQC_DEPTHS,
    _build_fold_encodings,
    _cz_ring_phase,
    _gradient_variance_witness,
    _matched_hidden,
    _mlp_parameter_count,
    _oof_and_validation,
    _qiskit_local_z,
    _real_states,
    _train_mlp,
    _train_vqc,
    _validate_ansatz,
    _vqc_logits,
    _z_signs,
)
from backend.ml.quantum_visual import PCA_COMPONENTS, QuantumVisualPreprocessor
from quantum_ml.visual_circuit import E2_N_QUBITS, QuantumVisualFeatureMap, nn_ring_pairs

DRIVER_SOURCE = Path(e3vqc.__file__).read_text(encoding="utf-8")

#: The driver's payload. Present only after a full run, so payload assertions skip.
REPORT_PATH = Path("reports_e3_trainable_vqc.json")

torch = pytest.importorskip("torch")


@pytest.fixture
def payload() -> dict:
    if not REPORT_PATH.exists():
        pytest.skip("reports_e3_trainable_vqc.json not present; run the driver first")
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


def _tensors(n_qubits: int = E2_N_QUBITS):
    cz = torch.tensor(_cz_ring_phase(n_qubits), dtype=torch.float64)
    zs = torch.tensor(_z_signs(n_qubits), dtype=torch.float64)
    return cz, zs


def _unit_states(n: int, n_qubits: int = E2_N_QUBITS, seed: int = 0) -> np.ndarray:
    """A batch of real unit-norm vectors -- the operative domain for the real E2 ansatz."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, 1 << n_qubits))
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
        assert "quantum_specific_advantage" in DRIVER_SOURCE


# --------------------------------------------------------------- frozen wiring
class TestFrozenWiring:
    def test_version_and_shared_constants(self):
        assert E3_TRAINABLE_VQC_VERSION == "v1-e3-trainable-vqc-1"
        assert E2_N_QUBITS == 8
        assert MOBILENET_DIM == 576
        assert C7_DIM == 1195
        assert PCA_COMPONENTS == 8
        assert C7_REFERENCE_PR_AUC == 0.913038

    def test_depths_are_shallow(self):
        # sqrt(T/N)-bounded: only genuinely shallow circuits are compared.
        assert VQC_DEPTHS == (1, 2)


# --------------------------------------------------- the torch forward is a real circuit
class TestRealCircuitArithmetic:
    def test_cz_ring_phase_is_plus_minus_one_and_matches_the_ring(self):
        phase = _cz_ring_phase(E2_N_QUBITS)
        assert phase.shape == (1 << E2_N_QUBITS,)
        assert set(np.unique(phase)).issubset({-1.0, 1.0})
        # State |...11...> on a single ring edge (and no other) must carry exactly one -1.
        ring = nn_ring_pairs(E2_N_QUBITS)
        a, b = ring[0]
        idx = (1 << a) | (1 << b)
        assert phase[idx] == -1.0
        assert phase[0] == 1.0  # |0...0>: no pair set

    def test_z_signs_are_plus_minus_one_with_correct_parity(self):
        signs = _z_signs(E2_N_QUBITS)
        assert signs.shape == (E2_N_QUBITS, 1 << E2_N_QUBITS)
        assert set(np.unique(signs)).issubset({-1.0, 1.0})
        # <Z_q> sign of basis state b is +1 iff qubit q is 0 (little-endian bit q).
        assert signs[0, 0] == 1.0 and signs[0, 1] == -1.0  # qubit 0 set in b=1
        assert signs[1, 1] == 1.0 and signs[1, 2] == -1.0  # qubit 1 set in b=2

    def test_torch_forward_matches_qiskit_to_1e_minus_10(self):
        """The honesty gate: torch <Z_i> == Qiskit Statevector.evolve, all depths + modes."""
        cz, zs = _tensors()
        check = _validate_ansatz(n_qubits=E2_N_QUBITS, cz_phase=cz, z_signs=zs, seed=SEED)
        assert check["passed"] is True
        assert check["max_abs_deviation"] <= check["tolerance"] == 1e-10
        # depths x {entangle, no-entangle} x samples per case.
        assert check["n_checks"] == len(VQC_DEPTHS) * 2 * e3vqc.ANSATZ_AER_SAMPLES

    def test_forward_matches_qiskit_on_an_independent_random_case(self):
        cz, zs = _tensors()
        rng = np.random.default_rng(7)
        state = _unit_states(1, seed=11)[0]
        thetas = rng.uniform(-np.pi, np.pi, size=(2, E2_N_QUBITS))
        with torch.no_grad():
            _, local_z = _vqc_logits(
                torch.tensor(state[None, :], dtype=torch.float64),
                torch.tensor(thetas, dtype=torch.float64),
                torch.zeros(E2_N_QUBITS, dtype=torch.float64),
                torch.zeros((), dtype=torch.float64),
                cz, zs, n_qubits=E2_N_QUBITS, entangle=True,
            )
        ref = _qiskit_local_z(state, thetas, E2_N_QUBITS, entangle=True)
        assert np.max(np.abs(local_z.numpy().ravel() - ref)) <= 1e-10

    def test_local_z_is_bounded_in_minus_one_one(self):
        cz, zs = _tensors()
        states = torch.tensor(_unit_states(9, seed=3), dtype=torch.float64)
        thetas = torch.tensor(np.random.default_rng(3).uniform(-np.pi, np.pi, (2, E2_N_QUBITS)), dtype=torch.float64)
        with torch.no_grad():
            _, local_z = _vqc_logits(
                states, thetas, torch.zeros(E2_N_QUBITS, dtype=torch.float64),
                torch.zeros((), dtype=torch.float64), cz, zs, n_qubits=E2_N_QUBITS, entangle=True,
            )
        assert bool((local_z.abs() <= 1.0 + 1e-9).all())

    def test_final_entangler_is_invisible_to_diagonal_readout_at_depth_one(self):
        """A CZ after the last Ry cannot change any <Z_q> (diagonal), so depth-1 entangle == not."""
        cz, zs = _tensors()
        states = torch.tensor(_unit_states(5, seed=5), dtype=torch.float64)
        thetas = torch.tensor(np.random.default_rng(5).uniform(-np.pi, np.pi, (1, E2_N_QUBITS)), dtype=torch.float64)
        with torch.no_grad():
            _, z_ent = _vqc_logits(states, thetas, torch.zeros(E2_N_QUBITS, dtype=torch.float64),
                                   torch.zeros((), dtype=torch.float64), cz, zs, n_qubits=E2_N_QUBITS, entangle=True)
            _, z_no = _vqc_logits(states, thetas, torch.zeros(E2_N_QUBITS, dtype=torch.float64),
                                  torch.zeros((), dtype=torch.float64), cz, zs, n_qubits=E2_N_QUBITS, entangle=False)
        assert np.allclose(z_ent.numpy(), z_no.numpy(), atol=1e-12)

    def test_real_states_accepts_the_real_e2_state_and_rejects_a_complex_one(self):
        # The genuine E2 statevector is real up to floating-point noise -> accepted as float64.
        x, _, _ = _synthetic(n_patients=6, seed=2)
        pre = QuantumVisualPreprocessor.fit(x, seed=SEED)
        states = QuantumVisualFeatureMap().statevectors(pre.to_angles(x[:4]))
        real = _real_states(states)
        assert real.dtype == np.float64 and real.shape == (4, 1 << E2_N_QUBITS)
        # A genuinely complex state must be refused -- the real shortcut would corrupt it.
        bad = np.zeros((1, 1 << E2_N_QUBITS), dtype=complex)
        bad[0, 0] = 0.6
        bad[0, 1] = 0.8j
        with pytest.raises(TrainableVQCError):
            _real_states(bad)


# --------------------------------------------------------- barren-plateau witness
class TestBarrenPlateauWitness:
    def test_witness_reports_a_healthy_non_vanishing_gradient(self):
        cz, zs = _tensors()
        states = _unit_states(40, seed=8)
        wit = _gradient_variance_witness(states, n_qubits=E2_N_QUBITS, depth=2, entangle=True,
                                         cz_phase=cz, z_signs=zs, seed=SEED)
        assert wit["gradient_variance"] > 1e-6
        assert wit["vanishing"] is False
        assert {"n_inits", "n_probe_states", "gradient_variance", "gradient_abs_mean", "observable"} <= set(wit)

    def test_witness_is_deterministic(self):
        cz, zs = _tensors()
        states = _unit_states(30, seed=9)
        a = _gradient_variance_witness(states, n_qubits=E2_N_QUBITS, depth=1, entangle=True, cz_phase=cz, z_signs=zs, seed=SEED)
        b = _gradient_variance_witness(states, n_qubits=E2_N_QUBITS, depth=1, entangle=True, cz_phase=cz, z_signs=zs, seed=SEED)
        assert a["gradient_variance"] == b["gradient_variance"]


# ------------------------------------------------------------- training dynamics
class TestTrainingDynamics:
    def _separable(self, n: int = 100, seed: int = 1):
        rng = np.random.default_rng(seed)
        st = rng.standard_normal((n, 1 << E2_N_QUBITS))
        st /= np.linalg.norm(st, axis=1, keepdims=True)
        w = rng.standard_normal(1 << E2_N_QUBITS)
        y = ((st @ w) > np.median(st @ w)).astype(int)
        return st, y

    def test_vqc_training_reduces_loss(self):
        cz, zs = _tensors()
        st, y = self._separable()
        fit = _train_vqc(st, y, n_qubits=E2_N_QUBITS, depth=2, entangle=True,
                         cz_phase=cz, z_signs=zs, seed=SEED, steps=200)
        assert fit["final_loss"] < fit["initial_loss"]

    def test_vqc_training_is_deterministic(self):
        cz, zs = _tensors()
        st, y = self._separable(seed=2)
        f1 = _train_vqc(st, y, n_qubits=E2_N_QUBITS, depth=1, entangle=True, cz_phase=cz, z_signs=zs, seed=SEED, steps=120)
        f2 = _train_vqc(st, y, n_qubits=E2_N_QUBITS, depth=1, entangle=True, cz_phase=cz, z_signs=zs, seed=SEED, steps=120)
        assert np.allclose(f1["scorer"](st), f2["scorer"](st), atol=0.0)

    def test_mlp_training_reduces_loss_and_is_deterministic(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal((80, PCA_COMPONENTS))
        y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
        f1 = _train_mlp(x, y, n_in=PCA_COMPONENTS, hidden=2, seed=SEED, steps=200)
        f2 = _train_mlp(x, y, n_in=PCA_COMPONENTS, hidden=2, seed=SEED, steps=200)
        assert f1["final_loss"] < f1["initial_loss"]
        assert np.allclose(f1["scorer"](x), f2["scorer"](x), atol=0.0)


# ------------------------------------------------------ matched-control sizing
class TestMatchedControls:
    def test_mlp_parameter_count_formula(self):
        # Linear(8,h) + Linear(h,1): 8h + h (bias) + h + 1.
        assert _mlp_parameter_count(8, 2) == 8 * 2 + 2 + 2 + 1
        assert _mlp_parameter_count(8, 1) == 8 + 1 + 1 + 1

    def test_matched_hidden_is_capacity_comparable(self):
        # VQC depth-1 has 8*1 + 8 + 1 = 17 trainable params; the matched MLP should be close.
        vqc_params = 8 * 1 + E2_N_QUBITS + 1
        hidden = _matched_hidden(vqc_params, PCA_COMPONENTS)
        assert hidden >= 1
        assert abs(_mlp_parameter_count(PCA_COMPONENTS, hidden) - vqc_params) <= 8


# ---------------------------------------------------- fold-honest, no leakage
class TestFoldHonestEncodings:
    def test_encodings_never_fit_on_a_holdout_row(self, monkeypatch):
        """Each per-fold ``QuantumVisualPreprocessor.fit`` sees only its fold-train rows.

        A spy records the exact rows every ``fit`` is given. ``_build_fold_encodings`` fits
        once per fold (in fold order) then once on all TRAIN for the validation column -- so
        the recorded fit-sets must be each fold's train rows (disjoint from its holdout) then
        all TRAIN. A holdout row in any fit-set would be the PCA/angle leakage the mission
        forbids, and it would poison every state and PCA vector built from that fold.
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

        monkeypatch.setattr(e3vqc, "QuantumVisualPreprocessor", SpyPre)
        enc = _build_fold_encodings(fmap, x, x_val, folds, seed=SEED)

        assert len(captured) == len(folds) + 1, "one fit per fold, plus one on all-train"

        def rowset(a: np.ndarray) -> set:
            return {row.tobytes() for row in np.asarray(a, dtype=np.float64)}

        for k, (train_idx, test_idx) in enumerate(folds):
            fit_rows = rowset(captured[k])
            assert fit_rows == rowset(x[train_idx])
            assert fit_rows.isdisjoint(rowset(x[test_idx]))
        assert rowset(captured[-1]) == rowset(x)

        # Encodings are real float64 and correctly shaped.
        assert enc.states_all_train.shape == (x.shape[0], 1 << E2_N_QUBITS)
        assert enc.states_validation.shape == (x_val.shape[0], 1 << E2_N_QUBITS)
        assert enc.states_all_train.dtype == np.float64
        assert enc.pca_all_train.shape == (x.shape[0], PCA_COMPONENTS)
        assert enc.pca_validation.shape == (x_val.shape[0], PCA_COMPONENTS)

    def test_oof_trains_on_fold_train_only_and_scores_the_holdout(self):
        """``_oof_and_validation`` trains per fold on fold-train states and scores the holdout."""
        x, y, patients = _synthetic(seed=3)
        folds, _ = grouped_folds(y, patients, seed=SEED, n_splits=N_FOLDS)
        # Distinguishable state stand-ins: row i -> the constant-i vector, so we can identify
        # exactly which rows any arm was trained on without running a real circuit.
        base = np.arange(x.shape[0], dtype=np.float64)[:, None] * np.ones((1, 4))
        fold_states = [(tr, te, base[tr], base[te]) for tr, te in folds]
        all_train = base
        validation = np.full((6, 4), -1.0)

        trained_on: list[set] = []

        def spy_train_fn(states, ysub):
            trained_on.append({float(r[0]) for r in np.asarray(states)})
            return {"scorer": lambda s: np.asarray(s)[:, 0].astype(float),
                    "final_loss": 0.0, "initial_loss": 1.0}

        oof, oof_ap, val_score, note = _oof_and_validation(
            fold_states, all_train, validation, y, train_fn=spy_train_fn
        )
        # Per fold, the training set equals the fold-train row ids and excludes the holdout's.
        for k, (tr, te) in enumerate(folds):
            assert trained_on[k] == set(tr.astype(float))
            assert trained_on[k].isdisjoint(set(te.astype(float)))
        # The scorer wrote each holdout row's own id into the OOF column (leakage-free identity).
        assert np.allclose(oof, np.arange(x.shape[0]))
        assert trained_on[-1] == set(np.arange(x.shape[0]).astype(float))  # all-train column


# --------------------------------------------------------------- emitted payload
class TestEmittedPayload:
    def test_c7_anchor_reproduces_the_baseline(self, payload: dict):
        anchor = payload["c7_anchor"]
        assert anchor["reproduced"] is True
        assert anchor["reproduced_pr_auc"] == C7_REFERENCE_PR_AUC

    def test_test_partition_is_declared_unused(self, payload: dict):
        assert payload["test_partition_used"] is False
        assert payload["primary_condition"] == PRIMARY

    def test_aer_and_ansatz_gates_passed(self, payload: dict):
        aer = payload["quantum_feature_map_aer_check"]
        assert aer["passed"] is True and aer["max_abs_deviation"] <= aer["tolerance"]
        ans = payload["trainable_ansatz_qiskit_check"]
        assert ans["passed"] is True and ans["max_abs_deviation"] <= ans["tolerance"] == 1e-10

    def test_four_arms_present_at_expected_dimensions(self, payload: dict):
        arms = payload["arms"]
        assert set(arms) == {"c7", "trainable_vqc", "trainable_vqc_no_entangler", "classical_mlp"}
        assert arms["c7"]["representation_dim"] == C7_DIM
        assert arms["trainable_vqc"]["representation_dim"] == E2_N_QUBITS  # 8 local <Z_i>
        assert arms["classical_mlp"]["representation_dim"] == PCA_COMPONENTS
        for arm in arms.values():
            boot = arm["validation_bootstrap"]
            assert boot["p2_5"] <= boot["p97_5"]

    def test_vqc_carries_a_barren_plateau_witness_and_depth(self, payload: dict):
        vqc = payload["arms"]["trainable_vqc"]
        assert vqc["selected_depth"] in VQC_DEPTHS
        wit = vqc["barren_plateau_witness"]
        assert {"gradient_variance", "gradient_abs_mean", "vanishing"} <= set(wit)
        # The recorded 'barren' verdict must agree with the witness it is derived from.
        assert payload["summary"]["trainable_vqc_barren_plateau"] == bool(wit["vanishing"])

    def test_sqrt_t_over_n_is_reported_and_bounded(self, payload: dict):
        vqc = payload["arms"]["trainable_vqc"]
        assert vqc["n_quantum_trainable_parameters"] == E2_N_QUBITS * vqc["selected_depth"]
        assert 0.0 < vqc["sqrt_t_over_n"] < 1.0  # a genuinely shallow, bounded circuit

    def test_increment_tests_present_for_vqc_and_mlp(self, payload: dict):
        inc = payload["increment_vs_c7"]
        assert set(inc) == {"trainable_vqc", "classical_mlp"}
        for block in inc.values():
            assert block["reference_only_pr_auc"] == C7_REFERENCE_PR_AUC
            assert isinstance(block["survives_gating_null"], bool)

    def test_verdict_is_internally_consistent_with_the_gating_null(self, payload: dict):
        summary = payload["summary"]
        vqc_survives = payload["increment_vs_c7"]["trainable_vqc"]["survives_gating_null"]
        mlp_survives = payload["increment_vs_c7"]["classical_mlp"]["survives_gating_null"]
        assert summary["trainable_vqc_increment_survives_gating_null"] == vqc_survives
        assert summary["classical_mlp_increment_survives_gating_null"] == mlp_survives
        assert summary["defensible_quantum_advantage_over_c7"] == vqc_survives
        # A quantum-specific claim needs the VQC to survive, the MLP not to, and entanglement to help.
        if summary["quantum_specific_advantage"]:
            assert vqc_survives and not mlp_survives and summary["trainable_vqc_beats_no_entangler"]

    def test_both_matched_controls_reported_beside_quantum(self, payload: dict):
        assert "classical_mlp" in payload["arms"]
        assert "trainable_vqc_no_entangler" in payload["arms"]
        assert "matched_control" in payload["protocol"] and "ablation_control" in payload["protocol"]

    def test_leaderboard_is_sorted_by_validation_pr_auc(self, payload: dict):
        board = payload["summary"]["leaderboard"]
        scores = [r["validation_pr_auc"] for r in board if r["validation_pr_auc"] is not None]
        assert scores == sorted(scores, reverse=True)
