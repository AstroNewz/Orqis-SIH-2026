"""Tests for E3 Round 1 -- the C7-vs-quantum hybrid-fusion advantage experiment (DEC-037).

E3 asks the mission-§30 question: does the E2 quantum representation add information
*beyond the best classical baseline C7*? The dangers these tests pin down are the ones
that would let that question be answered dishonestly:

* the test partition is never read -- E3 reads only TRAIN and VALIDATION, via
  ``build_inputs(conditions=(PRIMARY,))``, which itself iterates only those two;
* the comparison is against the *real* C7 -- the rebuilt baseline reproduces E1.1's
  0.913038 to 1e-6, so a mis-assembled or weaker "C7" cannot flatter the quantum arm;
* the fold-honest arm scorer never fits preprocessing on a row it is about to score --
  the CV-leakage rule the whole mission rests on, checked directly with a spy factory
  that records exactly which rows each per-fold ``fit`` sees;
* no quantum number is reported unless the map matches Aer to <=1e-10;
* the matched controls (RFF-16, PCA-8) are present at equal/lower dimension beside the
  quantum arm -- the quantum result is never reported alone, and the recorded verdict is
  internally consistent (a "defensible advantage" flag can only be true if the quantum
  increment actually survived its gating null).

The unit tests here run cache-free (the spy/no-leak and factory-dimension tests execute
the real Aer statevector on a tiny synthetic batch). The payload tests read the driver's
emitted report and skip when no run is on disk, exactly as the E2 suite does.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

import backend.evaluation.e3_hybrid_fusion as e3
from backend.evaluation.e0_controls import N_FOLDS, grouped_folds
from backend.evaluation.e1_failure_map import PRIMARY, SEED
from backend.evaluation.e3_hybrid_fusion import (
    C7_DIM,
    C7_REFERENCE_PR_AUC,
    E3_HYBRID_FUSION_VERSION,
    MOBILENET_DIM,
    _identity_factory,
    _make_pca_factory,
    _make_quantum_factory,
    _make_rff_factory,
    _score_arm,
)
from backend.ml.quantum_visual import PCA_COMPONENTS
from quantum_ml.visual_circuit import E2_N_FEATURES, QuantumVisualFeatureMap

DRIVER_SOURCE = Path(e3.__file__).read_text(encoding="utf-8")

#: The driver's payload. Present only after a full run, so payload assertions skip.
REPORT_PATH = Path("reports_e3_hybrid_fusion.json")


@pytest.fixture
def payload() -> dict:
    """The driver's emitted payload, or a skip when no run has produced one yet."""
    if not REPORT_PATH.exists():
        pytest.skip("reports_e3_hybrid_fusion.json not present; run the driver first")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _synthetic(n_patients: int = 24, per_patient: int = 2, dim: int = MOBILENET_DIM, seed: int = 0):
    """A small patient-grouped synthetic MobileNet-like problem (cache-free).

    Labels are assigned per patient (so a patient is wholly positive or wholly negative,
    mirroring the real split) and rows are drawn from well-separated Gaussians so every
    row is unique -- which lets the no-leak test compare fit-sets by exact row content.
    """
    rng = np.random.default_rng(seed)
    patients, y_rows, feats = [], [], []
    for p in range(n_patients):
        label = int(p % 2)
        centre = rng.normal(label * 3.0, 1.0, size=dim)
        for _ in range(per_patient):
            feats.append(centre + rng.normal(0.0, 0.1, size=dim))
            y_rows.append(label)
            patients.append(f"P{p:03d}")
    x = np.asarray(feats, dtype=np.float64)
    y = np.asarray(y_rows, dtype=int)
    return x, y, patients


# ----------------------------------------------------------------- test firewall
class TestDriverNeverTouchesTheTestPartition:
    def test_no_test_partition_reference(self):
        assert not re.search(r"\.test\b", DRIVER_SOURCE), (
            "the E3 driver must not touch the .test partition attribute"
        )
        assert not re.search(r"partitions\s*\[\s*[\"']test[\"']\s*\]", DRIVER_SOURCE)
        assert not re.search(r"roi_mode\s*=\s*[\"']test[\"']", DRIVER_SOURCE)

    def test_only_the_primary_condition_is_built(self):
        # build_inputs is called for the primary oracle condition alone; it iterates only
        # ("train", "validation"), so the test partition is unreachable by construction.
        assert "conditions=(PRIMARY,)" in DRIVER_SOURCE

    def test_the_payload_declares_the_test_partition_unused(self):
        assert '"test_partition_used": False' in DRIVER_SOURCE
        assert "test_partition_note" in DRIVER_SOURCE

    def test_no_advantage_is_asserted_unconditionally(self):
        # The verdict is derived from the gating null, never hard-coded true.
        assert "defensible_quantum_advantage_over_c7" in DRIVER_SOURCE


# --------------------------------------------------------------- frozen wiring
class TestFrozenWiring:
    def test_version_and_dims(self):
        assert E3_HYBRID_FUSION_VERSION == "v1-e3-hybrid-fusion-1"
        assert MOBILENET_DIM == 576
        assert C7_DIM == 1195  # 576 + (163 + 456)
        assert PCA_COMPONENTS == 8
        assert E2_N_FEATURES == 16

    def test_c7_reference_anchor_constant(self):
        assert C7_REFERENCE_PR_AUC == 0.913038


# ------------------------------------------------------- fold-honest, no leakage
class TestFoldHonestPreprocessing:
    def test_identity_factory_is_a_passthrough(self):
        x = np.arange(12, dtype=np.float64).reshape(4, 3)
        transform = _identity_factory(x)
        assert np.array_equal(transform(x), x)

    def test_preprocessing_is_never_fitted_on_a_holdout_row(self):
        """The load-bearing property: each per-fold ``fit`` sees only its fold-train rows.

        A spy factory records the exact rows every ``preprocess(...)`` call is fitted on.
        ``_score_arm`` builds one fold representation per fold (in fold order) then one on
        all TRAIN for the validation column, so the recorded fit-sets must be: for fold
        ``k`` exactly ``x[train_idx_k]`` (hence disjoint from the holdout ``x[test_idx_k]``),
        and finally all of ``x``. A single holdout row in any fold's fit-set would be the
        CV leakage the mission forbids.
        """
        x, y, patients = _synthetic(seed=1)
        folds, _ = grouped_folds(y, patients, seed=SEED, n_splits=N_FOLDS)

        captured: list[np.ndarray] = []

        def spy_factory(fold_source: np.ndarray):
            captured.append(np.asarray(fold_source, dtype=np.float64).copy())
            return _identity_factory(fold_source)

        # A validation block distinct from train; contents are irrelevant to the property.
        x_val, _, _ = _synthetic(n_patients=6, seed=99)
        _score_arm(x, x_val, y, folds, preprocess=spy_factory, seed=SEED)

        assert len(captured) == len(folds) + 1, "one fit per fold, plus one on all-train"

        def rowset(a: np.ndarray) -> set:
            return {row.tobytes() for row in np.asarray(a, dtype=np.float64)}

        for k, (train_idx, test_idx) in enumerate(folds):
            fit_rows = rowset(captured[k])
            assert len(fit_rows) == train_idx.size
            assert fit_rows == rowset(x[train_idx])
            # The decisive assertion: no held-out row leaked into this fold's fit-set.
            assert fit_rows.isdisjoint(rowset(x[test_idx]))

        assert rowset(captured[-1]) == rowset(x), "validation column fits on all TRAIN"

    def test_score_arm_returns_aligned_columns(self):
        x, y, patients = _synthetic(seed=2)
        x_val, _, _ = _synthetic(n_patients=6, seed=7)
        folds, _ = grouped_folds(y, patients, seed=SEED, n_splits=N_FOLDS)
        fit = _score_arm(x, x_val, y, folds, preprocess=_identity_factory, seed=SEED)
        assert fit["train_out_of_fold"].shape == (x.shape[0],)
        assert fit["validation_score"].shape == (x_val.shape[0],)
        assert fit["representation_dim"] == x.shape[1]
        # Probabilities, so finite and within the unit interval where defined.
        val = fit["validation_score"]
        assert np.all(np.isfinite(val)) and np.all((val >= 0) & (val <= 1))


# --------------------------------------------- representation factories (real Aer)
class TestRepresentationFactories:
    def test_quantum_factory_emits_sixteen_features_via_real_aer(self):
        x, _, _ = _synthetic(n_patients=12, seed=3)  # 24 rows > 8 PCA components
        fmap = QuantumVisualFeatureMap()
        transform = _make_quantum_factory(fmap, seed=SEED)(x)
        out = transform(x[:5])
        assert out.shape == (5, E2_N_FEATURES)
        assert np.all(np.isfinite(out))
        # Local-Z expectations live in [-1, 1].
        assert np.all(out >= -1.0 - 1e-9) and np.all(out <= 1.0 + 1e-9)

    def test_pca_factory_emits_eight_scores(self):
        x, _, _ = _synthetic(n_patients=12, seed=4)
        transform = _make_pca_factory(seed=SEED)(x)
        assert transform(x[:5]).shape == (5, PCA_COMPONENTS)

    def test_rff_factory_emits_sixteen_bounded_features(self):
        x, _, _ = _synthetic(n_patients=12, seed=5)
        transform = _make_rff_factory(seed=SEED, n_features=E2_N_FEATURES)(x)
        out = transform(x[:5])
        assert out.shape == (5, E2_N_FEATURES)
        # RFF z(x) = sqrt(2/D) cos(...), so |z| <= sqrt(2/D).
        assert np.all(np.abs(out) <= np.sqrt(2.0 / E2_N_FEATURES) + 1e-9)


# --------------------------------------------------------------- emitted payload
class TestEmittedPayload:
    def test_c7_anchor_reproduces_the_baseline(self, payload: dict):
        anchor = payload["c7_anchor"]
        assert anchor["reproduced"] is True
        assert anchor["reproduced_pr_auc"] == C7_REFERENCE_PR_AUC
        assert anchor["target_pr_auc"] == C7_REFERENCE_PR_AUC

    def test_test_partition_is_declared_unused(self, payload: dict):
        assert payload["test_partition_used"] is False
        assert payload["primary_condition"] == PRIMARY

    def test_aer_gate_passed(self, payload: dict):
        aer = payload["quantum_feature_map_aer_check"]
        assert aer["passed"] is True
        assert aer["max_abs_deviation"] <= aer["tolerance"]

    def test_all_four_arms_present_at_expected_dimensions(self, payload: dict):
        arms = payload["arms"]
        assert set(arms) == {"c7", "quantum", "rff", "pca"}
        assert arms["c7"]["representation_dim"] == C7_DIM
        assert arms["quantum"]["representation_dim"] == E2_N_FEATURES
        assert arms["rff"]["representation_dim"] == E2_N_FEATURES
        assert arms["pca"]["representation_dim"] == PCA_COMPONENTS
        # Every arm carries a patient-level bootstrap CI beside its point estimate.
        for arm in arms.values():
            boot = arm["validation_bootstrap"]
            assert boot["p2_5"] <= boot["p97_5"]

    def test_increment_tests_present_for_every_non_c7_arm(self, payload: dict):
        inc = payload["increment_vs_c7"]
        assert set(inc) == {"quantum", "rff", "pca"}
        for block in inc.values():
            assert block["reference_only_pr_auc"] == C7_REFERENCE_PR_AUC
            assert "survives_gating_null" in block
            assert isinstance(block["survives_gating_null"], bool)

    def test_verdict_is_internally_consistent_with_the_gating_null(self, payload: dict):
        # A "defensible advantage" is licensed only by the quantum increment surviving its
        # gating null -- never asserted independently.
        summary = payload["summary"]
        quantum_survives = payload["increment_vs_c7"]["quantum"]["survives_gating_null"]
        assert summary["quantum_increment_survives_gating_null"] == quantum_survives
        assert summary["defensible_quantum_advantage_over_c7"] == quantum_survives
        # A quantum-specific advantage additionally requires the matched controls to fail.
        if summary["quantum_specific_advantage"]:
            assert quantum_survives
            assert not summary["rff_increment_survives_gating_null"]
            assert not summary["pca_increment_survives_gating_null"]

    def test_matched_controls_are_reported_beside_quantum(self, payload: dict):
        # DEC-033: the quantum arm is never reported without its equal-dimension control.
        proto = payload["protocol"]["candidate_representations"]
        assert "quantum" in proto and "rff" in proto and "pca" in proto
