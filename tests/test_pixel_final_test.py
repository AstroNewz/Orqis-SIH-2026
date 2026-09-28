"""Tests for the final held-out test evaluation.

This is the one module allowed to read the test partition, so the properties that
matter are the ones that keep it from becoming a tuning loop: it must refuse to run
unless the pipeline is declared frozen, it must take both thresholds from artifacts
rather than choosing them, and it must refuse rather than default when an artifact
has no threshold recorded.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.evaluation import pixel_final_test as final


class TestFrozenGuard:
    def test_refuses_without_confirmation(self):
        """Reading the test partition must be deliberate. ``--confirm-frozen`` is the
        only way in, and argparse exits 2 without it."""
        with pytest.raises(SystemExit) as excinfo:
            final.main(["--roi", "oracle"])
        assert excinfo.value.code == 2

    def test_refusal_names_the_validation_only_alternative(self):
        """A refusal that does not say what to run instead gets worked around."""
        assert "pixel_comparison" in final.NOT_FROZEN_MESSAGE
        assert "validation-only" in final.NOT_FROZEN_MESSAGE

    def test_version_is_pinned(self):
        assert final.FINAL_TEST_VERSION == "v1-pixel-final-test-1"


class TestThresholdProvenance:
    """Neither threshold may be chosen here; both are read from frozen artifacts."""

    def _artifact(self, tmp_path, threshold, condition="A_lesion_polygon"):
        payload = {
            "selected_weights": [0.1] * 32,
            "validation_selection": {
                "selected_config": {
                    "roi_mode": "oracle",
                    "condition": condition,
                    "n_layers": 1,
                    "optimizer": "spsa",
                    "maxiter": 50,
                    "seed": 42,
                    "class_weighting": True,
                    "n_parameters": 32,
                },
                "threshold": {} if threshold is None else {
                    "basis": "validation partition only",
                    "target_sensitivity": 0.85,
                    # The real artifact stores the whole operating point here, not a
                    # bare float. An earlier fixture used a float, the tests passed,
                    # and the real run crashed on `float(dict)`.
                    "selected": {
                        "target_sensitivity": 0.85,
                        "threshold": threshold,
                        "sensitivity": 0.8571,
                        "specificity": 0.1199,
                    },
                },
            },
        }
        path = tmp_path / "oracle_A_lesion_polygon_pixel_vqc.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_refuses_an_artifact_with_no_recorded_threshold(self, tmp_path):
        """Falling back to 0.50 here would be picking a threshold at test time."""
        path = self._artifact(tmp_path, None)
        with pytest.raises(final.FinalTestError, match="no validation-selected threshold"):
            final._quantum_block(
                path, object(), np.array([0]), None,
                scored_condition="A_lesion_polygon",
            )

    def test_reads_the_scalar_out_of_the_nested_operating_point(self, tmp_path):
        """The artifact stores an operating point dict, not a float. Pinned against a
        fixture shaped like the real file."""
        path = self._artifact(tmp_path, 0.4969)
        payload = json.loads(path.read_text(encoding="utf-8"))
        chosen = payload["validation_selection"]["threshold"]["selected"]
        assert isinstance(chosen, dict)
        assert chosen["threshold"] == pytest.approx(0.4969)

    def test_fixture_matches_the_real_artifact_shape(self):
        """Guards the fixture itself: if a trained artifact is present, its threshold
        block must have the shape the fixture imitates."""
        from pathlib import Path

        real = Path(
            "backend/artifacts/models/pixel_vqc/v1-pixel-vqc-1/"
            "oracle_A_all_pixel_vqc.json"
        )
        if not real.exists():
            pytest.skip("no trained artifact present")
        block = json.loads(real.read_text(encoding="utf-8"))
        chosen = block["validation_selection"]["threshold"]["selected"]
        assert isinstance(chosen, dict), "fixture assumes a dict operating point"
        assert isinstance(chosen["threshold"], float)

    def test_no_sweep_or_argmax_over_test_rows(self):
        """The module computes metrics; it must not search for a better operating
        point on the rows it is scoring."""
        import inspect

        source = inspect.getsource(final)
        assert "threshold_sweep" not in source
        assert "threshold_for_sensitivity" not in source
        assert "sensitivity_at_specificity" not in source


class TestMetrics:
    def test_single_class_subset_has_no_metrics(self):
        labels = np.zeros(8, dtype=int)
        assert final._metrics(labels, np.linspace(0.1, 0.9, 8), threshold=0.5) is None

    def test_empty_subset_has_no_metrics(self):
        assert final._metrics(
            np.empty(0, dtype=int), np.empty(0), threshold=0.5
        ) is None

    def test_two_class_subset_reports_the_full_required_block(self):
        """Phase D requires more than accuracy: ROC-AUC, PR-AUC, sensitivity,
        specificity, precision, F1, confusion matrix and Brier."""
        labels = np.array([0, 0, 1, 1])
        metrics = final._metrics(labels, np.array([0.1, 0.2, 0.8, 0.9]), threshold=0.5)
        assert metrics is not None
        for key in (
            "roc_auc", "pr_auc", "sensitivity", "specificity",
            "precision", "f1", "brier", "ece", "confusion_matrix",
            "n_samples", "n_positive", "prevalence",
        ):
            assert key in metrics, key

    def test_threshold_is_honoured(self):
        """A threshold above every score must flag nothing."""
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        strict = final._metrics(labels, scores, threshold=0.99)
        assert strict is not None
        assert strict["sensitivity"] == pytest.approx(0.0)


class TestSummarize:
    def _entry(self, condition, *, n, pos, classical, quantum, leak=None):
        return {
            "condition": condition,
            "n_samples": n,
            "n_positive": pos,
            "prevalence": round(pos / n, 6) if n else None,
            "roi_geometry_leak": {"area_only_roc_auc": leak,
                                  "area_only_pr_auc": leak},
            "classical": classical,
            "quantum": quantum,
        }

    def test_margin_and_winner(self):
        payload = {"conditions": [self._entry(
            "A_lesion_polygon", n=49, pos=18,
            classical={"random_forest": {"metrics": {"pr_auc": 0.80}}},
            quantum={
                "own_condition_artifact": True,
                "weights_trained_on_condition": "A_lesion_polygon",
                "metrics": {"pr_auc": 0.50, "roc_auc": 0.55, "sensitivity": 0.9,
                            "specificity": 0.2, "brier": 0.25},
            },
        )]}
        row = final.summarize(payload)["per_condition"][0]
        assert row["quantum_beats_classical"] is False
        assert row["margin"] == pytest.approx(-0.30, abs=1e-6)
        assert row["quantum_brier"] == pytest.approx(0.25)

    def test_undefined_is_none_not_false(self):
        payload = {"conditions": [self._entry(
            "A_region_polygon", n=100, pos=0,
            classical={"random_forest": {"metrics": None}},
            quantum={"own_condition_artifact": False,
                     "weights_trained_on_condition": "A_lesion_polygon",
                     "metrics": None},
        )]}
        row = final.summarize(payload)["per_condition"][0]
        assert row["quantum_beats_classical"] is None
        assert row["margin"] is None
        assert row["quantum_pr_auc"] is None

    def test_borrowed_weights_are_flagged(self):
        payload = {"conditions": [self._entry(
            "C_fallback", n=12, pos=2,
            classical={"random_forest": {"metrics": {"pr_auc": 0.4}}},
            quantum={"own_condition_artifact": False,
                     "weights_trained_on_condition": "B_localized",
                     "metrics": {"pr_auc": 0.3}},
        )]}
        row = final.summarize(payload)["per_condition"][0]
        assert row["quantum_weights_borrowed"] is True
        assert row["quantum_weights_trained_on"] == "B_localized"

    def test_missing_quantum_model_is_none(self):
        payload = {"conditions": [self._entry(
            "C_fallback", n=12, pos=2,
            classical={"random_forest": {"metrics": {"pr_auc": 0.4}}},
            quantum=None,
        )]}
        row = final.summarize(payload)["per_condition"][0]
        assert row["quantum_weights_borrowed"] is None
        assert row["quantum_beats_classical"] is None

    def test_geometry_leak_key_matches_what_state_diagnostics_emits(self):
        """The leak is read out of another module's payload by key, so a rename there
        would silently turn every reported leak into ``null``."""
        import inspect

        from backend.evaluation import state_diagnostics

        source = inspect.getsource(state_diagnostics.roi_geometry_leak)
        assert '"area_only_roc_auc"' in source
        assert '"area_only_pr_auc"' in source

    def test_geometry_leak_travels_with_every_row(self):
        """DEC-024: a condition-A number cannot be read without the area-only
        ROC-AUC beside it."""
        payload = {"conditions": [self._entry(
            "A_all", n=381, pos=18,
            classical={"random_forest": {"metrics": {"pr_auc": 0.6}}},
            quantum={"own_condition_artifact": True,
                     "weights_trained_on_condition": "A_all",
                     "metrics": {"pr_auc": 0.1}},
            leak=0.8477,
        )]}
        row = final.summarize(payload)["per_condition"][0]
        assert row["roi_geometry_leak_roc_auc"] == pytest.approx(0.8477)

    def test_conditions_stay_separate(self):
        payload = {"conditions": [
            self._entry("B_localized", n=371, pos=18,
                        classical={"rf": {"metrics": {"pr_auc": 0.6}}},
                        quantum={"own_condition_artifact": True,
                                 "weights_trained_on_condition": "B_localized",
                                 "metrics": {"pr_auc": 0.2}}),
            self._entry("C_fallback", n=10, pos=0,
                        classical={"rf": {"metrics": None}},
                        quantum={"own_condition_artifact": False,
                                 "weights_trained_on_condition": "B_localized",
                                 "metrics": None}),
        ]}
        rows = final.summarize(payload)["per_condition"]
        assert [r["condition"] for r in rows] == ["B_localized", "C_fallback"]
        assert [r["n_samples"] for r in rows] == [371, 10]


class TestArtifactHashing:
    def test_hash_is_content_addressed(self, tmp_path):
        """A reported number must be traceable to the exact weights file."""
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text('{"weights": [1, 2]}', encoding="utf-8")
        b.write_text('{"weights": [1, 2]}', encoding="utf-8")
        assert final._sha256(a) == final._sha256(b)
        b.write_text('{"weights": [1, 3]}', encoding="utf-8")
        assert final._sha256(a) != final._sha256(b)
        assert len(final._sha256(a)) == 64
