"""Tests for the row-aligned classical-vs-quantum comparison.

The comparison is where Phase D's headline claim is decided, so the properties worth
pinning are the ones that would silently produce a flattering number: merging ROI
conditions, comparing metrics computed on different row sets, or presenting borrowed
weights as if the condition had its own trained model.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.evaluation import pixel_comparison as cmp


def _entry(
    condition: str,
    *,
    n_samples: int,
    n_positive: int,
    classical: dict,
    quantum: dict | None,
) -> dict:
    return {
        "condition": condition,
        "n_samples": n_samples,
        "n_positive": n_positive,
        "prevalence": round(n_positive / n_samples, 6) if n_samples else None,
        "classical": classical,
        "quantum": quantum,
    }


def _quantum(pr_auc: float | None, *, trained_on: str, own: bool) -> dict:
    return {
        "weights_trained_on_condition": trained_on,
        "own_condition_artifact": own,
        "metrics": None if pr_auc is None else {"pr_auc": pr_auc},
    }


class TestPrimaryCondition:
    def test_both_roi_modes_have_a_primary(self):
        """Every mode the CLI accepts must name a condition to borrow from.

        Otherwise a fallback condition silently reports "no model to borrow" for a
        mode that in fact trained one.
        """
        assert set(cmp.PRIMARY_CONDITION) == {"oracle", "predicted"}

    def test_oracle_primary_is_the_annotated_condition(self):
        """Not ``A_all``: that pools annotated and unannotated rows, and the oracle
        ROI is only defined for the annotated ones."""
        assert cmp.PRIMARY_CONDITION["oracle"] == "A_lesion_polygon"

    def test_predicted_primary_is_the_localized_condition(self):
        assert cmp.PRIMARY_CONDITION["predicted"] == "B_localized"


class TestMetricsForSubset:
    def test_single_class_subset_has_no_metrics(self):
        """A subset with no positives has no defined ROC-AUC or PR-AUC. Returning a
        number here -- 0.0, or the prevalence -- would be inventing one."""
        labels = np.zeros(10, dtype=int)
        scores = np.linspace(0.1, 0.9, 10)
        assert cmp._metrics_for_subset(labels, scores) is None

    def test_all_positive_subset_has_no_metrics(self):
        labels = np.ones(10, dtype=int)
        assert cmp._metrics_for_subset(labels, np.linspace(0.1, 0.9, 10)) is None

    def test_empty_subset_has_no_metrics(self):
        assert cmp._metrics_for_subset(np.empty(0, dtype=int), np.empty(0)) is None

    def test_two_class_subset_reports_pr_auc(self):
        labels = np.array([0, 0, 1, 1])
        metrics = cmp._metrics_for_subset(labels, np.array([0.1, 0.2, 0.8, 0.9]))
        assert metrics is not None
        assert metrics["pr_auc"] == pytest.approx(1.0)


class TestVerdict:
    def test_margin_is_quantum_minus_classical(self):
        payload = {"conditions": [_entry(
            "A_lesion_polygon", n_samples=52, n_positive=21,
            classical={"random_forest": {"metrics": {"pr_auc": 0.790313}}},
            quantum=_quantum(0.526376, trained_on="A_lesion_polygon", own=True),
        )]}
        row = cmp._verdict(payload)["per_condition"][0]
        assert row["quantum_beats_classical"] is False
        assert row["margin"] == pytest.approx(-0.263937, abs=1e-6)

    def test_best_classical_is_the_strongest_baseline(self):
        payload = {"conditions": [_entry(
            "B_localized", n_samples=100, n_positive=20,
            classical={
                "logistic_regression": {"metrics": {"pr_auc": 0.41}},
                "random_forest": {"metrics": {"pr_auc": 0.63}},
                "svm": {"metrics": {"pr_auc": 0.55}},
            },
            quantum=_quantum(0.30, trained_on="B_localized", own=True),
        )]}
        row = cmp._verdict(payload)["per_condition"][0]
        assert row["best_classical"] == "random_forest"
        assert row["best_classical_pr_auc"] == pytest.approx(0.63)

    def test_undefined_comparison_is_none_not_false(self):
        """A condition with no positives cannot be won or lost. ``False`` would read
        as "the quantum model lost", which is a claim the data does not support."""
        payload = {"conditions": [_entry(
            "A_region_polygon", n_samples=185, n_positive=0,
            classical={"random_forest": {"metrics": None}},
            quantum=_quantum(None, trained_on="A_lesion_polygon", own=False),
        )]}
        row = cmp._verdict(payload)["per_condition"][0]
        assert row["quantum_beats_classical"] is None
        assert row["margin"] is None
        assert row["best_classical_pr_auc"] is None

    def test_missing_quantum_model_is_none_not_false(self):
        payload = {"conditions": [_entry(
            "C_fallback", n_samples=10, n_positive=3,
            classical={"random_forest": {"metrics": {"pr_auc": 0.5}}},
            quantum=None,
        )]}
        row = cmp._verdict(payload)["per_condition"][0]
        assert row["quantum_pr_auc"] is None
        assert row["quantum_beats_classical"] is None
        assert row["quantum_weights_borrowed"] is None
        assert row["quantum_weights_trained_on"] is None

    def test_borrowed_weights_are_flagged_in_the_verdict(self):
        """The fallback condition's number is produced by another condition's model.
        Reporting it without that flag would imply a model was selected on rows it
        never saw."""
        payload = {"conditions": [_entry(
            "C_fallback", n_samples=10, n_positive=3,
            classical={"random_forest": {"metrics": {"pr_auc": 0.5}}},
            quantum=_quantum(0.44, trained_on="B_localized", own=False),
        )]}
        row = cmp._verdict(payload)["per_condition"][0]
        assert row["quantum_weights_borrowed"] is True
        assert row["quantum_weights_trained_on"] == "B_localized"

    def test_own_artifact_is_not_flagged_as_borrowed(self):
        payload = {"conditions": [_entry(
            "B_localized", n_samples=100, n_positive=20,
            classical={"random_forest": {"metrics": {"pr_auc": 0.5}}},
            quantum=_quantum(0.44, trained_on="B_localized", own=True),
        )]}
        row = cmp._verdict(payload)["per_condition"][0]
        assert row["quantum_weights_borrowed"] is False

    def test_conditions_are_reported_separately(self):
        """A/B/C must never be collapsed into one ROI statistic. The verdict carries
        one row per condition with its own sample and positive counts."""
        payload = {"conditions": [
            _entry("B_localized", n_samples=353, n_positive=18,
                   classical={"rf": {"metrics": {"pr_auc": 0.6}}},
                   quantum=_quantum(0.2, trained_on="B_localized", own=True)),
            _entry("C_fallback", n_samples=10, n_positive=3,
                   classical={"rf": {"metrics": {"pr_auc": 0.4}}},
                   quantum=_quantum(0.3, trained_on="B_localized", own=False)),
        ]}
        rows = cmp._verdict(payload)["per_condition"]
        assert [r["condition"] for r in rows] == ["B_localized", "C_fallback"]
        assert [r["n_samples"] for r in rows] == [353, 10]
        assert [r["n_positive"] for r in rows] == [18, 3]

    def test_every_row_carries_its_own_prevalence(self):
        """PR-AUC is prevalence-sensitive, so a number without its prevalence cannot
        be compared against another condition's."""
        payload = {"conditions": [
            _entry("A_all", n_samples=363, n_positive=21,
                   classical={"rf": {"metrics": {"pr_auc": 0.63}}},
                   quantum=_quantum(0.09, trained_on="A_lesion_polygon", own=False)),
            _entry("A_lesion_polygon", n_samples=52, n_positive=21,
                   classical={"rf": {"metrics": {"pr_auc": 0.79}}},
                   quantum=_quantum(0.52, trained_on="A_lesion_polygon", own=True)),
        ]}
        rows = cmp._verdict(payload)["per_condition"]
        assert rows[0]["prevalence"] == pytest.approx(0.057851, abs=1e-6)
        assert rows[1]["prevalence"] == pytest.approx(0.403846, abs=1e-6)
        assert all(r["prevalence"] is not None for r in rows)


class TestNoTestPartition:
    def test_compare_never_names_the_test_partition(self):
        """The comparison reads ``data.validation`` and nothing else.

        Asserted on the source rather than by running ``compare``, which needs the
        full descriptor and pixel caches. A ``.test`` access appearing here is the
        failure this guards against.
        """
        import inspect

        source = inspect.getsource(cmp.compare)
        assert "data.validation" in source
        assert ".test" not in source.replace("test_partition_used", "")

    def test_payload_declares_the_partition_and_the_test_flag(self):
        import inspect

        source = inspect.getsource(cmp.compare)
        assert '"partition": "validation"' in source
        assert '"test_partition_used": False' in source

    def test_comparison_version_is_pinned(self):
        assert cmp.COMPARISON_VERSION == "v1-pixel-comparison-1"
