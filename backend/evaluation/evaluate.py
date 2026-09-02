"""Final held-out evaluation: the quantum model against the classical baselines.

This is the only module in the repository that reads the ``test`` partition. Every
training stage refuses to touch it and returns ``test_partition_used: False``;
this one returns ``True``.

Run it once
-----------
PART 15 requires the held-out set be used for final evaluation only. That is a
discipline, not a switch, and nothing in code can fully enforce it -- but re-running
this after changing a model *is* tuning on test, so a previous report is treated as
a finding: the script warns, records how many times the partition has been scored,
and keeps the earlier run's headline numbers in the new report so a drift is
visible rather than overwritten.

How the comparison is made fair
-------------------------------
Two traps make quantum-vs-classical comparisons meaningless, and both are avoided
here.

**Feature parity.** Every model is scored on the matrix produced by the *same*
persisted pipeline, loaded and never refitted. A difference in the numbers is a
difference in the models.

**Operating-point parity.** A threshold chosen on the test set inflates whichever
model it was chosen for. Each model therefore gets its threshold picked on the
**validation** partition for the configured target sensitivity, and that threshold
is then applied unchanged to test. Metrics are also reported at the configured
``SCREENING_THRESHOLD`` for reference, but that value is not a per-model operating
point and is not what the ranking uses.

What the calibration metrics do and do not compare
--------------------------------------------------
The VQC's probabilities pass through the fitted calibrator. The baselines' come
from scikit-learn under ``class_weight="balanced"``, which deliberately distorts
them away from the observed base rate -- balanced weighting is what makes the
baselines learn the minority class at all, but it means their Brier and ECE are
not measuring the same thing as the VQC's. Ranking is therefore on PR-AUC, which
is rank-based and unaffected. The calibration columns are reported per model with
that caveat attached rather than silently compared.

Usage::

    python -m backend.evaluation.evaluate
    python -m backend.evaluation.evaluate --model-version v1-handcrafted
    python -m backend.evaluation.evaluate --mode noisy_simulation
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.evaluation.metrics import (
    ClassificationMetrics,
    compare_models,
    evaluate_predictions,
    sensitivity_at_specificity,
    threshold_for_sensitivity,
    threshold_sweep,
)
from backend.ml import baselines as baselines_module
from backend.ml.artifacts import CALIBRATION_FILE, QUANTUM_FILE, ArtifactStore
from backend.training.data import load_partitions
from backend.training.train_classical import BASELINES_FILE
from backend.training.train_quantum import _configure_logging
from quantum_ml.backends import build_backend
from quantum_ml.calibration import ProbabilityCalibrator, ScoreCalibrator
from quantum_ml.results import ExecutionMode
from quantum_ml.vqc_classifier import VariationalQuantumClassifier

logger = logging.getLogger(__name__)

EVALUATION_FILE = "evaluation.json"
QUANTUM_MODEL_NAME = "quantum_vqc_calibrated"


def _risk_band_reachability(
    calibrator: ScoreCalibrator,
    thresholds: ProbabilityCalibrator,
    observed_scores: np.ndarray,
) -> Dict[str, Any]:
    """Check whether the configured risk bands can actually be reached.

    A steep Platt fit compresses most of the raw range towards zero. If the highest
    calibrated probability the model can emit sits below ``HIGH_RISK_THRESHOLD``,
    the HIGH band is dead code: the API can never return it, and no amount of
    lesion severity will change that. Better to measure it here than to discover it
    from a clinician asking why nothing is ever flagged as high risk.
    """
    observed = np.asarray(observed_scores, dtype=np.float64)
    observed_calibrated = calibrator.transform(observed)
    # The raw score is (1 - <Z_0>)/2 and so is bounded in [0, 1] by construction;
    # the endpoints bound what the calibrator could ever emit.
    theoretical = calibrator.transform(np.array([0.0, 1.0], dtype=np.float64))
    observed_max = float(observed_calibrated.max())
    theoretical_max = float(theoretical.max())

    payload: Dict[str, Any] = {
        "screening_threshold": thresholds.threshold,
        "high_risk_threshold": thresholds.high_risk_threshold,
        "bands_source": thresholds.bands_source,
        "observed_raw_range": [float(observed.min()), float(observed.max())],
        "observed_calibrated_range": [
            float(observed_calibrated.min()),
            observed_max,
        ],
        "theoretical_calibrated_range": [
            float(theoretical.min()),
            theoretical_max,
        ],
        "screening_threshold_reachable_observed": observed_max >= thresholds.threshold,
        "high_risk_threshold_reachable_observed": observed_max
        >= thresholds.high_risk_threshold,
        "high_risk_threshold_reachable_theoretical": theoretical_max
        >= thresholds.high_risk_threshold,
    }
    if not payload["high_risk_threshold_reachable_observed"]:
        payload["note"] = (
            f"No score this model produces reaches HIGH_RISK_THRESHOLD="
            f"{thresholds.high_risk_threshold}: the highest calibrated probability "
            f"observed is {observed_max:.4f}. The HIGH risk band is unreachable in "
            "practice, so the band boundaries need to be set from this model's "
            "actual probability distribution rather than from round numbers."
        )
    return payload


def _operating_threshold(
    name: str,
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    target_sensitivity: float,
) -> Dict[str, Any]:
    """Pick ``name``'s operating point on validation, for use on test.

    Validation was also used to select baseline hyperparameters, so a threshold
    derived from it is mildly optimistic. It is still the correct choice: the
    alternative is deriving it from test, which is not mildly anything.
    """
    payload = dict(
        threshold_for_sensitivity(
            y_validation, validation_probabilities, target_sensitivity
        )
    )
    payload["model"] = name
    payload["chosen_on"] = "validation"
    payload["is_clinically_validated"] = False
    if payload["threshold"] is None:
        payload["threshold"] = 0.0
        payload["fallback"] = (
            "No threshold on validation reached the target sensitivity; flagging "
            "everything (threshold 0.0) is used so the test metrics are still "
            "computable, and the model should be read as not meeting the target."
        )
    return payload


def _score_baselines(
    baselines_payload: Dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    matrices: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, Any]]:
    """Rebuild each baseline from its record and score the given matrices.

    Logistic regression is restored from its persisted weights; the tree ensembles
    are refitted on the training partition from their records, which is exact
    because the estimators are constructed serially with a recorded seed. Refitting
    is not a shortcut around missing artifacts -- it is the no-pickle policy, and
    it doubles as a check that the record fully determines the model.
    """
    scored: Dict[str, Dict[str, Any]] = {}
    for name, entry in baselines_payload.get("baselines", {}).items():
        record = baselines_module.BaselineRecord.from_dict(entry["record"])
        if entry.get("weights"):
            model: Any = baselines_module.LogisticBaseline.from_dict(entry)
            restored_by = "persisted_weights"
        else:
            model = baselines_module.refit_from_record(record, x_train, y_train)
            restored_by = "refit_from_record"
        scored[name] = {
            "record": record,
            "restored_by": restored_by,
            "probabilities": {
                partition: model.predict_proba(matrix)
                for partition, matrix in matrices.items()
            },
        }
        logger.info(
            "  %-20s restored via %-18s hyperparameters %s",
            name,
            restored_by,
            json.dumps(record.hyperparameters, sort_keys=True),
        )
    return scored


def _previous_report(store: ArtifactStore, version: str) -> Optional[Dict[str, Any]]:
    try:
        return store.read_component(version, EVALUATION_FILE)
    except Exception:  # noqa: BLE001 - absence and unreadability are both "no prior run"
        return None


def evaluate(
    *,
    model_version: Optional[str] = None,
    extractor_name: str = "handcrafted_lab",
    target_sensitivity: Optional[float] = None,
    execution_mode: Optional[str] = None,
    include_sweep: bool = True,
    config: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Score the held-out test partition once and rank every model on it.

    Raises:
        ArtifactError: the version lacks a pipeline, a quantum model, or a calibrator.
    """
    cfg = config or default_settings
    resolved_target_sensitivity = (
        cfg.SCREENING_TARGET_SENSITIVITY if target_sensitivity is None else target_sensitivity
    )
    store = ArtifactStore.from_settings(cfg)
    version = store.resolve_version(model_version)

    previous = _previous_report(store, version)
    scored_count = 1 if previous is None else int(previous.get("times_scored", 1)) + 1
    if previous is not None:
        logger.warning(
            "The test partition has already been scored for %s (%s). This run makes "
            "it %d times. Re-evaluating after any model change is tuning on test; "
            "the earlier headline numbers are kept in the report for comparison.",
            version,
            previous.get("evaluated_at", "time not recorded"),
            scored_count,
        )

    pipeline = store.load_pipeline(version)
    quantum_payload = store.read_component(version, QUANTUM_FILE)
    calibration_payload = store.read_component(version, CALIBRATION_FILE)
    calibrator = ScoreCalibrator.from_dict(calibration_payload)
    baselines_payload = store.read_component(version, BASELINES_FILE)

    resolved_mode = ExecutionMode(execution_mode or cfg.QUANTUM_EXECUTION_MODE)
    backend = build_backend(
        mode=resolved_mode,
        shots=cfg.QUANTUM_SHOTS,
        seed=cfg.RANDOM_SEED,
        noise_device=cfg.QUANTUM_NOISE_MODEL,
        ibm_backend_name=cfg.IBMQ_BACKEND_NAME,
        ibm_api_key=cfg.IBMQ_API_KEY,
        ibm_instance=cfg.IBMQ_INSTANCE,
        ibm_channel=cfg.IBMQ_CHANNEL,
        allow_fallback=cfg.QUANTUM_ALLOW_HARDWARE_FALLBACK,
    )
    if backend.resolution.fell_back:
        logger.warning(
            "Requested %s but resolved to %s: %s",
            resolved_mode.value,
            backend.effective_mode.value,
            backend.resolution.fallback_reason,
        )
    calibrated_in_mode = calibration_payload.get("scored_in_mode")
    if calibrated_in_mode and calibrated_in_mode != backend.effective_mode.value:
        logger.warning(
            "The calibrator was fitted on %s scores but this evaluation runs in %s. "
            "The mapping was fitted to a different score distribution, so the "
            "calibrated probabilities below are not trustworthy. Re-run "
            "backend.training.calibrate --mode %s first.",
            calibrated_in_mode,
            backend.effective_mode.value,
            backend.effective_mode.value,
        )

    model = VariationalQuantumClassifier.from_dict(quantum_payload, backend=backend)
    dataset = load_partitions(extractor_name=extractor_name, config=cfg, pipeline=pipeline)
    x_train, y_train = dataset.xy("train")
    x_validation, y_validation = dataset.xy("validation")
    x_test, y_test = dataset.xy("test")
    logger.info(
        "Test partition: %d images, %d patients, %d positive (prevalence %.4f). "
        "Scoring in %s on %s.",
        dataset.test.n,
        dataset.test.n_patients,
        dataset.test.n_positive,
        dataset.test.prevalence,
        backend.effective_mode.value,
        backend.backend_name,
    )

    # --- quantum: raw scores, then the fitted calibrator ------------------------
    raw_validation = model.predict_proba(x_validation)
    raw_test = model.predict_proba(x_test)
    calibrated_validation = calibrator.transform(raw_validation)
    calibrated_test = calibrator.transform(raw_test)

    # --- baselines: rebuilt from their records, same feature matrices -----------
    logger.info("Restoring %d baselines:", len(baselines_payload.get("baselines", {})))
    baseline_scores = _score_baselines(
        baselines_payload,
        x_train,
        y_train,
        {"validation": x_validation, "test": x_test},
    )

    probabilities: Dict[str, Dict[str, np.ndarray]] = {
        QUANTUM_MODEL_NAME: {
            "validation": calibrated_validation,
            "test": calibrated_test,
        },
    }
    for name, entry in baseline_scores.items():
        probabilities[name] = entry["probabilities"]

    # The bands this model version actually ships with, taken from its calibration
    # artifact. Using the global fallback here would measure reachability against
    # numbers the inference path does not use.
    thresholds = ProbabilityCalibrator.from_calibration_payload(
        calibration_payload, config=cfg
    )
    logger.info(
        "Risk bands: screening %.4f, high-risk %.4f (source: %s).",
        thresholds.threshold,
        thresholds.high_risk_threshold,
        thresholds.bands_source,
    )
    operating_points: Dict[str, Dict[str, Any]] = {
        name: _operating_threshold(
            name, y_validation, per_partition["validation"], resolved_target_sensitivity
        )
        for name, per_partition in probabilities.items()
    }

    test_metrics: Dict[str, ClassificationMetrics] = {}
    at_configured: Dict[str, ClassificationMetrics] = {}
    for name, per_partition in probabilities.items():
        test_metrics[name] = evaluate_predictions(
            y_test, per_partition["test"], threshold=operating_points[name]["threshold"]
        )
        at_configured[name] = evaluate_predictions(
            y_test, per_partition["test"], threshold=thresholds.threshold
        )

    ranking = compare_models(test_metrics)
    logger.info(
        "\nHeld-out test results, each model at its own validation-chosen threshold "
        "for sensitivity >= %.2f:",
        resolved_target_sensitivity,
    )
    for row in ranking:
        logger.info("  %d. %-24s %s", row["rank"], row["model"], "")
        logger.info("     %s", test_metrics[row["model"]].summary_line())

    quantum_rank = next(
        row["rank"] for row in ranking if row["model"] == QUANTUM_MODEL_NAME
    )
    best_model = ranking[0]["model"]
    verdict = _verdict(quantum_rank, best_model, test_metrics, ranking)
    logger.info("\n%s", verdict["statement"])

    reachability = _risk_band_reachability(calibrator, thresholds, raw_test)
    if not reachability["high_risk_threshold_reachable_observed"]:
        logger.warning("%s", reachability["note"])

    report: Dict[str, Any] = {
        "model_version": version,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "times_scored": scored_count,
        "test_partition_used": True,
        "partition_summary": {
            name: dataset.partitions[name].summary()
            for name in ("train", "validation", "test")
        },
        "feature_representation": {
            "extractor": extractor_name,
            "feature_mode": pipeline.feature_mode,
            "n_reduced_features": int(x_test.shape[1]),
            "preprocessing_version": pipeline.preprocessing_version,
            "pipeline_version": pipeline.version,
            "note": (
                "Every model below was scored on this matrix, produced by the "
                "persisted pipeline and never refitted. Differences are model "
                "differences, not preprocessing differences."
            ),
        },
        "quantum": {
            "n_qubits": model.num_qubits,
            "n_layers": model.num_layers,
            "n_parameters": model.num_params,
            "ansatz_depth": model.ansatz_depth,
            "amplitude_dimension": 2**model.num_qubits,
            "execution_mode": backend.effective_mode.value,
            "backend_name": backend.backend_name,
            "shots": backend.effective_shots,
            "requested_mode": resolved_mode.value,
            "fell_back": backend.resolution.fell_back,
            "fallback_reason": backend.resolution.fallback_reason,
            "calibration_method": calibrator.method.value,
            "calibrator_fitted_on": calibrator.fitted_on,
            "calibrator_scored_in_mode": calibrated_in_mode,
            "raw_test_score_range": [
                round(float(raw_test.min()), 6),
                round(float(raw_test.max()), 6),
            ],
        },
        "target_sensitivity": resolved_target_sensitivity,
        "operating_points": operating_points,
        "test_metrics": {name: metrics.to_dict() for name, metrics in test_metrics.items()},
        "test_metrics_at_configured_threshold": {
            "threshold": thresholds.threshold,
            "bands_source": thresholds.bands_source,
            "note": (
                "The model's shipped screening threshold applied to every model. "
                "Reported for reference only: it is one number applied to models "
                "whose score distributions differ, so it is not a fair comparison "
                "and is not what the ranking uses."
            ),
            "metrics": {name: metrics.to_dict() for name, metrics in at_configured.items()},
        },
        "sensitivity_at_specificity_090": {
            name: sensitivity_at_specificity(y_test, per_partition["test"], 0.90)
            for name, per_partition in probabilities.items()
        },
        "ranking": ranking,
        "verdict": verdict,
        "risk_band_reachability": reachability,
        "calibration_comparability_caveat": (
            "Brier, log-loss and ECE are comparable across models only with care. "
            "The quantum probabilities are calibrated; the baselines' come from "
            "scikit-learn under class_weight='balanced', which shifts them away "
            "from the base rate by design. PR-AUC and ROC-AUC are rank-based and "
            "unaffected, which is why the ranking uses PR-AUC."
        ),
        "clinical_disclaimer": (
            "These are research metrics for an AI-assisted oral-cancer screening "
            "risk estimate on a single small dataset. They do not establish "
            "clinical validity, and no threshold here is a validated clinical "
            "cut-off. The system does not replace professional clinical assessment "
            "or histopathological confirmation."
        ),
    }
    if include_sweep:
        report["threshold_sweeps"] = {
            name: threshold_sweep(y_test, per_partition["test"])
            for name, per_partition in probabilities.items()
        }
    if previous is not None:
        report["previous_run"] = {
            "evaluated_at": previous.get("evaluated_at"),
            "ranking": [
                {key: row.get(key) for key in ("rank", "model", "pr_auc", "roc_auc")}
                for row in previous.get("ranking", [])
            ],
            "note": (
                "Kept so a change between test evaluations is visible. A model that "
                "improved here after being changed in response to an earlier test "
                "run has been tuned on the test set."
            ),
        }

    path = store.write_component(version, EVALUATION_FILE, report)
    store.update_metadata(
        version,
        {
            "evaluation": {
                "evaluated_at": report["evaluated_at"],
                "times_scored": scored_count,
                "execution_mode": backend.effective_mode.value,
                "best_model": best_model,
                "quantum_rank": quantum_rank,
                "quantum_test_pr_auc": test_metrics[QUANTUM_MODEL_NAME].pr_auc,
                "target_sensitivity": resolved_target_sensitivity,
            }
        },
    )
    logger.info("Report -> %s", path)

    return {
        "model_version": version,
        "artifact": str(path),
        "test_partition_used": True,
        "times_scored": scored_count,
        "test_partition": dataset.test.summary(),
        "execution_mode": backend.effective_mode.value,
        "target_sensitivity": resolved_target_sensitivity,
        "ranking": [
            {
                "rank": row["rank"],
                "model": row["model"],
                "threshold": row["threshold"],
                "pr_auc": row["pr_auc"],
                "roc_auc": row["roc_auc"],
                "sensitivity": row["sensitivity"],
                "specificity": row["specificity"],
                "precision": row["precision"],
                "brier": row["brier"],
            }
            for row in ranking
        ],
        "verdict": verdict,
        "risk_band_reachability": report["risk_band_reachability"],
        "clinical_disclaimer": report["clinical_disclaimer"],
    }


def _verdict(
    quantum_rank: int,
    best_model: str,
    test_metrics: Dict[str, ClassificationMetrics],
    ranking: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """State the quantum contribution as measured, in either direction.

    PART 10 requires evaluating the quantum contribution rather than assuming it.
    That cuts both ways, so this builds the sentence from the numbers instead of
    from a preferred conclusion -- and it refuses to call a gap meaningful when the
    positive count cannot support the claim.
    """
    quantum = test_metrics[QUANTUM_MODEL_NAME]
    best = test_metrics[best_model]
    n_positive = quantum.n_positive
    quantum_pr = quantum.pr_auc
    best_pr = best.pr_auc
    gap = None if (quantum_pr is None or best_pr is None) else best_pr - quantum_pr

    if quantum_rank == 1:
        statement = (
            f"The quantum model ranks 1 of {len(ranking)} on the held-out test set "
            f"(PR-AUC {_fmt(quantum_pr)})."
        )
    else:
        statement = (
            f"The quantum model ranks {quantum_rank} of {len(ranking)} on the "
            f"held-out test set: PR-AUC {_fmt(quantum_pr)} against {best_model} at "
            f"{_fmt(best_pr)}. On this dataset the variational circuit does not "
            "outperform a classical baseline."
        )

    # With this few positives the ranking is not a stable finding, whichever way it
    # falls. Saying so is part of reporting it honestly.
    underpowered = n_positive < 20
    if underpowered:
        statement += (
            f" The test partition contains only {n_positive} positive images, so "
            "the ordering between models is not statistically meaningful and should "
            "not be quoted as a result about quantum machine learning in general."
        )
    return {
        "statement": statement,
        "quantum_rank": quantum_rank,
        "n_models": len(ranking),
        "best_model": best_model,
        "quantum_pr_auc": quantum_pr,
        "best_pr_auc": best_pr,
        "pr_auc_gap": None if gap is None else round(gap, 6),
        "n_test_positive": n_positive,
        "statistically_underpowered": underpowered,
        "quantum_outperforms_classical": quantum_rank == 1,
    }


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Final held-out test evaluation. Run this once per trained model."
    )
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--extractor", default="handcrafted_lab")
    parser.add_argument(
        "--target-sensitivity", type=float, default=None,
        help="Each model's threshold is chosen on validation to reach this "
             "sensitivity. Defaults to SCREENING_TARGET_SENSITIVITY.",
    )
    parser.add_argument(
        "--mode", dest="execution_mode", default=None,
        choices=[mode.value for mode in ExecutionMode],
        help="Evaluate in this execution mode. Warns if it differs from the mode "
             "the calibrator was fitted in.",
    )
    parser.add_argument(
        "--no-sweep", dest="include_sweep", action="store_false",
        help="Omit the per-model threshold sweeps from the report.",
    )
    args = parser.parse_args(argv)

    _configure_logging()
    summary = evaluate(
        model_version=args.model_version,
        extractor_name=args.extractor,
        target_sensitivity=args.target_sensitivity,
        execution_mode=args.execution_mode,
        include_sweep=args.include_sweep,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
