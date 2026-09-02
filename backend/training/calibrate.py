"""Fit the probability calibrator on the trained quantum model's raw scores.

The third and last training stage. The VQC produces ``(1 - <Z_0>)/2``, which is
bounded and monotone in the measurement but is not a frequency: on this dataset a
trained model's scores sit in roughly [0.39, 0.59] with a mean near 0.48, against
an observed positive rate of about 6%. Reporting that to a clinician as "48% risk"
would be wrong by almost an order of magnitude.

The fitting discipline
----------------------
PART 14 is explicit that calibration may use training and validation data only.
This script therefore concatenates the ``train`` and ``validation`` partitions and
never loads a test score. :func:`~quantum_ml.calibration.fit_calibrator` chooses
between Platt scaling and isotonic regression by **out-of-fold** cross-validation
inside that fitting set, so the metrics it reports are out-of-sample even though
every row was used for the final fit.

Patient grouping
----------------
The cross-validation folds are stratified by label, not grouped by patient, and
that is a known limitation rather than an oversight: two images of one patient can
land in different folds, which makes the reported out-of-fold calibration metrics
slightly optimistic. Grouping would be better, but with 125 positives across
train+validation the fold sizes become too small for a stable isotonic fit. The
metric that is *not* affected is the held-out test evaluation, since patients never
cross partitions -- see :mod:`backend.evaluation.evaluate`.

Usage::

    python -m backend.training.calibrate
    python -m backend.training.calibrate --method platt
    python -m backend.training.calibrate --method isotonic --cv-seed 7
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.evaluation.metrics import (
    evaluate_predictions,
    sensitivity_at_specificity,
    threshold_for_sensitivity,
    threshold_sweep,
)
from backend.ml.artifacts import CALIBRATION_FILE, QUANTUM_FILE, ArtifactStore
from backend.training.data import load_partitions
from backend.training.train_quantum import _configure_logging
from quantum_ml.backends import build_backend
from quantum_ml.calibration import CalibrationMethod, ProbabilityCalibrator, fit_calibrator
from quantum_ml.results import ExecutionMode
from quantum_ml.vqc_classifier import VariationalQuantumClassifier

logger = logging.getLogger(__name__)


def _recommend_threshold(calibrator, target_sensitivity: float) -> Dict[str, Any]:
    """Derive an operating threshold from the calibrator's out-of-fold probabilities.

    Returns a payload including the full sweep, so a reader can pick a different
    operating point rather than having to trust this one. The threshold is labelled
    experimental everywhere it appears: PART 14 forbids presenting an invented
    cut-off as clinically validated.
    """
    if calibrator.held_out_probabilities is None or calibrator.held_out_labels is None:
        return {
            "threshold": None,
            "target_sensitivity": target_sensitivity,
            "note": (
                "The calibrator has no out-of-sample probabilities (identity "
                "fallback), so no threshold can be recommended."
            ),
        }
    probabilities = calibrator.held_out_probabilities
    labels = calibrator.held_out_labels
    payload = dict(threshold_for_sensitivity(labels, probabilities, target_sensitivity))
    payload["estimated_on"] = calibrator.fit_metrics.get("estimated_on", "out-of-fold")
    payload["is_clinically_validated"] = False
    payload["basis"] = (
        "Highest threshold reaching the target sensitivity on out-of-fold "
        "calibrated probabilities from the train+validation partitions. An "
        "experimental screening operating point, not a validated clinical cut-off."
    )
    payload["sweep"] = threshold_sweep(labels, probabilities)
    payload["sensitivity_at_specificity_090"] = sensitivity_at_specificity(
        labels, probabilities, 0.90
    )
    return payload


def _recommend_high_risk_threshold(
    calibrator, target_specificity: float, screening_threshold: Optional[float]
) -> Dict[str, Any]:
    """Derive the HIGH-risk band boundary from out-of-fold probabilities.

    The upper band is derived at high *specificity* rather than high sensitivity,
    because it answers a different question: the screening threshold decides who
    gets looked at, the HIGH band decides who gets looked at first. A boundary that
    flags a third of the cohort as high risk would not triage anything.

    Measured consequence of not doing this: the previous round-number default of
    0.70 sat above every probability this model can emit, so the HIGH band was
    unreachable and the API could never return it.
    """
    if calibrator.held_out_probabilities is None or calibrator.held_out_labels is None:
        return {
            "threshold": None,
            "target_specificity": target_specificity,
            "note": (
                "The calibrator has no out-of-sample probabilities (identity "
                "fallback), so no band boundary can be recommended."
            ),
        }
    probabilities = calibrator.held_out_probabilities
    labels = calibrator.held_out_labels
    payload = dict(sensitivity_at_specificity(labels, probabilities, target_specificity))
    payload["threshold"] = payload.get("threshold")
    payload["estimated_on"] = calibrator.fit_metrics.get("estimated_on", "out-of-fold")
    payload["is_clinically_validated"] = False
    payload["basis"] = (
        "Threshold reaching the target specificity on out-of-fold calibrated "
        "probabilities from the train+validation partitions. An experimental "
        "triage boundary, not a validated clinical cut-off."
    )
    # A HIGH boundary at or below the screening threshold would make every flagged
    # case high risk, collapsing the two bands into one.
    if (
        payload["threshold"] is not None
        and screening_threshold is not None
        and payload["threshold"] <= screening_threshold
    ):
        payload["degenerate"] = True
        payload["note"] = (
            f"The derived HIGH boundary ({payload['threshold']:.4f}) is not above the "
            f"screening threshold ({screening_threshold:.4f}), so the two bands "
            "coincide. This model does not separate the classes well enough to "
            "support a meaningful triage tier."
        )
    return payload


def calibrate(
    *,
    model_version: Optional[str] = None,
    extractor_name: str = "handcrafted_lab",
    method: Optional[str] = None,
    cv_seed: Optional[int] = None,
    target_sensitivity: Optional[float] = None,
    target_specificity: Optional[float] = None,
    execution_mode: Optional[str] = None,
    config: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Fit and persist the calibrator for a trained quantum model.

    Raises:
        ArtifactError: the version has no pipeline or no trained quantum model.
    """
    cfg = config or default_settings
    resolved_method = method or cfg.CALIBRATION_METHOD
    resolved_cv_seed = cfg.RANDOM_SEED if cv_seed is None else cv_seed
    resolved_target_sensitivity = (
        cfg.SCREENING_TARGET_SENSITIVITY if target_sensitivity is None else target_sensitivity
    )
    resolved_target_specificity = (
        cfg.HIGH_RISK_TARGET_SPECIFICITY if target_specificity is None else target_specificity
    )
    store = ArtifactStore.from_settings(cfg)
    version = store.resolve_version(model_version)

    pipeline = store.load_pipeline(version)
    quantum_payload = store.read_component(version, QUANTUM_FILE)

    # Scores are produced in the mode the calibrator will be used in. Calibrating
    # on ideal-simulator scores and then serving noisy ones would apply a mapping
    # fitted to a different score distribution -- the fit is mode-specific even
    # though the trained weights are not.
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
    model = VariationalQuantumClassifier.from_dict(quantum_payload, backend=backend)
    if not model.trained:
        raise RuntimeError(
            f"The quantum artifact for {version} is not marked trained. Run "
            "`python -m backend.training.train_quantum` first."
        )

    dataset = load_partitions(
        extractor_name=extractor_name, config=cfg, pipeline=pipeline
    )
    x_train, y_train = dataset.xy("train")
    x_validation, y_validation = dataset.xy("validation")

    logger.info(
        "Scoring %d train + %d validation rows through the VQC (%s on %s)...",
        x_train.shape[0],
        x_validation.shape[0],
        backend.effective_mode.value,
        backend.backend_name,
    )
    train_scores = model.predict_proba(x_train)
    validation_scores = model.predict_proba(x_validation)

    # The test partition is deliberately absent from everything below.
    fit_scores = np.concatenate([train_scores, validation_scores])
    fit_labels = np.concatenate([y_train, y_validation])
    logger.info(
        "Fitting calibrator on %d rows (%d positive), method=%s, cv_seed=%d.",
        fit_labels.size,
        int(np.sum(fit_labels == 1)),
        resolved_method,
        resolved_cv_seed,
    )

    calibrator = fit_calibrator(
        fit_scores,
        fit_labels,
        method=resolved_method,
        fitted_on="train+validation",
        cv_seed=resolved_cv_seed,
    )
    logger.info("  selected %s -- %s", calibrator.method.value, calibrator.selection_basis)
    for name, metrics in calibrator.candidate_metrics.items():
        logger.info(
            "    %-9s out-of-fold Brier %.5f  log-loss %.5f  ECE %.5f",
            name,
            metrics["brier"],
            metrics["log_loss"],
            metrics["ece"],
        )

    # Reported on validation, which was part of the fitting set, so these numbers
    # are in-sample for the calibrator and optimistic. The honest ones are the
    # out-of-fold metrics above and the test metrics in backend.evaluation.evaluate.
    in_sample = calibrator.evaluate(validation_scores, y_validation)
    calibrated_validation = calibrator.transform(validation_scores)
    thresholds = ProbabilityCalibrator.from_settings(cfg)
    validation_metrics = evaluate_predictions(
        y_validation, calibrated_validation, threshold=thresholds.threshold
    )
    logger.info(
        "  validation (in-sample for the calibrator): Brier %.5f -> %.5f, "
        "mean predicted %.4f vs observed %.4f",
        in_sample["brier_uncalibrated"],
        in_sample["brier"],
        in_sample["mean_predicted"],
        in_sample["observed_rate"],
    )
    logger.info("  %s", validation_metrics.summary_line())

    # A calibrated model at ~6% prevalence rarely exceeds 0.50, so the configured
    # default threshold detects almost nothing. Recommend one from the out-of-fold
    # probabilities -- out-of-sample, so it does not inherit the calibrator's
    # in-sample optimism, and without touching the test partition.
    recommendation = _recommend_threshold(calibrator, resolved_target_sensitivity)
    high_risk = _recommend_high_risk_threshold(
        calibrator, resolved_target_specificity, recommendation.get("threshold")
    )
    if recommendation["threshold"] is None:
        logger.warning(
            "Could not derive an operating threshold for sensitivity %.2f: %s",
            resolved_target_sensitivity,
            recommendation.get("note", "no reason recorded"),
        )
    else:
        logger.info(
            "  recommended operating threshold %.4f for sensitivity >= %.2f "
            "(out-of-fold: sens %.3f, spec %.3f, precision %s). Configured "
            "SCREENING_THRESHOLD is %.2f, which gives sensitivity %.3f.",
            recommendation["threshold"],
            resolved_target_sensitivity,
            recommendation["sensitivity"],
            recommendation["specificity"],
            recommendation["precision"],
            thresholds.threshold,
            validation_metrics.sensitivity,
        )
        if abs(recommendation["threshold"] - thresholds.threshold) > 0.05:
            logger.warning(
                "SCREENING_THRESHOLD=%.2f is far from the recommended %.4f. The "
                "inference path should read the recommended value from this "
                "artifact rather than the configured fallback.",
                thresholds.threshold,
                recommendation["threshold"],
            )
    if high_risk.get("threshold") is None:
        logger.warning(
            "Could not derive a HIGH-risk band boundary at specificity %.2f: %s",
            resolved_target_specificity,
            high_risk.get("note", "no reason recorded"),
        )
    else:
        logger.info(
            "  recommended HIGH-risk boundary %.4f at specificity >= %.2f "
            "(out-of-fold: achieved spec %.3f, sens %.3f).",
            high_risk["threshold"],
            resolved_target_specificity,
            high_risk["achieved_specificity"],
            high_risk["sensitivity"],
        )
        if high_risk.get("degenerate"):
            logger.warning("%s", high_risk["note"])

    # The bands the inference path should actually use for this model version.
    # Persisted with the calibrator so they travel with the model rather than
    # living in a global constant that is wrong for every other version.
    resolved_bands = {
        "screening_threshold": recommendation.get("threshold"),
        "high_risk_threshold": high_risk.get("threshold"),
        "target_sensitivity": resolved_target_sensitivity,
        "target_specificity": resolved_target_specificity,
        "derived_from": "out-of-fold calibrated probabilities on train+validation",
        "is_clinically_validated": False,
        "configured_fallback": thresholds.describe(),
        "note": (
            "Model-specific experimental band boundaries. They depend on this "
            "model's calibrated score distribution, so they are not transferable "
            "to another model version."
        ),
    }

    payload = calibrator.to_dict()
    payload["model_version"] = version
    payload["scored_in_mode"] = backend.effective_mode.value
    payload["scored_on_backend"] = backend.backend_name
    payload["thresholds"] = thresholds.describe()
    payload["recommended_threshold"] = recommendation
    payload["recommended_high_risk_threshold"] = high_risk
    payload["resolved_bands"] = resolved_bands
    payload["in_sample_validation"] = {
        **in_sample,
        "metrics": validation_metrics.to_dict(),
        "sensitivity_at_specificity_090": sensitivity_at_specificity(
            y_validation, calibrated_validation, 0.90
        ),
        "note": (
            "Validation rows were part of the calibrator's fitting set, so these "
            "are in-sample. Use fit_metrics (out-of-fold) or the held-out test "
            "report for honest numbers."
        ),
    }
    path = store.write_component(version, CALIBRATION_FILE, payload)
    store.update_metadata(
        version,
        {
            "calibration": {
                "method": calibrator.method.value,
                "fitted_on": calibrator.fitted_on,
                "n_fit_samples": calibrator.n_fit_samples,
                "n_fit_positive": calibrator.n_fit_positive,
                "out_of_fold_brier": calibrator.fit_metrics.get("brier"),
                "scored_in_mode": backend.effective_mode.value,
                "recommended_threshold": recommendation.get("threshold"),
                "recommended_threshold_target_sensitivity": resolved_target_sensitivity,
                "recommended_high_risk_threshold": high_risk.get("threshold"),
                "recommended_high_risk_target_specificity": resolved_target_specificity,
            }
        },
    )
    logger.info("Calibrator -> %s", path)

    return {
        "model_version": version,
        "artifact": str(path),
        "method": calibrator.method.value,
        "is_calibrated": calibrator.is_calibrated,
        "selection_basis": calibrator.selection_basis,
        "candidate_metrics": calibrator.candidate_metrics,
        "out_of_fold_metrics": calibrator.fit_metrics,
        "n_fit_samples": calibrator.n_fit_samples,
        "n_fit_positive": calibrator.n_fit_positive,
        "fitted_on": calibrator.fitted_on,
        "cv_seed": resolved_cv_seed,
        "scored_in_mode": backend.effective_mode.value,
        "thresholds": thresholds.describe(),
        "recommended_threshold": {
            key: value for key, value in recommendation.items() if key != "sweep"
        },
        "recommended_high_risk_threshold": high_risk,
        "resolved_bands": resolved_bands,
        "in_sample_validation": {
            "brier_uncalibrated": round(in_sample["brier_uncalibrated"], 6),
            "brier_calibrated": round(in_sample["brier"], 6),
            "ece": round(in_sample["ece"], 6),
            "mean_predicted": round(in_sample["mean_predicted"], 6),
            "observed_rate": round(in_sample["observed_rate"], 6),
            "metrics": validation_metrics.to_dict(),
        },
        "next_step": "python -m backend.evaluation.evaluate",
        "test_partition_used": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit the probability calibrator on train+validation scores."
    )
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--extractor", default="handcrafted_lab")
    parser.add_argument(
        "--method", default=None,
        choices=["auto", *[m.value for m in CalibrationMethod]],
        help="Defaults to CALIBRATION_METHOD. 'auto' picks by out-of-fold CV.",
    )
    parser.add_argument(
        "--cv-seed", type=int, default=None,
        help="Seed for the calibration cross-validation folds.",
    )
    parser.add_argument(
        "--target-sensitivity", type=float, default=None,
        help="Sensitivity the recommended operating threshold aims for. Defaults "
             "to SCREENING_TARGET_SENSITIVITY.",
    )
    parser.add_argument(
        "--target-specificity", type=float, default=None,
        help="Specificity the recommended HIGH-risk band boundary aims for. "
             "Defaults to HIGH_RISK_TARGET_SPECIFICITY.",
    )
    parser.add_argument(
        "--mode", dest="execution_mode", default=None,
        choices=[mode.value for mode in ExecutionMode],
        help="Score in this mode. The calibrator is mode-specific.",
    )
    args = parser.parse_args(argv)

    _configure_logging()
    summary = calibrate(
        model_version=args.model_version,
        extractor_name=args.extractor,
        method=args.method,
        cv_seed=args.cv_seed,
        target_sensitivity=args.target_sensitivity,
        target_specificity=args.target_specificity,
        execution_mode=args.execution_mode,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
