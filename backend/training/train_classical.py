"""Fit the classical pipeline and the PART 15 baselines, and persist them.

This is the first of the three training entry points. It does two things:

1. Fits the classical pipeline -- clinical encoder, feature fusion, dimensionality
   reduction -- **on the training partition only**, and saves it under a model
   version. Everything downstream (quantum training, calibration, evaluation,
   serving) loads that same fitted pipeline rather than refitting, so a single
   model version means one fixed transform.
2. Fits logistic regression, random forest and gradient boosting on the same
   reduced features the quantum model will see, selecting hyperparameters on the
   validation partition.

The baselines exist so the quantum model can be judged rather than assumed
superior (PART 10). They are trained here, in the same run and on the same
features, precisely so that comparison is apples-to-apples.

The test partition is not touched. It is loaded -- the loader always builds all
three -- but no metric is computed on it and nothing is selected using it. The
held-out report is :mod:`backend.evaluation.evaluate`, run once at the end.

Usage::

    python -m backend.training.train_classical
    python -m backend.training.train_classical --feature-mode image_only
    python -m backend.training.train_classical --model-version v2-mobilenet \\
        --extractor mobilenet_v3_small
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.evaluation.metrics import (
    ClassificationMetrics,
    compare_models,
    evaluate_predictions,
    sensitivity_at_specificity,
)
from backend.ml import baselines as baselines_module
from backend.ml.artifacts import BASELINES_FILE, ArtifactStore
from backend.ml.pipeline import PIPELINE_VERSION
from backend.training.data import class_balance, load_partitions

logger = logging.getLogger(__name__)

__all__ = ["BASELINES_FILE", "main", "train_classical"]
# BASELINES_FILE moved to backend.ml.artifacts, next to the other artifact
# filenames, so the inference path can name the file without importing this
# module -- serving must not pull scikit-learn in behind it. Re-exported here
# because callers already import it from this module.


def _provenance(dataset) -> Dict[str, Any]:
    """Feature-representation provenance recorded with every baseline.

    PART 15 asks for the exact preprocessing and feature representation per
    baseline. Since all three share one fitted pipeline, that is one dict --
    but it is stored on each record so a baseline artifact is self-describing.
    """
    return {
        "feature_mode": dataset.pipeline.feature_mode,
        "extractor": dataset.extractor_name,
        "preprocessing_version": dataset.pipeline.preprocessing_version,
        "pipeline_version": dataset.pipeline.version,
    }


def train_classical(
    *,
    extractor_name: str = "handcrafted_lab",
    feature_mode: Optional[str] = None,
    n_qubits: Optional[int] = None,
    model_version: Optional[str] = None,
    seed: Optional[int] = None,
    baseline_names: Sequence[str] = baselines_module.AVAILABLE_BASELINES,
    threshold: Optional[float] = None,
    set_current: bool = True,
    config: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Fit the pipeline and the baselines, persist both, and report on validation."""
    cfg = config or default_settings
    resolved_seed = cfg.RANDOM_SEED if seed is None else seed
    resolved_threshold = cfg.SCREENING_THRESHOLD if threshold is None else threshold
    version = model_version or cfg.MODEL_VERSION
    store = ArtifactStore.from_settings(cfg)

    dataset = load_partitions(
        extractor_name=extractor_name,
        config=cfg,
        feature_mode=feature_mode,
        n_qubits=n_qubits,
    )
    x_train, y_train = dataset.xy("train")
    x_validation, y_validation = dataset.xy("validation")

    logger.info(
        "Fitting %d baselines on %d train rows (%d positive), selecting on %d "
        "validation rows (%d positive).",
        len(baseline_names),
        x_train.shape[0],
        int(np.sum(y_train == 1)),
        x_validation.shape[0],
        int(np.sum(y_validation == 1)),
    )

    fitted = baselines_module.fit_all_baselines(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        names=baseline_names,
        seed=resolved_seed,
        metadata=_provenance(dataset),
    )

    validation_metrics: Dict[str, ClassificationMetrics] = {}
    baseline_payloads: Dict[str, Any] = {}
    for name, model in fitted.items():
        probabilities = model.predict_proba(x_validation)
        metrics = evaluate_predictions(
            y_validation, probabilities, threshold=resolved_threshold
        )
        validation_metrics[name] = metrics
        logger.info("  %-22s %s", name, metrics.summary_line())
        baseline_payloads[name] = {
            **model.to_dict(),
            "validation_metrics": metrics.to_dict(),
            "validation_sensitivity_at_specificity_090": sensitivity_at_specificity(
                y_validation, probabilities, 0.90
            ),
        }

    # The pipeline is saved before the baselines so a partially-failed run leaves a
    # loadable pipeline rather than baselines that reference a version with no
    # transform behind it.
    directory = store.save_pipeline(
        dataset.pipeline,
        model_version=version,
        extra_metadata={
            "training": {
                "entry_point": "backend.training.train_classical",
                "random_seed": resolved_seed,
                "screening_threshold": resolved_threshold,
                "dataset": dataset.summary(),
                "class_balance": {
                    name: class_balance(partition)
                    for name, partition in dataset.partitions.items()
                },
                "cache_metadata": dataset.cache_metadata,
            }
        },
        set_current=set_current,
    )
    store.write_component(
        version,
        BASELINES_FILE,
        {
            "version": baselines_module.BASELINE_VERSION,
            "model_version": version,
            "selected_on": "validation",
            "screening_threshold": resolved_threshold,
            "random_seed": resolved_seed,
            "feature_representation": _provenance(dataset),
            "n_reduced_features": int(x_train.shape[1]),
            "baselines": baseline_payloads,
            "validation_ranking": compare_models(validation_metrics),
            "note": (
                "Metrics here are on the validation partition, which was also used "
                "to select hyperparameters, so they are optimistic. The honest "
                "numbers come from backend.evaluation.evaluate on the held-out test "
                "partition."
            ),
        },
    )
    logger.info("Pipeline and baselines -> %s", directory)

    ranking = compare_models(validation_metrics)
    return {
        "model_version": version,
        "artifact_dir": str(directory),
        "pipeline_version": PIPELINE_VERSION,
        "feature_mode": dataset.pipeline.feature_mode,
        "extractor": extractor_name,
        "n_reduced_features": int(x_train.shape[1]),
        "random_seed": resolved_seed,
        "screening_threshold": resolved_threshold,
        "dataset": dataset.summary(),
        "baselines": {
            name: {
                "hyperparameters": model.record.hyperparameters,
                "selection_metric": model.record.selection_metric,
                "validation": validation_metrics[name].to_dict(),
            }
            for name, model in fitted.items()
        },
        "validation_ranking": [
            {"rank": row["rank"], "model": row["model"], "pr_auc": row["pr_auc"],
             "roc_auc": row["roc_auc"], "brier": row["brier"]}
            for row in ranking
        ],
        "best_on_validation": ranking[0]["model"] if ranking else None,
        "test_partition_used": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit the classical pipeline and the PART 15 baselines."
    )
    parser.add_argument(
        "--extractor", default="handcrafted_lab",
        help="handcrafted_lab | mobilenet_v3_small",
    )
    parser.add_argument(
        "--feature-mode", default=None,
        choices=["image_only", "clinical_only", "multimodal"],
        help="Overrides FEATURE_MODE, for the PART 16 ablations.",
    )
    parser.add_argument(
        "--n-qubits", type=int, default=None,
        help="Overrides N_QUBITS, which sets the reduction target dimension.",
    )
    parser.add_argument("--model-version", default=None, help="Artifact version name.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--baselines", nargs="*", default=list(baselines_module.AVAILABLE_BASELINES),
        help="Subset of baselines to fit.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--no-set-current", action="store_true",
        help="Do not repoint 'current' at this version. Use for ablation runs.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    unknown = [name for name in args.baselines if name not in baselines_module.AVAILABLE_BASELINES]
    if unknown:
        parser.error(
            f"Unknown baseline(s): {', '.join(unknown)}. "
            f"Available: {', '.join(baselines_module.AVAILABLE_BASELINES)}."
        )

    summary = train_classical(
        extractor_name=args.extractor,
        feature_mode=args.feature_mode,
        n_qubits=args.n_qubits,
        model_version=args.model_version,
        seed=args.seed,
        baseline_names=args.baselines,
        threshold=args.threshold,
        set_current=not args.no_set_current,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
