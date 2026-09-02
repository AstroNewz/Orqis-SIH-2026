"""Train the variational quantum classifier on the fitted pipeline's features.

This loads the pipeline saved by :mod:`backend.training.train_classical` rather
than fitting its own. That is the point: the VQC and the classical baselines must
see byte-identical feature matrices, or the comparison PART 10 asks for measures
the preprocessing difference instead of the model difference.

Training runs on the ideal simulator by default and needs no IBM credentials
(PART 12, PART 29). ``--mode noisy_simulation`` trains through a device noise
model; ``--mode ibm_hardware`` is accepted but is a poor idea for *training*,
because SPSA needs hundreds of objective evaluations and each one would be a
queued hardware job. The recommended sequence is to train ideally and then
*evaluate* under noise or on hardware, which is what the backend's execution-mode
switch exists for.

What the raw quantum score is not
---------------------------------
``fit`` optimises a weighted log-loss over ``(1 - <Z_0>)/2``. That output is
bounded and monotone but it is not a calibrated probability -- on this dataset the
trained model's scores span roughly [0.38, 0.60] against a 6% prevalence. Turning
it into something reportable is :mod:`backend.training.calibrate`, which must run
after this script.

Usage::

    python -m backend.training.train_quantum
    python -m backend.training.train_quantum --maxiter 300 --optimizer spsa
    python -m backend.training.train_quantum --n-qubits 10 --layers 3
    python -m backend.training.train_quantum --mode noisy_simulation --maxiter 40
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.evaluation.metrics import evaluate_predictions, sensitivity_at_specificity
from backend.ml.artifacts import ArtifactStore, QUANTUM_FILE
from backend.training.data import load_partitions
from quantum_ml.backends import build_backend
from quantum_ml.results import ExecutionMode, OptimizerName
from quantum_ml.vqc_classifier import VariationalQuantumClassifier

logger = logging.getLogger(__name__)


def _configure_logging(level: int = logging.INFO) -> None:
    """INFO for our modules, WARNING for Qiskit's per-pass transpiler chatter."""
    logging.basicConfig(level=level, format="%(message)s")
    for noisy in ("qiskit", "qiskit.transpiler", "qiskit.passmanager", "stevedore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def train_quantum(
    *,
    model_version: Optional[str] = None,
    extractor_name: str = "handcrafted_lab",
    feature_mode: Optional[str] = None,
    n_qubits: Optional[int] = None,
    n_layers: Optional[int] = None,
    optimizer: str = OptimizerName.SPSA.value,
    maxiter: int = 200,
    learning_rate: Optional[float] = None,
    perturbation: Optional[float] = None,
    execution_mode: Optional[str] = None,
    shots: Optional[int] = None,
    seed: Optional[int] = None,
    progress_every: int = 25,
    config: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Train the VQC, persist it beside its pipeline, and report validation metrics.

    Raises:
        ArtifactError: no pipeline has been trained for ``model_version`` yet.
    """
    cfg = config or default_settings
    resolved_seed = cfg.RANDOM_SEED if seed is None else seed
    store = ArtifactStore.from_settings(cfg)
    version = store.resolve_version(model_version)

    # Loaded, never refitted: the pipeline defines the feature space this model is
    # trained in, and refitting it here would silently produce a model whose
    # artifact no longer matches its transform.
    pipeline = store.load_pipeline(version)
    dataset = load_partitions(
        extractor_name=extractor_name,
        config=cfg,
        feature_mode=feature_mode,
        pipeline=pipeline,
    )
    x_train, y_train = dataset.xy("train")
    x_validation, y_validation = dataset.xy("validation")

    resolved_qubits = pipeline.n_qubits if n_qubits is None else n_qubits
    resolved_layers = cfg.QUANTUM_CIRCUIT_DEPTH if n_layers is None else n_layers
    resolved_shots = cfg.QUANTUM_SHOTS if shots is None else shots
    resolved_mode = ExecutionMode(execution_mode or cfg.QUANTUM_EXECUTION_MODE)

    if resolved_mode is ExecutionMode.IBM_HARDWARE:
        logger.warning(
            "Training on IBM hardware submits one job per objective evaluation "
            "(roughly %d for maxiter=%d). Train on a simulator and switch mode at "
            "inference instead unless you specifically intend this.",
            maxiter * 2,
            maxiter,
        )

    backend = build_backend(
        mode=resolved_mode,
        shots=resolved_shots,
        seed=resolved_seed,
        noise_device=cfg.QUANTUM_NOISE_MODEL,
        ibm_backend_name=cfg.IBMQ_BACKEND_NAME,
        ibm_api_key=cfg.IBMQ_API_KEY,
        ibm_instance=cfg.IBMQ_INSTANCE,
        ibm_channel=cfg.IBMQ_CHANNEL,
        allow_fallback=cfg.QUANTUM_ALLOW_HARDWARE_FALLBACK,
    )
    if backend.resolution.fell_back:
        logger.warning(
            "Requested %s but the backend resolved to %s: %s",
            resolved_mode.value,
            backend.effective_mode.value,
            backend.resolution.fallback_reason,
        )

    model = VariationalQuantumClassifier(
        num_qubits=resolved_qubits,
        num_layers=resolved_layers,
        shots=resolved_shots,
        backend=backend,
        seed=resolved_seed,
        class_weighting=True,
        model_version=version,
    )
    logger.info(
        "VQC: %d qubits, %d layers, %d parameters, ansatz depth %d, %s on %s | "
        "%d train rows (%d positive), %d features -> %d amplitudes",
        model.num_qubits,
        model.num_layers,
        model.num_params,
        model.ansatz_depth,
        backend.effective_mode.value,
        backend.backend_name,
        x_train.shape[0],
        int(np.sum(y_train == 1)),
        x_train.shape[1],
        2**model.num_qubits,
    )

    record = model.fit(
        x_train,
        y_train,
        optimizer=optimizer,
        maxiter=maxiter,
        learning_rate=learning_rate,
        perturbation=perturbation,
        validation_data=(x_validation, y_validation),
        progress_every=progress_every,
        notes=f"pipeline={version}; extractor={extractor_name}",
    )
    logger.info(
        "Trained in %.1fs: objective %.5f -> %.5f over %d iterations (%d circuit evaluations).",
        record.training_duration_seconds,
        record.initial_objective,
        record.final_objective,
        record.n_iterations,
        record.n_circuit_evaluations,
    )

    train_scores = model.predict_proba(x_train)
    validation_scores = model.predict_proba(x_validation)
    train_metrics = evaluate_predictions(y_train, train_scores, threshold=0.50)
    validation_metrics = evaluate_predictions(y_validation, validation_scores, threshold=0.50)
    logger.info("  train      %s", train_metrics.summary_line())
    logger.info("  validation %s", validation_metrics.summary_line())
    logger.info(
        "  raw score range [%.4f, %.4f] mean %.4f against prevalence %.4f -- "
        "uncalibrated, run backend.training.calibrate next.",
        float(validation_scores.min()),
        float(validation_scores.max()),
        float(validation_scores.mean()),
        validation_metrics.prevalence,
    )

    payload = model.to_dict()
    payload["trained_on"] = {
        "model_version": version,
        "extractor": extractor_name,
        "feature_mode": dataset.pipeline.feature_mode,
        "n_reduced_features": int(x_train.shape[1]),
        "n_train_samples": int(x_train.shape[0]),
        "n_train_positive": int(np.sum(y_train == 1)),
    }
    # Reported at threshold 0.50 on the *raw* score, which is not a probability.
    # Kept anyway as the pre-calibration reference point the calibrator improves on.
    payload["uncalibrated_metrics"] = {
        "train": train_metrics.to_dict(),
        "validation": validation_metrics.to_dict(),
        "validation_sensitivity_at_specificity_090": sensitivity_at_specificity(
            y_validation, validation_scores, 0.90
        ),
        "note": (
            "Threshold 0.50 on the raw quantum score. The score is bounded and "
            "monotone but not a probability; Brier and ECE here are expected to be "
            "poor and are the baseline the calibrator is measured against."
        ),
    }
    path = store.write_component(version, QUANTUM_FILE, payload)
    logger.info("Quantum model -> %s", path)

    return {
        "model_version": version,
        "artifact": str(path),
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
            "device_calibrated_noise": backend.resolution.device_calibrated_noise,
        },
        "training": record.model_dump(mode="json", exclude={"objective_history"}),
        "objective_history_length": len(record.objective_history),
        "uncalibrated": {
            "train": train_metrics.to_dict(),
            "validation": validation_metrics.to_dict(),
        },
        "raw_score_range": [
            round(float(validation_scores.min()), 6),
            round(float(validation_scores.max()), 6),
        ],
        "calibrated": False,
        "next_step": "python -m backend.training.calibrate",
        "test_partition_used": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the variational quantum classifier.")
    parser.add_argument("--model-version", default=None, help="Defaults to the current version.")
    parser.add_argument("--extractor", default="handcrafted_lab")
    parser.add_argument(
        "--feature-mode", default=None,
        choices=["image_only", "clinical_only", "multimodal"],
    )
    parser.add_argument(
        "--n-qubits", type=int, default=None,
        help="Defaults to the pipeline's qubit count. Changing it here without "
             "refitting the pipeline changes the padding, not the features.",
    )
    parser.add_argument("--layers", type=int, default=None, help="Ansatz repetitions.")
    parser.add_argument(
        "--optimizer", default=OptimizerName.SPSA.value,
        choices=[name.value for name in OptimizerName],
    )
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--perturbation", type=float, default=None)
    parser.add_argument(
        "--mode", dest="execution_mode", default=None,
        choices=[mode.value for mode in ExecutionMode],
        help="Defaults to QUANTUM_EXECUTION_MODE. Falls back to a simulator when "
             "hardware is unavailable.",
    )
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args(argv)

    _configure_logging()
    summary = train_quantum(
        model_version=args.model_version,
        extractor_name=args.extractor,
        feature_mode=args.feature_mode,
        n_qubits=args.n_qubits,
        n_layers=args.layers,
        optimizer=args.optimizer,
        maxiter=args.maxiter,
        learning_rate=args.learning_rate,
        perturbation=args.perturbation,
        execution_mode=args.execution_mode,
        shots=args.shots,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
