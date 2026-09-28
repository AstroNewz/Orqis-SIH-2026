"""Final held-out test evaluation for the 16-qubit pixel VQC.

This is the only module in the Phase D pipeline that reads the test partition, and it
exists so that reading it is a deliberate, single-purpose act rather than something a
tuning loop can reach by accident. It refuses to run without ``--confirm-frozen``.

Nothing is selected here. Both decision thresholds are read out of frozen artifacts:
the quantum threshold from each model's ``validation_selection.threshold.selected``,
the classical one from the baselines artifact's ``screening_threshold``. Both were
chosen on validation. This module computes metrics and nothing else -- there is no
sweep, no argmax over test rows, and no calibration fit.

Every model file is hashed and the hash is recorded next to the numbers it produced,
so a reported figure can be traced to the exact weights that produced it.

The three ROI conditions stay separate, and rows whose localization was rejected stay
in their own entry with their own counts. A condition with no positives reports its
counts and ``null`` metrics rather than an invented number.

Usage::

    python -m backend.evaluation.pixel_final_test --roi oracle --confirm-frozen
    python -m backend.evaluation.pixel_final_test --roi predicted --confirm-frozen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.evaluation.metrics import evaluate_predictions
from backend.evaluation.pixel_comparison import PRIMARY_CONDITION
from backend.evaluation.state_diagnostics import roi_geometry_leak
from backend.ml.artifacts import ArtifactStore, BASELINES_FILE
from backend.ml.baselines import BaselineRecord, refit_from_record
from backend.ml.pixel_pipeline import V1_PIXEL_COUNT, V1_QUBIT_COUNT
from backend.training.data import load_partitions
from backend.training.pixel_data import (
    CONDITION_DESCRIPTIONS,
    LoadedPixelDataset,
    PixelPartition,
    condition_rows,
    load_pixel_partitions,
)
from backend.training.train_pixel_vqc import (
    PIXEL_VQC_VERSION,
    Candidate,
    PixelVQCConfig,
    probabilities_for_rows,
    _rebuild_model,
)

logger = logging.getLogger(__name__)

FINAL_TEST_VERSION = "v1-pixel-final-test-1"

#: Refusal text for the missing-confirmation case. Spelled out because the whole
#: point of the guard is that someone reads it.
NOT_FROZEN_MESSAGE = (
    "Final test evaluation requires --confirm-frozen. The test partition may be read "
    "only after the complete pipeline is frozen: trained weights, ansatz depth, "
    "optimizer settings and decision threshold all selected on validation and written "
    "to artifacts. If any of those is still being chosen, use "
    "`python -m backend.evaluation.pixel_comparison` instead, which is validation-only."
)


class FinalTestError(RuntimeError):
    """Raised when the test evaluation cannot be run as specified."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def classical_test_scores(
    *, config: Optional[Settings] = None, model_version: Optional[str] = None
) -> Dict[str, Any]:
    """``{baseline_name: {image_id: score}}`` on the test partition, plus provenance.

    Deliberately a separate function from
    :func:`~backend.evaluation.pixel_comparison.classical_validation_scores` rather
    than a partition argument on it: that module is asserted to contain no test-set
    access at all, and a shared code path would weaken the assertion to a runtime
    argument check.

    The models are refit from their frozen records on the *train* partition. Test rows
    are scored, never fitted.
    """
    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)
    version = store.resolve_version(model_version)
    payload = store.read_component(version, BASELINES_FILE)
    if not payload or "baselines" not in payload:
        raise FinalTestError(
            f"No baselines artifact for model version {version!r}; there is no "
            "classical reference to compare against."
        )

    data = load_partitions(config=cfg)
    if data.train.features is None or data.test.features is None:
        raise FinalTestError(
            "load_partitions returned unreduced features; the classical reference "
            "cannot be rebuilt without the fitted pipeline."
        )

    scores: Dict[str, Dict[str, float]] = {}
    records: Dict[str, Any] = {}
    for name, entry in payload["baselines"].items():
        record = BaselineRecord.from_dict(entry["record"])
        fitted = refit_from_record(record, data.train.features, data.train.labels)
        probabilities = np.asarray(
            fitted.predict_proba(data.test.features), dtype=np.float64
        )
        scores[name] = {
            image_id: float(p)
            for image_id, p in zip(data.test.image_ids, probabilities)
        }
        records[name] = {
            "hyperparameters": record.hyperparameters,
            "random_seed": record.random_seed,
            "n_train_samples": record.n_train_samples,
            "n_train_positive": record.n_train_positive,
        }
        logger.info("  refitted %s on %d train rows", name, record.n_train_samples)

    threshold = payload.get("screening_threshold")
    if threshold is None:
        raise FinalTestError(
            "The baselines artifact records no screening_threshold. A threshold must "
            "come from validation; this module will not pick one on test rows."
        )
    return {
        "scores": scores,
        "records": records,
        "threshold": float(threshold),
        "threshold_source": f"{BASELINES_FILE}:screening_threshold (selected on validation)",
        "model_version": version,
        "n_test_rows_in_descriptor_cache": len(data.test.image_ids),
    }


def _metrics(
    labels: np.ndarray, scores: np.ndarray, *, threshold: float
) -> Optional[Dict[str, Any]]:
    """Full metric block, or ``None`` where the subset makes it undefined."""
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        return None
    return evaluate_predictions(labels, scores, threshold=threshold).to_dict()


def _quantum_block(
    artifact: Path,
    partition: PixelPartition,
    rows: np.ndarray,
    cfg: Settings,
    *,
    scored_condition: str,
) -> Dict[str, Any]:
    """Score frozen weights on test ``rows`` at the frozen threshold."""
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    selection = payload["validation_selection"]
    selected = selection["selected_config"]
    threshold_block = selection.get("threshold") or {}
    # ``selected`` is the whole operating point the validation sweep chose -- the
    # scalar plus the sensitivity/specificity it achieved there -- not a bare float.
    chosen = threshold_block.get("selected")
    threshold = chosen.get("threshold") if isinstance(chosen, dict) else chosen
    if threshold is None:
        raise FinalTestError(
            f"{artifact.name} records no validation-selected threshold. Choosing one "
            "here would be tuning against the test set."
        )

    config = PixelVQCConfig(
        roi_mode=selected["roi_mode"],
        condition=selected["condition"],
        n_layers=int(selected["n_layers"]),
        optimizer=selected["optimizer"],
        maxiter=int(selected["maxiter"]),
        seed=int(selected["seed"]),
        class_weighting=bool(selected["class_weighting"]),
    )
    candidate = Candidate(
        config=config,
        weights=np.asarray(payload["selected_weights"], dtype=np.float64),
        initial_objective=0.0,
        final_objective=0.0,
        best_objective=0.0,
        objective_history=[],
        validation_trace=[],
        n_iterations=0,
        n_objective_evaluations=0,
        n_circuit_evaluations=0,
        duration_seconds=0.0,
        converged=False,
        train_metrics={},
        validation_metrics={},
        class_weights=(1.0, 1.0),
    )
    model = _rebuild_model(candidate, cfg)
    scores = probabilities_for_rows(model, partition, rows)
    labels = partition.labels[rows]
    own = selected["condition"] == scored_condition
    return {
        "artifact": artifact.name,
        "artifact_sha256": _sha256(artifact),
        "weights_trained_on_condition": selected["condition"],
        "own_condition_artifact": own,
        "n_parameters": selected["n_parameters"],
        "n_layers": selected["n_layers"],
        "n_qubits": V1_QUBIT_COUNT,
        "amplitudes_per_state": V1_PIXEL_COUNT,
        "threshold": float(threshold),
        "threshold_source": (
            f"{artifact.name}:validation_selection.threshold.selected.threshold "
            f"({threshold_block.get('basis', 'validation')})"
        ),
        # What validation promised at this threshold, so the test row below can be
        # read against it rather than against nothing.
        "validation_operating_point": chosen if isinstance(chosen, dict) else None,
        "score_map": "(1 - <Z0>)/2, uncalibrated",
        "metrics": _metrics(labels, scores, threshold=float(threshold)),
        "score_distribution": {
            "min": round(float(scores.min()), 6) if scores.size else None,
            "max": round(float(scores.max()), 6) if scores.size else None,
            "mean": round(float(scores.mean()), 6) if scores.size else None,
            "std": round(float(scores.std()), 6) if scores.size else None,
        },
    }


def evaluate(
    *,
    roi_mode: str = "oracle",
    config: Optional[Settings] = None,
    dataset: Optional[LoadedPixelDataset] = None,
    model_version: Optional[str] = None,
    vqc_version: str = PIXEL_VQC_VERSION,
) -> Dict[str, Any]:
    """Final test metrics per ROI condition for both paths, on identical rows."""
    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)
    data = dataset or load_pixel_partitions(config=cfg, roi_mode=roi_mode)

    if data.qubit_count != V1_QUBIT_COUNT:
        raise FinalTestError(
            f"Pixel cache reports {data.qubit_count} qubits; V1 is fixed at "
            f"{V1_QUBIT_COUNT}."
        )

    logger.info("Refitting the classical reference from its frozen record...")
    classical = classical_test_scores(config=cfg, model_version=model_version)
    by_name = classical["scores"]
    classical_threshold = classical["threshold"]

    partition = data.test
    conditions = condition_rows(partition)
    directory = store.pixel_vqc_dir(vqc_version)

    rows_out: List[Dict[str, Any]] = []
    for name, rows in sorted(conditions.items()):
        labels = partition.labels[rows]
        image_ids = [partition.image_ids[int(r)] for r in rows]

        entry: Dict[str, Any] = {
            "condition": name,
            "description": CONDITION_DESCRIPTIONS.get(name, ""),
            "partition": "test",
            "n_samples": int(rows.size),
            "n_positive": int(labels.sum()),
            "n_negative": int(rows.size - labels.sum()),
            "n_patients": len({partition.patient_ids[int(r)] for r in rows}),
            "prevalence": round(float(labels.mean()), 6) if rows.size else None,
            "roi_geometry_leak": roi_geometry_leak(partition, rows),
            "classical": {},
            "quantum": None,
        }

        for baseline_name, scores_by_id in by_name.items():
            available = [i for i in image_ids if i in scores_by_id]
            mask = np.asarray([i in scores_by_id for i in image_ids], dtype=bool)
            entry["classical"][baseline_name] = {
                "n_scored": len(available),
                "n_not_in_descriptor_cache": len(image_ids) - len(available),
                "threshold": classical_threshold,
                "metrics": _metrics(
                    labels[mask],
                    np.asarray([scores_by_id[i] for i in available], dtype=np.float64),
                    threshold=classical_threshold,
                ),
            }

        artifact = directory / f"{roi_mode}_{name}_pixel_vqc.json"
        if artifact.exists():
            entry["quantum"] = _quantum_block(
                artifact, partition, rows, cfg, scored_condition=name
            )
        else:
            primary = PRIMARY_CONDITION.get(roi_mode)
            borrowed = (
                directory / f"{roi_mode}_{primary}_pixel_vqc.json" if primary else None
            )
            if borrowed is not None and primary != name and borrowed.exists():
                entry["quantum"] = _quantum_block(
                    borrowed, partition, rows, cfg, scored_condition=name
                )
                entry["quantum_note"] = (
                    f"No pixel VQC was trained on {name}; the model trained on "
                    f"{primary} was applied to these rows, which is what the deployed "
                    "pipeline does. The rows remain a separate entry and are not "
                    f"merged into the {primary} statistic."
                )
            else:
                entry["quantum_note"] = (
                    f"No trained pixel VQC for {name} and no {roi_mode} "
                    "primary-condition model to borrow."
                )
        rows_out.append(entry)

    return {
        "final_test_version": FINAL_TEST_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "roi_mode": roi_mode,
        "partition": "test",
        "n_qubits": V1_QUBIT_COUNT,
        "amplitudes_per_state": V1_PIXEL_COUNT,
        "nothing_selected_here": {
            "quantum_threshold_source": "each artifact's validation-selected threshold",
            "classical_threshold_source": classical["threshold_source"],
            "statement": (
                "No threshold, depth, seed or calibration parameter was chosen using "
                "test rows. This module computes metrics from frozen artifacts."
            ),
        },
        "classical_reference": {
            "model_version": classical["model_version"],
            "input": "fused descriptor + clinical features, PCA-reduced",
            "threshold": classical_threshold,
            "records": classical["records"],
            "rebuilt_from": BASELINES_FILE,
        },
        "alignment_note": (
            "Within a condition, both paths are scored on the same image ids, so "
            "prevalence is identical and PR-AUC is comparable. Across conditions the "
            "prevalences differ and the numbers are not comparable to each other."
        ),
        "conditions": rows_out,
    }


def summarize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """One row per condition: the head-to-head PR-AUC and whether quantum won."""
    out: List[Dict[str, Any]] = []
    for entry in payload["conditions"]:
        best_name, best_value = None, None
        for name, block in entry["classical"].items():
            metrics = block.get("metrics")
            if metrics and metrics.get("pr_auc") is not None:
                if best_value is None or metrics["pr_auc"] > best_value:
                    best_name, best_value = name, float(metrics["pr_auc"])

        quantum = entry.get("quantum")
        quantum_metrics = quantum.get("metrics") if quantum else None
        quantum_pr = (
            float(quantum_metrics["pr_auc"])
            if quantum_metrics and quantum_metrics.get("pr_auc") is not None
            else None
        )
        out.append({
            "condition": entry["condition"],
            "n_samples": entry["n_samples"],
            "n_positive": entry["n_positive"],
            "prevalence": entry["prevalence"],
            # DEC-024: a condition-A number is read against this, not against 0.5.
            "roi_geometry_leak_roc_auc": entry["roi_geometry_leak"].get(
                "area_only_roc_auc"
            ),
            "roi_geometry_leak_pr_auc": entry["roi_geometry_leak"].get(
                "area_only_pr_auc"
            ),
            "best_classical": best_name,
            "best_classical_pr_auc": best_value,
            "quantum_pr_auc": quantum_pr,
            "quantum_roc_auc": (
                quantum_metrics.get("roc_auc") if quantum_metrics else None
            ),
            "quantum_sensitivity": (
                quantum_metrics.get("sensitivity") if quantum_metrics else None
            ),
            "quantum_specificity": (
                quantum_metrics.get("specificity") if quantum_metrics else None
            ),
            "quantum_brier": quantum_metrics.get("brier") if quantum_metrics else None,
            "quantum_weights_trained_on": (
                quantum.get("weights_trained_on_condition") if quantum else None
            ),
            "quantum_weights_borrowed": (
                None if not quantum else not quantum.get("own_condition_artifact", True)
            ),
            "quantum_beats_classical": (
                None
                if quantum_pr is None or best_value is None
                else bool(quantum_pr > best_value)
            ),
            "margin": (
                None
                if quantum_pr is None or best_value is None
                else round(quantum_pr - best_value, 6)
            ),
        })
    return {"per_condition": out}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Final held-out test evaluation of the 16-qubit pixel VQC against "
        "the classical reference. Reads the test partition."
    )
    parser.add_argument("--roi", dest="roi_mode", default="oracle",
                        choices=("oracle", "predicted"))
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--vqc-version", default=PIXEL_VQC_VERSION)
    parser.add_argument("--out", default=None, help="Write the full payload here.")
    parser.add_argument(
        "--confirm-frozen", action="store_true",
        help="Required. Confirms the pipeline is frozen and the test partition may "
             "now be read.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("qiskit", "stevedore", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not args.confirm_frozen:
        parser.error(NOT_FROZEN_MESSAGE)

    payload = evaluate(
        roi_mode=args.roi_mode,
        model_version=args.model_version,
        vqc_version=args.vqc_version,
    )
    payload["summary"] = summarize(payload)
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Full payload -> %s", args.out)
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
