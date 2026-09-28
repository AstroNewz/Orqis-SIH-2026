"""Compare the classical baseline and the 16-qubit pixel VQC on identical rows.

The Phase D question is whether the quantum model beats the established classical
reference. Answering it from the two paths' headline numbers would be wrong: the
classical reference is measured on the whole validation partition (363 images, 5.8%
prevalence) and the honest oracle condition covers 52 of them at 40% prevalence.
PR-AUC is prevalence-sensitive, so those two numbers are not comparable, and a
comparison that ignored this would flatter whichever model was evaluated on the more
balanced subset.

This module therefore scores the classical baseline **per image**, then restricts it
to exactly the rows each ROI condition contains. Same patients, same images, same
prevalence -- only the input representation differs, which is the thing under test.

The classical model is rebuilt with :func:`~backend.ml.baselines.refit_from_record`
from the persisted record, so its hyperparameters, seed and training rows come from
the frozen artifact rather than being chosen here.

The test partition is not read. Every number this module produces is a validation
number, and the module refuses to run on test rows.

Usage::

    python -m backend.evaluation.pixel_comparison
    python -m backend.evaluation.pixel_comparison --roi predicted
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.evaluation.metrics import evaluate_predictions
from backend.ml.artifacts import ArtifactStore, BASELINES_FILE
from backend.ml.baselines import BaselineRecord, refit_from_record
from backend.ml.pixel_pipeline import V1_PIXEL_COUNT, V1_QUBIT_COUNT
from backend.training.data import load_partitions
from backend.training.pixel_data import (
    CONDITION_DESCRIPTIONS,
    LoadedPixelDataset,
    condition_rows,
    load_pixel_partitions,
)
from backend.training.train_pixel_vqc import (
    PIXEL_VQC_VERSION,
    PixelVQCConfig,
    Candidate,
    probabilities_for_rows,
    _rebuild_model,
)

logger = logging.getLogger(__name__)

COMPARISON_VERSION = "v1-pixel-comparison-1"

#: The condition whose trained model is applied to conditions that have none of their
#: own. ``C_fallback`` is the case that matters: its validation subset is single-class,
#: so training and selecting on it is not possible, but the spec still requires those
#: rows to be reported and to stay separately identifiable. Applying the model trained
#: on the mode's primary condition is what the deployed pipeline would do, and the
#: entry records that it was borrowed.
PRIMARY_CONDITION = {
    "oracle": "A_lesion_polygon",
    "predicted": "B_localized",
}


class ComparisonError(RuntimeError):
    """Raised when the two paths cannot be compared on aligned rows."""


def classical_validation_scores(
    *, config: Optional[Settings] = None, model_version: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """``{baseline_name: {image_id: score}}`` on the validation partition.

    Scores are keyed by image id rather than by row index because the descriptor
    cache and the pixel cache accept different images -- quality control runs on the
    ROI that was actually used, and an image can be usable in one and not the other.
    Aligning on ids makes the intersection explicit instead of assuming the two
    caches are row-parallel.
    """
    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)
    version = store.resolve_version(model_version)
    payload = store.read_component(version, BASELINES_FILE)
    if not payload or "baselines" not in payload:
        raise ComparisonError(
            f"No baselines artifact for model version {version!r}. Train the classical "
            "reference with `python -m backend.training.train_classical` first."
        )

    data = load_partitions(config=cfg)
    if data.train.features is None or data.validation.features is None:
        raise ComparisonError(
            "load_partitions returned unreduced features; the classical reference "
            "cannot be rebuilt without the fitted pipeline."
        )
    x_train = data.train.features
    y_train = data.train.labels
    x_validation = data.validation.features

    scores: Dict[str, Dict[str, float]] = {}
    for name, entry in payload["baselines"].items():
        record = BaselineRecord.from_dict(entry["record"])
        fitted = refit_from_record(record, x_train, y_train)
        probabilities = np.asarray(fitted.predict_proba(x_validation), dtype=np.float64)
        scores[name] = {
            image_id: float(p)
            for image_id, p in zip(data.validation.image_ids, probabilities)
        }
        logger.info(
            "  refitted %s (%s, seed %d) on %d train rows",
            name,
            record.hyperparameters,
            record.random_seed,
            record.n_train_samples,
        )
    return scores


def _metrics_for_subset(
    labels: np.ndarray, scores: np.ndarray, *, threshold: float = 0.50
) -> Optional[Dict[str, Any]]:
    """Metrics, or ``None`` when the subset is single-class and they are undefined."""
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        return None
    return evaluate_predictions(labels, scores, threshold=threshold).to_dict()


def compare(
    *,
    roi_mode: str = "oracle",
    config: Optional[Settings] = None,
    dataset: Optional[LoadedPixelDataset] = None,
    model_version: Optional[str] = None,
    vqc_version: str = PIXEL_VQC_VERSION,
) -> Dict[str, Any]:
    """Classical and quantum metrics per ROI condition, on identical rows."""
    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)
    data = dataset or load_pixel_partitions(config=cfg, roi_mode=roi_mode)

    if data.qubit_count != V1_QUBIT_COUNT:
        raise ComparisonError(
            f"Pixel cache reports {data.qubit_count} qubits; V1 is fixed at "
            f"{V1_QUBIT_COUNT}."
        )

    logger.info("Refitting the classical reference from its persisted record...")
    classical = classical_validation_scores(config=cfg, model_version=model_version)

    partition = data.validation
    conditions = condition_rows(partition)
    directory = store.pixel_vqc_dir(vqc_version)

    rows_out: List[Dict[str, Any]] = []
    for name, rows in sorted(conditions.items()):
        labels = partition.labels[rows]
        image_ids = [partition.image_ids[int(r)] for r in rows]

        entry: Dict[str, Any] = {
            "condition": name,
            "description": CONDITION_DESCRIPTIONS.get(name, ""),
            "partition": "validation",
            "n_samples": int(rows.size),
            "n_positive": int(labels.sum()),
            "n_patients": len({partition.patient_ids[int(r)] for r in rows}),
            "prevalence": round(float(labels.mean()), 6) if rows.size else None,
            "classical": {},
            "quantum": None,
        }

        # Classical, restricted to the images this condition contains. Images the
        # descriptor cache rejected are named rather than silently dropped.
        for baseline_name, by_id in classical.items():
            available = [i for i in image_ids if i in by_id]
            missing = len(image_ids) - len(available)
            mask = np.asarray([i in by_id for i in image_ids])
            subset_metrics = _metrics_for_subset(
                labels[mask], np.asarray([by_id[i] for i in available])
            )
            entry["classical"][baseline_name] = {
                "n_scored": len(available),
                "n_not_in_descriptor_cache": missing,
                "metrics": subset_metrics,
            }

        artifact = directory / f"{roi_mode}_{name}_pixel_vqc.json"
        if artifact.exists():
            entry["quantum"] = _quantum_entry(artifact, data, rows, cfg, scored_condition=name)
        else:
            primary = PRIMARY_CONDITION.get(roi_mode)
            borrowed = directory / f"{roi_mode}_{primary}_pixel_vqc.json" if primary else None
            if borrowed is not None and primary != name and borrowed.exists():
                entry["quantum"] = _quantum_entry(
                    borrowed, data, rows, cfg, scored_condition=name
                )
                entry["quantum_note"] = (
                    f"No pixel VQC was trained on {name}; the model trained on "
                    f"{primary} was applied to these rows. This is the deployed "
                    "behaviour for a condition that cannot be selected on, and the "
                    "rows stay separately identifiable rather than being merged into "
                    f"the {primary} statistic."
                )
            else:
                entry["quantum_note"] = (
                    f"No trained pixel VQC for this condition at {artifact.name}, and "
                    f"no {roi_mode} primary-condition model to borrow. Conditions are "
                    "trained separately and are not interchangeable."
                )
        rows_out.append(entry)

    return {
        "comparison_version": COMPARISON_VERSION,
        "roi_mode": roi_mode,
        "partition": "validation",
        "test_partition_used": False,
        "amplitudes_per_state": V1_PIXEL_COUNT,
        "n_qubits": V1_QUBIT_COUNT,
        "alignment_note": (
            "Classical and quantum metrics on each row are computed over the same "
            "image ids, so prevalence is identical and PR-AUC is comparable. Numbers "
            "across different conditions are not comparable to each other, because "
            "their prevalences differ."
        ),
        "classical_reference": {
            "model_version": store.resolve_version(model_version),
            "input": "181 fused descriptor + clinical features, PCA-reduced",
            "rebuilt_from": BASELINES_FILE,
        },
        "conditions": rows_out,
    }


def _quantum_entry(
    artifact: Path,
    data: LoadedPixelDataset,
    rows: np.ndarray,
    cfg: Settings,
    *,
    scored_condition: str,
) -> Dict[str, Any]:
    """Re-score a persisted pixel VQC on ``rows`` and report it.

    Re-scored from the stored weights rather than read back from the training
    artifact's own metric block: that proves the persisted weights reproduce the
    reported number, which is the property serialisation tests assert in the small
    and this asserts on the real data.

    ``scored_condition`` names the condition whose rows are being scored. When the
    artifact was fitted on a different condition the weights were borrowed, the row
    sets differ, and the reproducibility check is not applicable rather than failing.
    """
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    selected = payload["validation_selection"]["selected_config"]
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
    scores = probabilities_for_rows(model, data.validation, rows)
    labels = data.validation.labels[rows]
    reported = payload["validation_selection"]["selected_validation_pr_auc"]
    rescored = _metrics_for_subset(labels, scores)
    own = selected["condition"] == scored_condition
    if not own:
        matches: Optional[bool] = None
    elif rescored is None or reported is None or rescored.get("pr_auc") is None:
        matches = None
    else:
        matches = abs(float(rescored["pr_auc"]) - float(reported)) < 1e-6
    return {
        "artifact": artifact.name,
        "selected_label": payload["validation_selection"]["selected_label"],
        "weights_trained_on_condition": selected["condition"],
        "own_condition_artifact": own,
        "n_parameters": selected["n_parameters"],
        "n_layers": selected["n_layers"],
        "metrics": rescored,
        "reported_pr_auc_at_training_time": reported if own else None,
        "rescoring_matches_training_artifact": matches,
        "score_map": "(1 - <Z0>)/2, uncalibrated",
    }


def _verdict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Per condition: does the quantum model beat the best classical baseline?

    Stated as a measured comparison with the margin, not as a judgement. Where the
    quantum model was not trained, or the subset is single-class, the answer is
    "not measured" rather than a default.
    """
    out: List[Dict[str, Any]] = []
    for entry in payload["conditions"]:
        quantum = entry.get("quantum")
        best_classical = None
        best_name = None
        for name, block in entry["classical"].items():
            metrics = block.get("metrics")
            if metrics and metrics.get("pr_auc") is not None:
                if best_classical is None or metrics["pr_auc"] > best_classical:
                    best_classical, best_name = float(metrics["pr_auc"]), name
        quantum_pr = (
            quantum["metrics"]["pr_auc"]
            if quantum and quantum.get("metrics") and quantum["metrics"].get("pr_auc")
            is not None
            else None
        )
        out.append({
            "condition": entry["condition"],
            "n_samples": entry["n_samples"],
            "n_positive": entry["n_positive"],
            "prevalence": entry["prevalence"],
            "best_classical": best_name,
            "best_classical_pr_auc": best_classical,
            "quantum_pr_auc": quantum_pr,
            "quantum_weights_trained_on": (
                quantum.get("weights_trained_on_condition") if quantum else None
            ),
            "quantum_weights_borrowed": (
                None if not quantum else not quantum.get("own_condition_artifact", True)
            ),
            "quantum_beats_classical": (
                None
                if quantum_pr is None or best_classical is None
                else bool(quantum_pr > best_classical)
            ),
            "margin": (
                None
                if quantum_pr is None or best_classical is None
                else round(quantum_pr - best_classical, 6)
            ),
        })
    return {"per_condition": out}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the classical reference and the 16-qubit pixel VQC on "
        "identical validation rows."
    )
    parser.add_argument("--roi", dest="roi_mode", default="oracle",
                        choices=("oracle", "predicted"))
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--vqc-version", default=PIXEL_VQC_VERSION)
    parser.add_argument("--out", default=None, help="Write the full payload here.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("qiskit", "stevedore", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    payload = compare(
        roi_mode=args.roi_mode,
        model_version=args.model_version,
        vqc_version=args.vqc_version,
    )
    payload["verdict"] = _verdict(payload)
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Full payload -> %s", args.out)
    print(json.dumps(payload["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
