"""Train the 16-qubit V1 VQC on the 65,536-amplitude pixel path.

This is the pixel-path counterpart of :mod:`backend.training.train_quantum`, which
trains the same ansatz on the 163-value descriptor vector. They are deliberately
separate modules writing separate artifacts: one reads engineered features through a
fitted PCA, the other reads raw localised pixels through no fitted transform at all,
and a single trainer with a mode flag would make a loaded model's input
representation ambiguous.

What this module does *not* do
-----------------------------
It does not re-derive amplitudes. Normalisation lives in exactly one place --
:func:`backend.ml.pixel_pipeline.amplitudes_from_grayscale_batch` -- and this module
calls it through :meth:`backend.training.pixel_data.PixelPartition.amplitudes`. It
also does not call :meth:`VariationalQuantumClassifier.fit`, because that method
encodes its input with :class:`~quantum_ml.quantum_encoder.QuantumEncoder` first,
which would be a *second* normalisation of an already-normalised state. Everything
else on the classifier -- the ansatz, the binding, the observable, the loss, the
serialisation -- is reused unchanged.

Memory
------
A 1,692-sample float64 amplitude matrix is 846 MB, against 3.59 GB measured
available. The objective therefore streams: the uint8 cache block stays resident
(111 MB) and amplitudes are derived per chunk inside every evaluation. The loss is
accumulated as a weighted sum across chunks, which is arithmetically identical to
:meth:`VariationalQuantumClassifier.loss_from_states` on the whole matrix -- asserted
by ``test_chunked_loss_matches_whole_batch_loss``.

Train / validation / test discipline
------------------------------------
Fitting reads the train partition only. Validation chooses the ansatz depth, the
optimiser settings and the decision threshold. The test partition is not loaded by
this module at all -- final evaluation is a separate entry point that runs after the
artifact is frozen, and ``test_partition_used: false`` is written into every artifact
this module produces.

ROI conditions
--------------
One invocation trains one condition, named explicitly. Conditions are never pooled,
and the oracle condition is trained on ``A_lesion_polygon`` as well as ``A_all``
because the pooled oracle cache's ROI *selection* encodes the label (DEC-024). Each
condition gets its own candidate sweep and its own selected configuration.

Usage::

    python -m backend.training.train_pixel_vqc --roi oracle --condition A_lesion_polygon
    python -m backend.training.train_pixel_vqc --roi predicted --condition B_localized \\
        --layers 1 2 3 --maxiter 120
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.evaluation.metrics import (
    evaluate_predictions,
    sensitivity_at_specificity,
    threshold_for_sensitivity,
    threshold_sweep,
)
from backend.ml.artifacts import (
    ArtifactStore,
    PIXEL_VQC_FILE,
    PIXEL_VQC_SELECTION_FILE,
)
from backend.ml.pixel_pipeline import (
    V1_PIXEL_COUNT,
    V1_PIXEL_PREPROCESSING_VERSION,
    V1_QUBIT_COUNT,
    V1_RESAMPLE_NAME,
    V1_ROI_EDGE_PX,
)
from backend.training.pixel_data import (
    CONDITION_DESCRIPTIONS,
    LoadedPixelDataset,
    PixelPartition,
    condition_rows,
    load_pixel_partitions,
)
from quantum_ml.backends import build_backend
from quantum_ml.optimizers import build_optimizer
from quantum_ml.results import ExecutionMode, OptimizerName
from quantum_ml.vqc_classifier import EPSILON, VariationalQuantumClassifier

logger = logging.getLogger(__name__)

PIXEL_VQC_VERSION = "v1-pixel-vqc-1"

# Rows per amplitude derivation inside the objective. 256 float64 states is 134 MB,
# which keeps the objective's working set near 250 MB including the resident uint8
# block. Larger chunks do not measurably speed Aer up, because Aer already submits
# the whole chunk as one job and spreads it across cores.
CHUNK_ROWS = 256

# IBM Eagle/Heron native gate set, for the hardware-cost report. Fixed here rather
# than queried from a live backend so the reported numbers are reproducible without
# credentials.
IBM_BASIS_GATES = ("rz", "sx", "x", "cx")
TRANSPILE_OPTIMIZATION_LEVEL = 1


class PixelVQCError(RuntimeError):
    """Raised when the pixel-path VQC cannot be trained or persisted."""


# ------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class PixelVQCConfig:
    """One fully specified training configuration.

    Frozen and serialised verbatim into the artifact: a configuration that cannot be
    read back exactly is not a reproducible one.
    """

    roi_mode: str
    condition: str
    n_layers: int
    optimizer: str = "spsa"
    maxiter: int = 100
    learning_rate: Optional[float] = None
    perturbation: Optional[float] = None
    seed: int = 42
    class_weighting: bool = True

    @property
    def label(self) -> str:
        return (
            f"{self.roi_mode}/{self.condition}/{self.optimizer}"
            f"/L{self.n_layers}/it{self.maxiter}/s{self.seed}"
        )

    @property
    def n_parameters(self) -> int:
        return 2 * V1_QUBIT_COUNT * self.n_layers

    def describe(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["n_qubits"] = V1_QUBIT_COUNT
        payload["n_parameters"] = self.n_parameters
        payload["amplitudes_per_state"] = V1_PIXEL_COUNT
        return payload


@dataclass
class Candidate:
    """One trained configuration and everything measured about it."""

    config: PixelVQCConfig
    weights: np.ndarray
    initial_objective: float
    final_objective: float
    best_objective: float
    objective_history: List[float]
    validation_trace: List[Dict[str, float]]
    n_iterations: int
    n_objective_evaluations: int
    n_circuit_evaluations: int
    duration_seconds: float
    converged: bool
    train_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    class_weights: Tuple[float, float]
    notes: List[str] = field(default_factory=list)

    @property
    def selection_score(self) -> float:
        """Validation PR-AUC. ``-inf`` when undefined, so it can never be selected."""
        value = self.validation_metrics.get("pr_auc")
        return float("-inf") if value is None else float(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.describe(),
            "label": self.config.label,
            "weights": [float(w) for w in self.weights],
            "training": {
                "initial_objective": round(self.initial_objective, 8),
                "final_objective": round(self.final_objective, 8),
                "best_objective": round(self.best_objective, 8),
                "objective_improvement": round(
                    self.initial_objective - self.final_objective, 8
                ),
                "n_iterations": self.n_iterations,
                "n_objective_evaluations": self.n_objective_evaluations,
                "n_circuit_evaluations": self.n_circuit_evaluations,
                "duration_seconds": round(self.duration_seconds, 2),
                "converged": self.converged,
                "objective_history": [round(v, 8) for v in self.objective_history],
                "validation_trace": self.validation_trace,
            },
            "train_metrics_uncalibrated": self.train_metrics,
            "validation_metrics_uncalibrated": self.validation_metrics,
            "selection_score_validation_pr_auc": (
                None if self.selection_score == float("-inf") else self.selection_score
            ),
            "notes": list(self.notes),
        }


# ------------------------------------------------------------------- the objective
def probabilities_for_rows(
    model: VariationalQuantumClassifier,
    partition: PixelPartition,
    rows: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    chunk_rows: int = CHUNK_ROWS,
) -> np.ndarray:
    """``(n,)`` uncalibrated scores :math:`(1 - \\langle Z_0 \\rangle)/2`.

    Chunked so peak memory is one amplitude block. The chunk boundary cannot change
    the result: the samples are independent, and each chunk goes through the same
    :meth:`~quantum_ml.vqc_classifier.VariationalQuantumClassifier.expectation_states`
    call a whole-batch evaluation would use.
    """
    rows = np.asarray(rows, dtype=int)
    if rows.size == 0:
        return np.empty(0, dtype=np.float64)
    out = np.empty(rows.size, dtype=np.float64)
    cursor = 0
    for start in range(0, rows.size, max(1, chunk_rows)):
        block = rows[start:start + chunk_rows]
        values = model.probabilities_from_states(partition.amplitudes(block), weights)
        out[cursor:cursor + values.size] = values
        cursor += values.size
    return out


def loss_from_probabilities(
    probabilities: np.ndarray,
    y: np.ndarray,
    class_weights: Tuple[float, float],
) -> float:
    """Class-weighted binary cross-entropy from scores that are already computed.

    The arithmetic is copied from
    :meth:`~quantum_ml.vqc_classifier.VariationalQuantumClassifier.loss_from_states`
    -- same ``EPSILON``, same clip, same weighted average -- with only the
    expectation-value step removed, because chunked evaluation has already done it.
    ``test_chunked_loss_matches_whole_batch_loss`` asserts the two agree exactly, so
    this is a factoring of one implementation rather than a second one.
    """
    p = np.clip(np.asarray(probabilities, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    y = np.asarray(y, dtype=np.float64).ravel()
    weight_negative, weight_positive = class_weights
    sample_weights = np.where(y == 1, weight_positive, weight_negative)
    losses = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.average(losses, weights=sample_weights))


# --------------------------------------------------------------------- one candidate
def train_candidate(
    config: PixelVQCConfig,
    dataset: LoadedPixelDataset,
    *,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    validation_every: int = 0,
    progress_every: int = 10,
    chunk_rows: int = CHUNK_ROWS,
    app_config: Optional[Settings] = None,
) -> Candidate:
    """Fit one configuration on train rows and score it on validation rows.

    Args:
        validation_every: record a validation loss every N objective evaluations.
            Reporting only -- it is never used to stop the run. SPSA's trajectory is
            stochastic, so stopping on a noisy validation reading would select the
            parameters that fit the noise. Selection happens *across* candidates,
            after each has run its full budget.
    """
    cfg = app_config or default_settings
    train = dataset.train
    validation = dataset.validation
    train_rows = np.asarray(train_rows, dtype=int)
    validation_rows = np.asarray(validation_rows, dtype=int)

    y_train = train.labels[train_rows]
    y_validation = validation.labels[validation_rows]
    if y_train.size == 0:
        raise PixelVQCError(f"Condition {config.condition!r} has no training rows.")
    if len(set(y_train.tolist())) < 2:
        raise PixelVQCError(
            f"Condition {config.condition!r} has a single class in train "
            f"({int(y_train.sum())} positive of {y_train.size}); cross-entropy has no "
            "gradient and no metric is defined."
        )

    backend = build_backend(
        mode=ExecutionMode.IDEAL_SIMULATION, shots=cfg.QUANTUM_SHOTS, seed=config.seed
    )
    model = VariationalQuantumClassifier(
        num_qubits=V1_QUBIT_COUNT,
        num_layers=config.n_layers,
        shots=cfg.QUANTUM_SHOTS,
        backend=backend,
        seed=config.seed,
        class_weighting=config.class_weighting,
        model_version=PIXEL_VQC_VERSION,
    )
    if model.num_params != config.n_parameters:
        raise PixelVQCError(
            f"Ansatz has {model.num_params} parameters but the configuration declares "
            f"{config.n_parameters}."
        )

    class_weights = model._class_weights(y_train)  # noqa: SLF001 - one shared rule
    spec = build_optimizer(
        config.optimizer,
        maxiter=config.maxiter,
        learning_rate=config.learning_rate,
        perturbation=config.perturbation,
        seed=config.seed,
    )

    validation_trace: List[Dict[str, float]] = []
    state = {"calls": 0}

    def objective(theta: np.ndarray) -> float:
        probabilities = probabilities_for_rows(
            model, train, train_rows, theta, chunk_rows=chunk_rows
        )
        return loss_from_probabilities(probabilities, y_train, class_weights)

    def on_evaluation(index: int, value: float) -> None:
        state["calls"] = index
        if progress_every and index % progress_every == 0:
            logger.info("    eval %4d  train loss %.6f", index, value)
        if validation_every and index % validation_every == 0:
            probabilities = probabilities_for_rows(
                model, validation, validation_rows, model.weights, chunk_rows=chunk_rows
            )
            validation_trace.append(
                {
                    "objective_evaluation": int(index),
                    "train_loss": round(float(value), 8),
                    "validation_loss": round(
                        loss_from_probabilities(
                            probabilities, y_validation, class_weights
                        ),
                        8,
                    ),
                }
            )

    logger.info(
        "  %s: %d train rows (%d positive), %d parameters, class weights "
        "(neg %.3f, pos %.3f)",
        config.label,
        train_rows.size,
        int(y_train.sum()),
        model.num_params,
        class_weights[0],
        class_weights[1],
    )
    started = time.perf_counter()
    result = spec.minimize(objective, model.weights, on_evaluation=on_evaluation)
    duration = time.perf_counter() - started

    model.weights = result.parameters
    model.trained = True

    train_probabilities = probabilities_for_rows(
        model, train, train_rows, chunk_rows=chunk_rows
    )
    validation_probabilities = probabilities_for_rows(
        model, validation, validation_rows, chunk_rows=chunk_rows
    )
    train_metrics = evaluate_predictions(y_train, train_probabilities, threshold=0.50)
    validation_metrics = evaluate_predictions(
        y_validation, validation_probabilities, threshold=0.50
    )

    notes = [
        "Scores are (1 - <Z0>)/2, bounded and monotone but not calibrated "
        "probabilities. Metrics at threshold 0.50 are reported for comparability, "
        "not as an operating point.",
        f"Validation loss recorded every {validation_every} objective evaluations for "
        "reporting only; the run was not stopped on it."
        if validation_every
        else "No validation trace recorded for this candidate.",
    ]

    logger.info(
        "    loss %.6f -> %.6f in %.1fs | validation ROC-AUC %s PR-AUC %s",
        result.initial_objective if result.initial_objective is not None else float("nan"),
        result.objective,
        duration,
        validation_metrics.roc_auc,
        validation_metrics.pr_auc,
    )

    return Candidate(
        config=config,
        weights=np.asarray(result.parameters, dtype=np.float64),
        initial_objective=float(
            result.initial_objective
            if result.initial_objective is not None
            else result.objective
        ),
        final_objective=float(result.objective),
        best_objective=float(
            result.best_objective if result.best_objective is not None else result.objective
        ),
        objective_history=[float(v) for v in result.history],
        validation_trace=validation_trace,
        n_iterations=int(result.n_iterations),
        n_objective_evaluations=int(result.n_function_evaluations),
        n_circuit_evaluations=int(result.n_function_evaluations) * int(train_rows.size),
        duration_seconds=duration,
        converged=bool(result.converged),
        train_metrics=train_metrics.to_dict(),
        validation_metrics=validation_metrics.to_dict(),
        class_weights=(float(class_weights[0]), float(class_weights[1])),
        notes=notes,
    )


# ------------------------------------------------------------- hardware feasibility
def hardware_feasibility(n_layers: int, *, seed: int = 42) -> Dict[str, Any]:
    """Transpiled cost of the circuit, reported separately from simulator results.

    Two costs, kept apart because they differ by three orders of magnitude:

    * the **ansatz**, which is what the variational parameters live in, and
    * the **state preparation**, which loads 65,536 amplitudes into the register.

    Training uses Aer's ``set_statevector`` for the second one. That is a simulator
    instruction: it writes the amplitudes into the simulator's memory directly. It is
    *not* a circuit and it is not a claim about hardware. The honest hardware cost of
    preparing an arbitrary 16-qubit state is the Shende-Bullock-Markov bound below,
    measured at widths that actually transpile and extrapolated with the measurement
    that supports the extrapolation shown.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import StatePreparation

    from quantum_ml.vqc_classifier import VariationalQuantumClassifier as _VQC

    backend = build_backend(mode=ExecutionMode.IDEAL_SIMULATION, shots=1024, seed=seed)
    model = _VQC(
        num_qubits=V1_QUBIT_COUNT,
        num_layers=n_layers,
        shots=1024,
        backend=backend,
        seed=seed,
    )
    bound = model.bind_ansatz(model.weights)
    transpiled = transpile(
        bound,
        basis_gates=list(IBM_BASIS_GATES),
        optimization_level=TRANSPILE_OPTIMIZATION_LEVEL,
        seed_transpiler=seed,
    )
    ansatz = {
        "n_qubits": bound.num_qubits,
        "logical_depth": int(bound.depth()),
        "logical_gate_counts": {k: int(v) for k, v in bound.count_ops().items()},
        "transpiled_depth": int(transpiled.depth()),
        "transpiled_gate_counts": {k: int(v) for k, v in transpiled.count_ops().items()},
        "transpiled_cx": int(transpiled.count_ops().get("cx", 0)),
        "basis_gates": list(IBM_BASIS_GATES),
        "optimization_level": TRANSPILE_OPTIMIZATION_LEVEL,
        "coupling_map": "none (all-to-all assumed; a real device adds SWAP overhead)",
    }

    # Measured state-preparation cost at widths that transpile in seconds. 14 qubits
    # already needs ~16k CX; 16 is included only if it completes, because the
    # transpiler's own runtime grows with the same 2**n.
    rng = np.random.default_rng(seed)
    measured: List[Dict[str, Any]] = []
    for n in (6, 8, 10, 12):
        amplitudes = rng.random(1 << n)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        circuit = QuantumCircuit(n)
        circuit.append(StatePreparation(amplitudes), range(n))
        prepared = transpile(
            circuit,
            basis_gates=list(IBM_BASIS_GATES),
            optimization_level=TRANSPILE_OPTIMIZATION_LEVEL,
            seed_transpiler=seed,
        )
        measured.append(
            {
                "n_qubits": n,
                "state_dimension": 1 << n,
                "transpiled_depth": int(prepared.depth()),
                "transpiled_cx": int(prepared.count_ops().get("cx", 0)),
            }
        )

    bound_cx = (1 << (V1_QUBIT_COUNT + 1)) - 2 * V1_QUBIT_COUNT - 2
    ratios = [row["transpiled_cx"] / row["state_dimension"] for row in measured]
    extrapolated_cx = int(round(float(np.mean(ratios)) * V1_PIXEL_COUNT))
    dominance = int(round(extrapolated_cx / max(1, ansatz["transpiled_cx"])))
    return {
        "note": (
            "Reported separately from every simulator result. No simulator number in "
            "this artifact should be read as a hardware feasibility claim."
        ),
        "simulator_shortcut": {
            "instruction": "qiskit_aer set_statevector",
            "is_a_circuit": False,
            "statement": (
                "Training loads the 65,536 amplitudes with Aer's set_statevector, "
                "which writes them into simulator memory. It is not state "
                "preparation on hardware and implies nothing about hardware "
                "feasibility."
            ),
        },
        "ansatz": ansatz,
        "state_preparation_measured": measured,
        "state_preparation_at_16_qubits": {
            "measured": False,
            "reason": (
                "Transpiling an arbitrary 65,536-amplitude StatePreparation is itself "
                "2**16-scale work; the smaller widths above are measured instead and "
                "the extrapolation is shown rather than asserted."
            ),
            "cx_per_state_dimension_measured": [round(r, 4) for r in ratios],
            "extrapolated_cx": extrapolated_cx,
            "shende_bullock_markov_upper_bound_cx": int(bound_cx),
            "interpretation": (
                f"Preparing one arbitrary 16-qubit state costs on the order of "
                f"{extrapolated_cx:,} CX gates, against an upper bound of "
                f"{bound_cx:,}. The ansatz itself needs {ansatz['transpiled_cx']} CX. "
                f"State preparation therefore dominates the circuit by roughly "
                f"{dominance:,}x, and against current hardware coherence times this "
                "circuit is not runnable. That is a property of amplitude-encoding "
                "65,536 values, not of the ansatz."
            ),
        },
    }


# --------------------------------------------------------------------- orchestration
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_condition(
    dataset: LoadedPixelDataset, condition: str
) -> Tuple[np.ndarray, np.ndarray]:
    """``(train_rows, validation_rows)`` for a named condition, or a clear failure."""
    train_conditions = condition_rows(dataset.train)
    validation_conditions = condition_rows(dataset.validation)
    if condition not in train_conditions:
        raise PixelVQCError(
            f"Condition {condition!r} does not exist in the {dataset.train.roi_mode!r} "
            f"cache. Available: {sorted(train_conditions)}."
        )
    if condition not in validation_conditions:
        raise PixelVQCError(
            f"Condition {condition!r} exists in train but not in validation, so it "
            "cannot be selected on. Available in validation: "
            f"{sorted(validation_conditions)}."
        )
    return train_conditions[condition], validation_conditions[condition]


def train(
    *,
    roi_mode: str = "oracle",
    condition: str = "A_all",
    layers: Sequence[int] = (1, 2, 3),
    optimizers: Sequence[str] = ("spsa",),
    maxiter: int = 100,
    learning_rate: Optional[float] = None,
    perturbation: Optional[float] = None,
    seed: Optional[int] = None,
    validation_every: int = 0,
    progress_every: int = 10,
    chunk_rows: int = CHUNK_ROWS,
    dataset: Optional[LoadedPixelDataset] = None,
    config: Optional[Settings] = None,
    skip_hardware_report: bool = False,
    budget_note: str = "",
) -> Dict[str, Any]:
    """Sweep candidates, select on validation, persist. Test is never loaded.

    Returns the artifact payload, which is also written under
    ``models/pixel_vqc/<version>/``.
    """
    cfg = config or default_settings
    resolved_seed = cfg.RANDOM_SEED if seed is None else seed
    store = ArtifactStore.from_settings(cfg)
    data = dataset or load_pixel_partitions(config=cfg, roi_mode=roi_mode)

    if data.qubit_count != V1_QUBIT_COUNT:
        raise PixelVQCError(
            f"Cache reports {data.qubit_count} qubits; V1 is fixed at {V1_QUBIT_COUNT}."
        )
    if data.train.raw.shape[1] != V1_PIXEL_COUNT:
        raise PixelVQCError(
            f"Cache rows carry {data.train.raw.shape[1]} pixels; "
            f"{V1_QUBIT_COUNT} qubits address exactly {V1_PIXEL_COUNT}."
        )

    train_rows, validation_rows = _resolve_condition(data, condition)
    logger.info(
        "Condition %s (%s): %d train rows / %d positive, %d validation rows / %d "
        "positive",
        condition,
        roi_mode,
        train_rows.size,
        int(data.train.labels[train_rows].sum()),
        validation_rows.size,
        int(data.validation.labels[validation_rows].sum()),
    )

    grid = [
        PixelVQCConfig(
            roi_mode=roi_mode,
            condition=condition,
            n_layers=int(n),
            optimizer=str(name),
            maxiter=int(maxiter),
            learning_rate=learning_rate,
            perturbation=perturbation,
            seed=resolved_seed,
            class_weighting=True,
        )
        for name in optimizers
        for n in layers
    ]
    logger.info("Sweeping %d candidate configurations...", len(grid))

    candidates: List[Candidate] = []
    for candidate_config in grid:
        candidates.append(
            train_candidate(
                candidate_config,
                data,
                train_rows=train_rows,
                validation_rows=validation_rows,
                validation_every=validation_every,
                progress_every=progress_every,
                chunk_rows=chunk_rows,
                app_config=cfg,
            )
        )

    ranked = sorted(candidates, key=lambda c: -c.selection_score)
    best = ranked[0]
    if best.selection_score == float("-inf"):
        raise PixelVQCError(
            "No candidate produced a defined validation PR-AUC, so none can be "
            "selected. The validation rows for this condition are probably "
            "single-class."
        )
    logger.info(
        "Selected %s on validation PR-AUC %.6f", best.config.label, best.selection_score
    )

    threshold = _select_threshold(
        best,
        data,
        validation_rows=validation_rows,
        chunk_rows=chunk_rows,
        target_sensitivity=cfg.SCREENING_TARGET_SENSITIVITY,
        app_config=cfg,
    )

    payload = _assemble_payload(
        data=data,
        store=store,
        roi_mode=roi_mode,
        condition=condition,
        candidates=candidates,
        best=best,
        threshold=threshold,
        seed=resolved_seed,
        train_rows=train_rows,
        validation_rows=validation_rows,
        skip_hardware_report=skip_hardware_report,
        budget_note=budget_note,
    )

    directory = store.pixel_vqc_dir(PIXEL_VQC_VERSION)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / f"{roi_mode}_{condition}_{PIXEL_VQC_FILE}"
    artifact.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    selection = directory / f"{roi_mode}_{condition}_{PIXEL_VQC_SELECTION_FILE}"
    selection.write_text(
        json.dumps(payload["validation_selection"], indent=2), encoding="utf-8"
    )
    payload["artifact"] = str(artifact)
    payload["artifact_sha256"] = _sha256(artifact)
    logger.info("Artifact -> %s", artifact)
    return payload


def _select_threshold(
    best: Candidate,
    data: LoadedPixelDataset,
    *,
    validation_rows: np.ndarray,
    chunk_rows: int,
    target_sensitivity: float,
    app_config: Settings,
) -> Dict[str, Any]:
    """Choose the decision threshold on validation, and record the sweep behind it.

    The screening target sensitivity comes from configuration
    (``SCREENING_TARGET_SENSITIVITY``), not from anything measured here, so the
    operating point is a stated project intent rather than a number tuned to make a
    result look good. The full sweep is persisted so a different choice can be made
    on the evidence.
    """
    model = _rebuild_model(best, app_config)
    y = data.validation.labels[validation_rows]
    probabilities = probabilities_for_rows(
        model, data.validation, validation_rows, chunk_rows=chunk_rows
    )
    chosen = threshold_for_sensitivity(y, probabilities, target_sensitivity)
    payload: Dict[str, Any] = {
        "basis": "validation partition only",
        "target_sensitivity": target_sensitivity,
        "target_sensitivity_source": "Settings.SCREENING_TARGET_SENSITIVITY",
        "selected": chosen,
        "at_specificity_0_90": sensitivity_at_specificity(y, probabilities, 0.90),
        "sweep": threshold_sweep(y, probabilities),
        "score_distribution": {
            "min": round(float(probabilities.min()), 8),
            "max": round(float(probabilities.max()), 8),
            "mean": round(float(probabilities.mean()), 8),
            "sd": round(float(probabilities.std(ddof=1)), 8)
            if probabilities.size > 1
            else 0.0,
        },
        "note": (
            "Chosen on the uncalibrated score (1 - <Z0>)/2. Calibration is a separate "
            "stage and is also fitted without the test partition."
        ),
    }
    selected_threshold = chosen.get("threshold")
    if selected_threshold is not None:
        metrics = evaluate_predictions(y, probabilities, threshold=selected_threshold)
        payload["validation_metrics_at_selected_threshold"] = metrics.to_dict()
    return payload


def _rebuild_model(
    candidate: Candidate, app_config: Settings
) -> VariationalQuantumClassifier:
    """Reconstruct a trained model from a candidate's recorded weights."""
    backend = build_backend(
        mode=ExecutionMode.IDEAL_SIMULATION,
        shots=app_config.QUANTUM_SHOTS,
        seed=candidate.config.seed,
    )
    model = VariationalQuantumClassifier(
        num_qubits=V1_QUBIT_COUNT,
        num_layers=candidate.config.n_layers,
        shots=app_config.QUANTUM_SHOTS,
        backend=backend,
        seed=candidate.config.seed,
        class_weighting=candidate.config.class_weighting,
        model_version=PIXEL_VQC_VERSION,
    )
    model.weights = np.asarray(candidate.weights, dtype=np.float64)
    model.trained = True
    return model


def _assemble_payload(
    *,
    data: LoadedPixelDataset,
    store: ArtifactStore,
    roi_mode: str,
    condition: str,
    candidates: List[Candidate],
    best: Candidate,
    threshold: Dict[str, Any],
    seed: int,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    skip_hardware_report: bool,
    budget_note: str = "",
) -> Dict[str, Any]:
    """Everything needed to reproduce, audit, or reject this run."""
    manifest_path = store.split_manifest_path
    y_train = data.train.labels[train_rows]
    y_validation = data.validation.labels[validation_rows]

    return {
        "pixel_vqc_version": PIXEL_VQC_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_partition_used": False,
        "test_partition_note": (
            "This module never loads the test partition. Final evaluation is a "
            "separate entry point that runs after this artifact is frozen."
        ),
        "roi_mode": roi_mode,
        "condition": condition,
        "condition_description": CONDITION_DESCRIPTIONS.get(condition, ""),
        "quantum": {
            "n_qubits": V1_QUBIT_COUNT,
            "amplitudes_per_state": V1_PIXEL_COUNT,
            "encoding": "amplitude encoding: 256x256 grayscale / 255, L2-normalised",
            "observable": "Z on qubit 0",
            "score_map": "(1 - <Z0>)/2",
            "loss": "class-weighted binary cross-entropy",
            "execution_mode": ExecutionMode.IDEAL_SIMULATION.value,
            "shots": None,
            "shots_note": (
                "Exact statevector evaluation. No shot sampling, so there is no shot "
                "noise to report."
            ),
            "ansatz": (
                "per layer: ry(theta) and rz(psi) on every qubit, then a linear "
                "nearest-neighbour CNOT cascade with circular closure"
            ),
        },
        "preprocessing": {
            "preprocessing_version": V1_PIXEL_PREPROCESSING_VERSION,
            "roi_edge_px": V1_ROI_EDGE_PX,
            "resample_filter": V1_RESAMPLE_NAME,
            "cache_metadata": {
                key: (None if value is None else str(value))
                for key, value in data.cache_metadata.items()
            },
        },
        "dataset": {
            "split_manifest": str(manifest_path),
            "split_manifest_sha256": (
                _sha256(manifest_path) if manifest_path.exists() else None
            ),
            "split_seed": data.manifest.seed,
            "positive_classes": sorted(data.manifest.positive_classes),
            "train": {
                "n_samples": int(train_rows.size),
                "n_positive": int(y_train.sum()),
                "n_negative": int(train_rows.size - y_train.sum()),
                "prevalence": round(float(y_train.mean()), 6),
                "n_patients": len(
                    {data.train.patient_ids[int(r)] for r in train_rows}
                ),
            },
            "validation": {
                "n_samples": int(validation_rows.size),
                "n_positive": int(y_validation.sum()),
                "n_negative": int(validation_rows.size - y_validation.sum()),
                "prevalence": round(float(y_validation.mean()), 6),
                "n_patients": len(
                    {data.validation.patient_ids[int(r)] for r in validation_rows}
                ),
            },
            "partition_summary": data.summary(),
        },
        "seeds": {
            "random_seed": seed,
            "split_seed": data.manifest.seed,
            "note": (
                "seed_everything() seeds numpy and qiskit_algorithms.algorithm_globals "
                "inside OptimizerSpec.minimize; weight initialisation uses its own "
                "Generator(seed) so it cannot depend on optimiser call order."
            ),
        },
        "class_imbalance_handling": {
            "method": "inverse-frequency class weights normalised to mean 1",
            "applied_in": "the training loss only; no resampling, no synthetic samples",
            "weights_negative_positive": [
                round(best.class_weights[0], 6),
                round(best.class_weights[1], 6),
            ],
            "computed_by": "VariationalQuantumClassifier._class_weights",
        },
        "validation_selection": {
            "selection_metric": "PR-AUC on the validation partition",
            "selection_metric_rationale": (
                "The positive class is small, so PR-AUC discriminates between "
                "candidates where accuracy and ROC-AUC do not."
            ),
            "compute_budget": {
                "maxiter_per_candidate": [c.config.maxiter for c in candidates],
                "n_candidates": len(candidates),
                "seconds_per_objective_evaluation_measured": round(
                    best.duration_seconds / max(1, best.n_objective_evaluations), 3
                ),
                "note": budget_note or (
                    "Budget as passed on the command line; no reduction applied."
                ),
            },
            "selected_label": best.config.label,
            "selected_config": best.config.describe(),
            "selected_validation_pr_auc": best.selection_score,
            "early_stopping": {
                "used": False,
                "reason": (
                    "SPSA's trajectory is stochastic; stopping on a noisy validation "
                    "reading selects for the noise. Every candidate ran its full "
                    "iteration budget and selection happened across candidates."
                ),
            },
            "threshold": threshold,
            "ranking": [
                {
                    "label": c.config.label,
                    "validation_pr_auc": c.validation_metrics.get("pr_auc"),
                    "validation_roc_auc": c.validation_metrics.get("roc_auc"),
                    "final_train_loss": round(c.final_objective, 8),
                }
                for c in sorted(candidates, key=lambda c: -c.selection_score)
            ],
        },
        "candidates": [c.to_dict() for c in candidates],
        "selected_weights": [float(w) for w in best.weights],
        "hardware_feasibility": (
            None
            if skip_hardware_report
            else hardware_feasibility(best.config.n_layers, seed=seed)
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the 16-qubit V1 VQC on the 65,536-amplitude pixel path."
    )
    parser.add_argument(
        "--roi", dest="roi_mode", default="oracle", choices=("oracle", "predicted")
    )
    parser.add_argument(
        "--condition", default="A_all",
        help="Named ROI condition, e.g. A_all, A_lesion_polygon, B_localized.",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[1, 2, 3],
        help="Ansatz repetitions to sweep. Selection is on validation PR-AUC.",
    )
    parser.add_argument(
        "--optimizer", dest="optimizers", nargs="+", default=["spsa"],
        choices=[name.value for name in OptimizerName],
    )
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--perturbation", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--validation-every", type=int, default=0,
        help="Record validation loss every N objective evaluations (reporting only).",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    parser.add_argument(
        "--skip-hardware-report", action="store_true",
        help="Skip the transpiled-cost measurement (it takes a couple of minutes).",
    )
    parser.add_argument(
        "--budget-note", default="",
        help="Recorded in the artifact: why this condition got this iteration budget. "
             "Per-sample expectation cost falls with batch size (~29ms/sample at 215 "
             "rows, ~5.7ms/sample at 1692), so a budget cannot be inferred from the "
             "row count and the reason has to be stated.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("qiskit", "qiskit.transpiler", "qiskit.passmanager", "stevedore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    payload = train(
        roi_mode=args.roi_mode,
        condition=args.condition,
        layers=args.layers,
        optimizers=args.optimizers,
        maxiter=args.maxiter,
        learning_rate=args.learning_rate,
        perturbation=args.perturbation,
        seed=args.seed,
        validation_every=args.validation_every,
        progress_every=args.progress_every,
        chunk_rows=args.chunk_rows,
        skip_hardware_report=args.skip_hardware_report,
        budget_note=args.budget_note,
    )
    selection = payload["validation_selection"]
    print(json.dumps({
        "artifact": payload["artifact"],
        "artifact_sha256": payload["artifact_sha256"],
        "roi_mode": payload["roi_mode"],
        "condition": payload["condition"],
        "n_qubits": payload["quantum"]["n_qubits"],
        "amplitudes_per_state": payload["quantum"]["amplitudes_per_state"],
        "train": payload["dataset"]["train"],
        "validation": payload["dataset"]["validation"],
        "selected": selection["selected_label"],
        "selected_validation_pr_auc": selection["selected_validation_pr_auc"],
        "ranking": selection["ranking"],
        "threshold": selection["threshold"]["selected"],
        "test_partition_used": payload["test_partition_used"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
