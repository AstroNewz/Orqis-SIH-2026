"""Do the 16-qubit input states carry class-discriminative structure at all?

This runs *before* VQC training, and it exists to make one distinction reportable:
whether a poor quantum result means "the ansatz did not find the signal" or "there
was no signal in the state to find". Those need different responses and they are
indistinguishable from a PR-AUC alone.

What is measured
----------------
Per state (streamed, never all 1,692 at once):

* ``l2_norm`` -- must be 1. Reported as a deviation, because the whole encoding
  argument rests on it and an assertion that is never printed is not evidence.
* ``scaled_l2_norm`` -- ``||pixel/255||`` before normalisation. This is the one
  quantity normalisation destroys, so two ROIs differing only in brightness are
  indistinguishable to the circuit; recording it says how much was discarded.
* ``uniform_overlap`` -- :math:`\\langle u | \\psi \\rangle` against the uniform
  state. Grayscale pixel vectors are non-negative and mostly mid-range, so this is
  large by construction. It is a diagnostic, not a defect.
* ``residual_norm`` -- :math:`\\sqrt{1 - \\langle u|\\psi\\rangle^2}`, the part of
  the state that is *not* the common component. This is the entire budget any
  classifier has to work with.
* ``max_amplitude`` and ``participation_ratio`` -- how concentrated the state is.
  A participation ratio near 65,536 means a near-uniform state.

Across states: the Gram matrix :math:`\\langle \\psi_i | \\psi_j \\rangle`, split
into positive-positive, negative-negative and positive-negative blocks. If
within-class overlaps are not higher than between-class overlaps, no
kernel-shaped model -- and a VQC on amplitude-encoded states is kernel-shaped --
can separate the classes from these states.

The cheap baseline
------------------
The same Gram matrix *is* the linear kernel on the amplitude vectors, so a
precomputed-kernel SVM on it is the best linear classifier in the exact
65,536-dimensional space the circuit receives, for a few seconds of compute. That
is the cheap baseline the experiment design asks for before expensive VQC
training: it bounds what a variational circuit reading :math:`\\langle Z_0
\\rangle` off a linear evolution of the same state could plausibly reach, and it
is measured on the same partitions with the same labels.

It is *not* a substitute for the established classical reference baseline, which
lives in ``baselines.json`` and uses the 181-feature descriptor path. Both are
reported; they answer different questions.

ROI conditions
--------------
Every quantity is computed per ROI condition and never pooled across them, and
condition A is subdivided by which annotation supplied the ROI. That subdivision
is not decoration: measured on this dataset, **every** positive in the oracle
cache has a lesion-polygon ROI and **every** region-polygon and centre-crop row is
negative, so ROI area alone scores the label at ROC-AUC 0.78 (train) / 0.72
(validation) / 0.85 (test). :func:`roi_geometry_leak` measures that on whatever
rows it is given, so a condition-A result can be read against it instead of being
mistaken for tissue signal. See DEC-024.

The test partition is excluded by default and requires ``include_test=True``,
which only the final frozen evaluation passes.

Usage::

    python -m backend.evaluation.state_diagnostics --roi oracle
    python -m backend.evaluation.state_diagnostics --roi predicted --layers 2
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.evaluation.metrics import evaluate_predictions
from backend.ml.artifacts import ArtifactStore
from backend.ml.pixel_pipeline import V1_PIXEL_COUNT, V1_QUBIT_COUNT
from backend.training.pixel_data import (
    CONDITION_DESCRIPTIONS,
    LoadedPixelDataset,
    PixelPartition,
    condition_rows,
    load_pixel_partitions,
)
from quantum_ml.backends import build_backend
from quantum_ml.results import ExecutionMode
from quantum_ml.vqc_classifier import VariationalQuantumClassifier

logger = logging.getLogger(__name__)

STATE_DIAGNOSTICS_VERSION = "v1-state-diagnostics-1"

# 256 rows of float64 amplitudes is 134 MB. Measured available memory on the
# development machine is 3.59 GB, and the Gram accumulation holds two blocks at
# once, so this keeps the working set near 270 MB rather than the 846 MB a full
# training-partition amplitude matrix would need.
CHUNK_ROWS = 256

# Cap on the number of rows entering the Gram matrix and the cheap kernel
# baseline. 1,692 x 1,692 float64 is 23 MB, so the cap is not about the matrix --
# it is about the amplitude derivations, which cost one pass per block pair. At
# 2,048 the whole V1 training partition fits under the cap and nothing is dropped.
MAX_GRAM_ROWS = 2048


class DiagnosticsError(RuntimeError):
    """Raised when diagnostics cannot be computed from the given rows."""


def _summary(values: np.ndarray, places: int = 6) -> Dict[str, Any]:
    """Distribution summary. ``None`` throughout when there is nothing to describe."""
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {
        "n": int(finite.size),
        "mean": round(float(finite.mean()), places),
        "sd": round(float(finite.std(ddof=1)) if finite.size > 1 else 0.0, places),
        "min": round(float(finite.min()), places),
        "p05": round(float(np.percentile(finite, 5)), places),
        "median": round(float(np.median(finite)), places),
        "p95": round(float(np.percentile(finite, 95)), places),
        "max": round(float(finite.max()), places),
    }


def _as_rows(partition: PixelPartition, rows: Optional[Sequence[int]]) -> np.ndarray:
    if rows is None:
        return np.arange(len(partition), dtype=int)
    selected = np.asarray(rows, dtype=int)
    if selected.size and (selected.min() < 0 or selected.max() >= len(partition)):
        raise DiagnosticsError(
            f"Row indices out of range for partition {partition.name!r} of length "
            f"{len(partition)}."
        )
    return selected


def amplitude_blocks(
    partition: PixelPartition,
    rows: Sequence[int],
    *,
    chunk_rows: int = CHUNK_ROWS,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(row_indices, amplitudes)`` blocks, bounding peak memory.

    Deriving amplitudes per block rather than materialising the whole matrix is the
    difference between a 270 MB and an 846 MB working set on the training
    partition. The derivation itself is the single shared implementation in
    :mod:`backend.ml.pixel_pipeline`, so streaming cannot introduce a second
    normalisation.
    """
    rows = np.asarray(rows, dtype=int)
    for start in range(0, rows.size, max(1, chunk_rows)):
        block = rows[start:start + chunk_rows]
        yield block, partition.amplitudes(block)


def state_statistics(
    partition: PixelPartition,
    rows: Optional[Sequence[int]] = None,
    *,
    chunk_rows: int = CHUNK_ROWS,
) -> Dict[str, np.ndarray]:
    """Per-state scalars for the selected rows, in row order."""
    rows = _as_rows(partition, rows)
    dim = V1_PIXEL_COUNT
    uniform_scale = 1.0 / np.sqrt(dim)

    l2_norm: List[np.ndarray] = []
    uniform_overlap: List[np.ndarray] = []
    max_amplitude: List[np.ndarray] = []
    participation: List[np.ndarray] = []
    scaled_norm: List[np.ndarray] = []

    for block, states in amplitude_blocks(partition, rows, chunk_rows=chunk_rows):
        l2_norm.append(np.linalg.norm(states, axis=1))
        # <u|psi> for the uniform state is just the amplitude sum, scaled.
        uniform_overlap.append(states.sum(axis=1) * uniform_scale)
        max_amplitude.append(states.max(axis=1))
        probabilities = states * states
        participation.append(1.0 / np.sum(probabilities * probabilities, axis=1))
        # ||pixel/255||: recoverable from the raw uint8 block without re-deriving
        # the normalised state, and the quantity normalisation throws away.
        scaled = partition.raw[block].astype(np.float64) / 255.0
        scaled_norm.append(np.linalg.norm(scaled, axis=1))

    if not l2_norm:
        empty = np.empty(0, dtype=np.float64)
        return {
            "l2_norm": empty,
            "uniform_overlap": empty,
            "residual_norm": empty,
            "max_amplitude": empty,
            "participation_ratio": empty,
            "scaled_l2_norm": empty,
        }

    overlap = np.concatenate(uniform_overlap)
    return {
        "l2_norm": np.concatenate(l2_norm),
        "uniform_overlap": overlap,
        "residual_norm": np.sqrt(np.maximum(0.0, 1.0 - overlap * overlap)),
        "max_amplitude": np.concatenate(max_amplitude),
        "participation_ratio": np.concatenate(participation),
        "scaled_l2_norm": np.concatenate(scaled_norm),
    }


def gram_matrix(
    partition: PixelPartition,
    rows: Optional[Sequence[int]] = None,
    *,
    chunk_rows: int = CHUNK_ROWS,
) -> np.ndarray:
    """``(m, m)`` state overlaps :math:`\\langle \\psi_i | \\psi_j \\rangle`.

    Accumulated block-pairwise so peak memory is two amplitude blocks rather than
    the whole matrix. Only the upper triangle of block pairs is computed; the
    states are real, so the Gram matrix is symmetric.
    """
    rows = _as_rows(partition, rows)
    if rows.size > MAX_GRAM_ROWS:
        raise DiagnosticsError(
            f"Refusing to build a {rows.size}x{rows.size} Gram matrix; the cap is "
            f"{MAX_GRAM_ROWS}. Subsample explicitly so the report says what was used."
        )
    blocks = [
        (start, rows[start:start + chunk_rows])
        for start in range(0, rows.size, max(1, chunk_rows))
    ]
    out = np.empty((rows.size, rows.size), dtype=np.float64)
    for i, (start_i, block_i) in enumerate(blocks):
        states_i = partition.amplitudes(block_i)
        out[start_i:start_i + block_i.size, start_i:start_i + block_i.size] = (
            states_i @ states_i.T
        )
        for start_j, block_j in blocks[i + 1:]:
            states_j = partition.amplitudes(block_j)
            product = states_i @ states_j.T
            out[start_i:start_i + block_i.size, start_j:start_j + block_j.size] = product
            out[start_j:start_j + block_j.size, start_i:start_i + block_i.size] = product.T
    return out


def pairwise_overlap_stats(gram: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Off-diagonal overlap distributions, split by class pairing.

    The comparison that matters is ``positive_positive`` against
    ``positive_negative``: if a positive state is no more similar to another
    positive than to a negative, the states do not encode the class in a way any
    overlap-based model can read.
    """
    labels = np.asarray(labels, dtype=int).ravel()
    if gram.shape[0] != labels.size:
        raise DiagnosticsError(
            f"Gram matrix is {gram.shape[0]}x{gram.shape[1]} but {labels.size} labels "
            "were supplied."
        )
    if labels.size < 2:
        return {"note": "Fewer than two states; pairwise overlap is undefined."}

    upper = np.triu(np.ones_like(gram, dtype=bool), k=1)
    positive = labels == 1
    same_positive = np.outer(positive, positive) & upper
    same_negative = np.outer(~positive, ~positive) & upper
    cross = (np.outer(positive, ~positive) | np.outer(~positive, positive)) & upper

    payload = {
        "all_pairs": _summary(gram[upper]),
        "positive_positive": _summary(gram[same_positive]),
        "negative_negative": _summary(gram[same_negative]),
        "positive_negative": _summary(gram[cross]),
    }
    within = payload["positive_positive"]["mean"]
    between = payload["positive_negative"]["mean"]
    payload["within_minus_between_positive"] = (
        None if within is None or between is None else round(within - between, 6)
    )
    payload["interpretation"] = (
        "positive_positive minus positive_negative is the class structure available "
        "to any overlap-based model. At or below zero, the states do not separate "
        "the classes regardless of the ansatz."
    )
    return payload


def kernel_baseline(
    train_gram: np.ndarray,
    train_labels: np.ndarray,
    eval_gram: np.ndarray,
    eval_labels: np.ndarray,
    *,
    regularisation: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    seed: int = 42,
) -> Dict[str, Any]:
    """Best linear classifier in the 65,536-dimensional amplitude space.

    A precomputed-kernel SVM on the state Gram matrix. Because the kernel *is*
    :math:`\\langle \\psi_i | \\psi_j \\rangle`, this is exactly a linear model on
    the amplitudes -- the same vectors the circuit receives, with no feature
    engineering and no dimensionality reduction. It is the cheap upper reference the
    experiment design asks for before committing hours to SPSA.

    ``C`` is selected on the evaluation partition, which makes the reported number
    optimistic *for the baseline*. That direction is deliberate: the baseline is
    being used as a ceiling on the available signal, and a ceiling should not be
    understated. It is labelled as such in the payload.
    """
    from sklearn.svm import SVC

    train_labels = np.asarray(train_labels, dtype=int).ravel()
    eval_labels = np.asarray(eval_labels, dtype=int).ravel()
    if len(set(train_labels.tolist())) < 2:
        return {
            "note": "Training rows contain a single class; no baseline can be fitted."
        }

    candidates: List[Dict[str, Any]] = []
    for c in regularisation:
        model = SVC(
            C=float(c),
            kernel="precomputed",
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(train_gram, train_labels)
        scores = model.decision_function(eval_gram)
        metrics = evaluate_predictions(
            eval_labels, _rank_to_unit(scores), threshold=0.5
        )
        candidates.append(
            {
                "C": float(c),
                "n_support_vectors": int(model.n_support_.sum()),
                "roc_auc": metrics.roc_auc,
                "pr_auc": metrics.pr_auc,
            }
        )

    ranked = sorted(
        candidates, key=lambda row: (row["pr_auc"] is None, -(row["pr_auc"] or 0.0))
    )
    return {
        "model": "linear-kernel SVM on the 65,536 amplitudes (precomputed Gram)",
        "selected": ranked[0],
        "candidates": candidates,
        "n_train": int(train_labels.size),
        "n_train_positive": int(np.sum(train_labels == 1)),
        "n_eval": int(eval_labels.size),
        "n_eval_positive": int(np.sum(eval_labels == 1)),
        "selection_basis": (
            "C chosen by PR-AUC on the evaluation partition, so this figure is "
            "optimistic for the baseline. Intended as a ceiling on the linearly "
            "accessible signal in the quantum input, not as a deployable model."
        ),
    }


def _rank_to_unit(scores: np.ndarray) -> np.ndarray:
    """Map arbitrary real scores into (0, 1) monotonically.

    ROC-AUC and PR-AUC depend only on the ordering, and ``evaluate_predictions``
    validates its input as a probability. A rank transform satisfies both without
    pretending an SVM margin is a probability -- Brier and ECE from this mapping
    are meaningless and are not read from it.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.size == 0:
        return scores
    order = np.argsort(np.argsort(scores))
    return (order + 0.5) / scores.size


def observable_expectations(
    model: VariationalQuantumClassifier,
    partition: PixelPartition,
    rows: Optional[Sequence[int]] = None,
    *,
    weights: Optional[np.ndarray] = None,
    chunk_rows: int = CHUNK_ROWS,
) -> np.ndarray:
    """:math:`\\langle Z_0 \\rangle` per row, streamed through the given circuit."""
    rows = _as_rows(partition, rows)
    if rows.size == 0:
        return np.empty(0, dtype=np.float64)
    out = np.empty(rows.size, dtype=np.float64)
    cursor = 0
    for _, states in amplitude_blocks(partition, rows, chunk_rows=chunk_rows):
        values = model.expectation_states(states, weights)
        out[cursor:cursor + values.size] = values
        cursor += values.size
    return out


def observable_distribution(
    expectations: np.ndarray, labels: np.ndarray
) -> Dict[str, Any]:
    """The observable's distribution overall and per class, plus its discrimination.

    ``score`` is :math:`(1 - \\langle Z_0 \\rangle)/2`, the same monotone map the
    classifier uses, so the AUCs here are directly comparable to a trained model's.
    At untrained weights they are the null reference: whatever training achieves has
    to be read against this, not against 0.5.
    """
    expectations = np.asarray(expectations, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    if expectations.size != labels.size:
        raise DiagnosticsError(
            f"{expectations.size} expectations for {labels.size} labels."
        )
    payload: Dict[str, Any] = {
        "expectation_all": _summary(expectations),
        "expectation_positive": _summary(expectations[labels == 1]),
        "expectation_negative": _summary(expectations[labels == 0]),
    }
    positive_mean = payload["expectation_positive"]["mean"]
    negative_mean = payload["expectation_negative"]["mean"]
    payload["mean_gap_positive_minus_negative"] = (
        None
        if positive_mean is None or negative_mean is None
        else round(positive_mean - negative_mean, 8)
    )
    if expectations.size and len(set(labels.tolist())) == 2:
        scores = np.clip((1.0 - expectations) / 2.0, 0.0, 1.0)
        metrics = evaluate_predictions(labels, scores, threshold=0.5)
        payload["discrimination"] = {
            "roc_auc": metrics.roc_auc,
            "pr_auc": metrics.pr_auc,
            "prevalence": round(metrics.prevalence, 6),
            "note": (
                "PR-AUC is read against the prevalence, not against 0.5: at ~5% "
                "positive, chance-level PR-AUC is ~0.05."
            ),
        }
    else:
        payload["discrimination"] = {
            "roc_auc": None,
            "pr_auc": None,
            "note": "Single-class rows; discrimination is undefined.",
        }
    return payload


def roi_geometry_leak(
    partition: PixelPartition, rows: Optional[Sequence[int]] = None
) -> Dict[str, Any]:
    """How well the *crop geometry* alone predicts the label (DEC-024).

    Not a property of the quantum state; a property of how the ROI was chosen. It
    belongs in this report because a condition-A discrimination number cannot be
    interpreted without it. Score is ``1 - area_fraction``, i.e. "the tighter the
    crop, the more suspicious", which is the direction annotation practice creates.
    """
    rows = _as_rows(partition, rows)
    labels = partition.labels[rows]
    area = partition.roi_area_fraction[rows]
    finite = np.isfinite(area)
    sources = [partition.roi_sources[int(r)] for r in rows]
    payload: Dict[str, Any] = {
        "roi_area_fraction_all": _summary(area),
        "roi_area_fraction_positive": _summary(area[labels == 1]),
        "roi_area_fraction_negative": _summary(area[labels == 0]),
        "roi_source_label_counts": {
            source: {
                "n": sources.count(source),
                "n_positive": int(
                    sum(
                        1
                        for position, name in enumerate(sources)
                        if name == source and labels[position] == 1
                    )
                ),
            }
            for source in sorted(set(sources))
        },
    }
    if finite.sum() >= 2 and len(set(labels[finite].tolist())) == 2:
        metrics = evaluate_predictions(
            labels[finite], _rank_to_unit(-area[finite]), threshold=0.5
        )
        payload["area_only_roc_auc"] = metrics.roc_auc
        payload["area_only_pr_auc"] = metrics.pr_auc
    else:
        payload["area_only_roc_auc"] = None
        payload["area_only_pr_auc"] = None
    payload["interpretation"] = (
        "A high area-only ROC-AUC means the ROI selection rule already encodes the "
        "label, so any classifier reading these crops can score above chance "
        "without learning anything about tissue. Compare a model's ROC-AUC against "
        "this number, not against 0.5."
    )
    return payload


@dataclass
class ConditionDiagnostics:
    """Every diagnostic for one (partition, ROI condition) pair."""

    partition: str
    roi_mode: str
    condition: str
    n_samples: int
    n_positive: int
    n_negative: int
    n_patients: int
    states: Dict[str, Any] = field(default_factory=dict)
    pairwise: Dict[str, Any] = field(default_factory=dict)
    observable: Dict[str, Any] = field(default_factory=dict)
    roi_geometry: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "partition": self.partition,
            "roi_mode": self.roi_mode,
            "condition": self.condition,
            "condition_description": CONDITION_DESCRIPTIONS.get(self.condition, ""),
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_patients": self.n_patients,
            "states": self.states,
            "pairwise_overlaps": self.pairwise,
            "observable_at_untrained_weights": self.observable,
            "roi_geometry_leak": self.roi_geometry,
        }
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


def diagnose_condition(
    partition: PixelPartition,
    condition: str,
    rows: Sequence[int],
    *,
    model: Optional[VariationalQuantumClassifier] = None,
    gram_subsample: Optional[int] = None,
    seed: int = 42,
    chunk_rows: int = CHUNK_ROWS,
) -> ConditionDiagnostics:
    """All diagnostics for one ROI condition within one partition.

    Args:
        gram_subsample: cap on rows entering the Gram matrix. When the condition has
            more rows than this, **every positive is kept** and negatives are drawn
            with a seeded generator, because a head-of-array slice of this cache
            contains no positives at all and would silently produce nan class
            statistics.
    """
    rows = _as_rows(partition, rows)
    labels = partition.labels[rows]
    notes: List[str] = []

    diagnostics = ConditionDiagnostics(
        partition=partition.name,
        roi_mode=partition.roi_mode,
        condition=condition,
        n_samples=int(rows.size),
        n_positive=int(labels.sum()),
        n_negative=int(rows.size - labels.sum()),
        n_patients=len({partition.patient_ids[int(r)] for r in rows}),
    )
    if rows.size == 0:
        notes.append("No rows in this condition.")
        diagnostics.notes = notes
        return diagnostics

    stats = state_statistics(partition, rows, chunk_rows=chunk_rows)
    norm_deviation = float(np.max(np.abs(stats["l2_norm"] - 1.0)))
    diagnostics.states = {
        "qubit_count": V1_QUBIT_COUNT,
        "amplitudes_per_state": V1_PIXEL_COUNT,
        "l2_norm_max_deviation_from_one": float(f"{norm_deviation:.3e}"),
        "l2_norm": _summary(stats["l2_norm"], places=12),
        "scaled_l2_norm": _summary(stats["scaled_l2_norm"]),
        "uniform_overlap": _summary(stats["uniform_overlap"]),
        "uniform_overlap_positive": _summary(stats["uniform_overlap"][labels == 1]),
        "uniform_overlap_negative": _summary(stats["uniform_overlap"][labels == 0]),
        "residual_norm": _summary(stats["residual_norm"]),
        "residual_norm_positive": _summary(stats["residual_norm"][labels == 1]),
        "residual_norm_negative": _summary(stats["residual_norm"][labels == 0]),
        "max_amplitude": _summary(stats["max_amplitude"]),
        "participation_ratio": _summary(stats["participation_ratio"], places=2),
    }

    gram_rows = rows
    cap = gram_subsample or MAX_GRAM_ROWS
    if rows.size > cap:
        positives = rows[labels == 1]
        negatives = rows[labels == 0]
        take = max(0, cap - positives.size)
        rng = np.random.default_rng(seed)
        drawn = (
            rng.choice(negatives, size=min(take, negatives.size), replace=False)
            if negatives.size
            else negatives
        )
        gram_rows = np.sort(np.concatenate([positives, drawn]))
        notes.append(
            f"Pairwise overlaps and the kernel baseline use {gram_rows.size} of "
            f"{rows.size} rows: all {positives.size} positives plus "
            f"{drawn.size} negatives drawn with seed {seed}."
        )
    gram = gram_matrix(partition, gram_rows, chunk_rows=chunk_rows)
    diagnostics.pairwise = pairwise_overlap_stats(gram, partition.labels[gram_rows])
    diagnostics.pairwise["n_states"] = int(gram_rows.size)

    if model is not None:
        expectations = observable_expectations(
            model, partition, rows, chunk_rows=chunk_rows
        )
        diagnostics.observable = observable_distribution(expectations, labels)
        diagnostics.observable["weights"] = "untrained (seeded initialisation)"
        diagnostics.observable["n_layers"] = model.num_layers
        diagnostics.observable["n_parameters"] = model.num_params
        diagnostics.observable["seed"] = model.seed
        diagnostics.observable["observable"] = "Z on qubit 0"

    diagnostics.roi_geometry = roi_geometry_leak(partition, rows)
    diagnostics.notes = notes
    return diagnostics


def _build_untrained_model(
    *, n_layers: int, seed: int, config: Settings
) -> VariationalQuantumClassifier:
    # Ideal mode resolves its own ``shots=None`` internally -- that is what makes
    # ``backend.is_exact`` true and routes the 16-qubit batch through Aer rather
    # than through a 68 GB materialised unitary. The int is passed anyway so the
    # constructor's ``int(shots)`` has something to read.
    backend = build_backend(
        mode=ExecutionMode.IDEAL_SIMULATION, shots=config.QUANTUM_SHOTS, seed=seed
    )
    return VariationalQuantumClassifier(
        num_qubits=V1_QUBIT_COUNT,
        num_layers=n_layers,
        shots=config.QUANTUM_SHOTS,
        backend=backend,
        seed=seed,
        class_weighting=True,
        model_version="",
    )


def run_diagnostics(
    *,
    roi_mode: str = "oracle",
    n_layers: int = 2,
    seed: Optional[int] = None,
    include_test: bool = False,
    gram_subsample: Optional[int] = None,
    skip_observable: bool = False,
    chunk_rows: int = CHUNK_ROWS,
    dataset: Optional[LoadedPixelDataset] = None,
    config: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Diagnose every ROI condition in every allowed partition, and persist it.

    ``include_test`` defaults to False. The diagnostics are label-conditional --
    positive-versus-negative overlaps, positive-versus-negative observable
    distributions -- so running them on the held-out partition before the pipeline
    is frozen would be looking at test labels. Only the final frozen evaluation
    passes True.
    """
    cfg = config or default_settings
    resolved_seed = cfg.RANDOM_SEED if seed is None else seed
    store = ArtifactStore.from_settings(cfg)
    data = dataset or load_pixel_partitions(config=cfg, roi_mode=roi_mode)

    if data.qubit_count != V1_QUBIT_COUNT:
        raise DiagnosticsError(
            f"Cache reports {data.qubit_count} qubits; V1 is fixed at {V1_QUBIT_COUNT}."
        )

    model = (
        None
        if skip_observable
        else _build_untrained_model(n_layers=n_layers, seed=resolved_seed, config=cfg)
    )
    wanted = ["train", "validation"] + (["test"] if include_test else [])

    results: List[Dict[str, Any]] = []
    for name in wanted:
        partition = data.partitions[name]
        for condition, rows in condition_rows(partition).items():
            logger.info(
                "%s / %s: %d rows (%d positive)",
                name,
                condition,
                rows.size,
                int(partition.labels[rows].sum()),
            )
            diagnostics = diagnose_condition(
                partition,
                condition,
                rows,
                model=model,
                gram_subsample=gram_subsample,
                seed=resolved_seed,
                chunk_rows=chunk_rows,
            )
            results.append(diagnostics.to_dict())
            overlap = diagnostics.states["uniform_overlap"]["mean"]
            gap = diagnostics.pairwise.get("within_minus_between_positive")
            logger.info(
                "    uniform overlap %.5f | pos-pos minus pos-neg %s | "
                "untrained PR-AUC %s",
                overlap,
                "n/a" if gap is None else f"{gap:+.5f}",
                diagnostics.observable.get("discrimination", {}).get("pr_auc", "n/a"),
            )

    baseline = _cheap_baseline(
        data, roi_mode=roi_mode, seed=resolved_seed, chunk_rows=chunk_rows
    )

    payload: Dict[str, Any] = {
        "diagnostics_version": STATE_DIAGNOSTICS_VERSION,
        "roi_mode": roi_mode,
        "qubit_count": data.qubit_count,
        "amplitudes_per_state": V1_PIXEL_COUNT,
        "cache_metadata": {
            key: (None if value is None else str(value))
            for key, value in data.cache_metadata.items()
        },
        "split_seed": data.manifest.seed,
        "random_seed": resolved_seed,
        "n_ansatz_layers": None if model is None else model.num_layers,
        "test_partition_used": include_test,
        "partitions_diagnosed": wanted,
        "dataset_summary": data.summary(),
        "conditions": results,
        "cheap_baseline": baseline,
        "condition_descriptions": CONDITION_DESCRIPTIONS,
    }
    path = store.state_diagnostics_path(roi_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Diagnostics -> %s", path)
    payload["artifact"] = str(path)
    return payload


def _cheap_baseline(
    data: LoadedPixelDataset,
    *,
    roi_mode: str,
    seed: int,
    chunk_rows: int,
) -> Dict[str, Any]:
    """Linear-kernel baseline on the exact quantum input, train -> validation.

    Fitted on train, scored on validation. Never on test: this is a pre-training
    reference, and the held-out partition is not available for one.
    """
    train = data.train
    validation = data.validation
    train_rows = _as_rows(train, None)
    if train_rows.size > MAX_GRAM_ROWS:
        positives = train_rows[train.labels[train_rows] == 1]
        negatives = train_rows[train.labels[train_rows] == 0]
        rng = np.random.default_rng(seed)
        take = max(0, MAX_GRAM_ROWS - positives.size)
        train_rows = np.sort(
            np.concatenate(
                [positives, rng.choice(negatives, size=min(take, negatives.size),
                                       replace=False)]
            )
        )

    logger.info(
        "Cheap baseline: linear kernel on %d train x %d validation states...",
        train_rows.size,
        len(validation),
    )
    train_gram = gram_matrix(train, train_rows, chunk_rows=chunk_rows)
    cross = _cross_gram(validation, train, train_rows, chunk_rows=chunk_rows)
    payload = kernel_baseline(
        train_gram,
        train.labels[train_rows],
        cross,
        validation.labels,
        seed=seed,
    )
    payload["roi_mode"] = roi_mode
    payload["fitted_on"] = "train"
    payload["scored_on"] = "validation"
    payload["test_partition_used"] = False
    selected = payload.get("selected", {})
    logger.info(
        "  best C=%s -> validation ROC-AUC %s, PR-AUC %s",
        selected.get("C"),
        selected.get("roc_auc"),
        selected.get("pr_auc"),
    )
    return payload


def _cross_gram(
    left: PixelPartition,
    right: PixelPartition,
    right_rows: Sequence[int],
    *,
    chunk_rows: int = CHUNK_ROWS,
) -> np.ndarray:
    """``(len(left), len(right_rows))`` overlaps between two partitions' states."""
    right_rows = np.asarray(right_rows, dtype=int)
    out = np.empty((len(left), right_rows.size), dtype=np.float64)
    right_blocks = [
        (start, right_rows[start:start + chunk_rows])
        for start in range(0, right_rows.size, max(1, chunk_rows))
    ]
    cursor = 0
    for _, left_states in amplitude_blocks(
        left, _as_rows(left, None), chunk_rows=chunk_rows
    ):
        for start_j, block_j in right_blocks:
            right_states = right.amplitudes(block_j)
            out[cursor:cursor + left_states.shape[0],
                start_j:start_j + block_j.size] = left_states @ right_states.T
        cursor += left_states.shape[0]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure class-discriminative structure in the 16-qubit input states."
    )
    parser.add_argument(
        "--roi", dest="roi_mode", default="oracle", choices=("oracle", "predicted")
    )
    parser.add_argument("--layers", type=int, default=2, help="Ansatz repetitions.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--gram-subsample", type=int, default=None,
        help="Cap rows entering the Gram matrix. Positives are always kept.",
    )
    parser.add_argument(
        "--skip-observable", action="store_true",
        help="Skip the untrained <Z0> pass (the slow part, ~6 ms per state).",
    )
    parser.add_argument(
        "--include-test", action="store_true",
        help="Diagnose the held-out partition too. Only for the final frozen "
             "evaluation: these diagnostics read test labels.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("qiskit", "qiskit.transpiler", "qiskit.passmanager", "stevedore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    summary = run_diagnostics(
        roi_mode=args.roi_mode,
        n_layers=args.layers,
        seed=args.seed,
        include_test=args.include_test,
        gram_subsample=args.gram_subsample,
        skip_observable=args.skip_observable,
    )
    print(json.dumps({
        "artifact": summary["artifact"],
        "roi_mode": summary["roi_mode"],
        "qubit_count": summary["qubit_count"],
        "conditions": [
            {
                "partition": row["partition"],
                "condition": row["condition"],
                "n_samples": row["n_samples"],
                "n_positive": row["n_positive"],
                "uniform_overlap_mean": row["states"]["uniform_overlap"]["mean"],
                "within_minus_between": row["pairwise_overlaps"].get(
                    "within_minus_between_positive"
                ),
                "untrained_pr_auc": row["observable_at_untrained_weights"].get(
                    "discrimination", {}
                ).get("pr_auc"),
                "area_only_roc_auc": row["roi_geometry_leak"]["area_only_roc_auc"],
            }
            for row in summary["conditions"]
        ],
        "cheap_baseline": summary["cheap_baseline"].get("selected"),
        "test_partition_used": summary["test_partition_used"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
