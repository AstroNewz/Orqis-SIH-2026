"""E4 quantum rung -- a quantum feature map on PTB-XL, and the same-shape classical control
that is the actual experiment.

Round 7.9's prior-art sweep left this phase with exactly one surviving novelty claim: no
paper in that sweep controls a quantum arm against a **fused classical baseline of the same
shape**. Every quantum study found there compares against a single model, a parallel twin,
or a teacher. That makes the control the experiment rather than a formality, and it is why
this module is organised around pairs of arms rather than around a quantum arm with a
baseline footnote.

The protocol is pre-registered in ``docs/PHASE_E4_DATA_EXPANSION.md`` Section 7.10, written
before the first circuit was fitted. The load-bearing commitments, restated here because a
reader of the code should not have to trust a document:

* **The "same shape" contract.** An arm is a map ``R^8 -> R^36`` fitted on TRAIN rows only,
  followed by an *identical* logistic head with an *identical* ``C`` grid selected on the
  *identical* inner fold. Three maps compete: the Havlicek ZZ feature map read out as 8
  single-qubit and 28 pair ``Z`` observables; ``poly2``, which is the same index set built
  classically (8 values and their 28 pairwise products); and ``rff36``, 36 random Fourier
  features. All three have zero trainable parameters in the map.

* **``poly2`` is the tight control.** The ZZ map's 36 observables are indexed by exactly the
  8 singletons and 28 pairs of an 8-element set -- the index set of a degree-2 polynomial on
  the same inputs. Same width, same input, same parameter count, same combinatorial
  structure; only the function on each index differs. If ``zz`` beats ``poly2``, "pairwise
  interactions became available" is not an available explanation.

* **The headline is not the bar.** It is the paired patient-clustered delta
  ``fusion@cnn+gbm+zz - fusion@cnn+gbm+poly2``. An interval spanning zero is a null *however
  either arm scores against 0.946293*, because a quantum fusion that beats a single classical
  model may only be showing an ensembling effect that a classical fusion also shows.

* **The search budget is deliberately tiny.** ``reps in {1, 2}`` and ``C in {0.01, 0.1, 1,
  10}`` -- the same grid size each classical control gets. arXiv:2507.11401 is in the record
  as the failure mode being avoided: 400 entanglement topologies sampled, the 16% that beat
  baseline reported. No topology search, no ansatz sweep, no angle sweep happens here.

* **Members are recovered, not re-selected.** ``cnn@resnet_small`` (epoch 17) and ``gbm@f97``
  are rebuilt from what Section 7.8 recorded and must reproduce their recorded fold-9
  ROC-AUCs exactly, or the run raises. A drifting member would mean the fusion is being
  compared against a different bar than the one in the record.

Fold 10 is never read and its signals were never downloaded.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from backend.dataset.ptbxl import TEST_FOLDS, TRAIN_FOLDS, VALIDATION_FOLDS
from backend.evaluation.e3_feature_map import (
    N_OBSERVABLES,
    N_QUBITS,
    REPS_GRID,
    RichQuantumFeatureMap,
    validate_against_qiskit_zzmap,
    validate_readout_against_aer,
)
from backend.evaluation.e4_classical_baseline import (
    BOOTSTRAP_DRAWS,
    FIXED_SPECIFICITY,
    SEED,
    Partition,
    bootstrap_roc_auc,
    build_estimator,
    build_partition,
    guard_partitions,
    out_of_fold_scores,
    paired_bootstrap_delta,
    primary_labels,
    raw_scores,
    roc_auc,
    select_threshold,
)
from backend.evaluation.e4_deep_baseline import (
    BATCH_SIZE,
    INNER_VALIDATION_FOLD,
    MAX_EPOCHS,
    _logit,
    load_reference_arm,
    predict_logits,
    split_inner,
    train_model,
)
from backend.evaluation.metrics import evaluate_predictions, sensitivity_at_specificity
from backend.ml.ecg_transform import fit_ecg_transform
from backend.ml.quantum_visual import AngleScaler
from backend.training.prepare_ecg_features import (
    ECG_CACHE_VERSION,
    FEATURE_SET_VERSION,
    EcgCohort,
    load_cohort,
)
from backend.training.prepare_ecg_signals import SignalCache, load_signal_cache

logger = logging.getLogger(__name__)

#: Bump when the Round-7.10 protocol semantics change so a stored report cannot be
#: reinterpreted against different rules than the ones it ran under.
E4_QUANTUM_VERSION = "v1-e4-quantum-1"

#: One qubit per component of the compact representation. Section 7.6 measured that 8
#: dimensions retain 60.0% of the feature variance and reach 97.3% of the 97-feature
#: classical ceiling, which is what makes a null here informative: unlike E3's crushed
#: images, the input handed to the circuit still carries most of the task.
QUANTUM_DIM = N_QUBITS

#: The entire quantum search. See the module docstring on arXiv:2507.11401.
HEAD_C_GRID: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)

#: The ZZ re-uploading depths this rung will actually score -- **not** E3's ``REPS_GRID``.
#:
#: ``reps=1`` is excluded on a structural proof, not on a score, and the exclusion was made
#: before any arm had been scored on any partition. After the single Hadamard layer, every
#: remaining gate in a Havlicek block (the ``P(2 x_i)`` phases and the ``CX-P-CX`` gadgets)
#: is diagonal in the computational basis, so it changes only phases and leaves ``|psi|^2``
#: exactly uniform. Every observable here is a function of ``|psi|^2`` alone, so all 36 of
#: them are identically zero for every input: measured max ``|<O>|`` = 2.2e-16 and per-column
#: TRAIN std = 6.8e-17, i.e. floating-point noise. A ``reps=1`` arm is a constant vector
#: wearing a circuit, and a head fitted on it can only learn an intercept.
#:
#: Excluding it is the conservative direction twice over: it removes a cell that could only
#: have scored at chance, and it shrinks the quantum grid to four cells -- exactly the number
#: each classical control gets, which makes the same-shape contract tighter rather than
#: looser. :func:`fit_map` additionally refuses any degenerate map at runtime, so this
#: constant is a statement of intent and the guard is the enforcement.
QUANTUM_REPS_GRID: Tuple[int, ...] = (2,)

#: A map whose TRAIN output has no variance in any column is not a feature map. Scoring one
#: would report the head's intercept as a model result.
DEGENERATE_MAP_FLOOR = 1e-9

#: Robust-quantile angle range for the ZZ encoding. Havlicek's map takes ``x`` in a bounded
#: range and builds phases ``2 x_i`` and ``2 (pi - x_i)(pi - x_j)``; ``[0, pi]`` keeps the
#: first-order phase inside a single period. This is a fixed TRAIN-only scaling constant
#: chosen once, not swept -- Section 7.10 forbids an angle sweep, because sweeping the
#: encoding and reporting the best cell is the multiple-comparisons failure this protocol
#: exists to avoid.
ANGLE_MAX = float(np.pi)
ANGLE_QUANTILE = 0.02

#: Random Fourier feature count, pinned to the quantum map's width so the control is
#: same-shape rather than merely "also classical".
RFF_FEATURES = N_OBSERVABLES

#: Correctness gates on the map, run on a small sample and recorded in the report.
MAP_VALIDATION_SAMPLE = 8
MAP_VALIDATION_TOLERANCE = 1e-10

#: A connected correlation below this in absolute mean means the entanglers did nothing and
#: the map is effectively a product state -- in which case "the circuit was not entangling"
#: is a measured fact rather than a post-hoc excuse for a null.
ENTANGLING_WITNESS_FLOOR = 1e-3

DEFAULT_FEATURE_CACHE = Path("backend/artifacts/dataset/ptbxl/features_ecg-v1.npz")
DEFAULT_SIGNAL_DIR = Path("backend/artifacts/dataset")
DEFAULT_DEEP_REPORT = Path("backend/artifacts/reports/e4_deep_baseline.json")

#: The recovered members are cached here, because rebuilding the CNN costs about 45 minutes
#: and nothing about it changes between quantum runs.
DEFAULT_MEMBER_CACHE = Path("backend/artifacts/reports/e4_members.npz")

_EPS = 1e-12


class E4QuantumError(RuntimeError):
    """Raised when the quantum rung cannot be built, fitted or trusted."""


# ------------------------------------------------------------------- member recovery
@dataclass(frozen=True)
class RecoveredMembers:
    """The two classical fusion members, out-of-sample on fold 8 and scored on fold 9.

    Both vectors per member are needed and they come from *different* fits: the fold-8
    vector must come from a model that never saw fold 8 (so the blender is not learning to
    trust a member on data it memorised), and the fold-9 vector from the model refitted on
    all of TRAIN (so fold 9 is scored by the arm the record describes).
    """

    deep_inner: np.ndarray
    deep_validation: np.ndarray
    tabular_inner: np.ndarray
    tabular_validation: np.ndarray
    deep_architecture: str
    deep_epoch: int
    deep_inner_roc_auc: float
    deep_validation_roc_auc: float
    tabular_validation_roc_auc: float
    tabular_params: Dict[str, Any]
    reproduced_from_cache: bool

    def summary(self) -> Dict[str, Any]:
        return {
            "deep_architecture": self.deep_architecture,
            "deep_epoch": self.deep_epoch,
            "deep_inner_roc_auc": round(float(self.deep_inner_roc_auc), 6),
            "deep_validation_roc_auc": float(self.deep_validation_roc_auc),
            "tabular_validation_roc_auc": float(self.tabular_validation_roc_auc),
            "tabular_selected_params": dict(self.tabular_params),
            "reproduced_from_cache": bool(self.reproduced_from_cache),
        }


def load_recorded_deep_arm(
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Recover the recorded deep winner: its architecture, epoch and published ROC-AUC."""
    path = Path(report_path) if report_path is not None else DEFAULT_DEEP_REPORT
    if not path.exists():
        raise E4QuantumError(
            f"{path} not found. Run `python -m backend.evaluation.e4_deep_baseline` first; "
            "the quantum rung fuses with its recorded winner."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    arms = report.get("arms") or []
    deep = next(
        (entry for entry in arms if str(entry.get("arm", "")).startswith("cnn@")), None
    )
    if deep is None:
        raise E4QuantumError(f"{path} records no cnn@ arm; found {[a.get('arm') for a in arms]}.")
    fusion = next(
        (entry for entry in arms if str(entry.get("arm", "")) == "fusion@cnn+gbm"), None
    )
    selection = report.get("architecture_selection") or []
    architecture = str(deep["arm"].split("@", 1)[1])
    inner = next(
        (s for s in selection if s.get("architecture") == architecture), None
    )
    if inner is None or inner.get("best_inner_roc_auc") is None:
        raise E4QuantumError(
            f"{path} records no inner score for {architecture!r}; the fold-8 member cannot "
            "be reproduced and asserted."
        )
    return {
        "architecture": architecture,
        "epoch": int(deep["selected_epoch"]),
        "published_roc_auc": float(deep["validation_roc_auc"]),
        "published_inner_roc_auc": float(inner["best_inner_roc_auc"]),
        "published_fusion_roc_auc": (
            float(fusion["validation_roc_auc"]) if fusion else None
        ),
        "report_path": str(path),
        "deep_version": report.get("deep_version"),
    }


def _assert_reproduces(label: str, measured: Optional[float], recorded: float, tol: float) -> None:
    if measured is None or abs(float(measured) - float(recorded)) > tol:
        raise E4QuantumError(
            f"{label} scores {measured} here but {recorded} in the record (tolerance "
            f"{tol:g}). The quantum fusion would be compared against a different baseline "
            "than the one on record; refusing to continue."
        )


def recover_members(
    train: Partition,
    validation: Partition,
    signal_cache: SignalCache,
    inner_train_mask: np.ndarray,
    inner_validation_mask: np.ndarray,
    *,
    classical_report: Optional[Path] = None,
    deep_report: Optional[Path] = None,
    member_cache: Optional[Path] = None,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    refresh: bool = False,
) -> RecoveredMembers:
    """Rebuild ``cnn@resnet_small`` and ``gbm@f97`` and assert they reproduce the record."""
    recorded_deep = load_recorded_deep_arm(deep_report)
    reference = load_reference_arm(classical_report)
    cache_path = Path(member_cache) if member_cache is not None else DEFAULT_MEMBER_CACHE

    # ---------------------------------------------------------------- tabular member
    # Cheap (one 8-fold OOF pass plus one refit), so it is always recomputed rather than
    # cached: caching it would save minutes and add a way for the report to describe an
    # arm that is not the arm that ran.
    logger.info("  refitting tabular member %s", reference["arm"])
    tabular_inner_full, _ = out_of_fold_scores(
        reference["model"], reference["selected_params"], train, seed=seed
    )
    tabular_inner = tabular_inner_full[inner_validation_mask]
    estimator = build_estimator(reference["model"], reference["selected_params"], seed=seed)
    estimator.fit(train.features, train.y)
    tabular_validation = raw_scores(estimator, validation.features)
    tabular_validation_auc = roc_auc(validation.y, tabular_validation)
    _assert_reproduces(
        reference["arm"], tabular_validation_auc, float(reference["published_roc_auc"]), 1e-9
    )

    # ------------------------------------------------------------------- deep member
    cached: Optional[Dict[str, np.ndarray]] = None
    if cache_path.exists() and not refresh:
        with np.load(cache_path, allow_pickle=False) as handle:
            payload = {key: handle[key] for key in handle.files}
        same_shape = (
            payload.get("deep_inner") is not None
            and payload["deep_inner"].size == int(inner_validation_mask.sum())
            and payload["deep_validation"].size == validation.n_records
        )
        matching_ids = (
            "validation_ecg_ids" in payload
            and np.array_equal(
                payload["validation_ecg_ids"].astype(np.int64), validation.ecg_ids
            )
            and "inner_ecg_ids" in payload
            and np.array_equal(
                payload["inner_ecg_ids"].astype(np.int64),
                train.ecg_ids[inner_validation_mask],
            )
        )
        if same_shape and matching_ids:
            cached = payload
            logger.info("  deep member restored from %s", cache_path)
        else:
            logger.warning(
                "  %s does not align with the current partitions; refitting the deep member",
                cache_path,
            )

    if cached is not None:
        deep_inner = np.asarray(cached["deep_inner"], dtype=np.float64)
        deep_validation = np.asarray(cached["deep_validation"], dtype=np.float64)
    else:
        train_rows = signal_cache.rows_for(train.ecg_ids)
        validation_rows = signal_cache.rows_for(validation.ecg_ids)
        inner_rows = train_rows[inner_validation_mask]
        inner_y = train.y[inner_validation_mask]
        architecture = recorded_deep["architecture"]

        # The *selection* call is reproduced exactly as Section 7.8 ran it -- same code path,
        # same early stopping, same kept-best weights -- rather than replaced by a cheaper
        # fixed-epoch fit on folds 1-7. A fixed-epoch fit would probably land on the same
        # weights, and "probably" is not a basis for asserting the member reproduces.
        logger.info("  selection fit for the deep member: %s (this takes ~30 min)", architecture)
        _, history = train_model(
            architecture,
            signal_cache.signals,
            train_rows[inner_train_mask],
            train.y[inner_train_mask],
            inner_rows=inner_rows,
            inner_y=inner_y,
            max_epochs=max_epochs,
            batch_size=batch_size,
            seed=seed,
        )
        selection_model, _ = train_model(
            architecture,
            signal_cache.signals,
            train_rows[inner_train_mask],
            train.y[inner_train_mask],
            fixed_epochs=int(history.best_epoch),
            max_epochs=max_epochs,
            batch_size=batch_size,
            seed=seed,
        )
        deep_inner = predict_logits(
            selection_model, signal_cache.signals, inner_rows, batch_size=batch_size
        )
        logger.info("  refit on folds 1-8 for %d epochs", recorded_deep["epoch"])
        final_model, _ = train_model(
            architecture,
            signal_cache.signals,
            train_rows,
            train.y,
            fixed_epochs=int(recorded_deep["epoch"]),
            max_epochs=max_epochs,
            batch_size=batch_size,
            seed=seed,
        )
        deep_validation = predict_logits(
            final_model, signal_cache.signals, validation_rows, batch_size=batch_size
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            deep_inner=deep_inner,
            deep_validation=deep_validation,
            inner_ecg_ids=train.ecg_ids[inner_validation_mask],
            validation_ecg_ids=validation.ecg_ids,
        )
        logger.info("  deep member cached to %s", cache_path)

    inner_y = train.y[inner_validation_mask]
    deep_inner_auc = roc_auc(inner_y, deep_inner)
    deep_validation_auc = roc_auc(validation.y, deep_validation)
    # The inner score is recorded to 6 decimals in the report, so it is asserted at that
    # resolution; the fold-9 score is recorded in full and is asserted exactly.
    _assert_reproduces(
        f"cnn@{recorded_deep['architecture']} (fold {INNER_VALIDATION_FOLD})",
        deep_inner_auc,
        recorded_deep["published_inner_roc_auc"],
        5e-7,
    )
    _assert_reproduces(
        f"cnn@{recorded_deep['architecture']} (fold 9)",
        deep_validation_auc,
        recorded_deep["published_roc_auc"],
        1e-9,
    )
    return RecoveredMembers(
        deep_inner=np.asarray(deep_inner, dtype=np.float64),
        deep_validation=np.asarray(deep_validation, dtype=np.float64),
        tabular_inner=np.asarray(tabular_inner, dtype=np.float64),
        tabular_validation=np.asarray(tabular_validation, dtype=np.float64),
        deep_architecture=recorded_deep["architecture"],
        deep_epoch=int(recorded_deep["epoch"]),
        deep_inner_roc_auc=float(deep_inner_auc),
        deep_validation_roc_auc=float(deep_validation_auc),
        tabular_validation_roc_auc=float(tabular_validation_auc),
        tabular_params=dict(reference["selected_params"]),
        reproduced_from_cache=cached is not None,
    )


# --------------------------------------------------------------------------- the maps
def _standardiser(fold_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column mean/std from fold-train, unit std for a constant column."""
    mean = np.asarray(fold_train, dtype=np.float64).mean(axis=0)
    std = np.asarray(fold_train, dtype=np.float64).std(axis=0)
    return mean, np.where(std > _EPS, std, 1.0)


def _pair_indices(n: int = QUANTUM_DIM) -> List[Tuple[int, int]]:
    """``(i, j)`` pairs in the order :func:`all_pair_masks` emits them, so ``poly2``'s
    columns line up index-for-index with the quantum map's pair observables."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def poly2_expand(values: np.ndarray) -> np.ndarray:
    """``(n, 36)``: the 8 inputs and their 28 pairwise products, in observable order.

    This is the quantum map's index set, computed classically. It is not an approximation
    of the quantum map and is not meant to be one -- it is the answer to "would any map
    over the same singletons and pairs have done this?"
    """
    arr = np.atleast_2d(np.asarray(values, dtype=np.float64))
    if arr.shape[1] != QUANTUM_DIM:
        raise E4QuantumError(
            f"poly2 expects {QUANTUM_DIM} columns, got {arr.shape[1]}."
        )
    pairs = np.column_stack([arr[:, i] * arr[:, j] for i, j in _pair_indices()])
    out = np.hstack([arr, pairs])
    if out.shape[1] != N_OBSERVABLES:
        raise E4QuantumError(
            f"poly2 produced {out.shape[1]} columns but the quantum map has {N_OBSERVABLES}; "
            "the same-shape contract is broken."
        )
    return out


@dataclass(frozen=True)
class RandomFourierMap:
    """``(n, 36)`` random Fourier features for an RBF kernel, TRAIN-fitted bandwidth.

    ``cos(Wx + b)`` with ``W ~ N(0, gamma)``: 36 draws, no trainable parameter, matching the
    quantum map's width and parameter count. The bandwidth comes from the median pairwise
    distance on a TRAIN subsample -- the standard heuristic, fixed rather than tuned, so the
    control gets the same single-shot treatment the quantum map gets.
    """

    weights: np.ndarray
    offsets: np.ndarray
    gamma: float

    @classmethod
    def fit(
        cls, fold_train: np.ndarray, *, seed: int, n_features: int = RFF_FEATURES
    ) -> "RandomFourierMap":
        arr = np.atleast_2d(np.asarray(fold_train, dtype=np.float64))
        rng = np.random.default_rng(seed)
        sample = arr[rng.choice(arr.shape[0], size=min(1000, arr.shape[0]), replace=False)]
        diffs = sample[:, None, :] - sample[None, :, :]
        distances = np.sqrt(np.einsum("ijk,ijk->ij", diffs, diffs))
        upper = distances[np.triu_indices_from(distances, k=1)]
        median = float(np.median(upper)) if upper.size else 1.0
        gamma = 1.0 / (2.0 * max(median, _EPS) ** 2)
        weights = rng.normal(
            loc=0.0, scale=np.sqrt(2.0 * gamma), size=(arr.shape[1], int(n_features))
        )
        offsets = rng.uniform(0.0, 2.0 * np.pi, size=int(n_features))
        return cls(weights=weights, offsets=offsets, gamma=float(gamma))

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.atleast_2d(np.asarray(values, dtype=np.float64))
        projected = arr @ self.weights + self.offsets
        return np.sqrt(2.0 / self.weights.shape[1]) * np.cos(projected)


@dataclass(frozen=True)
class FittedMap:
    """One map plus the TRAIN-only affines around it, ready to score any partition."""

    family: str
    reps: Optional[int]
    _encode: Any
    _mean: np.ndarray
    _std: np.ndarray
    telemetry: Dict[str, Any]

    def transform(self, components: np.ndarray) -> np.ndarray:
        raw = self._encode(components)
        return (raw - self._mean) / self._std


def fit_map(
    family: str,
    fold_train_components: np.ndarray,
    *,
    reps: Optional[int] = None,
    seed: int = SEED,
) -> FittedMap:
    """Fit one map family on ``fold_train_components`` (PCA-8 scores) only.

    Each family receives the input encoding that suits it, which is a deliberate choice in
    the control's favour: ``zz`` gets robust-quantile angles in ``[0, pi]`` because a phase
    encoding needs a bounded range, and ``poly2``/``rff36`` get zero-mean unit-variance
    values because a polynomial over non-negative angles would lose all sign information and
    a hobbled control manufactures a quantum win. Both encodings are deterministic TRAIN-only
    functions of the same representation, and both add zero trainable parameters.
    """
    train = np.atleast_2d(np.asarray(fold_train_components, dtype=np.float64))
    if train.shape[1] != QUANTUM_DIM:
        raise E4QuantumError(
            f"Maps expect {QUANTUM_DIM} components, got {train.shape[1]}."
        )
    if family == "zz":
        if reps not in REPS_GRID:
            raise E4QuantumError(f"reps must be in {REPS_GRID}; got {reps!r}.")
        scaler = AngleScaler.fit(train, quantile=ANGLE_QUANTILE, max_angle=ANGLE_MAX)
        fmap = RichQuantumFeatureMap(reps=int(reps), n_qubits=QUANTUM_DIM, seed=seed)

        def encode(components: np.ndarray) -> np.ndarray:
            return fmap.transform(scaler.transform(components))

        telemetry = dict(fmap.telemetry())
        telemetry["angle_scaler"] = {
            "quantile": ANGLE_QUANTILE,
            "max_angle": ANGLE_MAX,
            "fitted_on": "fold-train rows only",
        }
    elif family == "poly2":
        mean, std = _standardiser(train)

        def encode(components: np.ndarray) -> np.ndarray:
            return poly2_expand((np.asarray(components, dtype=np.float64) - mean) / std)

        telemetry = {
            "map": "degree_2_polynomial",
            "n_qubits": None,
            "n_encoding_parameters": QUANTUM_DIM,
            "n_trainable_parameters": 0,
            "n_observables": N_OBSERVABLES,
            "index_set": "8 singletons + 28 pairs, identical to the ZZ observable set",
        }
    elif family == "rff36":
        mean, std = _standardiser(train)
        rff = RandomFourierMap.fit((train - mean) / std, seed=seed)

        def encode(components: np.ndarray) -> np.ndarray:
            return rff.transform((np.asarray(components, dtype=np.float64) - mean) / std)

        telemetry = {
            "map": "random_fourier_features",
            "n_qubits": None,
            "n_encoding_parameters": QUANTUM_DIM,
            "n_trainable_parameters": 0,
            "n_observables": RFF_FEATURES,
            "gamma": round(float(rff.gamma), 6),
            "bandwidth": "median pairwise distance on a TRAIN subsample",
        }
    else:
        raise E4QuantumError(f"Unknown map family {family!r}.")

    encoded_train = np.asarray(encode(train), dtype=np.float64)
    # Guard on the *raw* spread. _standardiser floors a constant column's divisor at 1.0 so
    # that scaling stays finite, so its returned std can never report degeneracy -- reading it
    # here would make this check unfireable.
    widest = float(np.max(encoded_train.std(axis=0)))
    if widest <= DEGENERATE_MAP_FLOOR:
        raise E4QuantumError(
            f"Map {family!r} (reps={reps!r}) is degenerate on TRAIN: the largest per-column "
            f"standard deviation is {widest:.3e}, at or below "
            f"{DEGENERATE_MAP_FLOOR:g}. Every output column is constant, so a head fitted on "
            "it can only learn an intercept. Refusing to score a constant as a model."
        )
    out_mean, out_std = _standardiser(encoded_train)
    return FittedMap(
        family=family,
        reps=int(reps) if reps is not None else None,
        _encode=encode,
        _mean=out_mean,
        _std=out_std,
        telemetry=telemetry,
    )


def entangling_witness(
    fold_train_components: np.ndarray, *, reps: int, seed: int = SEED, sample: int = 256
) -> Dict[str, Any]:
    """Mean ``|<Z_iZ_j> - <Z_i><Z_j>|`` on a TRAIN subsample.

    If this sits at the floor, the ZZ map is effectively a product state and a null means
    "the entanglers did nothing here", which is a different and much weaker statement than
    "entanglement did not help". Measuring it turns that distinction into a fact.
    """
    arr = np.atleast_2d(np.asarray(fold_train_components, dtype=np.float64))
    rng = np.random.default_rng(seed)
    rows = rng.choice(arr.shape[0], size=min(int(sample), arr.shape[0]), replace=False)
    scaler = AngleScaler.fit(arr, quantile=ANGLE_QUANTILE, max_angle=ANGLE_MAX)
    fmap = RichQuantumFeatureMap(reps=int(reps), n_qubits=QUANTUM_DIM, seed=seed)
    correlations = fmap.connected_correlations(scaler.transform(arr[rows]))
    mean_abs = float(np.mean(np.abs(correlations)))
    return {
        "reps": int(reps),
        "n_samples": int(rows.size),
        "mean_abs_connected_correlation": round(mean_abs, 8),
        "max_abs_connected_correlation": round(float(np.max(np.abs(correlations))), 8),
        "floor": ENTANGLING_WITNESS_FLOOR,
        "is_entangling": bool(mean_abs > ENTANGLING_WITNESS_FLOOR),
    }


def validate_map(*, reps: int, seed: int = SEED) -> Dict[str, Any]:
    """Run both E3 correctness gates on the map before any number derived from it is kept."""
    fmap = RichQuantumFeatureMap(reps=int(reps), n_qubits=QUANTUM_DIM, seed=seed)
    rng = np.random.default_rng(seed)
    probe = rng.uniform(0.0, ANGLE_MAX, size=(MAP_VALIDATION_SAMPLE, QUANTUM_DIM))
    return {
        "readout_vs_aer": validate_readout_against_aer(
            fmap, probe, tolerance=MAP_VALIDATION_TOLERANCE
        ),
        "circuit_vs_qiskit_zzmap": validate_against_qiskit_zzmap(
            fmap, probe, tolerance=MAP_VALIDATION_TOLERANCE
        ),
    }


# ---------------------------------------------------------------------------- the head
def _fit_head(features: np.ndarray, y: np.ndarray, *, c: float, seed: int) -> LogisticRegression:
    head = LogisticRegression(C=float(c), max_iter=2000, random_state=int(seed))
    head.fit(features, np.asarray(y, dtype=int))
    return head


def score_map_arm(
    family: str,
    train_components: np.ndarray,
    train_y: np.ndarray,
    inner_train_mask: np.ndarray,
    inner_validation_mask: np.ndarray,
    validation_components: np.ndarray,
    *,
    reps_grid: Sequence[Optional[int]],
    c_grid: Sequence[float] = HEAD_C_GRID,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Select ``(reps, C)`` on inner fold 8, refit on all of TRAIN, score fold 9.

    Returns both score vectors, because the fusion needs the fold-8 one and the fold-8 one
    must come from a model fitted on folds 1-7 only.
    """
    sub_train = train_components[inner_train_mask]
    sub_y = np.asarray(train_y, dtype=int)[inner_train_mask]
    inner = train_components[inner_validation_mask]
    inner_y = np.asarray(train_y, dtype=int)[inner_validation_mask]

    grid: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_inner_scores: Optional[np.ndarray] = None
    for reps in reps_grid:
        fitted = fit_map(family, sub_train, reps=reps, seed=seed)
        design = fitted.transform(sub_train)
        inner_design = fitted.transform(inner)
        for c in c_grid:
            head = _fit_head(design, sub_y, c=c, seed=seed)
            scores = head.predict_proba(inner_design)[:, 1]
            auc = roc_auc(inner_y, scores)
            cell = {"reps": reps, "C": float(c), "inner_roc_auc": auc}
            grid.append(cell)
            if best is None or (auc or -1.0) > (best["inner_roc_auc"] or -1.0):
                best = cell
                best_inner_scores = scores
    if best is None or best_inner_scores is None:
        raise E4QuantumError(f"No cell was fitted for family {family!r}.")

    # Refit the selected cell on all of TRAIN; fold 9 is scored by that arm, once.
    final_map = fit_map(family, train_components, reps=best["reps"], seed=seed)
    final_head = _fit_head(
        final_map.transform(train_components), train_y, c=best["C"], seed=seed
    )
    validation_scores = final_head.predict_proba(
        final_map.transform(validation_components)
    )[:, 1]
    return {
        "family": family,
        "selected": {"reps": best["reps"], "C": best["C"]},
        "selected_inner_roc_auc": best["inner_roc_auc"],
        "grid": grid,
        "n_grid_cells": len(grid),
        "map_telemetry": final_map.telemetry,
        "inner_scores": np.asarray(best_inner_scores, dtype=np.float64),
        "validation_scores": np.asarray(validation_scores, dtype=np.float64),
        "selected_on": (
            f"TRAIN fold {INNER_VALIDATION_FOLD}, predicted by a map+head fitted on folds 1-7"
        ),
    }


# --------------------------------------------------------------------------- fusion
def blend(
    inner_columns: Sequence[np.ndarray],
    inner_y: np.ndarray,
    validation_columns: Sequence[np.ndarray],
    *,
    names: Sequence[str],
) -> Dict[str, Any]:
    """Logistic blender on fold-8 out-of-sample log-odds -- identical for every fusion.

    Identical is the point. The quantum fusion and the classical control fusion differ in
    exactly one column and in nothing else, so a difference between them cannot be a
    difference in how they were combined.
    """
    if len(inner_columns) != len(names) or len(validation_columns) != len(names):
        raise E4QuantumError("Column count and name count disagree.")
    blend_train = np.column_stack([np.asarray(col, dtype=np.float64) for col in inner_columns])
    blend_eval = np.column_stack(
        [np.asarray(col, dtype=np.float64) for col in validation_columns]
    )
    blender = LogisticRegression(max_iter=1000)
    blender.fit(blend_train, np.asarray(inner_y, dtype=int))
    inner_probability = blender.predict_proba(blend_train)[:, 1]
    return {
        "members": list(names),
        "coefficients": {
            name: round(float(blender.coef_[0][index]), 6)
            for index, name in enumerate(names)
        },
        "intercept": round(float(blender.intercept_[0]), 6),
        "inner_probability": inner_probability,
        "validation_probability": blender.predict_proba(blend_eval)[:, 1],
        "inner_roc_auc": roc_auc(np.asarray(inner_y, dtype=int), inner_probability),
        "blender": "logistic regression on the members' log-odds",
    }


def arm_payload(
    name: str,
    probability: np.ndarray,
    validation: Partition,
    threshold: Dict[str, Any],
    *,
    bootstrap_draws: int,
    seed: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Every arm reports the same block, so arms are comparable line by line."""
    metrics = evaluate_predictions(validation.y, probability, threshold=threshold["threshold"])
    payload: Dict[str, Any] = {
        "arm": name,
        "threshold": threshold,
        "validation": metrics.to_dict(),
        "validation_roc_auc": metrics.roc_auc,
        "validation_bootstrap_roc_auc": bootstrap_roc_auc(
            validation.y, probability, validation.patients, draws=bootstrap_draws, seed=seed
        ),
        "sensitivity_at_fixed_specificity": sensitivity_at_specificity(
            validation.y, probability, FIXED_SPECIFICITY
        ),
        "validation_scores": probability,
    }
    if extra:
        payload.update(extra)
    return payload


# ------------------------------------------------------------------------------- run
def run(
    cache: Optional[Path] = None,
    signals_dir: Optional[Path] = None,
    *,
    classical_report: Optional[Path] = None,
    deep_report: Optional[Path] = None,
    member_cache: Optional[Path] = None,
    families: Sequence[str] = ("zz", "poly2", "rff36"),
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    refresh_members: bool = False,
) -> Dict[str, Any]:
    """The Section 7.10 protocol. Fits on folds 1-8, scores fold 9 once, never reads 10."""
    started = time.perf_counter()
    cache = Path(cache) if cache is not None else DEFAULT_FEATURE_CACHE
    cohort: EcgCohort = load_cohort(cache)
    fold_array = np.asarray(cohort.strat_folds, dtype=int)
    if np.isin(fold_array, sorted(TEST_FOLDS)).any():
        raise E4QuantumError("The feature cache contains fold 10; refusing to run.")
    signal_cache: SignalCache = load_signal_cache(
        Path(signals_dir) if signals_dir is not None else DEFAULT_SIGNAL_DIR
    )
    if set(signal_cache.metadata.get("folds", [])) & set(TEST_FOLDS):
        raise E4QuantumError("The signal cache contains fold 10; refusing to run.")

    labels = primary_labels(cohort)

    # The quantum representation: PCA-8, fitted on TRAIN folds 1-8 only. This is the same
    # convention the matched-dimension ceiling 0.915006 was measured under (Section 7.7),
    # which is what makes these arms directly comparable to it. The PCA sees fold 8, so the
    # inner selection is not perfectly nested -- that is stated rather than hidden, and it is
    # identical for the quantum arm and both classical controls, so it cannot favour either.
    quantum_transform = fit_ecg_transform(cohort, reduction="pca", n_components=QUANTUM_DIM)
    if set(quantum_transform.fitted_on_folds) - set(TRAIN_FOLDS):
        raise E4QuantumError(
            "Quantum representation fitted on non-TRAIN folds "
            f"{quantum_transform.fitted_on_folds}."
        )
    q_train = build_partition(cohort, quantum_transform, labels, sorted(TRAIN_FOLDS), "TRAIN")
    q_validation = build_partition(
        cohort, quantum_transform, labels, sorted(VALIDATION_FOLDS), "VALIDATION"
    )
    guards = guard_partitions(q_train, q_validation)

    # The members live on the full 97-feature representation, exactly as Section 7.8 fitted
    # them.
    identity_transform = fit_ecg_transform(cohort, reduction="identity")
    f_train = build_partition(cohort, identity_transform, labels, sorted(TRAIN_FOLDS), "TRAIN")
    f_validation = build_partition(
        cohort, identity_transform, labels, sorted(VALIDATION_FOLDS), "VALIDATION"
    )
    if not np.array_equal(f_train.ecg_ids, q_train.ecg_ids) or not np.array_equal(
        f_validation.ecg_ids, q_validation.ecg_ids
    ):
        raise E4QuantumError(
            "The 8-dimensional and 97-feature partitions do not contain the same records in "
            "the same order; fusion columns would be misaligned."
        )

    inner_train_mask, inner_validation_mask = split_inner(f_train)
    inner_y = f_train.y[inner_validation_mask]
    logger.info(
        "TRAIN %d rec / %d pt   inner-val (fold %d) %d   VALIDATION %d rec / %d pt",
        f_train.n_records,
        f_train.n_patients,
        INNER_VALIDATION_FOLD,
        int(inner_validation_mask.sum()),
        f_validation.n_records,
        f_validation.n_patients,
    )

    # ------------------------------------------------------------------ correctness gates
    logger.info("Validating the quantum map against Aer and against the qiskit ZZFeatureMap")
    map_validation = {str(reps): validate_map(reps=reps, seed=seed) for reps in REPS_GRID}
    witnesses = {
        str(reps): entangling_witness(
            q_train.features[inner_train_mask], reps=reps, seed=seed
        )
        for reps in REPS_GRID
    }
    for reps, witness in witnesses.items():
        logger.info(
            "  reps=%s entangling witness mean|C_ij| = %.6f (%s)",
            reps,
            witness["mean_abs_connected_correlation"],
            "entangling" if witness["is_entangling"] else "EFFECTIVELY A PRODUCT STATE",
        )

    # ----------------------------------------------------------------------- the members
    logger.info("Recovering the two classical fusion members from the record")
    members = recover_members(
        f_train,
        f_validation,
        signal_cache,
        inner_train_mask,
        inner_validation_mask,
        classical_report=classical_report,
        deep_report=deep_report,
        member_cache=member_cache,
        max_epochs=max_epochs,
        batch_size=batch_size,
        seed=seed,
        refresh=refresh_members,
    )
    logger.info(
        "  members reproduce: cnn fold-9 %.10f, gbm fold-9 %.10f",
        members.deep_validation_roc_auc,
        members.tabular_validation_roc_auc,
    )

    # -------------------------------------------------------------------- the map arms
    map_arms: Dict[str, Dict[str, Any]] = {}
    for family in families:
        reps_grid: Sequence[Optional[int]] = (
            QUANTUM_REPS_GRID if family == "zz" else (None,)
        )
        logger.info(
            "Scoring map family %s (%d cells)", family, len(reps_grid) * len(HEAD_C_GRID)
        )
        elapsed = time.perf_counter()
        result = score_map_arm(
            family,
            q_train.features,
            f_train.y,
            inner_train_mask,
            inner_validation_mask,
            q_validation.features,
            reps_grid=reps_grid,
            seed=seed,
        )
        result["seconds"] = round(time.perf_counter() - elapsed, 1)
        map_arms[family] = result
        logger.info(
            "  %s selected %s, inner %.6f, fold-9 %.6f (%.1f s)",
            family,
            json.dumps(result["selected"]),
            result["selected_inner_roc_auc"] or float("nan"),
            roc_auc(f_validation.y, result["validation_scores"]) or float("nan"),
            result["seconds"],
        )

    # ---------------------------------------------------------------------------- arms
    arms: List[Dict[str, Any]] = []
    prefix = {"zz": "q", "poly2": "c", "rff36": "c"}
    for family, result in map_arms.items():
        threshold = select_threshold(inner_y, result["inner_scores"])
        threshold["selected_on"] = result["selected_on"]
        arms.append(
            arm_payload(
                f"{prefix.get(family, 'c')}@{family}",
                result["validation_scores"],
                f_validation,
                threshold,
                bootstrap_draws=bootstrap_draws,
                seed=seed,
                extra={
                    "kind": "quantum_map" if family == "zz" else "classical_control_map",
                    "selected": result["selected"],
                    "selected_inner_roc_auc": result["selected_inner_roc_auc"],
                    "n_grid_cells": result["n_grid_cells"],
                    "map_telemetry": result["map_telemetry"],
                    "input": f"{QUANTUM_DIM}-dim ecg-fit-1 PCA scores",
                    "input_dimension": QUANTUM_DIM,
                    "output_dimension": N_OBSERVABLES,
                    "seconds": result["seconds"],
                },
            )
        )

    # -------------------------------------------------------------------------- fusions
    base_inner = [members.deep_inner, _logit(members.tabular_inner)]
    base_validation = [members.deep_validation, _logit(members.tabular_validation)]
    base_names = ["deep", "tabular"]

    fusions: Dict[str, Dict[str, Any]] = {}
    fusion_specs: List[Tuple[str, Optional[str]]] = [("fusion@cnn+gbm", None)]
    fusion_specs += [(f"fusion@cnn+gbm+{family}", family) for family in map_arms]
    for name, family in fusion_specs:
        if family is None:
            inner_cols, eval_cols, names = base_inner, base_validation, base_names
        else:
            inner_cols = base_inner + [_logit(map_arms[family]["inner_scores"])]
            eval_cols = base_validation + [_logit(map_arms[family]["validation_scores"])]
            names = base_names + [family]
        fusion = blend(inner_cols, inner_y, eval_cols, names=names)
        fusions[name] = fusion
        threshold = select_threshold(inner_y, fusion["inner_probability"])
        threshold["selected_on"] = (
            f"TRAIN fold {INNER_VALIDATION_FOLD}, blended from members that never saw it"
        )
        arms.append(
            arm_payload(
                name,
                fusion["validation_probability"],
                f_validation,
                threshold,
                bootstrap_draws=bootstrap_draws,
                seed=seed,
                extra={
                    "kind": "quantum_fusion" if family == "zz" else "classical_fusion",
                    "members": fusion["members"],
                    "blender": fusion["blender"],
                    "blender_coefficients": fusion["coefficients"],
                    "blender_intercept": fusion["intercept"],
                    "inner_roc_auc": fusion["inner_roc_auc"],
                },
            )
        )

    # The recomputed bar must land on the recorded one, or the blender is not the one that
    # produced 0.946293 and every delta below is measured from the wrong origin.
    recorded_fusion = load_recorded_deep_arm(deep_report)["published_fusion_roc_auc"]
    recomputed_fusion = roc_auc(
        f_validation.y, fusions["fusion@cnn+gbm"]["validation_probability"]
    )
    if recorded_fusion is not None:
        _assert_reproduces("fusion@cnn+gbm", recomputed_fusion, float(recorded_fusion), 1e-9)

    # ---------------------------------------------------------------------- comparisons
    by_name = {arm["arm"]: arm for arm in arms}

    def delta(a: str, b: str) -> Dict[str, Any]:
        return paired_bootstrap_delta(
            f_validation.y,
            by_name[a]["validation_scores"],
            by_name[b]["validation_scores"],
            f_validation.patients,
            draws=bootstrap_draws,
            seed=seed,
        )

    headline = None
    comparisons: Dict[str, Any] = {}
    if "zz" in map_arms and "poly2" in map_arms:
        headline = delta("fusion@cnn+gbm+zz", "fusion@cnn+gbm+poly2")
        comparisons["HEADLINE_quantum_fusion_vs_same_shape_classical_fusion"] = headline
        comparisons["single_arm_zz_vs_poly2"] = delta("q@zz", "c@poly2")
    for family in map_arms:
        comparisons[f"fusion_with_{family}_vs_bar"] = delta(
            f"fusion@cnn+gbm+{family}", "fusion@cnn+gbm"
        )

    # ------------------------------------------------------------------------- verdict
    verdict: Dict[str, Any] = {
        "headline_comparison": "fusion@cnn+gbm+zz minus fusion@cnn+gbm+poly2",
        "pre_registered_in": "docs/PHASE_E4_DATA_EXPANSION.md Section 7.10",
    }
    if headline is not None and headline.get("measured"):
        favours_quantum = bool(headline["excludes_zero"] and headline["observed_delta"] > 0)
        verdict["quantum_beats_same_shape_classical_fusion"] = favours_quantum
        verdict["observed_delta"] = headline["observed_delta"]
        verdict["ci"] = [headline["p2_5"], headline["p97_5"]]
        zz_auc = by_name["fusion@cnn+gbm+zz"]["validation_roc_auc"]
        verdict["clears_absolute_bar"] = bool(
            zz_auc is not None
            and recorded_fusion is not None
            and zz_auc > float(recorded_fusion)
        )
        verdict["conclusion"] = (
            "CANDIDATE: the quantum map contributed beyond a same-shape classical map; "
            "Section 7.4 five conditions still apply before the word advantage is used."
            if favours_quantum
            else (
                "NULL: the paired interval spans zero, so the quantum map contributed "
                "nothing a same-shape classical map did not. Per Section 7.10 this is the "
                "conclusion regardless of how either arm scores against the absolute bar."
            )
        )
    else:
        verdict["conclusion"] = "NOT MEASURED: the headline comparison did not run."

    report: Dict[str, Any] = {
        "quantum_version": E4_QUANTUM_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "cache_version": ECG_CACHE_VERSION,
        "signal_cache_version": signal_cache.metadata.get("version"),
        "seed": seed,
        "bootstrap_draws": bootstrap_draws,
        "protocol": {
            **guards,
            "representation": f"ecg-fit-1 PCA-{QUANTUM_DIM}, fitted on folds 1-8 only",
            "inner_validation_fold": INNER_VALIDATION_FOLD,
            "search_budget_cells_per_family": {
                "zz": len(QUANTUM_REPS_GRID) * len(HEAD_C_GRID),
                "poly2": len(HEAD_C_GRID),
                "rff36": len(HEAD_C_GRID),
            },
            "zz_reps_scored": list(QUANTUM_REPS_GRID),
            "zz_reps_excluded_structurally": [
                reps for reps in REPS_GRID if reps not in QUANTUM_REPS_GRID
            ],
            "zz_reps_exclusion_reason": (
                "After the single Hadamard layer a Havlicek block is entirely diagonal, so "
                "|psi|^2 stays uniform and all 36 Z-observables are identically zero at "
                "reps=1. Excluded on that proof before any arm was scored, not on a "
                "measured ROC-AUC; the entangling_witness block records the measurement."
            ),
            "same_shape_contract": (
                f"every map is R^{QUANTUM_DIM} -> R^{N_OBSERVABLES} with zero trainable "
                "parameters, followed by an identical logistic head and an identical C grid"
            ),
            "pca_sees_fold_8": True,
            "pca_sees_fold_9": False,
            "test_partition_used": False,
        },
        "cohort": {
            "train": f_train.summary(),
            "validation": f_validation.summary(),
            "inner_validation_records": int(inner_validation_mask.sum()),
        },
        "map_validation": map_validation,
        "entangling_witness": witnesses,
        "members": members.summary(),
        "recorded_bar": {
            "arm": "fusion@cnn+gbm",
            "recorded_roc_auc": recorded_fusion,
            "recomputed_roc_auc": recomputed_fusion,
            "reproduces": bool(
                recorded_fusion is not None
                and recomputed_fusion is not None
                and abs(float(recomputed_fusion) - float(recorded_fusion)) < 1e-9
            ),
        },
        "arms": arms,
        "comparisons": comparisons,
        "verdict": verdict,
        "test_partition_used": False,
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    return report


def summary_lines(report: Dict[str, Any]) -> List[str]:
    """A human-readable digest, ordered so the control is read before the headline."""
    lines: List[str] = []
    lines.append(f"E4 quantum rung ({report['quantum_version']})")
    lines.append("")
    protocol = report.get("protocol", {})
    lines.append(f"  representation: {protocol.get('representation')}")
    lines.append(f"  contract: {protocol.get('same_shape_contract')}")
    for reps, witness in (report.get("entangling_witness") or {}).items():
        state = "entangling" if witness["is_entangling"] else "PRODUCT STATE"
        lines.append(
            f"  entangling witness reps={reps}: mean|C_ij| = "
            f"{witness['mean_abs_connected_correlation']:.6f} -> {state}"
        )
    lines.append("")
    lines.append("  arm                            fold-9 ROC-AUC   95% CI")
    for arm in report.get("arms", []):
        auc = arm.get("validation_roc_auc")
        if auc is None:
            lines.append(f"  {arm['arm']}")
            continue
        boot = arm.get("validation_bootstrap_roc_auc") or {}
        low, high = boot.get("p2_5"), boot.get("p97_5")
        interval = f"[{low:.4f}, {high:.4f}]" if low is not None and high is not None else ""
        lines.append(f"  {arm['arm']:<30} {auc:.6f}   {interval}")
    lines.append("")
    for name, comparison in (report.get("comparisons") or {}).items():
        if not comparison.get("measured"):
            continue
        verdict = "EXCLUDES ZERO" if comparison["excludes_zero"] else "spans zero"
        lines.append(
            f"  {name}: {comparison['observed_delta']:+.6f} "
            f"[{comparison['p2_5']:+.6f}, {comparison['p97_5']:+.6f}] {verdict}"
        )
    lines.append("")
    lines.append(f"  VERDICT: {report.get('verdict', {}).get('conclusion')}")
    return lines


def _strip_arrays(value: Any) -> Any:
    """Drop raw score vectors before serialising; keep the report auditable, not bulky."""
    dropped = {
        "validation_scores",
        "inner_scores",
        "inner_probability",
        "validation_probability",
    }
    if isinstance(value, dict):
        return {
            key: _strip_arrays(item) for key, item in value.items() if key not in dropped
        }
    if isinstance(value, list):
        return [_strip_arrays(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "E4 quantum rung: a quantum feature map and its same-shape classical control."
        )
    )
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--signals", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--classical-report", type=Path, default=None)
    parser.add_argument("--deep-report", type=Path, default=None)
    parser.add_argument("--member-cache", type=Path, default=None)
    parser.add_argument("--families", default=None, help="Comma-separated subset.")
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--refresh-members", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = run(
        cache=args.cache,
        signals_dir=args.signals,
        classical_report=args.classical_report,
        deep_report=args.deep_report,
        member_cache=args.member_cache,
        families=(
            tuple(name.strip() for name in args.families.split(",") if name.strip())
            if args.families
            else ("zz", "poly2", "rff36")
        ),
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
        refresh_members=args.refresh_members,
    )
    for line in summary_lines(report):
        logger.info("%s", line)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(_strip_arrays(report), indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info("")
        logger.info("Report written to %s", args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
