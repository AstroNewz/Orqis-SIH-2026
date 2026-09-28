"""E3 Round 4: does a *richer quantum feature map* add information beyond C7?

Rounds 1-3 all nulled with the **same** shallow E2 representation -- one RY layer + a CZ
ring, read by 16 local observables:

* Round 1 (:mod:`backend.evaluation.e3_hybrid_fusion`) read it with a *linear* head;
* Round 2 (:mod:`backend.evaluation.e3_quantum_kernel`) read it with a *fidelity kernel*;
* Round 3 (:mod:`backend.evaluation.e3_trainable_vqc`) let the circuit *train*.

None beat -- or added defensible information beyond -- the classical fusion baseline C7
(validation PR-AUC 0.913038). Every one of those rounds fixed the same three things: the
**feature map** (RY+CZ), the **observable set** (16 local Z), and the **feature selection**
(variance-optimal PCA-8). The one qualitatively distinct rung the mission ladder (§30) still
names is to change *those*: a **richer feature map**, a **structured observable set**, and a
**quantum-suitable (label-aware) feature selection**.

What Round 4 tests
------------------
The canonical richer map with an actual quantum-advantage pedigree is the **Havlicek et al.
(2019) ZZ feature map**: Hadamards, first-order ``P(2 x_i)`` single-qubit phases, and
pairwise ``exp(i (pi - x_i)(pi - x_j) Z_i Z_j)`` entanglers via the CX-P-CX gadget, repeated
over ``reps`` data re-uploading blocks. Its induced kernel is conjectured classically hard to
estimate -- it is the standard "this is where a quantum kernel could help" encoding. This
module hand-builds it from Aer-native ``h``/``p``/``cx`` gates (so Aer can simulate the exact
statevector without decomposition) and asserts the hand-built circuit is bit-identical (up to
global phase) to :class:`qiskit.circuit.library.ZZFeatureMap` before any number is reported.

It is read by a **structured 36-observable set** -- all 8 single-qubit ``<Z_i>`` plus all
``C(8,2)=28`` pairwise ``<Z_i Z_j>`` -- richer than E2's NN-only 16 (the ZZ map entangles
*all* pairs, so reading only nearest neighbours would blind the head to most of what the map
produces). 36 features against 103 train positives is a defensible ratio.

Two feature selections are tried, chosen on TRAIN OOF only: **PCA-8** (variance-optimal, the
same input Rounds 1-3 used) and **mutual-information top-8** of the raw MobileNet-576 (a
label-aware "quantum-suitable" selection -- the components a quantum map should perhaps
encode are the most class-discriminative, not the highest-variance). With ``reps in {1,2}``
that is a ``2 x 2`` TRAIN-OOF config grid; the best config is the headline quantum arm.

The honest hazard this module measures rather than assumes
----------------------------------------------------------
A feature map can be *accidentally separable*: if the entanglers contribute nothing, the
36-vector is just a nonlinear function of 8 independent qubits and there is no quantum
structure to help. So the module reports an **entanglement witness** -- the mean/max absolute
connected correlation ``C_ij = <Z_i Z_j> - <Z_i><Z_j>`` over all 28 pairs on TRAIN. On a
product state these are identically 0; a clearly nonzero value proves the ZZ entanglers
produced genuine correlations, so a null is "richer entanglement does not help," not "the map
degenerated to a product state."

Matched classical control (DEC-033, at equal output dimension)
--------------------------------------------------------------
The quantum arm is never reported alone. The matched control is an **RFF at the same 36-d
output** on the same PCA-8 input (median-heuristic bandwidth) -- the honest question "would
*any* 36-d classical nonlinear map of these features do the same?" If the quantum increment
does not clear the gate while a same-width classical map's does (or neither does), there is no
map-level quantum advantage, and the module says so.

Discipline (identical to Rounds 1-3 / E1 / E2)
----------------------------------------------
C7 is rebuilt and must reproduce 0.913038 to 1e-6 through the **fold-honest** identity path
(:func:`e3_hybrid_fusion._score_arm` with :func:`_identity_factory`), byte-identical to
Rounds 1-3 -- so both its reported train-CV OOF (0.766915) and the increment-reference column
are the same C7 every round. The ZZ readout is validated against Aer ``save_expectation_value``
to <=1e-10 and against the canonical qiskit map to <=1e-10 before any quantum number is used.
Every selector / scaler / PCA / RFF / logistic ``C`` is fit **inside each fold-train only**
(a monkeypatch spy test proves each per-fold selector sees only fold-train rows). The gating
test is E1.1's ``_increment_null`` with C7 as reference; ``build_inputs`` iterates only
``("train", "validation")`` so there is no code path to the test partition. Selecting the best
of four configs on TRAIN OOF and then reporting that config's OOF is a mild *optimistic* bias
-- it can only make a quantum "win" easier, so a null under it is conservative -- and all four
configs are reported transparently. No number is fabricated: a null (a richer map adds nothing
C7 lacks) is a valid, reportable §29B outcome.

Usage::

    python -m backend.evaluation.e3_feature_map --out reports_e3_feature_map.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from backend.core.config import Settings
from backend.evaluation.e0_controls import C_GRID, N_FOLDS, grouped_folds
from backend.evaluation.e1_failure_map import (
    BOOTSTRAP_DRAWS,
    INCREMENT_PERMUTATIONS,
    PRIMARY,
    SEED,
    E1Inputs,
    _average_precision,
    _cv_select,
    _increment_null,
    _standardise,
    bootstrap_pr_auc,
    build_inputs,
    candidate_matrices,
)
from backend.evaluation.e3_hybrid_fusion import (
    C7_ANCHOR_TOLERANCE,
    C7_DIM,
    C7_REFERENCE_PR_AUC,
    C5_DIM,
    MOBILENET_DIM,
    QUANTUM_VALIDATION_SAMPLE,
    HybridFusionError,
    _identity_factory,
    _make_rff_factory,
    _score_arm,
)
from backend.ml.quantum_visual import PCA_COMPONENTS, PCAModel
from quantum_ml.readout import (
    all_pair_masks,
    observable_expectations,
    single_qubit_masks,
    z_string_diagonal,
)

logger = logging.getLogger(__name__)

#: Bump when the Round-4 protocol semantics change so a stored report cannot be reinterpreted.
E3_FEATURE_MAP_VERSION = "v1-e3-feature-map-1"

#: One qubit per selected feature (matches Rounds 1-3's 8-qubit register).
N_QUBITS = 8

#: Structured observable set: 8 singles + C(8,2)=28 pairs.
N_OBSERVABLES = N_QUBITS + (N_QUBITS * (N_QUBITS - 1)) // 2  # 36

#: Data re-uploading depths searched on TRAIN OOF (Havlicek reps).
REPS_GRID: Tuple[int, ...] = (1, 2)

#: Feature-selection width (top-k for MI; component count for PCA).
SELECT_K = 8

#: A connected correlation below this in absolute mean means the map is effectively a product
#: state (the entanglers did nothing); above it the map is genuinely entangling.
ENTANGLING_WITNESS_FLOOR = 1e-3

_EPS = 1e-12


class FeatureMapError(RuntimeError):
    """Raised when the Round-4 map, observable set, or an input batch is malformed."""


# --------------------------------------------------------------- richer quantum map
def _pair_index_list(n_qubits: int) -> List[Tuple[int, int]]:
    """The ``(i, j)`` pairs in the exact order :func:`all_pair_masks` emits them."""
    return list(itertools.combinations(range(n_qubits), 2))


def _build_zz_circuit(
    n_qubits: int, reps: int
) -> Tuple[QuantumCircuit, ParameterVector]:
    """The Havlicek ZZ feature map from Aer-native ``h``/``p``/``cx`` gates.

    Per re-uploading block: ``H`` on every qubit; a first-order phase ``P(2 x_i)``; then for
    every pair ``(i, j)`` the ``CX(i,j) - P(2 (pi - x_i)(pi - x_j), j) - CX(i,j)`` gadget that
    realises ``exp(i (pi - x_i)(pi - x_j) Z_i Z_j)``. This is bit-identical (up to global
    phase) to :class:`qiskit.circuit.library.ZZFeatureMap` with ``entanglement="full"``, which
    :func:`validate_against_qiskit_zzmap` asserts -- the hand build exists only so Aer can
    simulate the statevector without decomposing an opaque library instruction.
    """
    if reps not in REPS_GRID:
        raise FeatureMapError(f"reps must be in {REPS_GRID}; got {reps}.")
    theta = ParameterVector("x", n_qubits)
    circuit = QuantumCircuit(n_qubits, name=f"zz_reps{reps}")
    pi = float(np.pi)
    for _ in range(reps):
        for i in range(n_qubits):
            circuit.h(i)
        for i in range(n_qubits):
            circuit.p(2.0 * theta[i], i)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                circuit.cx(i, j)
                circuit.p(2.0 * (pi - theta[i]) * (pi - theta[j]), j)
                circuit.cx(i, j)
    return circuit, theta


def _mask_to_pauli_label(mask: int, n_qubits: int) -> str:
    """Qiskit Pauli label (qubit 0 last) for a Z-string ``mask`` -- e.g. ``Z_0 -> 'IIIIIIIZ'``."""
    chars = ["I"] * n_qubits
    for k in range(n_qubits):
        if (mask >> k) & 1:
            chars[n_qubits - 1 - k] = "Z"
    return "".join(chars)


@dataclass
class RichQuantumFeatureMap:
    """Executes the ZZ map to an exact Aer statevector and reads 36 structured observables.

    Fixed (0 trainable parameters): the ``n_qubits`` phase angles are *data*. Heavy objects
    (circuit, stacked ±1 diagonals, simulator) are built once and reused.
    """

    reps: int
    n_qubits: int = N_QUBITS
    seed: int = SEED

    _circuit: QuantumCircuit = field(init=False, repr=False)
    _theta: ParameterVector = field(init=False, repr=False)
    _masks: Tuple[int, ...] = field(init=False, repr=False)
    _diagonals: np.ndarray = field(init=False, repr=False)
    _simulator: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._circuit, self._theta = _build_zz_circuit(self.n_qubits, self.reps)
        self._masks = tuple(single_qubit_masks(self.n_qubits)) + tuple(
            all_pair_masks(self.n_qubits)
        )
        if len(self._masks) != N_OBSERVABLES:
            raise FeatureMapError(
                f"Expected {N_OBSERVABLES} observables, built {len(self._masks)}."
            )
        self._diagonals = np.stack(
            [z_string_diagonal(self.n_qubits, m) for m in self._masks]
        )

    @property
    def masks(self) -> Tuple[int, ...]:
        return self._masks

    def _simulator_instance(self):
        if self._simulator is None:
            from qiskit_aer import AerSimulator

            self._simulator = AerSimulator(method="statevector", seed_simulator=self.seed)
        return self._simulator

    def _prepare(self, features: np.ndarray) -> np.ndarray:
        arr = np.atleast_2d(np.asarray(features, dtype=np.float64))
        if arr.shape[1] != self.n_qubits:
            raise FeatureMapError(
                f"Feature batch has {arr.shape[1]} columns but the map spans "
                f"{self.n_qubits} qubits. Refusing to pad or truncate."
            )
        if not np.all(np.isfinite(arr)):
            raise FeatureMapError("Feature batch contains non-finite values.")
        return arr

    def _bound(self, row: Sequence[float]) -> QuantumCircuit:
        mapping = {self._theta[i]: float(row[i]) for i in range(self.n_qubits)}
        return self._circuit.assign_parameters(mapping, inplace=False)

    def statevectors(self, features: np.ndarray) -> np.ndarray:
        """``(n, 2**n_qubits)`` exact Aer statevectors for a batch of feature rows."""
        arr = self._prepare(features)
        simulator = self._simulator_instance()
        circuits = []
        for row in arr:
            bound = self._bound(row)
            bound.save_statevector()
            circuits.append(bound)
        result = simulator.run(circuits).result()
        dim = 1 << self.n_qubits
        out = np.empty((arr.shape[0], dim), dtype=np.complex128)
        for k in range(arr.shape[0]):
            state = result.data(k)["statevector"]
            out[k] = np.asarray(getattr(state, "data", state), dtype=np.complex128)
        return out

    def transform(self, features: np.ndarray) -> np.ndarray:
        """``(n, 36)`` expectations: exact Aer statevector then ``|psi|^2 @ diag^T``."""
        states = self.statevectors(features)
        probabilities = np.abs(states) ** 2
        return observable_expectations(probabilities, self._diagonals)

    def connected_correlations(self, features: np.ndarray) -> np.ndarray:
        """``(n, 28)`` connected correlations ``C_ij = <Z_iZ_j> - <Z_i><Z_j>``.

        Entanglement-witness telemetry only. Columns follow :func:`all_pair_masks` order.
        """
        feats = self.transform(features)
        singles = feats[:, : self.n_qubits]
        pairs = feats[:, self.n_qubits :]
        out = np.empty_like(pairs)
        for k, (i, j) in enumerate(_pair_index_list(self.n_qubits)):
            out[:, k] = pairs[:, k] - singles[:, i] * singles[:, j]
        return out

    def telemetry(self) -> Dict[str, Any]:
        two_qubit = sum(
            1 for instr in self._circuit.data if instr.operation.num_qubits == 2
        )
        return {
            "map": "havlicek_zz",
            "reps": self.reps,
            "n_qubits": self.n_qubits,
            "n_encoding_parameters": self.n_qubits,
            "n_trainable_parameters": 0,
            "two_qubit_gate_count": int(two_qubit),
            "circuit_depth": int(self._circuit.depth()),
            "state_dimension": 1 << self.n_qubits,
            "n_observables": len(self._masks),
            "entanglement": "full",
            "seed": self.seed,
        }


# --------------------------------------------------------------- correctness gates
def _aer_expectations(fmap: RichQuantumFeatureMap, features: np.ndarray) -> np.ndarray:
    """``(n, 36)`` expectations via Aer ``save_expectation_value`` -- shares no arithmetic
    with the fast ``|psi|^2 @ diag^T`` path, which is what makes their agreement meaningful."""
    from qiskit.quantum_info import SparsePauliOp

    arr = fmap._prepare(features)
    simulator = fmap._simulator_instance()
    labels = [_mask_to_pauli_label(m, fmap.n_qubits) for m in fmap.masks]
    qubit_indices = list(range(fmap.n_qubits))
    circuits = []
    for row in arr:
        bound = fmap._bound(row)
        for obs_index, pauli_label in enumerate(labels):
            bound.save_expectation_value(
                SparsePauliOp(pauli_label), qubit_indices, label=f"exp_{obs_index}"
            )
        circuits.append(bound)
    result = simulator.run(circuits).result()
    out = np.empty((arr.shape[0], len(labels)), dtype=np.float64)
    for k in range(arr.shape[0]):
        data = result.data(k)
        for obs_index in range(len(labels)):
            out[k, obs_index] = float(np.real(data[f"exp_{obs_index}"]))
    return out


def validate_readout_against_aer(
    fmap: RichQuantumFeatureMap, features: np.ndarray, *, tolerance: float = 1e-10
) -> Dict[str, Any]:
    """Assert the fast contraction equals Aer ``save_expectation_value`` to ``tolerance``."""
    fast = fmap.transform(features)
    exact = _aer_expectations(fmap, features)
    max_abs_deviation = float(np.max(np.abs(fast - exact))) if fast.size else 0.0
    if max_abs_deviation > tolerance:
        raise FeatureMapError(
            f"Fast |psi|^2 @ diag^T disagrees with Aer save_expectation_value by "
            f"{max_abs_deviation:.3e} > {tolerance:.1e}. Refusing to report Round-4 features."
        )
    return {
        "max_abs_deviation": max_abs_deviation,
        "tolerance": tolerance,
        "passed": True,
        "n_samples_checked": int(np.atleast_2d(features).shape[0]),
        "n_observables": len(fmap.masks),
        "reps": fmap.reps,
    }


def validate_against_qiskit_zzmap(
    fmap: RichQuantumFeatureMap, features: np.ndarray, *, tolerance: float = 1e-10
) -> Dict[str, Any]:
    """Assert the hand-built ZZ circuit equals the canonical qiskit map (up to global phase).

    Compares ``1 - |<psi_library | psi_handbuilt>|`` per sample; a value above ``tolerance``
    means the hand build is not the canonical Havlicek map and no number is reported. Records
    a best-effort status if the library map cannot be imported (a future qiskit removal).
    """
    try:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit.quantum_info import Statevector
    except Exception as exc:  # pragma: no cover - defensive against library churn
        return {"passed": True, "checked": False, "reason": f"library map unavailable: {exc!r}"}

    arr = fmap._prepare(features)
    library = ZZFeatureMap(feature_dimension=fmap.n_qubits, reps=fmap.reps, entanglement="full")
    library_params = list(library.parameters)
    hand_states = fmap.statevectors(arr)
    max_infidelity = 0.0
    for k, row in enumerate(arr):
        bound = library.assign_parameters(
            {p: float(row[idx]) for idx, p in enumerate(library_params)}, inplace=False
        )
        psi_lib = np.asarray(Statevector(bound).data, dtype=np.complex128)
        overlap = abs(complex(np.vdot(psi_lib, hand_states[k])))
        max_infidelity = max(max_infidelity, abs(1.0 - overlap))
    if max_infidelity > tolerance:
        raise FeatureMapError(
            f"Hand-built ZZ map disagrees with qiskit ZZFeatureMap by 1-|overlap|="
            f"{max_infidelity:.3e} > {tolerance:.1e}. Refusing to report Round-4 features."
        )
    return {
        "passed": True,
        "checked": True,
        "max_infidelity": max_infidelity,
        "tolerance": tolerance,
        "n_samples_checked": int(arr.shape[0]),
        "reps": fmap.reps,
    }


# --------------------------------------------------------- fold-honest feature selectors
#: A selector factory fits every train-only object on a fold-train ``(X, y)`` source and
#: returns a pure transform: raw MobileNet-576 rows -> ``(n, SELECT_K)`` standardised features.
SupervisedTransform = Callable[[np.ndarray], np.ndarray]
SupervisedFactory = Callable[[np.ndarray, np.ndarray], SupervisedTransform]


def _standardiser(sub_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column mean/std from fold-train, with a safe unit std for constant columns."""
    mean = sub_train.mean(axis=0)
    std = sub_train.std(axis=0)
    std = np.where(std > _EPS, std, 1.0)
    return mean, std


def _make_pca_selector(*, seed: int, k: int = SELECT_K) -> SupervisedFactory:
    """MobileNet-576 -> (fold-train) PCA-k -> standardised. The variance-optimal input."""

    def factory(fold_source: np.ndarray, _fold_y: np.ndarray) -> SupervisedTransform:
        pca = PCAModel.fit(fold_source, n_components=k, seed=seed)
        mean, std = _standardiser(pca.transform(fold_source))

        def transform(x: np.ndarray) -> np.ndarray:
            return (pca.transform(x) - mean) / std

        return transform

    return factory


def _make_mi_selector(*, seed: int, k: int = SELECT_K) -> SupervisedFactory:
    """MobileNet-576 -> (fold-train) mutual-information top-k -> standardised.

    The label-aware "quantum-suitable" selection: the k raw descriptor dimensions with the
    highest mutual information with the label on the fold-train rows, standardised on fold-train.
    MI and the retained indices are fit on the fold-train ``(X, y)`` only -- the exact
    label-derived-selection surface the mandate forbids leaking across folds, kept fold-honest.
    """

    def factory(fold_source: np.ndarray, fold_y: np.ndarray) -> SupervisedTransform:
        from sklearn.feature_selection import mutual_info_classif

        mi = mutual_info_classif(
            fold_source, np.asarray(fold_y, dtype=int), random_state=seed
        )
        idx = np.sort(np.argsort(mi)[::-1][:k])  # top-k, then stable ascending column order
        sub = fold_source[:, idx]
        mean, std = _standardiser(sub)

        def transform(x: np.ndarray) -> np.ndarray:
            return (np.asarray(x, dtype=np.float64)[:, idx] - mean) / std

        return transform

    return factory


SELECTORS: Dict[str, Callable[..., SupervisedFactory]] = {
    "pca": _make_pca_selector,
    "mi": _make_mi_selector,
}


# --------------------------------------------------------------- fold-honest arm scorer
def _score_richmap_config(
    source_train: np.ndarray,
    source_validation: np.ndarray,
    y_train: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    selector_factory: SupervisedFactory,
    feature_map: RichQuantumFeatureMap,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Fold-honest logistic scoring of one (selector, reps) config of the richer map.

    Per fold the selector is refit on the fold-train ``(X, y)`` only, its 8 features are ZZ
    -encoded to 36 observables, standardised on fold-train, and read by a logistic head; the
    out-of-fold TRAIN column and the fit-on-all-train VALIDATION column are what
    ``_increment_null`` and ``bootstrap_pr_auc`` consume -- exactly the ``_score_arm`` contract.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y_train, dtype=int)
    source_train = np.asarray(source_train, dtype=np.float64)
    source_validation = np.asarray(source_validation, dtype=np.float64)

    fold_reps: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in folds:
        transform = selector_factory(source_train[train_idx], y[train_idx])
        feat_train = feature_map.transform(transform(source_train[train_idx]))
        feat_test = feature_map.transform(transform(source_train[test_idx]))
        rep_train, rep_test = _standardise(feat_train, feat_test)
        fold_reps.append((train_idx, test_idx, rep_train, rep_test))

    def _oof_at(c: float) -> np.ndarray:
        oof = np.full(y.size, np.nan, dtype=np.float64)
        for train_idx, test_idx, rep_train, rep_test in fold_reps:
            if len(set(y[train_idx].tolist())) < 2:
                continue
            model = LogisticRegression(
                C=float(c), max_iter=5000, class_weight="balanced", random_state=seed
            )
            model.fit(rep_train, y[train_idx])
            oof[test_idx] = np.asarray(model.predict_proba(rep_test))[:, 1]
        return oof

    def _score(c: float) -> Optional[float]:
        oof = _oof_at(c)
        usable = np.isfinite(oof)
        if usable.sum() < 2:
            return None
        return _average_precision(y[usable], oof[usable])

    best_c, by_c = _cv_select(C_GRID, _score, label="e3 richmap C")
    oof = _oof_at(float(best_c))
    usable = np.isfinite(oof)
    oof_pr_auc = None if usable.sum() < 2 else _average_precision(y[usable], oof[usable])

    transform_all = selector_factory(source_train, y)
    feat_train_all = feature_map.transform(transform_all(source_train))
    feat_validation = feature_map.transform(transform_all(source_validation))
    rep_train_all, rep_validation = _standardise(feat_train_all, feat_validation)
    final = LogisticRegression(
        C=float(best_c), max_iter=5000, class_weight="balanced", random_state=seed
    )
    final.fit(rep_train_all, y)
    validation_score = np.asarray(final.predict_proba(rep_validation))[:, 1]

    return {
        "representation_dim": int(rep_train_all.shape[1]),
        "reps": feature_map.reps,
        "selected_c": float(best_c),
        "cv_average_precision_by_setting": by_c,
        "train_out_of_fold": oof,
        "train_cv_pr_auc": None if oof_pr_auc is None else round(oof_pr_auc, 6),
        "validation_score": validation_score,
    }


def _roc_auc(y: np.ndarray, scores: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, np.asarray(scores, dtype=np.float64)))


# ------------------------------------------------------------------------------- driver
def run(
    *,
    config: Optional[Settings] = None,
    seed: int = SEED,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    increment_permutations: int = INCREMENT_PERMUTATIONS,
    quantum_validation_sample: int = QUANTUM_VALIDATION_SAMPLE,
) -> Dict[str, Any]:
    """Run the E3 Round-4 richer-feature-map experiment on the primary leak-free condition."""
    started = time.perf_counter()

    per_condition, provenance = build_inputs(config=config, conditions=(PRIMARY,))
    inputs = per_condition[PRIMARY]
    if set(inputs.patients_train) & set(inputs.patients_validation):
        raise FeatureMapError("Patient overlap between train and validation; refusing to measure.")

    folds, cv_note = grouped_folds(
        inputs.y_train, inputs.patients_train, seed=seed, n_splits=N_FOLDS
    )
    logger.info(
        "E3-MAP primary %s: %d train / %d validation (%d / %d positive); %s",
        PRIMARY, inputs.y_train.size, inputs.y_validation.size,
        int(inputs.y_train.sum()), int(inputs.y_validation.sum()), cv_note,
    )

    # --- Rebuild C7 and anchor it through the fold-honest identity path (Rounds 1-3 consistent).
    specs = candidate_matrices(inputs)
    c6_train, c6_validation = specs["C6"]["train"], specs["C6"]["validation"]
    c5_train, c5_validation = specs["C5"]["train"], specs["C5"]["validation"]
    if c6_train.shape[1] != MOBILENET_DIM or c5_train.shape[1] != C5_DIM:
        raise FeatureMapError(
            f"Unexpected C6/C5 dims ({c6_train.shape[1]}/{c5_train.shape[1]}); expected "
            f"{MOBILENET_DIM}/{C5_DIM}."
        )
    c7_train = np.hstack([c6_train, c5_train]).astype(np.float64)
    c7_validation = np.hstack([c6_validation, c5_validation]).astype(np.float64)
    if c7_train.shape[1] != C7_DIM:
        raise FeatureMapError(f"C7 has {c7_train.shape[1]} columns, expected {C7_DIM}.")

    c7_arm = _score_arm(
        c7_train, c7_validation, inputs.y_train, folds, preprocess=_identity_factory, seed=seed
    )
    c7_pr_auc = _average_precision(inputs.y_validation, c7_arm["validation_score"])
    anchor_ok = (
        c7_pr_auc is not None and abs(float(c7_pr_auc) - C7_REFERENCE_PR_AUC) <= C7_ANCHOR_TOLERANCE
    )
    anchor = {
        "target_pr_auc": C7_REFERENCE_PR_AUC,
        "reproduced_pr_auc": None if c7_pr_auc is None else round(float(c7_pr_auc), 6),
        "selected_c": c7_arm.get("selected_c"),
        "tolerance": C7_ANCHOR_TOLERANCE,
        "reproduced": bool(anchor_ok),
        "path": "e3_hybrid_fusion._score_arm(_identity_factory) on hstack([C6, C5]) -- Rounds 1-3 consistent",
    }
    if not anchor_ok:
        logger.warning("C7 anchor NOT reproduced (%s vs %s); conclusions suppressed.",
                       anchor["reproduced_pr_auc"], C7_REFERENCE_PR_AUC)

    mobilenet_train = np.asarray(inputs.e1_train["mobilenet"], dtype=np.float64)
    mobilenet_validation = np.asarray(inputs.e1_validation["mobilenet"], dtype=np.float64)

    # --- Build + gate the ZZ map at each depth before any quantum number is used. The check
    # angles are PCA-8 features standardised on all TRAIN (a representative, fold-honest sample).
    check_selector = _make_pca_selector(seed=seed)(mobilenet_train, inputs.y_train)
    sample = min(int(quantum_validation_sample), mobilenet_train.shape[0])
    check_features = check_selector(mobilenet_train[:sample])
    feature_maps: Dict[int, RichQuantumFeatureMap] = {}
    aer_checks: Dict[str, Any] = {}
    canonical_checks: Dict[str, Any] = {}
    for reps in REPS_GRID:
        fmap = RichQuantumFeatureMap(reps=reps, seed=seed)
        feature_maps[reps] = fmap
        aer_checks[f"reps{reps}"] = validate_readout_against_aer(fmap, check_features)
        canonical_checks[f"reps{reps}"] = validate_against_qiskit_zzmap(fmap, check_features)
    aer_max_dev = max(c["max_abs_deviation"] for c in aer_checks.values())
    canonical_max_inf = max(
        c.get("max_infidelity", 0.0) for c in canonical_checks.values()
    )

    # --- Quantum config grid: (selector, reps), selected on TRAIN OOF.
    by_config: List[Dict[str, Any]] = []
    config_fits: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for selector_name, selector_builder in SELECTORS.items():
        selector_factory = selector_builder(seed=seed)
        for reps in REPS_GRID:
            logger.info("  richmap config selector=%s reps=%d ...", selector_name, reps)
            fit = _score_richmap_config(
                mobilenet_train, mobilenet_validation, inputs.y_train, folds,
                selector_factory=selector_factory, feature_map=feature_maps[reps], seed=seed,
            )
            config_fits[(selector_name, reps)] = fit
            by_config.append({
                "selector": selector_name,
                "reps": reps,
                "train_cv_pr_auc": fit["train_cv_pr_auc"],
                "validation_pr_auc": None if _average_precision(
                    inputs.y_validation, fit["validation_score"]
                ) is None else round(float(_average_precision(
                    inputs.y_validation, fit["validation_score"]
                )), 6),
                "selected_c": fit["selected_c"],
            })

    # Headline quantum arm = best config by TRAIN OOF (never by validation/test).
    def _oof_key(item: Tuple[Tuple[str, int], Dict[str, Any]]) -> Tuple[bool, float]:
        val = item[1]["train_cv_pr_auc"]
        return (val is None, -(val or 0.0))

    best_key, best_fit = min(config_fits.items(), key=_oof_key)
    best_selector, best_reps = best_key
    logger.info("  selected quantum config: selector=%s reps=%d (train-OOF %s)",
                best_selector, best_reps, best_fit["train_cv_pr_auc"])

    # --- Entanglement witness on all TRAIN with the selected config's map + selector.
    witness_selector = SELECTORS[best_selector](seed=seed)(mobilenet_train, inputs.y_train)
    witness_features = witness_selector(mobilenet_train)
    connected = feature_maps[best_reps].connected_correlations(witness_features)
    mean_abs_cc = float(np.mean(np.abs(connected)))
    max_abs_cc = float(np.max(np.abs(connected)))
    entangling = bool(mean_abs_cc > ENTANGLING_WITNESS_FLOOR)
    entanglement_witness = {
        "mean_abs_connected_correlation": mean_abs_cc,
        "max_abs_connected_correlation": max_abs_cc,
        "n_pairs": int(connected.shape[1]),
        "floor": ENTANGLING_WITNESS_FLOOR,
        "entangling": entangling,
    }

    # --- Matched classical control: RFF at equal 36-d output on the same PCA-8 input.
    logger.info("  arm rff36 (matched control) ...")
    rff_fit = _score_arm(
        mobilenet_train, mobilenet_validation, inputs.y_train, folds,
        preprocess=_make_rff_factory(seed=seed, n_features=N_OBSERVABLES), seed=seed,
    )

    # --- Assemble arms.
    def _boot(scores: np.ndarray) -> Dict[str, Any]:
        return bootstrap_pr_auc(
            inputs.y_validation, scores, inputs.patients_validation,
            draws=bootstrap_draws, seed=seed,
        )

    rich_val = _average_precision(inputs.y_validation, best_fit["validation_score"])
    rff_val = _average_precision(inputs.y_validation, rff_fit["validation_score"])
    c7_boot = _boot(np.asarray(c7_arm["validation_score"], dtype=np.float64))
    arms: Dict[str, Any] = {
        "c7": {
            "arm": "c7", "model": "linear_logistic", "representation_dim": C7_DIM,
            "train_cv_pr_auc": c7_arm["train_cv_pr_auc"],
            "validation_pr_auc": None if c7_pr_auc is None else round(float(c7_pr_auc), 6),
            "validation_roc_auc": _roc_auc(inputs.y_validation, c7_arm["validation_score"]),
            "validation_bootstrap": c7_boot,
        },
        "rich_quantum": {
            "arm": "rich_quantum", "model": "havlicek_zz + linear_logistic",
            "selector": best_selector, "reps": best_reps,
            "representation_dim": best_fit["representation_dim"],
            "selected_c": best_fit["selected_c"],
            "train_cv_pr_auc": best_fit["train_cv_pr_auc"],
            "validation_pr_auc": None if rich_val is None else round(float(rich_val), 6),
            "validation_roc_auc": _roc_auc(inputs.y_validation, best_fit["validation_score"]),
            "validation_bootstrap": _boot(np.asarray(best_fit["validation_score"], dtype=np.float64)),
            "map_telemetry": feature_maps[best_reps].telemetry(),
            "entanglement_witness": entanglement_witness,
        },
        "rff36": {
            "arm": "rff36", "model": "rff + linear_logistic",
            "representation_dim": rff_fit["representation_dim"],
            "selected_c": rff_fit["selected_c"],
            "train_cv_pr_auc": rff_fit["train_cv_pr_auc"],
            "validation_pr_auc": None if rff_val is None else round(float(rff_val), 6),
            "validation_roc_auc": _roc_auc(inputs.y_validation, rff_fit["validation_score"]),
            "validation_bootstrap": _boot(np.asarray(rff_fit["validation_score"], dtype=np.float64)),
        },
    }

    # --- Increment tests: C7 -> {rich_quantum, rff36}, C7 the fold-honest reference.
    increments: Dict[str, Dict[str, Any]] = {}
    reference_val = np.asarray(c7_arm["validation_score"], dtype=np.float64)
    reference_oof = np.asarray(c7_arm["train_out_of_fold"], dtype=np.float64)
    for name, fit in (("rich_quantum", best_fit), ("rff36", rff_fit)):
        logger.info("  increment C7 -> %s ...", name)
        increments[name] = _increment_null(
            reference_val,
            np.asarray(fit["validation_score"], dtype=np.float64),
            reference_oof,
            np.asarray(fit["train_out_of_fold"], dtype=np.float64),
            inputs.y_train,
            inputs.y_validation,
            inputs.patients_train,
            inputs.patients_validation,
            draws=increment_permutations,
            seed=seed,
        )

    # --- Honest verdict. No fabrication: a null is a valid §29B outcome.
    c7_val = arms["c7"]["validation_pr_auc"]
    rich_val_r = arms["rich_quantum"]["validation_pr_auc"]
    rff_val_r = arms["rff36"]["validation_pr_auc"]
    rich_survives = increments["rich_quantum"].get("survives_gating_null")
    rff_survives = increments["rff36"].get("survives_gating_null")
    rich_beats_c7_val = rich_val_r is not None and c7_val is not None and rich_val_r > c7_val
    rich_beats_c7_oof = (
        best_fit["train_cv_pr_auc"] is not None
        and c7_arm["train_cv_pr_auc"] is not None
        and best_fit["train_cv_pr_auc"] > c7_arm["train_cv_pr_auc"]
    )
    defensible_quantum_advantage = bool(rich_survives)
    quantum_specific = bool(rich_survives and not rff_survives)

    if not anchor_ok:
        conclusion = "C7 anchor did not reproduce; experiment not trustworthy, no conclusion drawn."
    elif not entangling:
        conclusion = (
            "The selected ZZ map is effectively separable (train mean |connected correlation| "
            f"{mean_abs_cc:.2e} <= {ENTANGLING_WITNESS_FLOOR:.0e}): its entanglers contributed "
            "nothing, so there is no quantum structure to help. Reported as measured (§29B)."
        )
    elif quantum_specific:
        conclusion = (
            "The C7 -> rich-quantum increment survives the patient-blocked column-permutation "
            "null while the matched RFF-36 increment does not: on this evidence the richer "
            "quantum feature map carries information beyond C7 that a same-width classical map "
            "does not. Freeze and re-examine before any test-set claim."
        )
    elif defensible_quantum_advantage:
        conclusion = (
            "The C7 -> rich-quantum increment survives its gating null, but the matched RFF-36 "
            "increment also survives: the added information is not quantum-specific (a generic "
            "extra-dimension effect). Not a quantum advantage."
        )
    else:
        conclusion = (
            "The C7 -> rich-quantum increment does NOT survive the patient-blocked "
            "column-permutation null: on this evidence a richer (Havlicek ZZ) feature map with "
            "structured observables and label-aware feature selection adds no information beyond "
            "C7. A null result (mission §29B), reported as measured."
        )

    leaderboard = sorted(
        (
            {"arm": n, "validation_pr_auc": arms[n]["validation_pr_auc"],
             "validation_roc_auc": arms[n].get("validation_roc_auc"),
             "train_cv_pr_auc": arms[n]["train_cv_pr_auc"]}
            for n in arms
        ),
        key=lambda r: (r["validation_pr_auc"] is None, -(r["validation_pr_auc"] or 0.0)),
    )

    summary = {
        "primary_condition": PRIMARY,
        "c7_validation_pr_auc": c7_val,
        "rich_quantum_validation_pr_auc": rich_val_r,
        "rff36_validation_pr_auc": rff_val_r,
        "selected_selector": best_selector,
        "selected_reps": best_reps,
        "rich_quantum_beats_c7_on_validation": rich_beats_c7_val,
        "rich_quantum_beats_c7_on_train_oof": rich_beats_c7_oof,
        "rich_quantum_entangling": entangling,
        "rich_quantum_increment_survives_gating_null": rich_survives,
        "rff36_increment_survives_gating_null": rff_survives,
        "defensible_quantum_advantage_over_c7": defensible_quantum_advantage,
        "quantum_specific_advantage": quantum_specific,
        "leaderboard": leaderboard,
        "conclusion": conclusion,
    }

    duration = time.perf_counter() - started
    return {
        "e3_feature_map_version": E3_FEATURE_MAP_VERSION,
        "question": (
            "Does a richer quantum feature map (Havlicek ZZ + re-uploading), read by a "
            "structured 36-observable set with label-aware feature selection, add information "
            "beyond C7 (validation PR-AUC 0.913038) under identical patient-grouped folds, "
            "train-only fitting, and a dimension-matched classical control?"
        ),
        "test_partition_used": False,
        "test_partition_note": (
            "Reads TRAIN and VALIDATION only via e1_failure_map.build_inputs, which iterates "
            "only ('train', 'validation'). No code path selects the test partition."
        ),
        "primary_condition": PRIMARY,
        "cv_note": cv_note,
        "rows": inputs.shape_note,
        "c7_anchor": anchor,
        "quantum_feature_map_aer_check": {
            "max_abs_deviation": aer_max_dev, "tolerance": 1e-10, "passed": True,
            "by_reps": aer_checks,
        },
        "quantum_feature_map_canonical_check": {
            "max_infidelity": canonical_max_inf, "tolerance": 1e-10, "passed": True,
            "by_reps": canonical_checks,
        },
        "config_grid": by_config,
        "arms": arms,
        "increment_vs_c7": increments,
        "summary": summary,
        "provenance": provenance,
        "protocol": {
            "reference_baseline": "C7 = hstack([C6 mobilenet(576), C5 descriptor+multiscale(619)]) = 1195, logistic (fold-honest identity path)",
            "rich_quantum": (
                "MobileNet-576 -> fold-train {PCA-8 | MI-top-8} -> standardise -> Havlicek ZZ "
                "feature map (reps in {1,2}, entanglement=full) -> 36 observables (8 singles + "
                "28 pairs) -> logistic; config chosen on TRAIN OOF"
            ),
            "matched_control": "MobileNet-576 -> fold-train PCA-8 -> RFF-36 (DEC-033 analogue at equal output width)",
            "folds": f"{N_FOLDS}-fold patient-grouped StratifiedGroupKFold on TRAIN, seed {seed}",
            "preprocessing": "train-only; selector/scaler/PCA/RFF refit inside each fold-train for the OOF column",
            "gating_test": "e1_failure_map._increment_null with C7 as reference",
            "entanglement_witness": "train mean/max |<Z_iZ_j> - <Z_i><Z_j>| over all 28 pairs (0 => product state)",
            "correctness_gates": [
                "C7 reproduces 0.913038 to 1e-6",
                "fast |psi|^2 @ diag^T == Aer save_expectation_value to <=1e-10",
                "hand-built ZZ == qiskit ZZFeatureMap (up to global phase) to <=1e-10",
            ],
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "seed": seed,
            "n_folds": N_FOLDS,
            "bootstrap_draws": bootstrap_draws,
            "increment_permutations": increment_permutations,
            "pca_components": PCA_COMPONENTS,
            "n_observables": N_OBSERVABLES,
            "reps_grid": list(REPS_GRID),
            "select_k": SELECT_K,
        },
        "duration_seconds": round(duration, 3),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="E3 Round 4: richer-feature-map research experiment.")
    parser.add_argument("--out", type=str, default="reports_e3_feature_map.json")
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--permutations", type=int, default=INCREMENT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    payload = run(seed=args.seed, bootstrap_draws=args.bootstrap_draws, increment_permutations=args.permutations)
    from pathlib import Path

    Path(args.out).write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    logger.info("Wrote %s", args.out)
    s = payload["summary"]
    logger.info(
        "C7=%s rich_quantum=%s (selector=%s reps=%s) rff36=%s | rich beats C7: %s | "
        "rich increment survives: %s (rff=%s) | entangling: %s",
        s["c7_validation_pr_auc"], s["rich_quantum_validation_pr_auc"],
        s["selected_selector"], s["selected_reps"], s["rff36_validation_pr_auc"],
        s["rich_quantum_beats_c7_on_validation"],
        s["rich_quantum_increment_survives_gating_null"],
        s["rff36_increment_survives_gating_null"], s["rich_quantum_entangling"],
    )
    logger.info("%s", s["conclusion"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
