"""Hardware-aware variational quantum classifier.

PART 10 asks for explicit modules rather than one opaque blob, so the stages are
separated and individually testable:

===========================  ==================================================
Stage                        Where
===========================  ==================================================
Feature encoding             :class:`~quantum_ml.quantum_encoder.QuantumEncoder`
Ansatz                       :meth:`VariationalQuantumClassifier.build_ansatz`
Parameterised circuit        :meth:`VariationalQuantumClassifier.build_full_circuit`
Measurement + expectation    :meth:`VariationalQuantumClassifier.compute_expectation_value`
Classical optimisation       :meth:`VariationalQuantumClassifier.fit`
Inference                    :meth:`VariationalQuantumClassifier.infer`
===========================  ==================================================

The circuit
-----------
Per layer, every qubit gets :math:`R_z(\\psi) R_y(\\phi)` -- two angles, which
together span an arbitrary rotation up to a phase that :math:`Z_0` cannot see --
followed by a linear CNOT cascade with a circular closure. The cascade is the
"hardware-aware" part: superconducting devices have nearest-neighbour coupling, so
a linear chain transpiles without SWAP overhead, whereas all-to-all entanglement
would be inserted by the transpiler at a cost the circuit does not advertise.

Depth is configurable via ``num_layers`` (PART 10) and the parameter count is
``2 * num_qubits * num_layers``.

Readout is :math:`\\langle Z_0 \\rangle` on qubit 0, mapped by
:math:`P = (1 - \\langle Z_0 \\rangle) / 2`. That mapping is monotone and bounded
but it is **not** a calibrated probability -- see :mod:`quantum_ml.calibration`.

On quantum superiority
----------------------
Nothing here assumes the quantum model wins. It exposes the same
``fit``/``predict_proba`` surface as the classical baselines in
:mod:`backend.ml.baselines` precisely so the comparison can be run and reported
either way.

Which mode trains
-----------------
Training runs in **exact mode**. SPSA needs a few hundred objective evaluations
and each evaluation touches every training sample; at 1024 shots per sample that
is hundreds of millions of circuit executions, which is not a thing that finishes.
Exact statevector evaluation makes it a matrix product instead. Noisy and hardware
modes are for *evaluation and inference*, which is the sequence PART 12
prescribes: train ideal, then measure what noise and real hardware do to the
result.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

from quantum_ml.backends import (
    MAX_OPERATOR_DIM as _MAX_OPERATOR_DIM,
    QuantumBackend,
    build_backend,
    z0_diagonal,
)
from quantum_ml.optimizers import (
    DEFAULT_MAXITER,
    OptimizationResult,
    OptimizerSpec,
    build_optimizer,
)
from quantum_ml.quantum_encoder import QuantumEncoder
from quantum_ml.results import (
    ExecutionMode,
    OptimizerName,
    QuantumInferenceResult,
    QuantumTrainingRecord,
)

logger = logging.getLogger(__name__)

# Clamp on probabilities inside the log-loss. 1e-7 keeps the loss finite for a
# confidently wrong prediction without letting a single sample dominate the mean.
EPSILON = 1.0e-7

# Re-exported: the dimension above which the ansatz unitary is not materialised.
# The threshold and the reasoning behind it live with the code that allocates the
# matrix, in :mod:`quantum_ml.backends`.
MAX_OPERATOR_DIM = _MAX_OPERATOR_DIM

VQC_VERSION = "vqc-1"


class VQCError(RuntimeError):
    """Raised when the classifier is misconfigured or used before training."""


class VariationalQuantumClassifier:
    """Hardware-aware VQC for oral-cancer screening risk estimation.

    Args:
        num_qubits: register width. The amplitude-encoding dimension is ``2**n``.
        num_layers: variational depth (rotation + entanglement blocks).
        shots: shots per circuit in shot-based modes. Ignored in exact mode.
        backend_name: legacy label retained for the existing public surface and
            reported as ``metadata["backend"]``. The mode actually resolved is
            reported separately as ``metadata["resolved_backend"]``.
        backend: execution backend. Defaults to ideal local simulation, so no
            credentials and no network are needed.
        seed: seeds weight initialisation and the optimiser.
        class_weighting: weight the loss by inverse class frequency. On by
            default because the dataset is heavily negative-skewed and an
            unweighted loss is minimised well by predicting "no risk" for
            everyone.
        model_version: recorded on every inference result for traceability.
    """

    def __init__(
        self,
        num_qubits: int = 8,
        num_layers: int = 2,
        shots: int = 1024,
        backend_name: str = "aer_simulator",
        *,
        backend: Optional[QuantumBackend] = None,
        seed: int = 42,
        class_weighting: bool = True,
        model_version: str = "",
        encoder: Optional[QuantumEncoder] = None,
    ) -> None:
        if num_qubits < 1:
            raise VQCError(f"num_qubits must be >= 1, got {num_qubits}.")
        if num_layers < 1:
            raise VQCError(f"num_layers must be >= 1, got {num_layers}.")

        self.num_qubits = int(num_qubits)
        self.num_layers = int(num_layers)
        self.shots = int(shots)
        self.backend_name = backend_name
        self.seed = int(seed)
        self.class_weighting = class_weighting
        self.model_version = model_version
        self.version = VQC_VERSION

        self.encoder = encoder or QuantumEncoder(num_qubits=self.num_qubits)
        self.backend = backend or build_backend(
            mode=ExecutionMode.IDEAL_SIMULATION, shots=self.shots, seed=self.seed
        )

        self.num_params = 2 * self.num_qubits * self.num_layers

        # A local Generator rather than np.random.seed: the optimiser seeds global
        # NumPy state, and weight initialisation must not depend on whether an
        # optimiser has run first.
        self.weights = np.random.default_rng(self.seed).uniform(
            -np.pi, np.pi, self.num_params
        )

        self.trained = False
        self.training_record: Optional[QuantumTrainingRecord] = None

        # The parameterised ansatz is structural, so it is built once and reused.
        # Rebuilding it inside the objective would rebuild it a few hundred times
        # per training run for no benefit.
        self._ansatz_template, self._ansatz_parameters = self._build_ansatz_template()

    # ------------------------------------------------------------------ ansatz
    def _build_ansatz_template(self) -> Tuple[QuantumCircuit, ParameterVector]:
        circuit = QuantumCircuit(self.num_qubits, name="HardwareAwareVQC")
        params = ParameterVector("theta", self.num_params)

        index = 0
        for _ in range(self.num_layers):
            # Rotation layer: Rz(psi) Ry(phi) per qubit.
            for qubit in range(self.num_qubits):
                circuit.ry(params[index], qubit)
                circuit.rz(params[index + 1], qubit)
                index += 2
            # Entanglement layer: linear nearest-neighbour cascade.
            for qubit in range(self.num_qubits - 1):
                circuit.cx(qubit, qubit + 1)
            # Circular closure. Skipped at 2 qubits, where cx(1, 0) would merely
            # undo part of the cascade rather than adding connectivity.
            if self.num_qubits > 2:
                circuit.cx(self.num_qubits - 1, 0)
        return circuit, params

    def build_ansatz(self) -> Tuple[QuantumCircuit, ParameterVector]:
        """The parameterised ansatz and its parameter vector."""
        return self._ansatz_template, self._ansatz_parameters

    def bind_ansatz(self, weights: Optional[np.ndarray] = None) -> QuantumCircuit:
        """Ansatz with concrete angles substituted for its parameters."""
        theta = self._check_weights(weights)
        return self._ansatz_template.assign_parameters(
            dict(zip(self._ansatz_parameters, theta))
        )

    def _check_weights(self, weights: Optional[np.ndarray]) -> np.ndarray:
        theta = self.weights if weights is None else np.asarray(weights, dtype=np.float64)
        if theta.shape != (self.num_params,):
            raise VQCError(
                f"Expected {self.num_params} parameters for {self.num_qubits} qubits "
                f"x {self.num_layers} layers, got shape {theta.shape}."
            )
        return theta

    # -------------------------------------------------------- circuit assembly
    def build_full_circuit(
        self,
        features: Union[np.ndarray, List[float]],
        weights: Optional[np.ndarray] = None,
        measure: bool = True,
    ) -> QuantumCircuit:
        """State preparation -> ansatz -> (optional) measurement."""
        circuit = self.encoder.prepare_circuit(features)
        circuit.compose(self.bind_ansatz(weights), inplace=True)
        if measure:
            circuit.measure_all()
        return circuit

    def circuit_from_state(
        self,
        state: np.ndarray,
        weights: Optional[np.ndarray] = None,
        measure: bool = True,
    ) -> QuantumCircuit:
        """Same as :meth:`build_full_circuit` for an already-encoded state."""
        circuit = self.encoder.circuit_from_state(state)
        circuit.compose(self.bind_ansatz(weights), inplace=True)
        if measure:
            circuit.measure_all()
        return circuit

    @property
    def ansatz_depth(self) -> int:
        """Depth of the ansatz alone, before state-preparation synthesis."""
        return int(self._ansatz_template.depth())

    def transpiled_depth(self, weights: Optional[np.ndarray] = None) -> int:
        """Depth of a full circuit as the active backend would execute it.

        Reported rather than the ansatz depth because state preparation dominates
        an amplitude-encoded circuit, and quoting only the ansatz would understate
        the real cost by an order of magnitude.
        """
        uniform = np.full(self.encoder.dim, 1.0 / np.sqrt(self.encoder.dim))
        return self.backend.transpiled_depth(
            self.circuit_from_state(uniform, weights, measure=False)
        )

    # ------------------------------------------------------------ measurement
    def compute_expectation_value(
        self,
        features: Union[np.ndarray, List[float]],
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Execute one circuit and return :math:`\\langle Z_0 \\rangle` with metadata.

        ``metadata["backend"]`` is the legacy label from the constructor;
        ``metadata["resolved_backend"]`` names the backend that actually ran.
        """
        circuit = self.build_full_circuit(features, weights=weights, measure=False)
        expectation, counts = self.backend.expectation_single(
            circuit, n_qubits=self.num_qubits
        )
        probability_zero = (1.0 + expectation) / 2.0
        metadata: Dict[str, Any] = {
            "shots": self.backend.effective_shots,
            "counts": counts,
            "prob_0": probability_zero,
            "prob_1": 1.0 - probability_zero,
            "backend": self.backend_name,
            "resolved_backend": self.backend.backend_name,
            "execution_mode": self.backend.effective_mode.value,
            "num_qubits": self.num_qubits,
            "num_layers": self.num_layers,
        }
        return float(expectation), metadata

    def predict_probability(
        self,
        features: Union[np.ndarray, List[float]],
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Uncalibrated risk score :math:`(1 - \\langle Z_0 \\rangle) / 2`.

        Not a calibrated probability. :mod:`quantum_ml.calibration` turns this into
        one; using it directly as a clinical risk would be wrong.
        """
        expectation, metadata = self.compute_expectation_value(features, weights=weights)
        probability = float(np.clip((1.0 - expectation) / 2.0, 0.0, 1.0))
        metadata["expectation_value"] = expectation
        metadata["quantum_probability"] = probability
        return probability, metadata

    # ------------------------------------------------------- batch expectation
    def encode_batch(self, features: np.ndarray) -> np.ndarray:
        """Amplitude-encode a design matrix once, for reuse across iterations."""
        return self.encoder.transform_batch(np.atleast_2d(np.asarray(features, dtype=np.float64)))

    def expectation_states(
        self, states: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """:math:`\\langle Z_0 \\rangle` for a batch of encoded states.

        In exact mode this delegates to
        :meth:`~quantum_ml.backends.QuantumBackend.expectation_batch`, which picks
        between the materialised-unitary matrix product (narrow registers) and batched
        Aer simulation (the 16-qubit V1 register, where the unitary would be 68 GB).
        Both are exact. Shot-based modes always execute per circuit -- that is the
        point of them.
        """
        bound = self.bind_ansatz(weights)
        states = np.atleast_2d(np.asarray(states))
        if states.shape[0] == 0:
            return np.empty(0, dtype=np.float64)

        if self.backend.is_exact:
            return self.backend.expectation_batch(states, bound)

        circuits = [self.circuit_from_state(state, weights, measure=True) for state in states]
        return np.asarray(
            self.backend.run_batch_shots(circuits, n_qubits=self.num_qubits),
            dtype=np.float64,
        )

    def probabilities_from_states(
        self, states: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Uncalibrated scores in [0, 1] for a batch of encoded states."""
        return np.clip((1.0 - self.expectation_states(states, weights)) / 2.0, 0.0, 1.0)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Uncalibrated positive-class scores for a design matrix.

        Mirrors the classical baselines' signature so the two can be compared by
        the same evaluation code.
        """
        return self.probabilities_from_states(self.encode_batch(features))

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Hard labels at a threshold on the *uncalibrated* score.

        Provided for parity with the baselines. Clinical decisions use the
        calibrated probability and the configured screening threshold, not this.
        """
        return (self.predict_proba(features) >= threshold).astype(int)

    # ------------------------------------------------------------------- loss
    def _class_weights(self, y: np.ndarray) -> Tuple[float, float]:
        """Inverse-frequency weights, normalised to mean 1.

        Normalising keeps the loss on a comparable scale to the unweighted one, so
        a weighted and an unweighted run's objectives can be read side by side.
        """
        if not self.class_weighting:
            return 1.0, 1.0
        n_positive = int(np.sum(y == 1))
        n_negative = int(y.size - n_positive)
        if n_positive == 0 or n_negative == 0:
            return 1.0, 1.0
        weight_positive = y.size / (2.0 * n_positive)
        weight_negative = y.size / (2.0 * n_negative)
        return weight_negative, weight_positive

    def loss_from_states(
        self,
        states: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        *,
        class_weights: Optional[Tuple[float, float]] = None,
    ) -> float:
        """Class-weighted binary cross-entropy over encoded states."""
        probabilities = np.clip(
            self.probabilities_from_states(states, weights), EPSILON, 1.0 - EPSILON
        )
        y = np.asarray(y, dtype=np.float64)
        weight_negative, weight_positive = class_weights or self._class_weights(y)
        sample_weights = np.where(y == 1, weight_positive, weight_negative)
        losses = -(y * np.log(probabilities) + (1.0 - y) * np.log(1.0 - probabilities))
        return float(np.average(losses, weights=sample_weights))

    def _batch_loss(
        self, x_batch: np.ndarray, y_batch: np.ndarray, weights: np.ndarray
    ) -> float:
        """Binary cross-entropy over a raw feature batch (encodes first)."""
        return self.loss_from_states(self.encode_batch(x_batch), y_batch, weights)

    # -------------------------------------------------------------- training
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        optimizer: Optional[Union[str, OptimizerSpec]] = None,
        maxiter: int = DEFAULT_MAXITER,
        learning_rate: Optional[float] = None,
        perturbation: Optional[float] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        progress_every: int = 0,
        notes: str = "",
    ) -> QuantumTrainingRecord:
        """Train the variational parameters and return a reproducibility record.

        The whole training set is encoded **once** up front; the optimiser then
        only ever re-binds angles. Re-encoding inside the objective would repeat
        the state-preparation work on every one of a few hundred iterations.
        """
        x_train = np.atleast_2d(np.asarray(x_train, dtype=np.float64))
        y_train = np.asarray(y_train).astype(int).ravel()
        if x_train.shape[0] != y_train.size:
            raise VQCError(
                f"Feature/label mismatch: {x_train.shape[0]} samples vs {y_train.size} labels."
            )
        if y_train.size == 0:
            raise VQCError("Cannot train on an empty training set.")
        unexpected = set(np.unique(y_train)) - {0, 1}
        if unexpected:
            raise VQCError(f"Labels must be binary 0/1; found {sorted(unexpected)}.")

        spec = self._resolve_optimizer(optimizer, maxiter, learning_rate, perturbation)
        feature_dimension = int(x_train.shape[1])
        states = self.encode_batch(x_train)
        class_weights = self._class_weights(y_train)

        def objective(theta: np.ndarray) -> float:
            return self.loss_from_states(states, y_train, theta, class_weights=class_weights)

        callback = self._progress_callback(progress_every) if progress_every > 0 else None

        start = time.perf_counter()
        result: OptimizationResult = spec.minimize(
            objective, self.weights, on_evaluation=callback
        )
        duration = time.perf_counter() - start

        self.weights = result.parameters
        self.trained = True

        validation_note = self._validation_note(validation_data, class_weights)
        combined_notes = "; ".join(part for part in (notes, validation_note) if part)

        self.training_record = QuantumTrainingRecord(
            optimizer=spec.name,
            n_iterations=result.n_iterations,
            n_circuit_evaluations=result.n_function_evaluations * int(y_train.size),
            learning_rate=spec.learning_rate,
            perturbation=spec.perturbation,
            n_qubits=self.num_qubits,
            n_ansatz_layers=self.num_layers,
            n_parameters=self.num_params,
            random_seed=spec.seed,
            execution_mode=self.backend.effective_mode,
            backend_name=self.backend.backend_name,
            shots=self.backend.effective_shots,
            training_duration_seconds=duration,
            final_objective=result.objective,
            initial_objective=result.initial_objective,
            best_objective=result.best_objective,
            objective_history=[float(value) for value in result.history],
            n_train_samples=int(y_train.size),
            n_train_positive=int(np.sum(y_train == 1)),
            class_weighting=self.class_weighting,
            feature_dimension=feature_dimension,
            converged=result.converged,
            notes=combined_notes,
        )
        return self.training_record

    def _resolve_optimizer(
        self,
        optimizer: Optional[Union[str, OptimizerSpec]],
        maxiter: int,
        learning_rate: Optional[float],
        perturbation: Optional[float],
    ) -> OptimizerSpec:
        if isinstance(optimizer, OptimizerSpec):
            return optimizer
        return build_optimizer(
            optimizer or "spsa",
            maxiter=maxiter,
            learning_rate=learning_rate,
            perturbation=perturbation,
            seed=self.seed,
        )

    def _progress_callback(self, every: int) -> Callable[[int, float], None]:
        def report(index: int, value: float) -> None:
            if index % every == 0:
                logger.info("VQC objective eval %d: %.6f", index, value)

        return report

    def _validation_note(
        self,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]],
        class_weights: Tuple[float, float],
    ) -> str:
        """Validation loss at the fitted parameters, as a recorded note.

        Validation is *reported*, not used for early stopping: SPSA's trajectory is
        stochastic, so stopping on a noisy validation reading would select a
        parameter set for the noise rather than for its generalisation.
        """
        if validation_data is None:
            return ""
        x_validation, y_validation = validation_data
        y_validation = np.asarray(y_validation).astype(int).ravel()
        if y_validation.size == 0:
            return ""
        loss = self.loss_from_states(
            self.encode_batch(x_validation), y_validation, class_weights=class_weights
        )
        return f"validation_loss={loss:.6f} on n={y_validation.size}"

    def train_step_spsa(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        lr: float = 0.05,
        c: float = 0.1,
    ) -> float:
        """One manual SPSA step, retained from the original public surface.

        :meth:`fit` is the real training entry point -- it uses the calibrated
        gain schedule from ``qiskit-algorithms`` rather than the fixed ``lr``/``c``
        here, and it produces a :class:`QuantumTrainingRecord`. This method is a
        single gradient estimate, useful for tests and for stepping through the
        update by hand.
        """
        states = self.encode_batch(x_batch)
        y_batch = np.asarray(y_batch).astype(int).ravel()
        class_weights = self._class_weights(y_batch)

        delta = 2 * np.random.randint(0, 2, size=self.num_params) - 1
        loss_plus = self.loss_from_states(
            states, y_batch, self.weights + c * delta, class_weights=class_weights
        )
        loss_minus = self.loss_from_states(
            states, y_batch, self.weights - c * delta, class_weights=class_weights
        )
        gradient = (loss_plus - loss_minus) / (2.0 * c * delta)
        self.weights = self.weights - lr * gradient
        return self.loss_from_states(
            states, y_batch, self.weights, class_weights=class_weights
        )

    # ------------------------------------------------------------- inference
    def infer(
        self,
        features: Union[np.ndarray, Sequence[float]],
        *,
        inference_id: Optional[str] = None,
        circuit_depth: Optional[int] = None,
    ) -> QuantumInferenceResult:
        """One typed inference (PART 13).

        Args:
            features: the reduced feature vector for a single case.
            inference_id: UUID v4; generated when omitted.
            circuit_depth: precomputed transpiled depth. Transpiling costs real
                time and the depth is identical for every case at fixed qubit
                count and layer count, so a caller running many inferences should
                compute it once and pass it in.
        """
        start = time.perf_counter()
        expectation, metadata = self.compute_expectation_value(features)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        depth = circuit_depth if circuit_depth is not None else self.transpiled_depth()
        resolution = self.backend.resolution
        optimizer = (
            self.training_record.optimizer if self.training_record is not None else None
        )

        return QuantumInferenceResult(
            inference_id=inference_id or str(uuid.uuid4()),
            expectation_value=float(np.clip(expectation, -1.0, 1.0)),
            raw_score=float(np.clip(-expectation, -1.0, 1.0)),
            probability_uncalibrated=float(np.clip((1.0 - expectation) / 2.0, 0.0, 1.0)),
            n_qubits=self.num_qubits,
            circuit_depth=depth,
            n_ansatz_layers=self.num_layers,
            n_parameters=self.num_params,
            execution_mode=resolution.mode,
            backend_name=resolution.backend_name,
            shots=resolution.shots,
            execution_time_ms=elapsed_ms,
            model_version=self.model_version,
            optimizer=optimizer,
            requested_mode=resolution.requested_mode,
            fell_back=resolution.fell_back,
            fallback_reason=resolution.fallback_reason,
            measurement_counts=metadata.get("counts"),
        )

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        """Serialisable model state. JSON-safe: no pickle, no executable payload."""
        return {
            "version": self.version,
            "num_qubits": self.num_qubits,
            "num_layers": self.num_layers,
            "num_params": self.num_params,
            "weights": [float(value) for value in self.weights],
            "shots": self.shots,
            "seed": self.seed,
            "class_weighting": self.class_weighting,
            "model_version": self.model_version,
            "trained": self.trained,
            "encoder": self.encoder.describe(),
            "training_record": (
                self.training_record.model_dump(mode="json")
                if self.training_record is not None
                else None
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        *,
        backend: Optional[QuantumBackend] = None,
    ) -> "VariationalQuantumClassifier":
        """Rebuild a trained classifier.

        ``backend`` is supplied by the caller rather than restored from the
        payload: the execution mode is a deployment decision, so a model trained
        on the ideal simulator can be served noisily or on hardware without being
        retrained. That separation is what PART 29 requires.
        """
        model = cls(
            num_qubits=int(payload["num_qubits"]),
            num_layers=int(payload["num_layers"]),
            shots=int(payload.get("shots", 1024)),
            backend=backend,
            seed=int(payload.get("seed", 42)),
            class_weighting=bool(payload.get("class_weighting", True)),
            model_version=str(payload.get("model_version", "")),
        )
        weights = np.asarray(payload["weights"], dtype=np.float64)
        if weights.shape != (model.num_params,):
            raise VQCError(
                f"Stored weights have shape {weights.shape} but this configuration "
                f"needs {model.num_params} parameters. The artifact does not match "
                "the requested qubit/layer counts."
            )
        model.weights = weights
        model.trained = bool(payload.get("trained", True))
        record = payload.get("training_record")
        if record:
            model.training_record = QuantumTrainingRecord(**record)
        return model

    def describe(self) -> Dict[str, Any]:
        """Configuration summary for ``/model/info`` and training logs."""
        return {
            "version": self.version,
            "num_qubits": self.num_qubits,
            "num_layers": self.num_layers,
            "num_parameters": self.num_params,
            "state_dimension": self.encoder.dim,
            "ansatz_depth": self.ansatz_depth,
            "trained": self.trained,
            "class_weighting": self.class_weighting,
            "model_version": self.model_version,
            "seed": self.seed,
            "backend": self.backend.describe(),
        }
