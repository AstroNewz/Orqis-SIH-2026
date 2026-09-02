"""Quantum execution backends: ideal simulation, noisy simulation, IBM hardware.

PART 12 requires all three modes and requires that the absence of IBM
credentials never breaks anything. That shapes the whole module:

- **Ideal simulation** is the default. Exact statevector evaluation via Qiskit
  Aer. No credentials, no network, no shot noise.
- **Noisy simulation** builds an Aer ``NoiseModel`` from a real IBM device's
  published calibration data (``qiskit_ibm_runtime.fake_provider``). This is
  more honest than a hand-tuned depolarising model: the error rates are the ones
  IBM measured on the device. If ``qiskit-ibm-runtime`` is not installed it falls
  back to a documented synthetic model rather than failing.
- **IBM hardware** submits through ``QiskitRuntimeService``. Credentials come
  from configuration/environment only -- there is no hard-coded token anywhere in
  this file, and none is ever logged or returned in a result.

Fallback contract
-----------------
:meth:`QuantumBackend.resolve` never raises because of a missing credential or an
unreachable service. It returns a backend for the best available mode and records
``fell_back`` plus a human-readable reason on the result. The Flutter client
therefore cannot crash because hardware is unavailable, which PART 12 states
explicitly.

Setting ``QUANTUM_ALLOW_HARDWARE_FALLBACK=false`` makes hardware requests fail
loudly instead -- appropriate for a benchmarking run where a silent fallback
would corrupt the reported comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from quantum_ml.results import ExecutionMode

logger = logging.getLogger(__name__)

# Default IBM device whose calibration data seeds the noise model. A 127-qubit
# Eagle processor, so its error rates are representative of what a real run would
# see rather than of a small legacy device.
DEFAULT_NOISE_DEVICE = "fake_sherbrooke"

# Synthetic fallback error rates, used only when qiskit-ibm-runtime is absent.
# Chosen to sit in the range IBM publishes for superconducting devices circa 2024
# (~1e-3 single-qubit, ~1e-2 two-qubit). Documented here rather than tuned to
# make results look good.
SYNTHETIC_1Q_ERROR = 1.0e-3
SYNTHETIC_2Q_ERROR = 1.0e-2

# Above this amplitude-encoding dimension the ansatz unitary is not materialised.
# A D x D complex matrix costs 16 * D**2 bytes: D = 4096 (12 qubits) is already
# 268 MB, and the V1 register -- D = 65536 (16 qubits) -- would be 68 GB. Wider
# registers go through :meth:`QuantumBackend.expectation_batch_simulated` instead,
# which never allocates anything of order D**2.
MAX_OPERATOR_DIM = 4096

# Peak working-set target for one simulated batch. A 16-qubit statevector is 1 MiB,
# so a full 1692-sample training partition submitted in one job would ask Aer for
# 1.7 GiB of circuit payload at once. Chunking bounds that without changing results,
# because the samples are independent.
EXPECTATION_BATCH_BYTES = 256 * 1024 * 1024

_Z0_CACHE: Dict[int, np.ndarray] = {}


class QuantumBackendError(RuntimeError):
    """Raised when a requested backend cannot be provided and fallback is disabled."""


def z0_diagonal(n_qubits: int) -> np.ndarray:
    """Diagonal of the :math:`Z_0` observable in the computational basis.

    Qiskit orders basis states little-endian: for state index ``i``, qubit 0 is
    bit 0 of ``i``. So :math:`\\langle Z_0 \\rangle` is ``+1`` where that bit is
    clear and ``-1`` where it is set, and the expectation is a single weighted sum
    over the probability vector.
    """
    cached = _Z0_CACHE.get(n_qubits)
    if cached is None:
        indices = np.arange(2**n_qubits)
        cached = np.where(indices & 1, -1.0, 1.0)
        _Z0_CACHE[n_qubits] = cached
    return cached


def build_synthetic_noise_model() -> NoiseModel:
    """Depolarising noise model used when no device calibration data is available."""
    model = NoiseModel()
    model.add_all_qubit_quantum_error(
        depolarizing_error(SYNTHETIC_1Q_ERROR, 1), ["u1", "u2", "u3", "rz", "sx", "x", "ry"]
    )
    model.add_all_qubit_quantum_error(depolarizing_error(SYNTHETIC_2Q_ERROR, 2), ["cx", "cz"])
    return model


def device_noise_model(device: str = DEFAULT_NOISE_DEVICE) -> Tuple[NoiseModel, str, bool]:
    """Noise model from a real IBM device's calibration data.

    Returns:
        ``(model, description, is_device_calibrated)``. The third element is
        ``False`` when the synthetic fallback was used, so the caller can report
        honestly which one produced the numbers.
    """
    try:
        from qiskit_ibm_runtime import fake_provider
    except ImportError:
        logger.info(
            "qiskit-ibm-runtime not installed; using the synthetic depolarising noise model."
        )
        return build_synthetic_noise_model(), "synthetic_depolarizing", False

    # "fake_sherbrooke" -> "FakeSherbrooke"
    class_name = "".join(part.capitalize() for part in device.split("_"))
    backend_class = getattr(fake_provider, class_name, None)
    if backend_class is None:
        logger.warning(
            "Unknown noise device %r; using the synthetic depolarising noise model.", device
        )
        return build_synthetic_noise_model(), "synthetic_depolarizing", False

    try:
        return NoiseModel.from_backend(backend_class()), device, True
    except Exception as exc:  # noqa: BLE001 - any construction failure degrades, not fails
        logger.warning("Could not build a noise model from %s (%s); using synthetic.", device, exc)
        return build_synthetic_noise_model(), "synthetic_depolarizing", False


@dataclass
class BackendResolution:
    """What actually happened when a mode was requested."""

    mode: ExecutionMode
    backend_name: str
    shots: Optional[int]
    requested_mode: ExecutionMode
    fell_back: bool = False
    fallback_reason: Optional[str] = None
    device_calibrated_noise: bool = False


@dataclass
class QuantumBackend:
    """Executes circuits in one of the three modes.

    Instances are cheap to construct but hold a simulator, so reuse one per
    inference service rather than building one per request.
    """

    mode: ExecutionMode = ExecutionMode.IDEAL_SIMULATION
    shots: int = 1024
    seed: Optional[int] = None
    noise_device: str = DEFAULT_NOISE_DEVICE
    ibm_backend_name: str = ""
    ibm_api_key: Optional[str] = None
    ibm_instance: Optional[str] = None
    ibm_channel: str = "ibm_quantum_platform"
    allow_fallback: bool = True

    resolution: BackendResolution = field(init=False)
    _simulator: Any = field(default=None, init=False, repr=False)
    _service: Any = field(default=None, init=False, repr=False)
    _runtime_backend: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.resolution = self._resolve()

    # ------------------------------------------------------------- resolution
    def _resolve(self) -> BackendResolution:
        requested = self.mode

        if requested is ExecutionMode.IDEAL_SIMULATION:
            # max_parallel_experiments=0 lets Aer spread a batch of independent
            # circuits across cores. Each exact circuit is deterministic on its own,
            # so this changes wall-clock only, not results -- measured at 1.45x on a
            # 12-core machine for the 16-qubit batch path.
            self._simulator = AerSimulator(
                method="statevector",
                seed_simulator=self.seed,
                max_parallel_experiments=0,
            )
            return BackendResolution(
                mode=requested,
                backend_name="aer_simulator_statevector",
                shots=None,  # exact evaluation: shot noise does not apply
                requested_mode=requested,
            )

        if requested is ExecutionMode.NOISY_SIMULATION:
            model, description, calibrated = device_noise_model(self.noise_device)
            self._simulator = AerSimulator(noise_model=model, seed_simulator=self.seed)
            return BackendResolution(
                mode=requested,
                backend_name=f"aer_simulator_noisy[{description}]",
                shots=self.shots,
                requested_mode=requested,
                device_calibrated_noise=calibrated,
            )

        # --- IBM hardware ---
        reason = self._connect_hardware()
        if reason is None:
            return BackendResolution(
                mode=ExecutionMode.IBM_HARDWARE,
                backend_name=self._runtime_backend.name,
                shots=self.shots,
                requested_mode=requested,
            )

        if not self.allow_fallback:
            raise QuantumBackendError(
                f"IBM hardware was requested but is unavailable: {reason}. "
                "Set QUANTUM_ALLOW_HARDWARE_FALLBACK=true to fall back to local simulation."
            )

        # Fall back to *noisy* simulation, not ideal: a caller who asked for
        # hardware wants NISQ-realistic behaviour, so silently handing back
        # noiseless results would misrepresent what hardware would produce.
        logger.info("Falling back to noisy local simulation: %s", reason)
        model, description, calibrated = device_noise_model(self.noise_device)
        self._simulator = AerSimulator(noise_model=model, seed_simulator=self.seed)
        return BackendResolution(
            mode=ExecutionMode.NOISY_SIMULATION,
            backend_name=f"aer_simulator_noisy[{description}]",
            shots=self.shots,
            requested_mode=requested,
            fell_back=True,
            fallback_reason=reason,
            device_calibrated_noise=calibrated,
        )

    def _connect_hardware(self) -> Optional[str]:
        """Try to obtain a real IBM backend. Returns ``None`` on success.

        The returned string is shown to operators and stored in results, so it
        must never contain the token. Only the exception *type* and a short
        message are surfaced, never the credential itself.
        """
        if not self.ibm_api_key:
            return "no IBM Quantum API key configured (set IBMQ_API_KEY)"
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError:
            return "qiskit-ibm-runtime is not installed"

        try:
            kwargs: Dict[str, Any] = {"channel": self.ibm_channel, "token": self.ibm_api_key}
            if self.ibm_instance:
                kwargs["instance"] = self.ibm_instance
            service = QiskitRuntimeService(**kwargs)
            backend = (
                service.backend(self.ibm_backend_name)
                if self.ibm_backend_name
                else service.least_busy(operational=True, simulator=False)
            )
        except Exception as exc:  # noqa: BLE001 - network/auth failures vary widely
            # Deliberately reports only the exception class and a truncated
            # message: a full repr of an auth error can echo the token back.
            return f"could not reach IBM Quantum ({type(exc).__name__})"

        self._service = service
        self._runtime_backend = backend
        return None

    # ------------------------------------------------------------ properties
    @property
    def effective_mode(self) -> ExecutionMode:
        return self.resolution.mode

    @property
    def backend_name(self) -> str:
        return self.resolution.backend_name

    @property
    def effective_shots(self) -> Optional[int]:
        return self.resolution.shots

    @property
    def is_exact(self) -> bool:
        return self.resolution.shots is None

    def describe(self) -> Dict[str, Any]:
        return {
            "execution_mode": self.resolution.mode.value,
            "requested_mode": self.resolution.requested_mode.value,
            "backend_name": self.resolution.backend_name,
            "shots": self.resolution.shots,
            "fell_back": self.resolution.fell_back,
            "fallback_reason": self.resolution.fallback_reason,
            "device_calibrated_noise": self.resolution.device_calibrated_noise,
            "seed": self.seed,
        }

    # -------------------------------------------------------------- execution
    def transpiled_depth(self, circuit: QuantumCircuit) -> int:
        """Depth of the circuit as it would actually run on this backend."""
        target = self._runtime_backend or self._simulator
        try:
            return int(transpile(circuit, target, seed_transpiler=self.seed).depth())
        except Exception:  # noqa: BLE001 - depth reporting must never break inference
            return int(circuit.depth())

    def ansatz_operator(self, ansatz: QuantumCircuit) -> np.ndarray:
        """Unitary matrix of a bound ansatz, via Qiskit's own circuit simulation.

        Used only in exact mode, and only below :data:`MAX_OPERATOR_DIM`.
        Materialising :math:`U(\\theta)` once and applying it to a whole batch of
        encoded states is mathematically identical to running the statevector
        simulator on each sample's full circuit, because amplitude encoding makes the
        ansatz the same unitary for every sample. It is also orders of magnitude
        faster, which is what makes SPSA training feasible at all: a 1700-sample
        batch becomes one matrix product instead of 1700 circuit simulations.

        :func:`quantum_ml.backends.QuantumBackend.expectation_batch` is verified
        against per-circuit Aer execution in the test suite, so this shortcut is a
        measured equivalence rather than an assumption. It is not used in noisy or
        hardware mode, where per-circuit execution is the point.
        """
        return np.asarray(Operator(ansatz).data, dtype=np.complex128)

    def expectation_batch(
        self, states: np.ndarray, ansatz: QuantumCircuit
    ) -> np.ndarray:
        """Exact :math:`\\langle Z_0 \\rangle` for a batch of amplitude-encoded states.

        Dispatches on register width. Narrow registers use the materialised-unitary
        matrix product; at or above :data:`MAX_OPERATOR_DIM` that matrix does not fit,
        so the batch is simulated instead (see
        :meth:`expectation_batch_simulated`). Both paths are exact and are asserted
        to agree in the test suite.

        Args:
            states: ``(n_samples, 2**n_qubits)`` real, L2-normalised amplitudes.
            ansatz: the variational circuit with parameters already bound.

        Returns:
            ``(n_samples,)`` expectation values in [-1, 1].
        """
        if not self.is_exact:
            raise QuantumBackendError(
                "expectation_batch is exact-mode only. Use expectation_single for "
                f"{self.effective_mode.value}."
            )
        states = np.atleast_2d(np.asarray(states))
        if states.shape[0] == 0:
            return np.empty(0, dtype=np.float64)

        if states.shape[1] > MAX_OPERATOR_DIM:
            return self.expectation_batch_simulated(states, ansatz)

        unitary = self.ansatz_operator(ansatz)
        # |psi> = U |x>, batched as (n, D) @ (D, D)^T
        evolved = states.astype(np.complex128) @ unitary.T
        probabilities = np.abs(evolved) ** 2
        return probabilities @ z0_diagonal(ansatz.num_qubits)

    def expectation_batch_simulated(
        self, states: np.ndarray, ansatz: QuantumCircuit
    ) -> np.ndarray:
        """Exact :math:`\\langle Z_0 \\rangle` per sample, without materialising ``U``.

        Each sample becomes one Aer circuit whose initial state is set directly with
        ``set_statevector`` -- not synthesised by a state-preparation circuit, which
        for a 16-qubit register would need on the order of :math:`2^{16}` CX gates and
        would dominate the runtime while computing exactly the same thing. Aer then
        evolves and contracts :math:`Z_0` in C++, and the batch is submitted as one
        job so the per-circuit setup is paid once and the circuits spread across cores.

        Why this rather than a Python loop over ``Statevector.evolve``: measured on a
        16-qubit, 2-layer ansatz (32 ry, 32 rz, 32 cx, depth 36), the loop costs
        330 ms per sample and this costs 11 ms -- 30x, which is the difference between
        a 31-hour SPSA run and a one-hour one.

        Args:
            states: ``(n_samples, 2**n_qubits)`` L2-normalised amplitudes.
            ansatz: the variational circuit with parameters already bound.

        Returns:
            ``(n_samples,)`` expectation values in [-1, 1].
        """
        from qiskit.quantum_info import SparsePauliOp

        if not self.is_exact:
            raise QuantumBackendError(
                "expectation_batch_simulated is exact-mode only."
            )
        states = np.atleast_2d(np.asarray(states))
        if states.shape[0] == 0:
            return np.empty(0, dtype=np.float64)

        n_qubits = ansatz.num_qubits
        expected_dim = 1 << n_qubits
        if states.shape[1] != expected_dim:
            raise QuantumBackendError(
                f"Ansatz spans {n_qubits} qubits ({expected_dim} amplitudes) but the "
                f"batch rows carry {states.shape[1]}. Refusing to pad or truncate."
            )

        # Pauli strings are written most-significant-qubit first, so the readout
        # qubit 0 is the *last* character: "I...IZ" is Z_0.
        observable = SparsePauliOp("I" * (n_qubits - 1) + "Z") if n_qubits > 1 else SparsePauliOp("Z")
        qubit_indices = list(range(n_qubits))

        bytes_per_row = expected_dim * 16  # complex128 statevector payload
        chunk = max(1, EXPECTATION_BATCH_BYTES // max(1, bytes_per_row))

        out = np.empty(states.shape[0], dtype=np.float64)
        for start in range(0, states.shape[0], chunk):
            stop = min(start + chunk, states.shape[0])
            circuits = []
            for row in states[start:stop]:
                circuit = QuantumCircuit(n_qubits)
                circuit.set_statevector(Statevector(np.asarray(row, dtype=np.complex128)))
                circuit.compose(ansatz, inplace=True)
                circuit.save_expectation_value(observable, qubit_indices)
                circuits.append(circuit)
            result = self._simulator.run(circuits).result()
            for offset in range(stop - start):
                out[start + offset] = float(
                    np.real(result.data(offset)["expectation_value"])
                )
        return out

    def expectation_single(
        self, circuit: QuantumCircuit, *, n_qubits: int
    ) -> Tuple[float, Optional[Dict[str, int]]]:
        """Execute one full circuit and return :math:`\\langle Z_0 \\rangle`.

        Returns the shot histogram alongside, or ``None`` in exact mode.
        """
        if self.effective_mode is ExecutionMode.IBM_HARDWARE:
            return self._expectation_hardware(circuit, n_qubits=n_qubits)
        if self.is_exact:
            state = Statevector.from_instruction(circuit)
            probabilities = np.abs(np.asarray(state.data)) ** 2
            return float(probabilities @ z0_diagonal(n_qubits)), None
        return self._expectation_shots(circuit, n_qubits=n_qubits)

    def _expectation_shots(
        self, circuit: QuantumCircuit, *, n_qubits: int
    ) -> Tuple[float, Dict[str, int]]:
        measured = circuit.copy()
        if not measured.clbits:
            measured.measure_all()
        transpiled = transpile(measured, self._simulator, seed_transpiler=self.seed)
        counts = self._simulator.run(transpiled, shots=self.shots).result().get_counts()
        return expectation_from_counts(counts), {str(k): int(v) for k, v in counts.items()}

    def _expectation_hardware(
        self, circuit: QuantumCircuit, *, n_qubits: int
    ) -> Tuple[float, Dict[str, int]]:
        """Submit one circuit to real IBM hardware via the Sampler primitive."""
        from qiskit_ibm_runtime import SamplerV2

        measured = circuit.copy()
        if not measured.clbits:
            measured.measure_all()
        transpiled = transpile(measured, self._runtime_backend, seed_transpiler=self.seed)
        sampler = SamplerV2(mode=self._runtime_backend)
        result = sampler.run([transpiled], shots=self.shots).result()
        data = result[0].data
        register = next(iter(data.__dict__)) if hasattr(data, "__dict__") else "meas"
        counts = getattr(data, register).get_counts()
        return expectation_from_counts(counts), {str(k): int(v) for k, v in counts.items()}

    def run_batch_shots(
        self, circuits: Sequence[QuantumCircuit], *, n_qubits: int
    ) -> List[float]:
        """Execute several circuits in one job. Shot-based simulation only.

        One Aer job for many circuits avoids per-circuit setup overhead, which
        dominates when each circuit is small.
        """
        if self.is_exact:
            raise QuantumBackendError("run_batch_shots requires a shot-based mode.")
        if self.effective_mode is ExecutionMode.IBM_HARDWARE:
            return [
                self._expectation_hardware(circuit, n_qubits=n_qubits)[0]
                for circuit in circuits
            ]
        prepared = []
        for circuit in circuits:
            measured = circuit.copy()
            if not measured.clbits:
                measured.measure_all()
            prepared.append(measured)
        transpiled = transpile(prepared, self._simulator, seed_transpiler=self.seed)
        result = self._simulator.run(transpiled, shots=self.shots).result()
        return [
            expectation_from_counts(result.get_counts(i)) for i in range(len(prepared))
        ]


def expectation_from_counts(counts: Dict[str, int]) -> float:
    """:math:`\\langle Z_0 \\rangle = P(0) - P(1)` from a shot histogram.

    Qiskit renders bitstrings most-significant-qubit first, so qubit 0 -- the
    readout qubit -- is the **last** character. Registers are separated by spaces
    when a circuit has several, so those are stripped first.
    """
    zeros = 0
    ones = 0
    for bitstring, count in counts.items():
        cleaned = bitstring.replace(" ", "")
        if not cleaned:
            continue
        if cleaned[-1] == "0":
            zeros += count
        else:
            ones += count
    total = zeros + ones
    if total == 0:
        return 0.0
    return (zeros - ones) / total


def build_backend(
    *,
    mode: ExecutionMode = ExecutionMode.IDEAL_SIMULATION,
    shots: int = 1024,
    seed: Optional[int] = None,
    noise_device: str = DEFAULT_NOISE_DEVICE,
    ibm_backend_name: str = "",
    ibm_api_key: Optional[str] = None,
    ibm_instance: Optional[str] = None,
    ibm_channel: str = "ibm_quantum_platform",
    allow_fallback: bool = True,
) -> QuantumBackend:
    """Construct a backend for a mode, resolving availability immediately."""
    return QuantumBackend(
        mode=mode,
        shots=shots,
        seed=seed,
        noise_device=noise_device,
        ibm_backend_name=ibm_backend_name,
        ibm_api_key=ibm_api_key,
        ibm_instance=ibm_instance,
        ibm_channel=ibm_channel,
        allow_fallback=allow_fallback,
    )


def backend_from_settings(config: Optional[Any] = None) -> QuantumBackend:
    """Build the backend the application settings describe.

    Credentials are read from configuration only. This function is the single
    place the API key reaches the quantum layer.
    """
    from backend.core.config import settings as default_settings

    cfg = config or default_settings
    return build_backend(
        mode=ExecutionMode(cfg.QUANTUM_EXECUTION_MODE),
        shots=cfg.QUANTUM_SHOTS,
        seed=cfg.RANDOM_SEED,
        noise_device=cfg.QUANTUM_NOISE_MODEL,
        ibm_backend_name=cfg.IBMQ_BACKEND_NAME,
        ibm_api_key=cfg.IBMQ_API_KEY,
        ibm_instance=cfg.IBMQ_INSTANCE,
        ibm_channel=cfg.IBMQ_CHANNEL,
        allow_fallback=cfg.QUANTUM_ALLOW_HARDWARE_FALLBACK,
    )
