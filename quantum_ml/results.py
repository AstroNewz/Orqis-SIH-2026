"""Strongly typed quantum inference results.

PART 13 requires the quantum layer to return a structured object, not a bare
float and not a ``Dict[str, Any]``. Everything an audit needs -- which backend
ran the circuit, how deep it was, how many shots, how long it took, which model
version produced it -- travels with the score.

These are Pydantic models so the FastAPI layer can serialise them directly and
so malformed values are caught at the boundary rather than surfacing as a
nonsense probability three stages later.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ExecutionMode(str, Enum):
    """How the circuit was executed.

    The evaluation sequence PART 12 prescribes is ideal -> noisy -> hardware.
    ``IDEAL_SIMULATION`` is the default so local development needs no
    credentials and no network.
    """

    IDEAL_SIMULATION = "ideal_simulation"
    NOISY_SIMULATION = "noisy_simulation"
    IBM_HARDWARE = "ibm_hardware"


class OptimizerName(str, Enum):
    SPSA = "spsa"
    QNSPSA = "qnspsa"
    COBYLA = "cobyla"


class QuantumInferenceResult(BaseModel):
    """One circuit execution, fully described.

    The three score fields are deliberately distinct and must not be conflated:

    ``expectation_value``
        The physical measurement, :math:`\\langle Z_0 \\rangle \\in [-1, 1]`.
    ``raw_score``
        The model's decision score before any probabilistic interpretation.
        Equal to :math:`-\\langle Z_0 \\rangle` so that larger means higher risk.
    ``probability_uncalibrated``
        :math:`(1 - \\langle Z_0 \\rangle) / 2`, mapped to [0, 1]. This is a
        *number in the unit interval*, which is not the same thing as a
        calibrated probability -- see :mod:`quantum_ml.calibration`. It is named
        ``uncalibrated`` precisely so no caller mistakes it for a clinical risk.
    """

    model_config = ConfigDict(frozen=True)

    inference_id: str
    """UUID v4 for this execution. Pseudonymous: carries no patient identity."""

    expectation_value: float = Field(ge=-1.0, le=1.0)
    raw_score: float = Field(ge=-1.0, le=1.0)
    probability_uncalibrated: float = Field(ge=0.0, le=1.0)

    n_qubits: int = Field(ge=1)
    circuit_depth: int = Field(ge=0)
    """Depth of the transpiled circuit as executed, not the ansatz layer count."""

    n_ansatz_layers: int = Field(ge=1)
    n_parameters: int = Field(ge=0)

    execution_mode: ExecutionMode
    backend_name: str
    shots: Optional[int] = None
    """``None`` for exact statevector evaluation, where shot noise does not exist."""

    execution_time_ms: float = Field(ge=0.0)
    model_version: str = ""
    optimizer: Optional[OptimizerName] = None

    requested_mode: Optional[ExecutionMode] = None
    """What the caller asked for. Differs from ``execution_mode`` after a fallback."""

    fell_back: bool = False
    fallback_reason: Optional[str] = None
    """Why hardware or a noise model was unavailable. Never contains credentials."""

    measurement_counts: Optional[Dict[str, int]] = None
    """Raw shot histogram when shot-based. Omitted for exact evaluation."""

    @property
    def is_exact(self) -> bool:
        return self.shots is None

    @property
    def used_hardware(self) -> bool:
        return self.execution_mode is ExecutionMode.IBM_HARDWARE


class QuantumTrainingRecord(BaseModel):
    """Reproducibility record for one VQC training run (PART 11).

    Every field here is something needed to reproduce the run or to explain the
    result later. A model artifact without this is not reproducible, and an
    irreproducible clinical model is not usable.
    """

    model_config = ConfigDict(frozen=True)

    optimizer: OptimizerName
    n_iterations: int = Field(ge=0)
    n_circuit_evaluations: int = Field(ge=0)
    learning_rate: Optional[float] = None
    perturbation: Optional[float] = None
    n_qubits: int = Field(ge=1)
    n_ansatz_layers: int = Field(ge=1)
    n_parameters: int = Field(ge=0)
    random_seed: int
    execution_mode: ExecutionMode
    backend_name: str
    shots: Optional[int] = None

    training_duration_seconds: float = Field(ge=0.0)
    final_objective: float
    """Final training loss (mean binary cross-entropy)."""

    initial_objective: Optional[float] = None
    best_objective: Optional[float] = None
    objective_history: List[float] = Field(default_factory=list)

    n_train_samples: int = Field(ge=0)
    n_train_positive: int = Field(ge=0)
    class_weighting: bool = False
    """Whether the loss was class-weighted. Relevant: the dataset is ~6% positive."""

    feature_dimension: int = Field(ge=0)
    """Length of the reduced vector before zero-padding to 2**n_qubits."""

    converged: bool = False
    notes: str = ""

    @property
    def positive_rate(self) -> float:
        return self.n_train_positive / self.n_train_samples if self.n_train_samples else 0.0
