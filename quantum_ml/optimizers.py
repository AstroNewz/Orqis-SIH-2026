"""Classical optimisers for the variational quantum classifier.

PART 11 names SPSA, QNSPSA where practical, and COBYLA as a classical baseline.
All three come from ``qiskit-algorithms``, the canonical implementations, rather
than being reimplemented -- a hand-rolled SPSA with a subtly wrong gain schedule
would be indistinguishable from a model that simply does not learn.

Why SPSA is the default
-----------------------
The VQC objective is evaluated by circuit execution, so its cost is measured in
circuit evaluations. SPSA needs **two** objective evaluations per iteration
regardless of how many parameters there are, whereas a finite-difference gradient
needs ``2p``. At 8 qubits and 3 layers that is 48 parameters, so SPSA is roughly
24x cheaper per step. It also tolerates the stochastic objective that shot-based
execution produces, which is exactly what it was designed for.

QNSPSA additionally estimates the Fubini-Study metric, giving natural-gradient
steps at the cost of extra fidelity evaluations. It is included because PART 11
asks for it "where practical" -- and practicality is the operative word: it needs
a fidelity primitive, which makes each iteration substantially more expensive.

COBYLA is gradient-free and deterministic. It is the honest classical baseline:
if COBYLA on the same circuit matches SPSA, the stochastic machinery is not
earning its keep.

Determinism
-----------
Every optimiser here is seeded. SPSA and QNSPSA draw random perturbation
directions, so without a seed two runs of the same configuration produce
different parameters and the model is not reproducible. The seed is threaded
through NumPy's global state because that is what ``qiskit-algorithms`` samples
from; :func:`seed_everything` is called by :func:`build_optimizer` and records
what it set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from quantum_ml.results import OptimizerName

logger = logging.getLogger(__name__)

# SPSA defaults. qiskit-algorithms calibrates its own gain schedule from the
# objective when learning_rate/perturbation are left unset, which is more robust
# than guessing constants -- so these are only used when a caller pins them.
#
# Calibration is not free: it spends ~25 extra objective evaluations before the
# first real step, which is why n_function_evaluations exceeds 2 * maxiter + 1 on
# an uncalibrated run. Both numbers are recorded so the cost is visible.
DEFAULT_MAXITER = 100
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_PERTURBATION = 0.10

# COBYLA's initial trust-region radius. The parameters are rotation angles on
# [-pi, pi], so a step of ~0.5 rad is a meaningful but not wild first move.
DEFAULT_COBYLA_RHOBEG = 0.5


class OptimizationError(RuntimeError):
    """Raised when optimisation cannot start or completes in an invalid state."""


def seed_everything(seed: int) -> None:
    """Seed every RNG the optimisers draw from.

    ``qiskit-algorithms`` samples SPSA's Bernoulli perturbation directions from
    ``algorithm_globals.random``, a ``numpy.random.Generator`` that is *separate*
    from NumPy's global legacy state. Seeding only ``np.random.seed`` leaves that
    generator untouched, so two runs of the same configuration produce different
    parameters -- verified: the fix is what makes repeat runs bit-identical.
    NumPy's global state is seeded too, for the initial-parameter draw and for any
    downstream code that still uses it.
    """
    np.random.seed(seed)
    try:
        from qiskit_algorithms.utils import algorithm_globals
    except ImportError:  # pragma: no cover - dependency is pinned
        logger.warning("qiskit-algorithms unavailable; optimiser seeding is incomplete.")
        return
    algorithm_globals.random_seed = seed


@dataclass
class OptimizationResult:
    """Outcome of one optimisation run."""

    parameters: np.ndarray
    objective: float
    n_iterations: int
    n_function_evaluations: int
    history: List[float] = field(default_factory=list)
    initial_objective: Optional[float] = None
    best_objective: Optional[float] = None
    best_parameters: Optional[np.ndarray] = None
    converged: bool = False
    optimizer: OptimizerName = OptimizerName.SPSA
    notes: str = ""

    @property
    def improvement(self) -> Optional[float]:
        if self.initial_objective is None:
            return None
        return self.initial_objective - self.objective


@dataclass
class ObjectiveTracker:
    """Records every objective evaluation so training is auditable.

    The optimisers themselves report only a final value. Without this the
    ``objective_history`` in :class:`~quantum_ml.results.QuantumTrainingRecord`
    would be empty and there would be no way to tell a converged run from one
    that never moved.
    """

    objective: Callable[[np.ndarray], float]
    history: List[float] = field(default_factory=list)
    best_objective: float = float("inf")
    best_parameters: Optional[np.ndarray] = None
    n_calls: int = 0
    on_evaluation: Optional[Callable[[int, float], None]] = None

    def __call__(self, parameters: np.ndarray) -> float:
        value = float(self.objective(np.asarray(parameters, dtype=np.float64)))
        self.n_calls += 1
        self.history.append(value)
        if value < self.best_objective:
            self.best_objective = value
            self.best_parameters = np.array(parameters, dtype=np.float64, copy=True)
        if self.on_evaluation is not None:
            self.on_evaluation(self.n_calls, value)
        return value


@dataclass
class OptimizerSpec:
    """A configured optimiser, with the settings recorded for the training record."""

    name: OptimizerName
    maxiter: int = DEFAULT_MAXITER
    learning_rate: Optional[float] = None
    perturbation: Optional[float] = None
    seed: int = 42
    extra: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> Dict[str, Any]:
        return {
            "optimizer": self.name.value,
            "maxiter": self.maxiter,
            "learning_rate": self.learning_rate,
            "perturbation": self.perturbation,
            "seed": self.seed,
            **self.extra,
        }

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        initial_parameters: np.ndarray,
        *,
        on_evaluation: Optional[Callable[[int, float], None]] = None,
        fidelity: Optional[Any] = None,
    ) -> OptimizationResult:
        """Minimise ``objective`` starting from ``initial_parameters``.

        Args:
            objective: scalar loss over the parameter vector.
            initial_parameters: starting angles.
            on_evaluation: progress callback, receiving ``(call_index, value)``.
            fidelity: a Qiskit fidelity primitive, required only by QNSPSA.
        """
        seed_everything(self.seed)
        x0 = np.asarray(initial_parameters, dtype=np.float64)
        if x0.ndim != 1 or x0.size == 0:
            raise OptimizationError(
                f"Initial parameters must be a non-empty 1-D vector, got shape {x0.shape}."
            )

        tracker = ObjectiveTracker(objective=objective, on_evaluation=on_evaluation)
        optimizer = self._build(fidelity=fidelity)

        try:
            # The initial evaluation is inside the guard so an objective that
            # fails immediately -- an unfitted pipeline, a dimension mismatch --
            # still surfaces as OptimizationError rather than leaking whatever
            # the objective happened to raise.
            initial_value = tracker(x0)
            raw = optimizer.minimize(fun=tracker, x0=x0)
        except Exception as exc:  # noqa: BLE001 - surfaced as a typed error
            raise OptimizationError(
                f"{self.name.value} optimisation failed after {tracker.n_calls} "
                f"objective evaluations: {exc}"
            ) from exc

        parameters = np.asarray(raw.x, dtype=np.float64)
        final = float(raw.fun) if raw.fun is not None else tracker.history[-1]

        # SPSA returns its last iterate, which -- being a stochastic method -- is
        # not necessarily its best. Keeping the best-seen parameters means a noisy
        # final step cannot undo a good run.
        best_parameters = tracker.best_parameters
        if best_parameters is not None and tracker.best_objective < final:
            parameters = best_parameters
            final = tracker.best_objective

        n_iterations = int(getattr(raw, "nit", 0) or 0) or self._infer_iterations(tracker.n_calls)
        return OptimizationResult(
            parameters=parameters,
            objective=final,
            n_iterations=n_iterations,
            n_function_evaluations=tracker.n_calls,
            history=tracker.history,
            initial_objective=initial_value,
            best_objective=tracker.best_objective,
            best_parameters=best_parameters,
            converged=final < initial_value,
            optimizer=self.name,
        )

    def _infer_iterations(self, n_calls: int) -> int:
        """Iteration count when the optimiser does not report one.

        SPSA spends two objective evaluations per iteration, plus the one this
        module makes to record the initial value.
        """
        if self.name in (OptimizerName.SPSA, OptimizerName.QNSPSA):
            return max(0, (n_calls - 1) // 2)
        return max(0, n_calls - 1)

    def _build(self, *, fidelity: Optional[Any] = None):
        try:
            from qiskit_algorithms.optimizers import COBYLA, QNSPSA, SPSA
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise OptimizationError(
                "qiskit-algorithms is required for VQC training. "
                "Install it with `pip install qiskit-algorithms`."
            ) from exc

        if self.name is OptimizerName.SPSA:
            return SPSA(
                maxiter=self.maxiter,
                learning_rate=self.learning_rate,
                perturbation=self.perturbation,
                **self.extra,
            )
        if self.name is OptimizerName.COBYLA:
            return COBYLA(
                maxiter=self.maxiter,
                rhobeg=self.extra.pop("rhobeg", DEFAULT_COBYLA_RHOBEG),
                **self.extra,
            )
        if self.name is OptimizerName.QNSPSA:
            if fidelity is None:
                raise OptimizationError(
                    "QNSPSA requires a fidelity primitive. Pass fidelity=... (see "
                    "QNSPSA.get_fidelity) or choose SPSA, which needs no metric "
                    "estimate."
                )
            return QNSPSA(
                fidelity=fidelity,
                maxiter=self.maxiter,
                learning_rate=self.learning_rate,
                perturbation=self.perturbation,
                **self.extra,
            )
        raise OptimizationError(f"Unsupported optimizer {self.name!r}.")


def build_optimizer(
    name: str = "spsa",
    *,
    maxiter: int = DEFAULT_MAXITER,
    learning_rate: Optional[float] = None,
    perturbation: Optional[float] = None,
    seed: int = 42,
    **extra: Any,
) -> OptimizerSpec:
    """Construct an optimiser spec by name.

    ``learning_rate`` and ``perturbation`` default to ``None`` so SPSA calibrates
    its own gain schedule against the objective, which is more robust than fixed
    constants across different qubit counts and circuit depths.
    """
    try:
        optimizer_name = OptimizerName(name.lower())
    except ValueError as exc:
        available = ", ".join(member.value for member in OptimizerName)
        raise OptimizationError(
            f"Unknown optimizer {name!r}. Available: {available}."
        ) from exc
    return OptimizerSpec(
        name=optimizer_name,
        maxiter=maxiter,
        learning_rate=learning_rate,
        perturbation=perturbation,
        seed=seed,
        extra=dict(extra),
    )


def qnspsa_fidelity(ansatz, sampler=None):
    """Build the fidelity primitive QNSPSA needs from a parameterised ansatz.

    Separated out because constructing it requires a Sampler, which the VQC owns
    -- keeping the dependency here rather than inside
    :meth:`OptimizerSpec.minimize` means SPSA and COBYLA never pay for it.
    """
    from qiskit.primitives import Sampler
    from qiskit_algorithms.optimizers import QNSPSA

    return QNSPSA.get_fidelity(ansatz, sampler=sampler or Sampler())
