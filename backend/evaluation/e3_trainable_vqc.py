"""E3 Round 3: does a *trainable* shallow VQC on the E2 states add information beyond C7?

Round 1 (:mod:`backend.evaluation.e3_hybrid_fusion`) read the fixed E2 map with a *linear*
head; Round 2 (:mod:`backend.evaluation.e3_quantum_kernel`) read it with a *fidelity
kernel*. Both were clean nulls against the classical fusion baseline C7 (validation PR-AUC
0.913038). Both kept the circuit **fixed** -- the eight ``Ry`` angles are data, never
weights. The natural next rung of the mission ladder (§30) is to let the circuit *learn*: a
small **variational quantum classifier** whose gates are trained by gradient descent.

The circuit (trainability is the *only* new axis)
-------------------------------------------------
The fold-honest ``MobileNet->PCA-8->angle`` encoding and the exact 8-qubit E2 statevector
are reused unchanged (Round 1/2 identical), so this experiment isolates *trainability* from
representation. On top of the fixed E2 state we apply ``L`` trainable layers of
``{Ry(theta) on every qubit; nearest-neighbour CZ ring}`` and read the **8 single-qubit
local ``<Z_i>``** into a trainable logistic head. Trained end-to-end (circuit angles + head)
by Adam on class-weighted BCE, fold-honestly.

Two hazards, measured not assumed
---------------------------------
* **Overfitting / the ``sqrt(T/N)`` bound (Caro et al. 2022).** A trainable circuit *can*
  memorise; the generalisation gap scales like ``sqrt(T/N)``. With ``T = 8L + 9`` trainable
  parameters and ``N = 215`` train samples the bound is ~0.28 (L=1) / ~0.34 (L=2), and the
  honest estimate is the patient-grouped **train-CV out-of-fold** PR-AUC, never the fit.
* **Barren plateaus (McClean 2018, Cerezo 2021).** Deep circuits with *global* observables
  have gradients that vanish exponentially in qubit count, making training impossible. The
  guard is architectural -- **shallow depth + local (single-qubit) observables** provably
  avoid the plateau (Cerezo 2021) -- and it is *witnessed*: the module reports the variance
  of the loss gradient at random initialisation. A vanishing variance would be an honest
  "untrainable", not something hidden behind a fitted number.

Exact and cheap, and provably a real circuit
---------------------------------------------
Every gate here (``Ry``, ``CZ``) is a **real** matrix, so the E2 statevector and every
downstream state are real vectors: the whole forward pass is real ``float64`` and its
gradients are exact torch autodiff -- no parameter-shift estimator, no sampling noise. That
the torch forward *is* the quantum circuit is not asserted but **proven**: the trainable
ansatz is rebuilt in Qiskit and the torch ``<Z_i>`` are checked against
``Statevector.evolve`` (Qiskit's reference simulator, arithmetic-independent of this module)
to ``<=1e-10`` before any VQC number is reported.

Matched classical controls (DEC-033 discipline)
------------------------------------------------
The VQC is never reported alone. Two controls sit beside it on identical folds:

* a **capacity-matched classical MLP** on the same L2-normalised PCA-8 input (a trainable
  nonlinear map of comparable parameter count) -- the classical analogue of "let it learn";
* a **no-entangler ablation** of the VQC itself (CZ ring removed -> a product-state circuit)
  -- isolating whether entanglement, specifically, contributes anything.

If the VQC beats neither C7 nor its own classical/ablation controls, there is no trainable
quantum advantage, and the module says so.

Discipline (identical to Rounds 1-2 / E1 / E2)
----------------------------------------------
C7 is rebuilt and must reproduce 0.913038 to 1e-6; the E2 map is Aer-validated to <=1e-10;
the PCA/angle encoding is refit **inside each fold-train only**; VQC depth is selected on
**TRAIN out-of-fold PR-AUC only** (never validation), then confirmed once on validation;
training hyper-parameters are fixed a priori (never tuned on validation/test); scores are
the head logits (uncalibrated margin -- PR/ROC-AUC, the patient bootstrap and the
``_increment_null`` stack are all rank/standardise based); the gating test is E1.1's
``_increment_null`` with C7 as reference; TEST is never read. No number is fabricated -- a
sub-C7 result, an untrainable circuit, or a null increment is reported as measured (§29B).

Usage::

    python -m backend.evaluation.e3_trainable_vqc --out reports_e3_trainable_vqc.json
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from backend.core.config import Settings
from backend.evaluation.e0_controls import N_FOLDS, grouped_folds
from backend.evaluation.e1_failure_map import (
    BOOTSTRAP_DRAWS,
    INCREMENT_PERMUTATIONS,
    PRIMARY,
    SEED,
    E1Inputs,
    _average_precision,
    _increment_null,
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
    _score_arm,
)
from backend.ml.quantum_visual import PCA_COMPONENTS, QuantumVisualPreprocessor, _l2_normalise
from quantum_ml.visual_circuit import E2_N_QUBITS, QuantumVisualFeatureMap, nn_ring_pairs, validate_against_aer

logger = logging.getLogger(__name__)

#: Bump when the Round-3 protocol semantics change.
E3_TRAINABLE_VQC_VERSION = "v1-e3-trainable-vqc-1"

#: Trainable variational depths compared on TRAIN out-of-fold PR-AUC (never validation).
VQC_DEPTHS: Tuple[int, ...] = (1, 2)

#: Training hyper-parameters, fixed a priori (never tuned on validation/test). Full-batch
#: Adam on class-weighted BCE; a shallow real circuit on <=215 x 256 states trains in well
#: under a second, so a fixed generous step budget needs no early stopping (which on a
#: holdout would be validation leakage).
TRAIN_STEPS = 400
LEARNING_RATE = 0.05
WEIGHT_DECAY = 1e-3

#: Random initialisations averaged for the barren-plateau gradient-variance witness.
GRAD_VAR_INITS = 48

#: Torch/Qiskit agreement gate for the trainable ansatz (same tolerance E2 uses for Aer).
ANSATZ_AER_TOLERANCE = 1e-10
ANSATZ_AER_SAMPLES = 6

_REAL_STATE_ATOL = 1e-8  # E2 states are real by construction (Ry, CZ real); guard the cast.


class TrainableVQCError(RuntimeError):
    """Raised when the Round-3 VQC experiment is misconfigured or an invariant breaks."""


# ------------------------------------------------------- real-arithmetic circuit pieces
def _cz_ring_phase(n_qubits: int = E2_N_QUBITS) -> np.ndarray:
    """``(2**n,)`` real ``+/-1`` diagonal of the nearest-neighbour CZ ring.

    ``CZ`` flips the sign of every basis state with both ring-edge qubits set to 1; the
    whole ring is the product of those per-edge signs, so applying the ring to a real
    statevector is one elementwise multiply. Little-endian: bit ``q`` of index ``b`` is
    qubit ``q`` (matching Qiskit and :func:`quantum_ml.visual_circuit.build_encoding_circuit`).
    """
    dim = 1 << n_qubits
    ring = nn_ring_pairs(n_qubits)
    idx = np.arange(dim)
    phase = np.ones(dim, dtype=np.float64)
    for a, b in ring:
        both = ((idx >> a) & 1) & ((idx >> b) & 1)
        phase[both.astype(bool)] *= -1.0
    return phase


def _z_signs(n_qubits: int = E2_N_QUBITS) -> np.ndarray:
    """``(n, 2**n)`` real signs so ``<Z_q> = sum_b p(b) z_signs[q, b]``.

    ``z_signs[q, b] = +1`` if qubit ``q`` is 0 in basis state ``b`` else ``-1``. Diagonal in
    the computational basis, so the expectation depends only on ``|amplitude|^2``.
    """
    dim = 1 << n_qubits
    idx = np.arange(dim)
    out = np.empty((n_qubits, dim), dtype=np.float64)
    for q in range(n_qubits):
        out[q] = np.where((idx >> q) & 1, -1.0, 1.0)
    return out


def _real_states(states: np.ndarray) -> np.ndarray:
    """Cast exact E2 statevectors to real ``float64``, asserting the imaginary part is noise.

    The E2 circuit is ``Ry`` + ``CZ`` on ``|0>`` -- all real -- so its statevector is real up
    to floating point. Working in real arithmetic (not complex) is what makes the torch
    forward a trivially-differentiable ``float64`` computation; this guard makes the
    assumption explicit rather than silent.
    """
    arr = np.asarray(states)
    max_imag = float(np.max(np.abs(arr.imag))) if np.iscomplexobj(arr) else 0.0
    if max_imag > _REAL_STATE_ATOL:
        raise TrainableVQCError(
            f"E2 statevector has a non-negligible imaginary part ({max_imag:.2e} > "
            f"{_REAL_STATE_ATOL:.0e}); the real-arithmetic VQC forward would be wrong."
        )
    return np.ascontiguousarray(arr.real, dtype=np.float64)


# ------------------------------------------------------------------- torch VQC forward
def _apply_ry_layer(state_nd: torch.Tensor, thetas_row: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """Apply ``Ry(theta_q)`` to every qubit of a real state tensor ``(batch, 2, ..., 2)``.

    Little-endian axis map: qubit ``q`` is tensor axis ``n_qubits - q`` (batch is axis 0), so
    qubit 0 (LSB) is the last axis and qubit ``n-1`` (MSB) is axis 1 -- matching the reshape
    of a Qiskit statevector. The Torch/Qiskit gate is checked in :func:`_validate_ansatz`.
    """
    s = state_nd
    for q in range(n_qubits):
        half = thetas_row[q] * 0.5
        c, sn = torch.cos(half), torch.sin(half)
        ax = n_qubits - q
        moved = torch.movedim(s, ax, -1)  # (..., 2) along qubit q
        a0, a1 = moved[..., 0], moved[..., 1]
        # Ry = [[c, -s], [s, c]]
        moved = torch.stack([c * a0 - sn * a1, sn * a0 + c * a1], dim=-1)
        s = torch.movedim(moved, -1, ax)
    return s


def _vqc_logits(
    states: torch.Tensor,
    thetas: torch.Tensor,
    head_w: torch.Tensor,
    head_b: torch.Tensor,
    cz_phase: torch.Tensor,
    z_signs: torch.Tensor,
    *,
    n_qubits: int,
    entangle: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass: fixed E2 state -> ``L`` trainable ``{Ry ring; CZ ring}`` layers -> head.

    Returns ``(logits, local_z)`` where ``local_z`` is the ``(batch, n_qubits)`` matrix of
    single-qubit ``<Z_q>`` (the barren-plateau-safe local observables) and ``logits`` is the
    trainable logistic head over them. All real ``float64``; gradients are exact autodiff.
    """
    batch, dim = states.shape
    depth = thetas.shape[0]
    shape_nd = (batch,) + (2,) * n_qubits
    s = states
    for layer in range(depth):
        s = _apply_ry_layer(s.reshape(shape_nd), thetas[layer], n_qubits).reshape(batch, dim)
        if entangle:
            s = s * cz_phase
    probs = s * s  # real amplitudes -> Born-rule probabilities
    local_z = probs @ z_signs.t()  # (batch, n_qubits)
    logits = local_z @ head_w + head_b
    return logits, local_z


def _class_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights normalised to mean 1 (the VQC-classifier convention)."""
    y = np.asarray(y, dtype=np.float64)
    n_pos = float(np.sum(y == 1))
    n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.ones(y.size, dtype=np.float64)
    w_pos = y.size / (2.0 * n_pos)
    w_neg = y.size / (2.0 * n_neg)
    return np.where(y == 1, w_pos, w_neg).astype(np.float64)


def _init_parameters(
    n_qubits: int, depth: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic parameter init: circuit angles ``~U(-pi, pi)``, head ``~N(0, 0.1)``.

    The head is small but *non-zero* on purpose. A zero head would make ``d(logit)/d(theta)``
    identically zero at initialisation (theta reaches the logit only *through* the head), so
    the first optimiser step would carry no circuit gradient. A small random head lets the
    circuit angles and the head both receive signal from step 0. (The barren-plateau witness
    is head-independent by construction -- see :func:`_gradient_variance_witness` -- so this
    choice does not touch the trainability diagnostic.)
    """
    gen = torch.Generator().manual_seed(int(seed))
    thetas = (torch.rand(depth, n_qubits, generator=gen, dtype=torch.float64) * 2.0 - 1.0) * np.pi
    head_w = torch.randn(n_qubits, generator=gen, dtype=torch.float64) * 0.1
    head_b = torch.zeros((), dtype=torch.float64)
    return thetas, head_w, head_b


def _train_vqc(
    states_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_qubits: int,
    depth: int,
    entangle: bool,
    cz_phase: torch.Tensor,
    z_signs: torch.Tensor,
    seed: int,
    steps: int = TRAIN_STEPS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> Dict[str, Any]:
    """Fold-honest end-to-end training of circuit angles + logistic head (Adam, weighted BCE).

    A ``scorer`` closure is returned that maps any batch of (real) states to head logits, so
    the caller scores held-out folds and validation *only* through the trained parameters.
    """
    x = torch.tensor(_real_states(states_train), dtype=torch.float64)
    y = torch.tensor(np.asarray(y_train, dtype=np.float64), dtype=torch.float64)
    w = torch.tensor(_class_sample_weights(y_train), dtype=torch.float64)

    thetas, head_w, head_b = _init_parameters(n_qubits, depth, seed)
    thetas.requires_grad_(True)
    head_w.requires_grad_(True)
    head_b.requires_grad_(True)
    optimiser = torch.optim.Adam([thetas, head_w, head_b], lr=lr, weight_decay=weight_decay)
    bce = torch.nn.BCEWithLogitsLoss(weight=w, reduction="mean")

    initial_loss = None
    for step in range(steps):
        optimiser.zero_grad(set_to_none=True)
        logits, _ = _vqc_logits(
            x, thetas, head_w, head_b, cz_phase, z_signs, n_qubits=n_qubits, entangle=entangle
        )
        loss = bce(logits, y)
        loss.backward()
        optimiser.step()
        if step == 0:
            initial_loss = float(loss.detach())
    final_loss = float(loss.detach())

    theta_final = thetas.detach().clone()
    w_final = head_w.detach().clone()
    b_final = head_b.detach().clone()

    def scorer(states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xs = torch.tensor(_real_states(states), dtype=torch.float64)
            logits, _ = _vqc_logits(
                xs, theta_final, w_final, b_final, cz_phase, z_signs,
                n_qubits=n_qubits, entangle=entangle,
            )
            return logits.numpy().astype(np.float64)

    return {"scorer": scorer, "initial_loss": initial_loss, "final_loss": final_loss}


def _gradient_variance_witness(
    states_train: np.ndarray,
    *,
    n_qubits: int,
    depth: int,
    entangle: bool,
    cz_phase: torch.Tensor,
    z_signs: torch.Tensor,
    seed: int,
    n_inits: int = GRAD_VAR_INITS,
) -> Dict[str, Any]:
    """Barren-plateau witness: variance of the *observable* gradient over random circuit inits.

    Measures ``d<Z_0>/d(theta)`` -- the gradient of a single local observable (Pauli ``Z`` on
    qubit 0) for individual representative training states -- across ``n_inits`` random angle
    initialisations. This is exactly the McClean-2018 / Cerezo-2021 barren-plateau probe, and
    it is **head-independent** on purpose: it witnesses the *circuit's* trainability, not the
    logistic head's. Per-state gradients are *pooled*, never averaged across states or qubits
    (averaging cancels sign-varying components and would spuriously suppress the magnitude). A
    barren plateau is a variance that vanishes (exponentially in ``n_qubits``); a healthy
    trainable circuit keeps it well away from zero. Measured, never assumed -- shallow depth
    plus a local ``<Z_i>`` observable is exactly the Cerezo-2021 recipe for avoiding it.
    """
    x = torch.tensor(_real_states(states_train), dtype=torch.float64)
    zero_w = torch.zeros(n_qubits, dtype=torch.float64)
    zero_b = torch.zeros((), dtype=torch.float64)
    n_probe = int(min(8, x.shape[0]))

    grads: List[np.ndarray] = []
    for r in range(n_inits):
        base_thetas, _, _ = _init_parameters(n_qubits, depth, seed + r)
        for j in range(n_probe):
            thetas = base_thetas.clone().requires_grad_(True)
            _, local_z = _vqc_logits(
                x[j : j + 1], thetas, zero_w, zero_b, cz_phase, z_signs,
                n_qubits=n_qubits, entangle=entangle,
            )
            local_z[0, 0].backward()  # <Z_0> for this single state
            grads.append(thetas.grad.detach().numpy().ravel().copy())

    stacked = np.concatenate(grads) if grads else np.zeros(1)
    variance = float(np.var(stacked))
    return {
        "n_inits": int(n_inits),
        "n_probe_states": n_probe,
        "n_circuit_parameters": int(depth * n_qubits),
        "observable": "single local <Z_0>, per-state gradients pooled (head-independent, Cerezo-2021 local cost)",
        "gradient_variance": variance,
        "gradient_abs_mean": float(np.mean(np.abs(stacked))),
        # A generous floor: true barren plateaus drive this many orders of magnitude lower.
        "vanishing": bool(variance < 1e-6),
    }


# ------------------------------------------------------------------- honesty gate (Qiskit)
def _qiskit_local_z(state: np.ndarray, thetas: np.ndarray, n_qubits: int, entangle: bool) -> np.ndarray:
    """Reference ``<Z_q>`` from Qiskit ``Statevector.evolve`` for the trainable ansatz.

    Arithmetic-independent of the torch forward: Qiskit applies its own ``ry``/``cz`` gate
    definitions. Agreement to 1e-10 is what proves the torch VQC *is* the quantum circuit.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli, Statevector

    depth = thetas.shape[0]
    qc = QuantumCircuit(n_qubits)
    ring = nn_ring_pairs(n_qubits)
    for layer in range(depth):
        for q in range(n_qubits):
            qc.ry(float(thetas[layer, q]), q)
        if entangle:
            for a, b in ring:
                qc.cz(a, b)
    evolved = Statevector(np.asarray(state, dtype=complex)).evolve(qc)
    out = np.empty(n_qubits, dtype=np.float64)
    for q in range(n_qubits):
        label = ["I"] * n_qubits
        label[n_qubits - 1 - q] = "Z"  # Qiskit label is MSB-first; qubit q is index n-1-q
        out[q] = float(np.real(evolved.expectation_value(Pauli("".join(label)))))
    return out


def _validate_ansatz(
    *,
    n_qubits: int,
    cz_phase: torch.Tensor,
    z_signs: torch.Tensor,
    seed: int,
    tolerance: float = ANSATZ_AER_TOLERANCE,
    n_samples: int = ANSATZ_AER_SAMPLES,
    depths: Sequence[int] = VQC_DEPTHS,
) -> Dict[str, Any]:
    """Assert the torch ``<Z_q>`` equals Qiskit's to ``tolerance`` on random states/angles.

    Runs across the depths and both entangling modes actually used, on random *real* unit
    states (the E2 states are real, so this is the operative domain). Raises before any VQC
    number is reported if the deviation exceeds ``tolerance``.
    """
    rng = np.random.default_rng(seed)
    dim = 1 << n_qubits
    max_dev = 0.0
    checks = 0
    for depth in depths:
        for entangle in (True, False):
            for _ in range(n_samples):
                raw = rng.standard_normal(dim)
                state = raw / np.linalg.norm(raw)
                thetas = rng.uniform(-np.pi, np.pi, size=(depth, n_qubits))
                with torch.no_grad():
                    xs = torch.tensor(state[None, :], dtype=torch.float64)
                    _, local_z = _vqc_logits(
                        xs,
                        torch.tensor(thetas, dtype=torch.float64),
                        torch.zeros(n_qubits, dtype=torch.float64),
                        torch.zeros((), dtype=torch.float64),
                        cz_phase, z_signs, n_qubits=n_qubits, entangle=entangle,
                    )
                torch_z = local_z.numpy().ravel()
                ref_z = _qiskit_local_z(state, thetas, n_qubits, entangle)
                max_dev = max(max_dev, float(np.max(np.abs(torch_z - ref_z))))
                checks += 1
    passed = max_dev <= tolerance
    if not passed:
        raise TrainableVQCError(
            f"Torch VQC forward disagrees with Qiskit Statevector by {max_dev:.3e} > "
            f"{tolerance:.1e}; refusing to report VQC features (the circuit would not be real)."
        )
    return {
        "max_abs_deviation": max_dev,
        "tolerance": tolerance,
        "passed": True,
        "n_checks": checks,
        "reference": "qiskit Statevector.evolve (ry/cz), arithmetic-independent of the torch forward",
    }


# ----------------------------------------------------------------------- arm scorers
def _oof_and_validation(
    fold_states: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    states_all_train: np.ndarray,
    states_validation: np.ndarray,
    y_train: np.ndarray,
    *,
    train_fn,
) -> Tuple[np.ndarray, Optional[float], np.ndarray, Dict[str, Any]]:
    """Generic fold-honest OOF + validation column for a trainable arm.

    ``train_fn(states_train, y_sub)`` returns a dict with a ``scorer`` (states -> logits).
    Per fold the model is trained on fold-train states only and scores the held-out fold;
    the validation column is trained on all TRAIN and scores validation once.
    """
    y = np.asarray(y_train, dtype=int)
    oof = np.full(y.size, np.nan, dtype=np.float64)
    losses: List[float] = []
    for train_idx, test_idx, st_train, st_test in fold_states:
        if len(set(y[train_idx].tolist())) < 2:
            continue
        fit = train_fn(st_train, y[train_idx])
        oof[test_idx] = fit["scorer"](st_test)
        losses.append(fit["final_loss"])
    usable = np.isfinite(oof)
    oof_pr_auc = None if usable.sum() < 2 else _average_precision(y[usable], oof[usable])

    fit_all = train_fn(states_all_train, y)
    validation_score = fit_all["scorer"](states_validation)
    train_note = {
        "mean_fold_final_loss": float(np.mean(losses)) if losses else None,
        "all_train_final_loss": fit_all["final_loss"],
        "all_train_initial_loss": fit_all["initial_loss"],
    }
    return oof, oof_pr_auc, validation_score, train_note


def _score_vqc_arm(
    fold_states: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    states_all_train: np.ndarray,
    states_validation: np.ndarray,
    y_train: np.ndarray,
    *,
    n_qubits: int,
    entangle: bool,
    cz_phase: torch.Tensor,
    z_signs: torch.Tensor,
    seed: int,
    depths: Sequence[int] = VQC_DEPTHS,
) -> Dict[str, Any]:
    """Trainable VQC arm: select depth on TRAIN OOF PR-AUC, then confirm once on validation.

    Depth is the only architecture knob and it is chosen by the same TRAIN-only nested-CV
    criterion DEC-036 used for the E2 angle range -- never by peeking at validation. With
    ``entangle=False`` this is the no-entangler ablation on the identical protocol.
    """
    by_depth: Dict[int, Dict[str, Any]] = {}
    for depth in depths:
        def train_fn(st: np.ndarray, ysub: np.ndarray, _d=depth):
            return _train_vqc(
                st, ysub, n_qubits=n_qubits, depth=_d, entangle=entangle,
                cz_phase=cz_phase, z_signs=z_signs, seed=seed,
            )

        oof, oof_pr_auc, val_score, train_note = _oof_and_validation(
            fold_states, states_all_train, states_validation, y_train, train_fn=train_fn
        )
        by_depth[depth] = {
            "depth": depth,
            "train_out_of_fold": oof,
            "train_cv_pr_auc": None if oof_pr_auc is None else round(oof_pr_auc, 6),
            "validation_score": val_score,
            "train_note": train_note,
            "n_trainable_parameters": depth * n_qubits + n_qubits + 1,
            "n_quantum_trainable_parameters": depth * n_qubits,
        }

    ranked = sorted(
        by_depth.values(),
        key=lambda r: (r["train_cv_pr_auc"] is None, -(r["train_cv_pr_auc"] or 0.0)),
    )
    best = ranked[0]
    # Trainability diagnostic at the selected depth, on all TRAIN states (a property of the
    # circuit/loss surface, not a scored metric -- it never touches validation).
    witness = _gradient_variance_witness(
        states_all_train,
        n_qubits=n_qubits, depth=best["depth"], entangle=entangle,
        cz_phase=cz_phase, z_signs=z_signs, seed=seed,
    )
    n_train = np.asarray(y_train).size
    t = best["n_trainable_parameters"]
    return {
        "model": "trainable_vqc" if entangle else "trainable_vqc_no_entangler",
        "entangle": entangle,
        "representation_dim": n_qubits,  # 8 local-Z observables read into the head
        "selected_depth": best["depth"],
        "train_cv_pr_auc_by_depth": {str(d): by_depth[d]["train_cv_pr_auc"] for d in depths},
        "train_out_of_fold": best["train_out_of_fold"],
        "train_cv_pr_auc": best["train_cv_pr_auc"],
        "validation_score": best["validation_score"],
        "n_trainable_parameters": t,
        "n_quantum_trainable_parameters": best["n_quantum_trainable_parameters"],
        "sqrt_t_over_n": round(float(np.sqrt(t / n_train)), 4),
        "train_note": best["train_note"],
        "barren_plateau_witness": witness,
    }


def _train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_in: int,
    hidden: int,
    seed: int,
    steps: int = TRAIN_STEPS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> Dict[str, Any]:
    """Capacity-matched classical control: a tanh MLP on PCA-8, trained like the VQC."""
    x = torch.tensor(np.asarray(x_train, dtype=np.float64), dtype=torch.float64)
    y = torch.tensor(np.asarray(y_train, dtype=np.float64), dtype=torch.float64)
    w = torch.tensor(_class_sample_weights(y_train), dtype=torch.float64)

    torch.manual_seed(int(seed))
    model = torch.nn.Sequential(
        torch.nn.Linear(n_in, hidden, dtype=torch.float64),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1, dtype=torch.float64),
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = torch.nn.BCEWithLogitsLoss(weight=w, reduction="mean")
    for step in range(steps):
        optimiser.zero_grad(set_to_none=True)
        logits = model(x).squeeze(-1)
        loss = bce(logits, y)
        loss.backward()
        optimiser.step()
        if step == 0:
            initial_loss = float(loss.detach())
    final_loss = float(loss.detach())

    def scorer(x_new: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xs = torch.tensor(np.asarray(x_new, dtype=np.float64), dtype=torch.float64)
            return model(xs).squeeze(-1).numpy().astype(np.float64)

    return {"scorer": scorer, "initial_loss": initial_loss, "final_loss": final_loss}


def _mlp_parameter_count(n_in: int, hidden: int) -> int:
    return n_in * hidden + hidden + hidden + 1


def _matched_hidden(vqc_params: int, n_in: int) -> int:
    """Hidden width whose MLP parameter count is closest to the VQC's trainable-param count."""
    best_h, best_gap = 1, None
    for h in range(1, 33):
        gap = abs(_mlp_parameter_count(n_in, h) - vqc_params)
        if best_gap is None or gap < best_gap:
            best_h, best_gap = h, gap
    return best_h


def _score_mlp_arm(
    fold_inputs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    pca_all_train: np.ndarray,
    pca_validation: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden: int,
    seed: int,
) -> Dict[str, Any]:
    """Matched classical MLP control on fold-honest L2-normalised PCA-8."""
    n_in = int(pca_all_train.shape[1])

    def train_fn(x: np.ndarray, ysub: np.ndarray):
        return _train_mlp(x, ysub, n_in=n_in, hidden=hidden, seed=seed)

    oof, oof_pr_auc, val_score, train_note = _oof_and_validation(
        fold_inputs, pca_all_train, pca_validation, y_train, train_fn=train_fn
    )
    return {
        "model": "classical_mlp",
        "representation_dim": n_in,
        "hidden_units": hidden,
        "n_trainable_parameters": _mlp_parameter_count(n_in, hidden),
        "train_out_of_fold": oof,
        "train_cv_pr_auc": None if oof_pr_auc is None else round(oof_pr_auc, 6),
        "validation_score": val_score,
        "train_note": train_note,
    }


def _roc_auc(y: np.ndarray, scores: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, np.asarray(scores, dtype=np.float64)))


def _arm_summary(
    name: str,
    fitted: Dict[str, Any],
    inputs: E1Inputs,
    *,
    bootstrap_draws: int,
    seed: int,
) -> Dict[str, Any]:
    """PR-AUC + ROC-AUC + patient bootstrap for a logit-scored trainable arm."""
    scores = np.asarray(fitted["validation_score"], dtype=np.float64)
    validation_pr_auc = _average_precision(inputs.y_validation, scores)
    bootstrap = bootstrap_pr_auc(
        inputs.y_validation, scores, inputs.patients_validation, draws=bootstrap_draws, seed=seed
    )
    summary = {
        "arm": name,
        "model": fitted["model"],
        "representation_dim": fitted["representation_dim"],
        "train_cv_pr_auc": fitted["train_cv_pr_auc"],
        "validation_pr_auc": None if validation_pr_auc is None else round(validation_pr_auc, 6),
        "validation_roc_auc": _roc_auc(inputs.y_validation, scores),
        "n_trainable_parameters": fitted.get("n_trainable_parameters"),
        "score_note": "head logits (uncalibrated margin); PR/ROC-AUC are rank-based",
        "validation_bootstrap": bootstrap,
    }
    for key in (
        "selected_depth", "train_cv_pr_auc_by_depth", "n_quantum_trainable_parameters",
        "sqrt_t_over_n", "barren_plateau_witness", "hidden_units", "entangle", "train_note",
    ):
        if key in fitted:
            summary[key] = fitted[key]
    return summary


# --------------------------------------------------------------- fold-honest encodings
@dataclass
class _FoldEncodings:
    """Pre-computed fold-honest encodings shared by every arm (so folds are refit once)."""

    fold_states: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    fold_pca: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    states_all_train: np.ndarray
    states_validation: np.ndarray
    pca_all_train: np.ndarray
    pca_validation: np.ndarray


def _build_fold_encodings(
    feature_map: QuantumVisualFeatureMap,
    mobilenet_train: np.ndarray,
    mobilenet_validation: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
) -> _FoldEncodings:
    """Refit the PCA/angle encoding **inside each fold-train only**, then build the states.

    This is where the mission's leakage discipline lives for Round 3: the
    ``QuantumVisualPreprocessor`` (PCA + angle scaler) is fit on fold-train rows only, then
    used to encode both fold-train and the held-out fold before either the E2 statevectors
    (VQC input) or the L2-normalised PCA-8 (MLP input) are computed. The validation column is
    built from a preprocessor fit on all TRAIN and never sees validation during the fit. A
    spy on ``QuantumVisualPreprocessor.fit`` (see the tests) proves no holdout row is ever fit.
    """
    def _states(pre: QuantumVisualPreprocessor, source: np.ndarray) -> np.ndarray:
        return _real_states(feature_map.statevectors(pre.to_angles(source)))

    fold_states: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    fold_pca: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in folds:
        pre = QuantumVisualPreprocessor.fit(mobilenet_train[train_idx], seed=seed)
        fold_states.append(
            (train_idx, test_idx, _states(pre, mobilenet_train[train_idx]), _states(pre, mobilenet_train[test_idx]))
        )
        fold_pca.append(
            (train_idx, test_idx, _l2_normalise(pre.pca_scores(mobilenet_train[train_idx])), _l2_normalise(pre.pca_scores(mobilenet_train[test_idx])))
        )
    pre_all = QuantumVisualPreprocessor.fit(mobilenet_train, seed=seed)
    return _FoldEncodings(
        fold_states=fold_states,
        fold_pca=fold_pca,
        states_all_train=_states(pre_all, mobilenet_train),
        states_validation=_states(pre_all, mobilenet_validation),
        pca_all_train=_l2_normalise(pre_all.pca_scores(mobilenet_train)),
        pca_validation=_l2_normalise(pre_all.pca_scores(mobilenet_validation)),
    )


# ------------------------------------------------------------------------- driver
def run(
    *,
    config: Optional[Settings] = None,
    seed: int = SEED,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    increment_permutations: int = INCREMENT_PERMUTATIONS,
    quantum_validation_sample: int = QUANTUM_VALIDATION_SAMPLE,
) -> Dict[str, Any]:
    """Run the E3 Round-3 trainable-VQC experiment on the primary leak-free condition."""
    started = time.perf_counter()

    per_condition, provenance = build_inputs(config=config, conditions=(PRIMARY,))
    inputs = per_condition[PRIMARY]
    if set(inputs.patients_train) & set(inputs.patients_validation):
        raise HybridFusionError("Patient overlap between train and validation; refusing to measure.")

    folds, cv_note = grouped_folds(inputs.y_train, inputs.patients_train, seed=seed, n_splits=N_FOLDS)
    logger.info(
        "E3-VQC primary %s: %d train / %d validation (%d / %d positive); %s",
        PRIMARY, inputs.y_train.size, inputs.y_validation.size,
        int(inputs.y_train.sum()), int(inputs.y_validation.sum()), cv_note,
    )

    # --- Rebuild C7 exactly as Rounds 1-2 do, and anchor it.
    specs = candidate_matrices(inputs)
    c6_train, c6_validation = specs["C6"]["train"], specs["C6"]["validation"]
    c5_train, c5_validation = specs["C5"]["train"], specs["C5"]["validation"]
    if c6_train.shape[1] != MOBILENET_DIM or c5_train.shape[1] != C5_DIM:
        raise HybridFusionError(
            f"Unexpected C6/C5 dims ({c6_train.shape[1]}/{c5_train.shape[1]}); expected "
            f"{MOBILENET_DIM}/{C5_DIM}."
        )
    c7_train = np.hstack([c6_train, c5_train]).astype(np.float64)
    c7_validation = np.hstack([c6_validation, c5_validation]).astype(np.float64)
    if c7_train.shape[1] != C7_DIM:
        raise HybridFusionError(f"C7 has {c7_train.shape[1]} columns, expected {C7_DIM}.")

    # Score C7 through Rounds 1-2's exact fold-honest path (`_score_arm` + identity
    # factory): standardisation refit inside each fold-train, so the train-CV OOF column
    # and its PR-AUC are strictly leakage-free and identical to how DEC-037/DEC-038 scored
    # C7 (train-CV OOF 0.766915). The all-train validation fit reproduces the 0.913038
    # anchor. This is the single source of truth for C7 here -- both the reported train-CV
    # figure and the increment-test reference column come from it.
    c7_arm = _score_arm(
        c7_train, c7_validation, inputs.y_train, folds,
        preprocess=_identity_factory, seed=seed,
    )
    c7_pr_auc = _average_precision(inputs.y_validation, c7_arm["validation_score"])
    anchor_ok = c7_pr_auc is not None and abs(float(c7_pr_auc) - C7_REFERENCE_PR_AUC) <= C7_ANCHOR_TOLERANCE
    anchor = {
        "target_pr_auc": C7_REFERENCE_PR_AUC,
        "reproduced_pr_auc": None if c7_pr_auc is None else round(float(c7_pr_auc), 6),
        "selected_c": c7_arm.get("selected_c"),
        "tolerance": C7_ANCHOR_TOLERANCE,
        "reproduced": bool(anchor_ok),
    }
    if not anchor_ok:
        logger.warning("C7 anchor NOT reproduced (%s vs %s); conclusions suppressed.",
                       anchor["reproduced_pr_auc"], C7_REFERENCE_PR_AUC)

    # --- E2 feature map, Aer-validated; then the trainable ansatz, Qiskit-validated.
    mobilenet_train = np.asarray(inputs.e1_train["mobilenet"], dtype=np.float64)
    mobilenet_validation = np.asarray(inputs.e1_validation["mobilenet"], dtype=np.float64)
    feature_map = QuantumVisualFeatureMap()
    pre_check = QuantumVisualPreprocessor.fit(mobilenet_train, seed=seed)
    sample = min(int(quantum_validation_sample), mobilenet_train.shape[0])
    aer_check = validate_against_aer(feature_map, pre_check.to_angles(mobilenet_train[:sample]))

    cz_phase = torch.tensor(_cz_ring_phase(E2_N_QUBITS), dtype=torch.float64)
    z_signs = torch.tensor(_z_signs(E2_N_QUBITS), dtype=torch.float64)
    ansatz_check = _validate_ansatz(n_qubits=E2_N_QUBITS, cz_phase=cz_phase, z_signs=z_signs, seed=seed)

    # --- Fold-honest E2 states (VQC) and L2-normalised PCA-8 (MLP), refit per fold-train.
    enc = _build_fold_encodings(feature_map, mobilenet_train, mobilenet_validation, folds, seed=seed)

    # --- Arms: trainable VQC, its no-entangler ablation, and a capacity-matched MLP.
    logger.info("  arm trainable_vqc ...")
    vqc_fit = _score_vqc_arm(
        enc.fold_states, enc.states_all_train, enc.states_validation, inputs.y_train,
        n_qubits=E2_N_QUBITS, entangle=True, cz_phase=cz_phase, z_signs=z_signs, seed=seed,
    )
    logger.info("  arm trainable_vqc_no_entangler ...")
    noent_fit = _score_vqc_arm(
        enc.fold_states, enc.states_all_train, enc.states_validation, inputs.y_train,
        n_qubits=E2_N_QUBITS, entangle=False, cz_phase=cz_phase, z_signs=z_signs, seed=seed,
    )
    hidden = _matched_hidden(vqc_fit["n_trainable_parameters"], int(enc.pca_all_train.shape[1]))
    logger.info("  arm classical_mlp (hidden=%d) ...", hidden)
    mlp_fit = _score_mlp_arm(
        enc.fold_pca, enc.pca_all_train, enc.pca_validation, inputs.y_train, hidden=hidden, seed=seed
    )

    arms = {
        "trainable_vqc": _arm_summary("trainable_vqc", vqc_fit, inputs, bootstrap_draws=bootstrap_draws, seed=seed),
        "trainable_vqc_no_entangler": _arm_summary("trainable_vqc_no_entangler", noent_fit, inputs, bootstrap_draws=bootstrap_draws, seed=seed),
        "classical_mlp": _arm_summary("classical_mlp", mlp_fit, inputs, bootstrap_draws=bootstrap_draws, seed=seed),
    }
    c7_boot = bootstrap_pr_auc(
        inputs.y_validation, c7_arm["validation_score"], inputs.patients_validation,
        draws=bootstrap_draws, seed=seed,
    )
    arms["c7"] = {
        "arm": "c7",
        "model": "linear_logistic",
        "representation_dim": C7_DIM,
        "train_cv_pr_auc": c7_arm["train_cv_pr_auc"],
        "validation_pr_auc": None if c7_pr_auc is None else round(float(c7_pr_auc), 6),
        "validation_roc_auc": _roc_auc(inputs.y_validation, c7_arm["validation_score"]),
        "validation_bootstrap": c7_boot,
    }

    # --- Increment tests: C7 -> {trainable_vqc, classical_mlp}.
    increments: Dict[str, Dict[str, Any]] = {}
    reference_val = np.asarray(c7_arm["validation_score"], dtype=np.float64)
    reference_oof = np.asarray(c7_arm["train_out_of_fold"], dtype=np.float64)
    for name, fit in (("trainable_vqc", vqc_fit), ("classical_mlp", mlp_fit)):
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

    # --- Honest verdict.
    vqc_val = arms["trainable_vqc"]["validation_pr_auc"]
    noent_val = arms["trainable_vqc_no_entangler"]["validation_pr_auc"]
    mlp_val = arms["classical_mlp"]["validation_pr_auc"]
    c7_val = arms["c7"]["validation_pr_auc"]
    vqc_survives = increments["trainable_vqc"].get("survives_gating_null")
    mlp_survives = increments["classical_mlp"].get("survives_gating_null")
    vqc_beats_c7 = vqc_val is not None and c7_val is not None and vqc_val > c7_val
    vqc_beats_mlp = vqc_val is not None and mlp_val is not None and vqc_val > mlp_val
    vqc_beats_noent = vqc_val is not None and noent_val is not None and vqc_val > noent_val

    witness = arms["trainable_vqc"].get("barren_plateau_witness", {})
    barren = bool(witness.get("vanishing", False))
    defensible_quantum_advantage = bool(vqc_survives)
    # A trainable-quantum-specific advantage needs the increment to survive AND the classical
    # MLP not to AND entanglement to matter (VQC beats its own no-entangler ablation).
    quantum_specific = bool(vqc_survives and not mlp_survives and vqc_beats_noent)

    if not anchor_ok:
        conclusion = "C7 anchor did not reproduce; experiment not trustworthy, no conclusion drawn."
    elif barren:
        conclusion = (
            "The trainable VQC sits on a barren plateau (init gradient variance "
            f"{witness.get('gradient_variance'):.2e}): its parameters do not receive a usable "
            "training signal, so any score is an artefact. Reported as a measured negative (§29B)."
        )
    elif quantum_specific:
        conclusion = (
            "The C7 -> trainable-VQC increment survives the patient-blocked column-permutation "
            "null while the matched classical MLP does not, and the VQC beats its own "
            "no-entangler ablation: on this evidence a trainable entangled circuit carries "
            "information beyond C7 that a comparable classical model does not. Freeze and "
            "re-examine before any test-set claim."
        )
    elif defensible_quantum_advantage:
        conclusion = (
            "The C7 -> trainable-VQC increment survives its gating null, but so does the matched "
            "classical MLP (or entanglement does not help): the added information is not "
            "quantum-specific. Not a quantum advantage."
        )
    else:
        conclusion = (
            "The C7 -> trainable-VQC increment does NOT survive the patient-blocked "
            "column-permutation null: on this evidence a trainable shallow VQC adds no "
            "information beyond C7. A null result (§29B), reported as measured."
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
        "trainable_vqc_validation_pr_auc": vqc_val,
        "trainable_vqc_no_entangler_validation_pr_auc": noent_val,
        "classical_mlp_validation_pr_auc": mlp_val,
        "trainable_vqc_beats_c7": vqc_beats_c7,
        "trainable_vqc_beats_mlp_control": vqc_beats_mlp,
        "trainable_vqc_beats_no_entangler": vqc_beats_noent,
        "trainable_vqc_barren_plateau": barren,
        "trainable_vqc_increment_survives_gating_null": vqc_survives,
        "classical_mlp_increment_survives_gating_null": mlp_survives,
        "defensible_quantum_advantage_over_c7": defensible_quantum_advantage,
        "quantum_specific_advantage": quantum_specific,
        "selected_vqc_depth": vqc_fit["selected_depth"],
        "leaderboard": leaderboard,
        "conclusion": conclusion,
    }

    duration = time.perf_counter() - started
    return {
        "e3_trainable_vqc_version": E3_TRAINABLE_VQC_VERSION,
        "question": (
            "Does a trainable shallow VQC on the 8-qubit E2 states add information beyond C7 "
            "(validation PR-AUC 0.913038), under identical patient-grouped folds, train-only "
            "encoding + depth selection, sqrt(T/N)-bounded trainable parameters, a "
            "barren-plateau witness, and matched classical MLP + no-entangler controls?"
        ),
        "test_partition_used": False,
        "test_partition_note": (
            "Reads TRAIN and VALIDATION only via e1_failure_map.build_inputs, which iterates only "
            "('train', 'validation'). No code path selects the test partition."
        ),
        "primary_condition": PRIMARY,
        "cv_note": cv_note,
        "rows": inputs.shape_note,
        "c7_anchor": anchor,
        "quantum_feature_map_aer_check": aer_check,
        "trainable_ansatz_qiskit_check": ansatz_check,
        "arms": arms,
        "increment_vs_c7": increments,
        "summary": summary,
        "provenance": provenance,
        "protocol": {
            "reference_baseline": "C7 = hstack([C6 mobilenet(576), C5 descriptor+multiscale(619)]) = 1195, logistic",
            "trainable_vqc": (
                "fixed E2 encoding -> L trainable {Ry ring; CZ ring} layers -> 8 local <Z_i> -> "
                "logistic head; Adam on class-weighted BCE; depth in {1,2} selected on TRAIN OOF"
            ),
            "matched_control": "capacity-matched tanh MLP on the same L2-normalised PCA-8 input",
            "ablation_control": "identical VQC with the CZ ring removed (product-state circuit)",
            "folds": f"{N_FOLDS}-fold patient-grouped StratifiedGroupKFold on TRAIN, seed {seed}",
            "preprocessing": "train-only; PCA/angle encoding refit inside each fold-train",
            "generalisation_bound": "sqrt(T/N), Caro et al. 2022; T = 8L + 9, N = train size",
            "barren_plateau_guard": "shallow depth + local single-qubit observables (Cerezo 2021), witnessed by init gradient variance",
            "scores": "head logits (rank-based PR/ROC-AUC; increment stack standardises)",
            "gating_test": "e1_failure_map._increment_null with C7 as reference",
            "honesty_gate": "torch <Z_i> vs qiskit Statevector.evolve <=1e-10 before any VQC number",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "seed": seed,
            "n_folds": N_FOLDS,
            "bootstrap_draws": bootstrap_draws,
            "increment_permutations": increment_permutations,
            "pca_components": PCA_COMPONENTS,
            "train_steps": TRAIN_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "duration_seconds": round(duration, 3),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="E3 Round 3: trainable shallow VQC research experiment.")
    parser.add_argument("--out", type=str, default="reports_e3_trainable_vqc.json")
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
        "C7=%s VQC=%s (depth %s) no-ent=%s MLP=%s | VQC beats C7: %s | VQC increment survives: %s (mlp=%s) | barren: %s",
        s["c7_validation_pr_auc"], s["trainable_vqc_validation_pr_auc"], s["selected_vqc_depth"],
        s["trainable_vqc_no_entangler_validation_pr_auc"], s["classical_mlp_validation_pr_auc"],
        s["trainable_vqc_beats_c7"], s["trainable_vqc_increment_survives_gating_null"],
        s["classical_mlp_increment_survives_gating_null"], s["trainable_vqc_barren_plateau"],
    )
    logger.info("%s", s["conclusion"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
