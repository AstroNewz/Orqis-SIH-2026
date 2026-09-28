"""E2 angle-range selection experiment -- TRAIN partition ONLY (firewall-safe).

The first E2 run measured the fixed quantum map destroying signal: on the primary oracle
condition the 16 quantum features scored validation PR-AUC ~0.505 while the PCA-8 input
they are built from scored 0.775 and the matched RFF-16 control scored 0.871. The
analytic cause is angle folding: with two Ry re-uploading blocks the effective
single-qubit response folds like ``cos(2*theta)``, and over the originally-specced range
``[0, pi]`` that ``2*theta in [0, 2*pi]`` wraps, so two well-separated PCA values collapse
onto the same observable and the linear head cannot separate them.

This experiment selects the encoded angle range **honestly**, on TRAIN only, by nested
patient-grouped cross-validation:

* Outer: :func:`grouped_folds` splits TRAIN into patient-disjoint folds.
* Per outer fold the whole pipeline is refit on the fold's own train rows -- PCA, the
  angle scaler, the quantum feature map and the logistic head (whose L2 ``C`` is chosen by
  its own inner grouped CV) -- and scored on the held-out fold. No fold ever sees another
  fold's rows, and **no validation or test row is read anywhere in this module.**

The winner is ``argmax`` mean outer-fold PR-AUC. Entanglement (mean ``|connected
correlation|`` over TRAIN) is reported alongside so the choice can be read against the
demonstrator's requirement of genuine quantum content -- one block leaves the CZ ring
invisible to every Z-string observable (it commutes through), so its entanglement witness
is identically zero and it is disqualified as a *quantum* demonstrator regardless of its
score. The selection is model architecture, made on TRAIN; validation is scored once, by
the driver, only for the finally chosen configuration.

Usage::

    python -m backend.evaluation.e2_angle_experiment
    python -m backend.evaluation.e2_angle_experiment --out reports_e2_angle_experiment.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.evaluation.e0_controls import grouped_folds
from backend.evaluation.e0_readout import _metrics, fit_readout
from backend.evaluation.e2_quantum_visual import PRIMARY, build_inputs
from backend.ml.quantum_visual import QuantumVisualPreprocessor
from quantum_ml.visual_circuit import QuantumVisualFeatureMap

logger = logging.getLogger(__name__)

E2_ANGLE_EXPERIMENT_VERSION = "v1-e2-angle-experiment-1"
SEED = 42
N_OUTER_FOLDS = 5

#: The candidate configurations. One-block is included once, at the full range, purely to
#: demonstrate empirically that it carries zero quantum content (its CZ ring commutes
#: through every Z observable) even if it scores; it is not eligible to win as a quantum
#: demonstrator. The rest sweep the two-block angle ceiling.
CANDIDATES: Tuple[Tuple[int, float, str], ...] = (
    (1, float(np.pi), "1 block, [0, pi] -- CZ ring invisible to Z observables (control)"),
    (2, float(np.pi / 4.0), "2 blocks, [0, pi/4]"),
    (2, float(np.pi / 2.0), "2 blocks, [0, pi/2]"),
    (2, float(2.0 * np.pi / 3.0), "2 blocks, [0, 2pi/3]"),
    (2, float(3.0 * np.pi / 4.0), "2 blocks, [0, 3pi/4]"),
    (2, float(np.pi), "2 blocks, [0, pi] -- original spec, folds cos(2 theta)"),
)


def _cv_pr_auc(
    embeddings_train: np.ndarray,
    y_train: np.ndarray,
    patients_train: Sequence[str],
    *,
    n_blocks: int,
    max_angle: float,
    seed: int,
) -> Dict[str, Any]:
    """Nested patient-grouped CV PR-AUC of the quantum representation on TRAIN only.

    The preprocessor (PCA + angle scaler) and the head are refit inside every outer fold
    on that fold's train rows, so the held-out fold is genuinely unseen -- the estimate is
    of the *configuration*, not of a single fit.
    """
    folds, note = grouped_folds(y_train, patients_train, seed=seed, n_splits=N_OUTER_FOLDS)
    feature_map = QuantumVisualFeatureMap(n_blocks=n_blocks)
    labels = list(feature_map.observable_set.labels)
    per_fold: List[float] = []
    for fold_train_idx, fold_val_idx in folds:
        emb_tr = embeddings_train[fold_train_idx]
        emb_val = embeddings_train[fold_val_idx]
        y_tr = y_train[fold_train_idx]
        y_val = y_train[fold_val_idx]
        pat_tr = [patients_train[int(i)] for i in fold_train_idx]
        if len(set(y_val.tolist())) < 2 or len(set(y_tr.tolist())) < 2:
            continue
        pre = QuantumVisualPreprocessor.fit(emb_tr, seed=seed, max_angle=max_angle)
        x_tr = feature_map.transform(pre.to_angles(emb_tr))
        x_val = feature_map.transform(pre.to_angles(emb_val))
        try:
            readout = fit_readout(
                x_tr, y_tr, pat_tr, variant="e2_angle_sweep", labels=labels, seed=seed
            )
        except Exception:  # noqa: BLE001 -- a degenerate fold is dropped, never zeroed
            continue
        metrics = _metrics(y_val, readout.scores(x_val))
        if metrics and metrics.get("pr_auc") is not None:
            per_fold.append(float(metrics["pr_auc"]))
    values = np.asarray(per_fold, dtype=np.float64)
    return {
        "fold_note": note,
        "n_folds_used": int(values.size),
        "cv_pr_auc_mean": (round(float(values.mean()), 6) if values.size else None),
        "cv_pr_auc_std": (round(float(values.std(ddof=0)), 6) if values.size else None),
        "cv_pr_auc_min": (round(float(values.min()), 6) if values.size else None),
        "cv_pr_auc_folds": [round(v, 6) for v in per_fold],
    }


def _entanglement(
    embeddings_train: np.ndarray, *, n_blocks: int, max_angle: float, seed: int
) -> Dict[str, Any]:
    """Mean/max ``|connected correlation|`` over TRAIN -- the entanglement witness.

    Fit on all TRAIN rows (a descriptive statistic, no held-out claim), so it reflects the
    configuration the driver would persist. Zero for a product-state encoding.
    """
    pre = QuantumVisualPreprocessor.fit(embeddings_train, seed=seed, max_angle=max_angle)
    feature_map = QuantumVisualFeatureMap(n_blocks=n_blocks)
    features = feature_map.transform(pre.to_angles(embeddings_train))
    corr = np.abs(feature_map.connected_correlations(features))
    return {
        "mean_abs_connected_correlation": round(float(corr.mean()), 8),
        "max_abs_connected_correlation": round(float(corr.max()), 8),
        "is_genuinely_entangled": bool(corr.max() > 1e-6),
    }


def run(
    *, config=None, seed: int = SEED, out: Optional[Path] = None
) -> Dict[str, Any]:
    per_condition, provenance = build_inputs(config=config, conditions=(PRIMARY,))
    inputs = per_condition[PRIMARY]
    # TRAIN fields only. Validation is loaded by build_inputs but is never referenced here.
    emb = inputs.mobilenet_train
    y = inputs.y_train
    patients = inputs.patients_train

    results: List[Dict[str, Any]] = []
    for n_blocks, max_angle, description in CANDIDATES:
        logger.info("Config: %s", description)
        cv = _cv_pr_auc(emb, y, patients, n_blocks=n_blocks, max_angle=max_angle, seed=seed)
        ent = _entanglement(emb, n_blocks=n_blocks, max_angle=max_angle, seed=seed)
        results.append(
            {
                "n_blocks": n_blocks,
                "max_angle": round(max_angle, 8),
                "max_angle_over_pi": round(max_angle / float(np.pi), 6),
                "description": description,
                "eligible_quantum_demonstrator": bool(n_blocks >= 2 and ent["is_genuinely_entangled"]),
                **cv,
                "entanglement": ent,
            }
        )
        logger.info(
            "  CV PR-AUC %.4f (+/- %.4f) | mean|C| %.4g | eligible=%s",
            cv["cv_pr_auc_mean"] or float("nan"),
            cv["cv_pr_auc_std"] or float("nan"),
            ent["mean_abs_connected_correlation"],
            bool(n_blocks >= 2 and ent["is_genuinely_entangled"]),
        )

    eligible = [
        r for r in results if r["eligible_quantum_demonstrator"] and r["cv_pr_auc_mean"] is not None
    ]
    # Selection is on TRAIN-internal CV PR-AUC among genuinely-entangled configs only.
    winner = max(eligible, key=lambda r: r["cv_pr_auc_mean"]) if eligible else None

    payload = {
        "experiment": "e2-angle-range-selection",
        "version": E2_ANGLE_EXPERIMENT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "partition_used": "train",
        "test_partition_used": False,
        "validation_used_for_selection": False,
        "selection_rule": (
            "argmax mean outer-fold PR-AUC under nested patient-grouped CV on TRAIN, "
            "restricted to genuinely-entangled (>=2 block, nonzero connected correlation) "
            "configurations. Validation is scored once by the driver only for the winner."
        ),
        "primary_condition": PRIMARY,
        "seed": seed,
        "n_outer_folds": N_OUTER_FOLDS,
        "provenance": provenance,
        "candidates": results,
        "selected": (
            None
            if winner is None
            else {
                "n_blocks": winner["n_blocks"],
                "max_angle": winner["max_angle"],
                "max_angle_over_pi": winner["max_angle_over_pi"],
                "cv_pr_auc_mean": winner["cv_pr_auc_mean"],
                "mean_abs_connected_correlation": winner["entanglement"]["mean_abs_connected_correlation"],
            }
        ),
    }
    destination = out or Path("reports_e2_angle_experiment.json")
    destination.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    logger.info("Wrote %s", destination)
    if winner is not None:
        logger.info(
            "SELECTED: %d block(s), max_angle=%.4f (%.3f pi), TRAIN-CV PR-AUC=%.4f",
            winner["n_blocks"], winner["max_angle"],
            winner["max_angle_over_pi"], winner["cv_pr_auc_mean"],
        )
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="E2 TRAIN-only angle-range selection.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    for noisy in ("qiskit", "stevedore", "matplotlib", "PIL", "sklearn"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run(seed=args.seed, out=args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
