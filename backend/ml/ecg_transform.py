"""The fitted half of the PTB-XL pipeline: impute -> scale -> compact (Phase E4).

:mod:`backend.ml.features_ecg` is deliberately unfitted. Every value it emits is
a function of exactly one record, which is what makes it fold-honest *by
construction* rather than by convention -- no caller can break that property
because there is no pooled statistic to break. The cost of that guarantee is
that its output is not model-ready: it carries ``NaN`` where a feature was
unmeasurable, its columns span wildly different units (millivolts, milliseconds,
degrees, dimensionless ratios), and it is 97-dimensional where the near-term
quantum regime wants 8-32.

This module is where those three things are fixed, and it is a *separate* module
for one reason: every step here is **fitted**, and a fitted step is exactly what
must never live next to a record-local one. Putting a median inside the
extractor would silently pool statistics across folds; keeping it here makes the
fold it was fitted on an explicit, recorded, checkable property of the artifact.

The guarantees this module is responsible for
---------------------------------------------

1. **Fitted on TRAIN only.** :func:`fit_ecg_transform` refuses any fold outside
   PTB-XL's TRAIN folds 1-8 unless the caller passes ``allow_non_train=True``
   and says why. Validation statistics cannot leak into the transform by
   accident, only by an argument that reads like what it is.
2. **The fitting cohort is recorded, not asserted.** The fitted artifact carries
   the folds *and the patient ids* it was fitted on, so
   :func:`assert_no_fit_eval_patient_overlap` can prove disjointness against an
   evaluation split at any later point -- including after the transform has been
   saved and reloaded in a different process.
3. **Labels are only touched by the step that needs them.** Imputation, scaling
   and PCA are unsupervised and use every TRAIN record. Univariate selection is
   supervised and therefore sees only the *labeled* TRAIN rows; records whose
   label is unknown (``LABEL_UNKNOWN``) are excluded from it rather than
   silently counted as negatives.
4. **Nothing is pickled.** The fitted state persists as plain arrays plus a JSON
   metadata blob, mirroring :mod:`backend.ml.reduction`: an artifact that
   reproduces a reported number must still load in a year, without matching
   library versions.

Why not reuse :class:`backend.ml.reduction.DimensionalityReducer`
-----------------------------------------------------------------
That class is sized for **amplitude** encoding -- its target dimension is
``2 ** n_qubits`` and it pads to a power of two, which is the right contract
there and the wrong one here. The E4 compact representation is angle-encoded, so
8-32 is a free choice rather than a power of two, and it additionally needs
imputation and scaling, which ``reduction.py`` deliberately does not do. The PCA
maths is the same (SVD on the centred matrix, no pickle) and the rationale in
that module's docstring applies unchanged.

Usage::

    python -m backend.ml.ecg_transform --cache backend/artifacts/dataset/ptbxl/features_ecg-v1.npz
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from backend.dataset.ptbxl import TRAIN_FOLDS, VALIDATION_FOLDS
from backend.ml.features_ecg import FEATURE_SET_VERSION, feature_names
from backend.training.prepare_ecg_features import LABEL_UNKNOWN, EcgCohort

logger = logging.getLogger(__name__)

ECG_TRANSFORM_VERSION = "ecg-fit-1"
"""Bumped whenever the fitted transform's semantics or stored state change."""

REDUCTION_METHODS: Tuple[str, ...] = ("identity", "pca", "select")

DEFAULT_COMPONENTS = 16
"""Mid-range of the 8-32 dimensions a near-term angle-encoded circuit can carry."""


class EcgTransformError(ValueError):
    """Raised when the transform is misconfigured, misused, or used unfitted."""


# --------------------------------------------------------------------------
# The fitted artifact
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EcgFeatureTransform:
    """An imputer + scaler + reducer, fitted on one named set of folds.

    Immutable on purpose: a transform that could be re-fitted in place is a
    transform whose recorded fitting cohort might no longer be true.
    """

    feature_set_version: str
    transform_version: str
    input_feature_names: Tuple[str, ...]
    kept_indices: np.ndarray          # into the input feature layout
    medians: np.ndarray               # TRAIN medians of the kept columns
    means: np.ndarray                 # TRAIN means, post-imputation
    scales: np.ndarray                # TRAIN stds, post-imputation (1.0 where 0)
    reduction_method: str
    components: Optional[np.ndarray]  # (k, d) PCA basis in the scaled space
    explained_variance_ratio: Optional[np.ndarray]
    selected_indices: Optional[np.ndarray]  # into the scaled space, for 'select'
    selection_scores: Optional[np.ndarray]
    fitted_on_folds: Tuple[int, ...]
    train_patient_ids: np.ndarray
    n_train_records: int
    n_train_labeled: int
    dropped_features: Tuple[str, ...]
    drop_reasons: Dict[str, str]

    # ------------------------------------------------------------------ shape

    @property
    def input_dimension(self) -> int:
        return len(self.input_feature_names)

    @property
    def kept_dimension(self) -> int:
        return int(self.kept_indices.size)

    @property
    def output_dimension(self) -> int:
        if self.reduction_method == "pca":
            return int(self.components.shape[0])  # type: ignore[union-attr]
        if self.reduction_method == "select":
            return int(self.selected_indices.size)  # type: ignore[union-attr]
        return self.kept_dimension

    @property
    def n_train_patients(self) -> int:
        return int(np.unique(self.train_patient_ids).size)

    @property
    def total_explained_variance_ratio(self) -> Optional[float]:
        if self.explained_variance_ratio is None:
            return None
        return float(np.sum(self.explained_variance_ratio))

    def output_feature_names(self) -> Tuple[str, ...]:
        """Names for the compact representation, so a result stays interpretable."""
        kept = [self.input_feature_names[int(index)] for index in self.kept_indices]
        if self.reduction_method == "pca":
            return tuple(f"pc{index + 1:02d}" for index in range(self.output_dimension))
        if self.reduction_method == "select":
            return tuple(kept[int(index)] for index in self.selected_indices)  # type: ignore[union-attr]
        return tuple(kept)

    # -------------------------------------------------------------- transform

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """Apply the fitted transform to any split. Never re-fits anything."""
        data = np.asarray(matrix, dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] != self.input_dimension:
            raise EcgTransformError(
                f"Transform was fitted for {self.input_dimension}-dimensional "
                f"features, got {data.shape[1]}. Rebuild the cache or refit."
            )

        kept = data[:, self.kept_indices]
        # Non-finite rather than NaN: the extractor emits NaN for the
        # unmeasurable, but an inf arriving from anywhere must also be treated as
        # missing instead of propagating through the scaler into every component.
        missing = ~np.isfinite(kept)
        if missing.any():
            kept = np.where(missing, np.broadcast_to(self.medians, kept.shape), kept)

        scaled = (kept - self.means) / self.scales
        if self.reduction_method == "pca":
            reduced = scaled @ self.components.T  # type: ignore[union-attr]
        elif self.reduction_method == "select":
            reduced = scaled[:, self.selected_indices]  # type: ignore[union-attr]
        else:
            reduced = scaled

        if not np.isfinite(reduced).all():
            raise EcgTransformError(
                "Transform produced a non-finite value. A fitted median or scale "
                "is invalid; refusing to hand this to a model."
            )
        return reduced

    def transform_cohort(self, cohort: EcgCohort, folds: Iterable[int]) -> np.ndarray:
        mask = cohort.fold_mask(folds)
        return self.transform(cohort.features[mask])

    # ------------------------------------------------------------- reporting

    def summary(self) -> Dict[str, Any]:
        return {
            "transform_version": self.transform_version,
            "feature_set_version": self.feature_set_version,
            "fitted_on_folds": list(self.fitted_on_folds),
            "n_train_records": self.n_train_records,
            "n_train_patients": self.n_train_patients,
            "n_train_labeled": self.n_train_labeled,
            "input_dimension": self.input_dimension,
            "kept_dimension": self.kept_dimension,
            "output_dimension": self.output_dimension,
            "reduction_method": self.reduction_method,
            "total_explained_variance_ratio": self.total_explained_variance_ratio,
            "n_dropped_features": len(self.dropped_features),
            "dropped_features": list(self.dropped_features),
            "drop_reasons": dict(self.drop_reasons),
        }

    # --------------------------------------------------------------- storage

    def save(self, path: Path) -> Path:
        """Persist as plain arrays + JSON metadata. No pickle, no sklearn state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "feature_set_version": self.feature_set_version,
            "transform_version": self.transform_version,
            "input_feature_names": list(self.input_feature_names),
            "reduction_method": self.reduction_method,
            "fitted_on_folds": list(self.fitted_on_folds),
            "n_train_records": self.n_train_records,
            "n_train_labeled": self.n_train_labeled,
            "dropped_features": list(self.dropped_features),
            "drop_reasons": dict(self.drop_reasons),
        }
        arrays: Dict[str, np.ndarray] = {
            "kept_indices": self.kept_indices,
            "medians": self.medians,
            "means": self.means,
            "scales": self.scales,
            "train_patient_ids": self.train_patient_ids,
        }
        for name, value in (
            ("components", self.components),
            ("explained_variance_ratio", self.explained_variance_ratio),
            ("selected_indices", self.selected_indices),
            ("selection_scores", self.selection_scores),
        ):
            if value is not None:
                arrays[name] = np.asarray(value)
        np.savez_compressed(path, metadata=json.dumps(metadata), **arrays)
        return path

    @classmethod
    def load(cls, path: Path) -> "EcgFeatureTransform":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No fitted ECG transform at {path}.")
        with np.load(path, allow_pickle=True) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata.get("feature_set_version") != FEATURE_SET_VERSION:
                raise EcgTransformError(
                    f"{path} was fitted on feature set "
                    f"{metadata.get('feature_set_version')!r}, but this code is "
                    f"{FEATURE_SET_VERSION!r}. Refit; do not reuse."
                )
            if metadata.get("transform_version") != ECG_TRANSFORM_VERSION:
                raise EcgTransformError(
                    f"{path} was written by transform "
                    f"{metadata.get('transform_version')!r}, not "
                    f"{ECG_TRANSFORM_VERSION!r}. Refit; do not reuse."
                )
            optional = {
                name: (np.asarray(archive[name]) if name in archive.files else None)
                for name in (
                    "components",
                    "explained_variance_ratio",
                    "selected_indices",
                    "selection_scores",
                )
            }
            return cls(
                feature_set_version=metadata["feature_set_version"],
                transform_version=metadata["transform_version"],
                input_feature_names=tuple(metadata["input_feature_names"]),
                kept_indices=np.asarray(archive["kept_indices"], dtype=np.int64),
                medians=np.asarray(archive["medians"], dtype=np.float64),
                means=np.asarray(archive["means"], dtype=np.float64),
                scales=np.asarray(archive["scales"], dtype=np.float64),
                reduction_method=str(metadata["reduction_method"]),
                fitted_on_folds=tuple(int(f) for f in metadata["fitted_on_folds"]),
                train_patient_ids=np.asarray(archive["train_patient_ids"]),
                n_train_records=int(metadata["n_train_records"]),
                n_train_labeled=int(metadata["n_train_labeled"]),
                dropped_features=tuple(metadata["dropped_features"]),
                drop_reasons=dict(metadata["drop_reasons"]),
                **optional,
            )


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def _univariate_auc_scores(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """|AUC - 0.5| per column: how much each feature separates the classes alone.

    Rank-based, so it is invariant to the monotone scaling already applied and
    does not assume a linear relationship. A constant column scores 0.
    """
    from sklearn.metrics import roc_auc_score

    scores = np.zeros(matrix.shape[1], dtype=np.float64)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        if np.allclose(values, values[0]):
            continue
        scores[column] = abs(float(roc_auc_score(labels, values)) - 0.5)
    return scores


def fit_ecg_transform(
    cohort: EcgCohort,
    *,
    folds: Optional[Iterable[int]] = None,
    reduction: str = "pca",
    n_components: int = DEFAULT_COMPONENTS,
    min_coverage: float = 0.5,
    variance_epsilon: float = 1e-12,
    allow_non_train: bool = False,
) -> EcgFeatureTransform:
    """Fit imputer + scaler + reducer on the TRAIN folds of ``cohort``.

    Args:
        folds: folds to fit on. Defaults to PTB-XL's TRAIN folds 1-8.
        reduction: ``"pca"`` (unsupervised), ``"select"`` (supervised univariate
            AUC, labeled TRAIN rows only), or ``"identity"``.
        min_coverage: a feature measurable in fewer than this fraction of TRAIN
            records is dropped rather than imputed. Imputing a column that is
            90% missing manufactures a constant and calls it a measurement.
        allow_non_train: required to fit on anything but folds 1-8. The one
            legitimate use is the final refit on train+validation *after* an
            architecture is frozen; it is an explicit argument so that choice
            appears in the caller's code and in review.

    Raises:
        EcgTransformError: on a non-TRAIN fold without the override, an unknown
            reduction method, an empty fitting cohort, or a feature layout that
            does not match the extractor in use.
    """
    wanted = sorted(set(folds)) if folds is not None else sorted(TRAIN_FOLDS)
    if not wanted:
        raise EcgTransformError("No folds selected to fit on.")
    outside = [fold for fold in wanted if fold not in set(TRAIN_FOLDS)]
    if outside and not allow_non_train:
        leaked = sorted(set(outside) & set(VALIDATION_FOLDS))
        raise EcgTransformError(
            f"Refusing to fit on non-TRAIN folds {outside}. Fitting a median, a "
            "scale or a PCA basis on "
            + ("validation" if leaked else "held-out")
            + " data leaks its statistics into every subsequent evaluation. Pass "
            "allow_non_train=True only for a deliberate post-freeze refit."
        )
    if reduction not in REDUCTION_METHODS:
        raise EcgTransformError(
            f"Unknown reduction {reduction!r}; expected one of {REDUCTION_METHODS}."
        )

    expected_names = feature_names()
    if tuple(cohort.feature_names) != expected_names:
        raise EcgTransformError(
            f"Cohort carries {len(cohort.feature_names)} feature names, the "
            f"extractor emits {len(expected_names)}. Rebuild the cache before "
            "fitting; a mismatched layout would fit the wrong columns."
        )

    mask = cohort.fold_mask(wanted)
    if not mask.any():
        raise EcgTransformError(f"No records in folds {wanted} to fit on.")
    train = np.asarray(cohort.features[mask], dtype=np.float64)

    # ---- step 1: coverage. Drop before imputing, not after.
    finite = np.isfinite(train)
    coverage = finite.mean(axis=0)
    drop_reasons: Dict[str, str] = {}
    keep = coverage >= float(min_coverage)
    for index in np.flatnonzero(~keep):
        drop_reasons[expected_names[int(index)]] = (
            f"coverage {coverage[int(index)]:.3f} < {min_coverage}"
        )

    # ---- step 2: TRAIN medians for what survives, then impute.
    kept_indices = np.flatnonzero(keep).astype(np.int64)
    if kept_indices.size == 0:
        raise EcgTransformError(
            "Every feature fell below the coverage floor; nothing left to fit."
        )
    kept = train[:, kept_indices]
    with np.errstate(all="ignore"):
        medians = np.nanmedian(np.where(np.isfinite(kept), kept, np.nan), axis=0)
    # A column that is entirely non-finite cannot pass the coverage floor above,
    # so a NaN median here would be a logic error rather than a data property.
    if not np.isfinite(medians).all():
        raise EcgTransformError("A TRAIN median is non-finite; refusing to impute.")
    imputed = np.where(np.isfinite(kept), kept, medians)

    # ---- step 3: variance. A constant column carries no information and would
    # divide by ~0 in the scaler.
    stds = imputed.std(axis=0)
    varying = stds > float(variance_epsilon)
    for position in np.flatnonzero(~varying):
        name = expected_names[int(kept_indices[int(position)])]
        drop_reasons[name] = f"zero variance on TRAIN (std {stds[int(position)]:.3e})"
    if not varying.any():
        raise EcgTransformError("No TRAIN feature varies; nothing left to fit.")
    kept_indices = kept_indices[varying]
    medians = medians[varying]
    imputed = imputed[:, varying]

    # ---- step 4: standardise with TRAIN statistics.
    means = imputed.mean(axis=0)
    scales = imputed.std(axis=0)
    scales = np.where(scales > float(variance_epsilon), scales, 1.0)
    scaled = (imputed - means) / scales

    # ---- step 5: compact. Unsupervised by default; supervised only on demand.
    components: Optional[np.ndarray] = None
    explained: Optional[np.ndarray] = None
    selected: Optional[np.ndarray] = None
    selection_scores: Optional[np.ndarray] = None
    labels = np.asarray(cohort.labels)[mask]
    labeled = labels != LABEL_UNKNOWN

    if reduction == "pca":
        k = int(min(n_components, scaled.shape[1], max(scaled.shape[0] - 1, 1)))
        if k < 1:
            raise EcgTransformError("Too few TRAIN records to fit any component.")
        # SVD on the already-centred matrix; same maths as backend/ml/reduction.py.
        _, singular, right = np.linalg.svd(scaled, full_matrices=False)
        components = np.ascontiguousarray(right[:k])
        variances = (singular**2) / max(scaled.shape[0] - 1, 1)
        total = float(np.sum(variances))
        explained = (variances[:k] / total) if total > 0 else np.zeros(k)
    elif reduction == "select":
        if int(labeled.sum()) < 2 or np.unique(labels[labeled]).size < 2:
            raise EcgTransformError(
                "Supervised selection needs both classes among the labeled TRAIN "
                "rows; the unknown-label records are deliberately not usable here."
            )
        # Labeled TRAIN rows only: an unknown label is not a negative.
        selection_scores = _univariate_auc_scores(scaled[labeled], labels[labeled])
        k = int(min(n_components, scaled.shape[1]))
        selected = np.argsort(-selection_scores, kind="stable")[:k].astype(np.int64)
        selected = np.sort(selected)  # keep the original feature order readable

    dropped = tuple(
        name for name in expected_names if name in drop_reasons
    )
    return EcgFeatureTransform(
        feature_set_version=FEATURE_SET_VERSION,
        transform_version=ECG_TRANSFORM_VERSION,
        input_feature_names=expected_names,
        kept_indices=kept_indices,
        medians=medians,
        means=means,
        scales=scales,
        reduction_method=reduction,
        components=components,
        explained_variance_ratio=explained,
        selected_indices=selected,
        selection_scores=selection_scores,
        fitted_on_folds=tuple(wanted),
        train_patient_ids=np.unique(np.asarray(cohort.patient_ids)[mask]),
        n_train_records=int(mask.sum()),
        n_train_labeled=int(labeled.sum()),
        dropped_features=dropped,
        drop_reasons=drop_reasons,
    )


def assert_no_fit_eval_patient_overlap(
    transform: EcgFeatureTransform, patient_ids: Sequence[int]
) -> None:
    """Prove the fitting cohort shares no patient with an evaluation split.

    PTB-XL's ``strat_fold`` is already patient-level, so this should never fire.
    That is the point: it is cheap, it survives a save/load round trip, and it
    turns "the folds are patient-disjoint" from a claim into a check.
    """
    overlap = np.intersect1d(
        np.asarray(transform.train_patient_ids),
        np.unique(np.asarray(list(patient_ids))),
    )
    if overlap.size:
        raise EcgTransformError(
            f"{overlap.size} patient(s) appear in both the transform's fitting "
            f"cohort (folds {list(transform.fitted_on_folds)}) and the evaluation "
            f"split, e.g. {overlap[:5].tolist()}. Every metric downstream would "
            "be inflated."
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _format_summary(transform: EcgFeatureTransform, cohort: EcgCohort) -> str:
    summary = transform.summary()
    lines = [
        "PTB-XL fitted feature transform",
        "===============================",
        f"transform ........... {summary['transform_version']}"
        f" on {summary['feature_set_version']}",
        f"fitted on folds ..... {summary['fitted_on_folds']}",
        f"TRAIN cohort ........ {summary['n_train_records']} records, "
        f"{summary['n_train_patients']} patients "
        f"({summary['n_train_labeled']} labeled)",
        f"dimension ........... {summary['input_dimension']} in -> "
        f"{summary['kept_dimension']} kept -> {summary['output_dimension']} out"
        f" ({summary['reduction_method']})",
    ]
    if summary["total_explained_variance_ratio"] is not None:
        lines.append(
            f"variance retained ... {summary['total_explained_variance_ratio']:.4f}"
        )
    lines.append(f"dropped features .... {summary['n_dropped_features']}")
    for name, reason in list(transform.drop_reasons.items())[:10]:
        lines.append(f"  - {name}: {reason}")

    lines.append("")
    lines.append("applied to each partition (transform only, never re-fitted):")
    for label, folds in (
        ("train 1-8", sorted(TRAIN_FOLDS)),
        ("validation 9", sorted(VALIDATION_FOLDS)),
    ):
        mask = cohort.fold_mask(folds)
        if not mask.any():
            lines.append(f"  {label:<14} absent from this cache")
            continue
        reduced = transform.transform(cohort.features[mask])
        patients = np.unique(np.asarray(cohort.patient_ids)[mask]).size
        lines.append(
            f"  {label:<14} {reduced.shape[0]:>6} records / {patients:>5} patients"
            f" -> {reduced.shape[1]}d, finite={bool(np.isfinite(reduced).all())}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from backend.training.prepare_ecg_features import cache_path, load_cohort

    parser = argparse.ArgumentParser(
        description="Fit the TRAIN-only ECG feature transform (impute/scale/compact)"
    )
    parser.add_argument("--cache", type=Path, default=None, help="Feature cache .npz")
    parser.add_argument("--root", type=Path, default=None, help="Dataset root")
    parser.add_argument("--reduction", default="pca", choices=list(REDUCTION_METHODS))
    parser.add_argument("--components", type=int, default=DEFAULT_COMPONENTS)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None, help="Where to save it")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cache = args.cache
    if cache is None:
        if args.root is None:
            parser.error("pass --cache or --root")
        cache = cache_path(args.root)
    cohort = load_cohort(cache)

    transform = fit_ecg_transform(
        cohort,
        reduction=args.reduction,
        n_components=args.components,
        min_coverage=args.min_coverage,
    )
    # The guarantee, checked rather than asserted: the fitting cohort and the
    # validation split share no patient.
    validation = cohort.fold_mask(sorted(VALIDATION_FOLDS))
    if validation.any():
        assert_no_fit_eval_patient_overlap(
            transform, np.asarray(cohort.patient_ids)[validation].tolist()
        )

    print(_format_summary(transform, cohort))
    if args.out is not None:
        transform.save(args.out)
        print(f"\nFitted transform written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
