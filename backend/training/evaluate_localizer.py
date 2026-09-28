"""Final held-out evaluation of the **frozen** lesion-ROI localiser.

This module reads the test partition. That is the whole reason it is a separate
module from :mod:`backend.training.train_localizer`, which structurally cannot:

* the trainer's ``TRAINABLE_PARTITIONS`` excludes test, so no threshold, epoch,
  checkpoint or preprocessing constant can be chosen from test data;
* this module has no optimiser, no threshold sweep and no checkpoint selection. It
  loads one checkpoint, applies one threshold that was already written to disk, and
  measures. There is no code path here that could tune anything.

The split is enforced by construction rather than by discipline: the frozen
threshold is read out of ``localizer.json`` and compared against the value the
*validation* sweep selected in ``training_report.json``. If they disagree the
evaluation refuses to run, because that would mean the artifact was touched after
the development phase closed.

**"Accepted" is not "correct".** The localiser accepts a box when its confidence
clears the frozen threshold, and that is a statement about the model's own estimate,
not about the box. Correctness needs a ground-truth box, which exists for only part
of the test partition. So this report keeps three populations apart:

* lesion-annotated test images -- acceptance *and* IoU, so correctness is measured;
* unannotated test images -- acceptance only, labelled
  ``unverified_localization`` throughout, because there is nothing to check the box
  against;
* images whose annotation lies outside its own frame -- counted, never scored, for
  the reason recorded in DEC-021.

Usage::

    python -m backend.training.evaluate_localizer
    python -m backend.training.evaluate_localizer --no-save
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from backend.core.config import Settings, settings as default_settings
from backend.dataset.smartom import ImageRecord, SmartOmIndex, build_index
from backend.dataset.split import SplitManifest
from backend.ml.artifacts import ArtifactStore
from backend.ml.localizer import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LOCALIZER_INPUT_SIZE,
    LOCALIZER_VERSION,
    LesionLocalizer,
    LocalizationTally,
    box_iou,
)
from backend.ml.types import RoiSource
from backend.training.train_localizer import (
    OUT_OF_FRAME_TOLERANCE,
    TRAINABLE_PARTITIONS,
    _lesion_box_for,
    box_overshoot,
    build_localizer_dataset,
)

logger = logging.getLogger(__name__)

TEST_PARTITION = "test"

VERIFIED_IOU_FLOOR = 0.25
"""The IoU at which a box counts as verified-usable in this report.

Not a new decision: it is the same floor the validation threshold sweep optimised
against, carried here so the development and final numbers mean the same thing.
Reported alongside 0.5 and 0.75 so a reader who prefers a stricter definition has
the number without needing this file to be re-run.
"""

DETERMINISM_SAMPLE = 12
"""How many test images the repeat-prediction check covers.

Determinism is a property of the code path, not of the image, so a sample is
sufficient to catch a stochastic layer left in training mode; the check is cheap
enough that there is no reason to make it smaller.
"""


class LocalizerEvaluationError(RuntimeError):
    """Raised when the frozen artifact cannot be evaluated as frozen."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not len(values):
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 4),
        "median": round(float(np.median(array)), 4),
        "min": round(float(array.min()), 4),
        "max": round(float(array.max()), 4),
        "std": round(float(array.std(ddof=0)), 4),
    }


def _correlation(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """Pearson correlation, or an explicit reason it is not meaningful.

    Returned as ``None`` with a stated reason rather than 0.0: a correlation of zero
    is a finding, and "there was not enough variation to compute one" is a different
    finding that must not be reported as the first.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size < 3:
        return {"value": None, "n": int(x.size), "reason": "fewer_than_3_pairs"}
    if x.std() == 0.0 or y.std() == 0.0:
        return {"value": None, "n": int(x.size), "reason": "zero_variance"}
    return {
        "value": round(float(np.corrcoef(x, y)[0, 1]), 4),
        "n": int(x.size),
        "reason": None,
    }


# --------------------------------------------------------------- frozen artifact
def frozen_artifact_description(
    directory: Path, *, training_report: Dict[str, Any]
) -> Dict[str, Any]:
    """Identify exactly what is being evaluated, and check it is unmodified.

    Raises:
        LocalizerEvaluationError: the persisted threshold does not match the one the
            validation sweep selected. That mismatch is the signature of an artifact
            retuned after development closed, so it stops the evaluation rather than
            being reported as a footnote.
    """
    weights_path = directory / "localizer.pt"
    config_path = directory / "localizer.json"
    if not weights_path.exists() or not config_path.exists():
        raise LocalizerEvaluationError(
            f"No frozen localiser at {directory}. Train one with "
            "`python -m backend.training.train_localizer`."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected = training_report.get("threshold_selection", {}).get("selected_threshold")
    persisted = config.get("min_confidence")
    if selected is None or float(persisted) != float(selected):
        raise LocalizerEvaluationError(
            f"The persisted threshold ({persisted}) is not the one the validation "
            f"sweep selected ({selected}). The artifact has been modified since the "
            "development phase; retrain rather than evaluating it as frozen."
        )
    return {
        "artifact_dir": str(directory),
        "checkpoint": str(weights_path),
        "checkpoint_sha256": _sha256(weights_path),
        "checkpoint_bytes": weights_path.stat().st_size,
        "config_sha256": _sha256(config_path),
        "min_confidence": float(persisted),
        "threshold_selected_on": "validation",
        "threshold_objective": (
            f"F1 of accepted vs IoU>={training_report['threshold_selection']['min_iou_for_good']}"
        ),
        "selected_epoch": training_report.get("best_epoch"),
        "epoch_selected_on": "validation_mean_iou",
        "preprocessing": {
            "input_size": list(LOCALIZER_INPUT_SIZE),
            "resample_filter": config.get("resample_filter"),
            "input_normalisation": config.get("input_normalisation"),
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "normalisation_fitted_on_dataset": False,
        },
        "target": config.get("target"),
        "target_decision": config.get("target_decision"),
        "seed": config.get("seed"),
        "split_manifest_seed": config.get("split_manifest_seed"),
        "index_fingerprint": config.get("index_fingerprint"),
    }


# ------------------------------------------------------------------- test sweep
def evaluate_partition(
    localizer: LesionLocalizer,
    records: Dict[str, ImageRecord],
    image_ids: Sequence[str],
) -> Dict[str, Any]:
    """Localise every image and tally outcomes, keeping the populations separate.

    Correctness (IoU) is computed only where a usable ground-truth lesion box
    exists. Everywhere else the report says so instead of substituting acceptance
    for correctness.
    """
    annotated = LocalizationTally()
    unannotated = LocalizationTally()
    unusable = LocalizationTally()
    unreadable: Dict[str, str] = {}

    # Per-image rows for the annotated subset, so acceptance and IoU can be
    # cross-tabulated: "accepted but wrong" is the number that matters most and it
    # is invisible in separate acceptance and IoU summaries.
    annotated_rows: List[Dict[str, Any]] = []
    accepted_confidences: List[float] = []
    fallback_confidences: List[float] = []
    roi_source_counts: Dict[str, int] = {}

    for image_id in image_ids:
        record = records.get(image_id)
        if record is None:
            continue
        truth = _lesion_box_for(record)
        try:
            with Image.open(record.path) as handle:
                handle.load()
                result = localizer.localize(handle.convert("RGB"))
                overshoot = (
                    box_overshoot(truth, handle.width, handle.height)
                    if truth is not None
                    else 0.0
                )
        except (OSError, ValueError) as exc:
            # An image with no prediction is an image with no ROI. Recorded as a
            # rejection rather than dropped, so the denominator stays honest.
            unreadable[image_id] = f"{type(exc).__name__}: {exc}"
            continue

        if result.roi is not None:
            source = result.roi.source.value
            roi_source_counts[source] = roi_source_counts.get(source, 0) + 1
        if result.is_localized:
            accepted_confidences.append(float(result.confidence))
        elif result.used_fallback:
            fallback_confidences.append(float(result.confidence))

        if truth is None:
            unannotated.record(result)
            continue
        if overshoot > OUT_OF_FRAME_TOLERANCE:
            unusable.record(result)
            continue

        # IoU of the box actually used. A fallback centre crop is scored too:
        # its overlap with the lesion is what condition C costs the pipeline, and
        # omitting it would make the fallback look free.
        used_iou = box_iou(result.roi.box, truth) if result.roi is not None else 0.0
        accepted_iou = used_iou if result.is_localized else None
        annotated.record(result, iou=accepted_iou)
        annotated_rows.append(
            {
                "image_id": image_id,
                "diagnostic_class": record.diagnostic_class,
                "status": result.status.value,
                "confidence": round(float(result.confidence), 6),
                "iou_of_used_roi": round(float(used_iou), 6),
                "accepted": bool(result.is_localized),
                "reasons": list(result.reasons),
            }
        )

    groups = (annotated, unannotated, unusable)
    combined = LocalizationTally(
        localized=sum(g.localized for g in groups),
        fallback_used=sum(g.fallback_used for g in groups),
        rejected=sum(g.rejected for g in groups) + len(unreadable),
        reasons={
            key: sum(g.reasons.get(key, 0) for g in groups)
            for key in set().union(*(g.reasons for g in groups))
        },
        confidences=[c for g in groups for c in g.confidences],
        ious=annotated.ious,
    )
    if unreadable:
        combined.reasons["unreadable_image"] = combined.reasons.get(
            "unreadable_image", 0
        ) + len(unreadable)

    combined_summary = combined.summary()
    if "iou" in combined_summary:
        # The counts in this block span every image; the IoU block cannot, because
        # most of them have no ground-truth box. Saying so inside the block itself,
        # so a reader cannot take a whole-partition IoU from a whole-partition
        # section.
        combined_summary["iou"]["measured_on"] = "lesion_annotated_subset_only"
        combined_summary["iou"]["n"] = len(annotated.ious)

    return {
        "all_images": combined_summary,
        "lesion_annotated": _annotated_section(annotated, annotated_rows),
        "unannotated": _unannotated_section(unannotated),
        "unusable_annotation": {
            **unusable.summary(),
            "scored": False,
            "note": (
                "Annotation lies outside its own image (DEC-021). The image is "
                "localised; only its label is discarded, so no IoU is reported."
            ),
        },
        "unreadable_images": unreadable,
        "roi_source_counts": dict(sorted(roi_source_counts.items())),
        "confidence_by_outcome": {
            "accepted": _stats(accepted_confidences),
            "fallback": _stats(fallback_confidences),
        },
    }


def _annotated_section(
    tally: LocalizationTally, rows: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """The only population where "correct" is measurable, reported as such."""
    accepted = [r for r in rows if r["accepted"]]
    used_ious = [float(r["iou_of_used_roi"]) for r in rows]
    accepted_ious = [float(r["iou_of_used_roi"]) for r in accepted]
    verified = [iou for iou in accepted_ious if iou >= VERIFIED_IOU_FLOOR]

    section: Dict[str, Any] = {
        **tally.summary(),
        "verified_against_ground_truth": True,
        "iou_floor_for_verified": VERIFIED_IOU_FLOOR,
        # Acceptance and correctness cross-tabulated. "accepted_but_below_floor" is
        # the count requirement 5 exists to keep visible: a box whose confidence
        # cleared 0.30 and which is nevertheless wrong.
        "accepted": len(accepted),
        "accepted_and_verified": len(verified),
        "accepted_but_below_floor": len(accepted) - len(verified),
        "verified_share_of_all_annotated": (
            round(len(verified) / len(rows), 4) if rows else 0.0
        ),
        "verified_share_of_accepted": (
            round(len(verified) / len(accepted), 4) if accepted else 0.0
        ),
        "iou_of_accepted": _stats(accepted_ious),
        # The ROI the pipeline actually receives, fallbacks included. Lower than
        # iou_of_accepted by construction; it is what condition B+C delivers
        # together, and the honest number for "what the quantum path will see".
        "iou_of_used_roi_including_fallbacks": _stats(used_ious),
        "confidence_iou_correlation": _correlation(
            [float(r["confidence"]) for r in rows], used_ious
        ),
        "per_image": list(rows),
    }
    for floor in (0.25, 0.5, 0.75):
        share = (
            float(np.mean([iou >= floor for iou in accepted_ious]))
            if accepted_ious
            else 0.0
        )
        section[f"accepted_iou_at_least_{floor}"] = round(share, 4)
    by_class: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        bucket = by_class.setdefault(
            row["diagnostic_class"], {"n": 0, "accepted": 0, "ious": []}
        )
        bucket["n"] += 1
        bucket["accepted"] += int(row["accepted"])
        bucket["ious"].append(float(row["iou_of_used_roi"]))
    section["by_diagnostic_class"] = {
        name: {
            "n_images": bucket["n"],
            "accepted": bucket["accepted"],
            "iou_of_used_roi": _stats(bucket["ious"]),
        }
        for name, bucket in sorted(by_class.items())
    }
    return section


def _unannotated_section(tally: LocalizationTally) -> Dict[str, Any]:
    """Acceptance rates with the correctness claim explicitly withheld."""
    summary = tally.summary()
    return {
        **summary,
        "verified_against_ground_truth": False,
        "unverified_localization": True,
        "accepted": summary["localized"],
        "note": (
            "No ground-truth lesion box exists for these images, so "
            "'localized' means only that the confidence cleared the frozen "
            "threshold. It is an acceptance rate, not an accuracy: none of these "
            "boxes has been verified to contain a lesion."
        ),
    }


# ------------------------------------------------------------------ determinism
def check_determinism(
    localizer: LesionLocalizer,
    records: Dict[str, ImageRecord],
    image_ids: Sequence[str],
    *,
    sample: int = DETERMINISM_SAMPLE,
    reload_from: Optional[Path] = None,
) -> Dict[str, Any]:
    """Predict twice per image, and once more from a freshly loaded checkpoint.

    Two repeats catch a layer left in training mode; the reload catches a result
    that depends on process state rather than on the saved weights. Equality is
    exact, not approximate -- a deterministic path has no tolerance to spend.
    """
    checked: List[str] = []
    mismatches: List[Dict[str, Any]] = []
    reload_mismatches: List[Dict[str, Any]] = []
    reloaded = LesionLocalizer.load(reload_from) if reload_from is not None else None

    for image_id in list(image_ids)[:sample]:
        record = records.get(image_id)
        if record is None:
            continue
        with Image.open(record.path) as handle:
            handle.load()
            image = handle.convert("RGB")
        first = localizer.localize(image)
        second = localizer.localize(image)
        checked.append(image_id)
        if (
            first.predicted_box_normalised != second.predicted_box_normalised
            or first.confidence != second.confidence
            or first.status is not second.status
        ):
            mismatches.append(
                {
                    "image_id": image_id,
                    "first": first.describe(),
                    "second": second.describe(),
                }
            )
        if reloaded is not None:
            third = reloaded.localize(image)
            if (
                third.predicted_box_normalised != first.predicted_box_normalised
                or third.confidence != first.confidence
                or third.status is not first.status
            ):
                reload_mismatches.append(
                    {
                        "image_id": image_id,
                        "in_process": first.describe(),
                        "reloaded": third.describe(),
                    }
                )

    return {
        "n_images_checked": len(checked),
        "repeats_per_image": 2,
        "reloaded_checkpoint_compared": reloaded is not None,
        "identical_on_repeat": not mismatches,
        "identical_after_reload": not reload_mismatches,
        "mismatches": mismatches,
        "reload_mismatches": reload_mismatches,
    }


# ---------------------------------------------------------------- leakage audit
def audit_test_isolation(
    records: Sequence[ImageRecord],
    manifest: SplitManifest,
    *,
    training_report: Dict[str, Any],
    frozen: Dict[str, Any],
) -> Dict[str, Any]:
    """Check, from the artifacts, that nothing test-derived entered development.

    Every entry is a computed check rather than an assertion of intent, so a future
    change that starts reading test data fails this audit instead of passing it by
    inheriting a docstring.
    """
    dataset = build_localizer_dataset(records, manifest)
    fitted_ids = {s.image_id for s in dataset.train} | {
        s.image_id for s in dataset.validation
    }
    test_ids = set(manifest.images_in(TEST_PARTITION))
    test_patients = {
        r.patient_id for r in records if r.image_id in test_ids
    }
    development_patients = {s.patient_id for s in dataset.train} | {
        s.patient_id for s in dataset.validation
    }
    config = training_report.get("training_config", {})
    threshold_grid_n = training_report.get("threshold_selection", {}).get("n_good")

    checks = {
        "trainer_partitions_used": list(config.get("partitions_used", [])),
        "trainer_partitions_exclude_test": TEST_PARTITION
        not in set(config.get("partitions_used", [])),
        "trainable_partitions_constant": list(TRAINABLE_PARTITIONS),
        "test_partition_touched_flag": bool(config.get("test_partition_touched", True)),
        "n_test_images_in_fitted_set": len(fitted_ids & test_ids),
        "n_test_patients_in_development_patients": len(
            test_patients & development_patients
        ),
        "development_report_has_no_test_section": TEST_PARTITION
        not in set(training_report.get("dataset", {})),
        "threshold_selected_from_n_validation_images": threshold_grid_n,
        "threshold_selection_population": "validation_lesion_annotated",
        # Preprocessing is not fitted: the standardisation constants are the
        # published ImageNet ones, so no dataset statistic -- test or otherwise --
        # enters the input transform.
        "normalisation_constants_are_imagenet_published": (
            tuple(frozen["preprocessing"]["mean"]) == IMAGENET_MEAN
            and tuple(frozen["preprocessing"]["std"]) == IMAGENET_STD
        ),
        "normalisation_fitted_on_dataset": False,
        "checkpoint_selected_on": frozen["epoch_selected_on"],
        "threshold_selected_on": frozen["threshold_selected_on"],
    }
    failures = []
    if not checks["trainer_partitions_exclude_test"]:
        failures.append("the trainer recorded test among the partitions it used")
    if checks["test_partition_touched_flag"]:
        failures.append("the training config flags the test partition as touched")
    if checks["n_test_images_in_fitted_set"]:
        failures.append(
            f"{checks['n_test_images_in_fitted_set']} test images are in the fitted set"
        )
    if checks["n_test_patients_in_development_patients"]:
        failures.append(
            f"{checks['n_test_patients_in_development_patients']} test patients also "
            "appear among the images the localiser was fitted or selected on"
        )
    if not checks["development_report_has_no_test_section"]:
        failures.append("the development report contains a test section")
    if not checks["normalisation_constants_are_imagenet_published"]:
        failures.append("input normalisation constants are not the published ImageNet ones")
    checks["passed"] = not failures
    checks["failures"] = failures
    return checks


# ------------------------------------------------------------------ entry point
def evaluate(
    *,
    config: Optional[Settings] = None,
    index: Optional[SmartOmIndex] = None,
    save: bool = True,
    determinism_sample: int = DETERMINISM_SAMPLE,
) -> Dict[str, Any]:
    """Evaluate the frozen localiser on the test partition and write the report.

    Nothing here selects, fits or tunes. The threshold and checkpoint come from
    disk, are checked against the development record, and are reported alongside the
    result so the number can always be traced to the artifact that produced it.
    """
    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)
    directory = store.localizer_dir(LOCALIZER_VERSION)
    report_path = directory / "training_report.json"
    if not report_path.exists():
        raise LocalizerEvaluationError(
            f"No development report at {report_path}; the frozen threshold cannot be "
            "verified against the validation sweep that chose it."
        )
    training_report = json.loads(report_path.read_text(encoding="utf-8"))
    frozen = frozen_artifact_description(directory, training_report=training_report)

    localizer = LesionLocalizer.load(directory)
    if float(localizer.min_confidence) != float(frozen["min_confidence"]):
        raise LocalizerEvaluationError(
            f"Loaded localiser applies {localizer.min_confidence} but the frozen "
            f"configuration records {frozen['min_confidence']}."
        )

    dataset_index = index if index is not None else build_index(cfg.DATASET_ROOT)
    manifest = SplitManifest.load(store.split_manifest_path)
    manifest.assert_no_patient_leakage(dataset_index.records)
    records = {r.image_id: r for r in dataset_index.records}
    test_ids = sorted(manifest.images_in(TEST_PARTITION))

    logger.info(
        "Frozen localiser %s, threshold %.2f -- evaluating %d test images",
        localizer.version,
        localizer.min_confidence,
        len(test_ids),
    )
    started = time.perf_counter()
    test_section = evaluate_partition(localizer, records, test_ids)
    duration = time.perf_counter() - started

    determinism = check_determinism(
        localizer,
        records,
        test_ids,
        sample=determinism_sample,
        reload_from=directory,
    )
    audit = audit_test_isolation(
        dataset_index.records,
        manifest,
        training_report=training_report,
        frozen=frozen,
    )

    rates_path = directory / "localization_rates.json"
    development_rates = (
        json.loads(rates_path.read_text(encoding="utf-8")).get("partitions", {})
        if rates_path.exists()
        else {}
    )

    report: Dict[str, Any] = {
        "localizer_version": localizer.version,
        "frozen_artifact": frozen,
        # ---- development: copied from the training record, never recomputed here,
        # so this section cannot drift from what actually drove the decisions.
        "development": {
            "partitions": list(TRAINABLE_PARTITIONS),
            "dataset": training_report.get("dataset"),
            "selected_epoch": training_report.get("best_epoch"),
            "validation": training_report.get("final_validation"),
            "threshold_selection": training_report.get("threshold_selection"),
            "localization_rates": development_rates,
            "duration_seconds": training_report.get("duration_seconds"),
        },
        # ---- final: the only section computed from test data.
        "final_test": {
            "partition": TEST_PARTITION,
            "n_images_in_partition": len(test_ids),
            "n_patients": len({records[i].patient_id for i in test_ids if i in records}),
            "applied_threshold": float(localizer.min_confidence),
            "threshold_changed_for_this_evaluation": False,
            **test_section,
            "duration_seconds": round(duration, 1),
        },
        "conditions": _condition_breakdown(test_section),
        "determinism": determinism,
        "test_isolation_audit": audit,
        "go_no_go": {
            "documented_numeric_criteria": None,
            "note": (
                "No numeric acceptance criteria for the ROI stage exist in "
                "REQUIREMENTS.md, DESIGN.md, ARCHITECTURE.md, PROJECT.md or "
                "DECISIONS.md. The pass/fail decision therefore requires a project "
                "decision and is not asserted here."
            ),
        },
    }

    if save:
        out = directory / "test_evaluation_report.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(out)
        logger.info("Wrote %s", out)
    return report


def _condition_breakdown(test_section: Dict[str, Any]) -> Dict[str, Any]:
    """Conditions A, B and C as they stand on the test partition.

    A is annotation-derived and needs no localiser, so its count is the images that
    *have* a usable annotation. B and C come from the localiser's own outcome, which
    is why they are read off the ROI source rather than recomputed: the source is
    what the pixel cache records, so the report and the cache cannot disagree.
    """
    annotated = test_section["lesion_annotated"]
    sources = test_section["roi_source_counts"]
    return {
        "A_oracle_lesion_roi": {
            "definition": "ROI from the annotator's polygons; no localiser involved.",
            "n_test_images_with_usable_lesion_annotation": annotated["n_images"],
            "verifiable": True,
        },
        "B_predicted_roi": {
            "definition": "Localiser box accepted at the frozen threshold.",
            "roi_source": RoiSource.PREDICTED.value,
            "n": sources.get(RoiSource.PREDICTED.value, 0),
            "n_verified_against_annotation": annotated["accepted_and_verified"],
            "n_accepted_but_below_iou_floor": annotated["accepted_but_below_floor"],
        },
        "C_fallback_or_rejected": {
            "definition": (
                "Localisation ran and its output was refused; the centre crop was "
                "used and is labelled predicted_rejected."
            ),
            "roi_source": RoiSource.PREDICTED_REJECTED.value,
            "n": sources.get(RoiSource.PREDICTED_REJECTED.value, 0),
            "n_with_no_roi_at_all": test_section["all_images"]["rejected"],
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Final held-out evaluation of the frozen lesion-ROI localiser."
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Print the report without writing it."
    )
    parser.add_argument(
        "--determinism-sample", type=int, default=DETERMINISM_SAMPLE
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = evaluate(
        save=not args.no_save, determinism_sample=args.determinism_sample
    )
    # per_image is provenance for the artifact, not console output.
    trimmed = json.loads(json.dumps(report))
    trimmed["final_test"]["lesion_annotated"].pop("per_image", None)
    print(json.dumps(trimmed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
