"""Train the MobileNet lesion-ROI localiser on the training partition only.

Split discipline, which is the part of this module that matters most:

* the network is fitted on lesion-annotated images whose patient is in **train**;
* the **validation** partition selects the epoch and the confidence threshold;
* the **test** partition is never read here. There is no code path in this module
  that can reach it -- ``_subset`` refuses any partition name other than train or
  validation, so touching test would be an exception rather than an oversight.

The target is the lesion bounding box in normalised coordinates, per DEC-020. Only
318 images in the dataset carry a lesion polygon, 215 of them in the training
partition, so the training set is small; that is a consequence of the target
choice, not an accident, and it is why the run is short and heavily augmented.

Determinism: every seed that affects the result is set from one ``--seed``, cuDNN
is irrelevant (CPU only), and the augmentation draws from a seeded generator rather
than the global one. Two runs with the same seed produce the same weights. That is
asserted in ``tests/test_localizer.py`` on a short run rather than claimed here.

Loss: smooth-L1 on the four normalised coordinates, plus binary cross-entropy on
the confidence head against the IoU the predicted box achieved. The confidence
target is recomputed each step from the current prediction and detached, so the
head learns to predict localisation quality without back-propagating through its
own target -- otherwise the cheapest way to lower the loss would be to make the box
worse and the confidence agree with it.

Usage::

    python -m backend.training.train_localizer
    python -m backend.training.train_localizer --epochs 40 --seed 42
    python -m backend.training.train_localizer --report-only
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from backend.core.config import Settings, settings as default_settings
from backend.dataset.smartom import ImageRecord, SmartOmIndex, build_index
from backend.dataset.split import SplitManifest
from backend.ml.artifacts import ArtifactStore
from backend.ml.localizer import (
    LOCALIZER_INPUT_SIZE,
    LOCALIZER_VERSION,
    MAX_BOX_AREA_FRACTION,
    MIN_BOX_EDGE_FRACTION,
    MIN_CONFIDENCE,
    LesionLocalizer,
    LocalizationTally,
    box_iou,
    build_localizer_network,
    normalised_box_from_pixels,
    preprocess_for_localizer,
)
from backend.ml.roi import polygons_bounding_box

logger = logging.getLogger(__name__)

TRAINABLE_PARTITIONS = ("train", "validation")
"""The only partitions this module may read. ``test`` is excluded by design."""

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 3.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_SEED = 42

OUT_OF_FRAME_TOLERANCE = 0.10
"""How far past the frame edge a lesion annotation may reach and still be used.

Five of the 318 lesion annotations in this dataset extend beyond their image: two
overshoot by under 6% and three by 24-88%. The small overshoots are consistent with
annotator imprecision at the boundary or a mild resize between annotation and
storage, and clipping them to the edge loses nothing. The large ones mean the
polygon and the image disagree about their coordinate system -- ``Ca 4`` has a box
1,082 px wide on a 548 px image -- and clipping *those* would fabricate a label,
turning "the annotation is unusable" into "the lesion fills the frame". They are
dropped and counted instead. Deliberately not silently clamped: the count is
reported in the training summary so the training-set size always accounts for every
annotated image.
"""

CONFIDENCE_LOSS_WEIGHT = 0.5
"""Relative weight of the confidence head against the box regression.

Below 1 on purpose: the box is the product and the confidence is a diagnostic on
it. Weighting them equally made the network trade box accuracy for a
better-calibrated estimate of its own inaccuracy, which is the wrong bargain.
"""

# Threshold candidates swept on the validation partition. Deliberately a coarse
# grid: 52 validation images cannot resolve finer differences, and pretending
# otherwise would be fitting the threshold to noise.
THRESHOLD_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70)


class LocalizerTrainingError(RuntimeError):
    """Raised when the localiser cannot be trained from the current artifacts."""


# ------------------------------------------------------------------------ data
@dataclass
class LocalizerSample:
    """One supervised example: an image path and its normalised lesion box."""

    image_id: str
    patient_id: str
    path: Path
    diagnostic_class: str
    box_normalised: Tuple[float, float, float, float]
    source_size: Tuple[int, int]


@dataclass
class LocalizerDataset:
    """The train/validation samples, plus the images that have no lesion box.

    ``unannotated`` is carried rather than discarded because it is the population
    evaluation condition C describes: images the localiser must still handle at
    inference time and for which no ground-truth box exists to score it against.
    """

    train: List[LocalizerSample] = field(default_factory=list)
    validation: List[LocalizerSample] = field(default_factory=list)
    unannotated: Dict[str, List[str]] = field(default_factory=dict)
    unusable_annotations: Dict[str, str] = field(default_factory=dict)
    """Lesion-annotated images excluded, and why. Never silently absent."""

    def counts(self) -> Dict[str, Any]:
        return {
            "train": {
                "n_images": len(self.train),
                "n_patients": len({s.patient_id for s in self.train}),
                "classes": _class_counts(self.train),
            },
            "validation": {
                "n_images": len(self.validation),
                "n_patients": len({s.patient_id for s in self.validation}),
                "classes": _class_counts(self.validation),
            },
            "unannotated_by_partition": {
                name: len(ids) for name, ids in sorted(self.unannotated.items())
            },
            "excluded_unusable_annotations": dict(sorted(self.unusable_annotations.items())),
        }


def _class_counts(samples: Sequence[LocalizerSample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample.diagnostic_class] = counts.get(sample.diagnostic_class, 0) + 1
    return dict(sorted(counts.items()))


def _lesion_box_for(record: ImageRecord) -> Optional[Tuple[int, int, int, int]]:
    """The union lesion bounding box, or ``None`` when there is no lesion polygon.

    Region polygons are deliberately not consulted. See DEC-020: they are 1,531/1,536
    normal and cover 97% of the frame, so including them would make box size a label
    proxy.
    """
    return polygons_bounding_box(record.lesion_polygons)


def box_overshoot(
    box: Sequence[float], width: int, height: int
) -> float:
    """How far ``box`` reaches outside a ``width`` x ``height`` frame, as a fraction.

    Zero when the box is inside. Shared by the trainer and the rate report so both
    apply one definition of "usable annotation" -- otherwise the report would score
    the localiser against boxes the trainer refused to learn from, and the IoU it
    published would be measured against a label the model was never shown.
    """
    normalised = normalised_box_from_pixels(box, width, height)
    return max(0.0, max(normalised) - 1.0, -min(normalised))


def build_localizer_dataset(
    records: Sequence[ImageRecord], manifest: SplitManifest
) -> LocalizerDataset:
    """Collect lesion-annotated samples for train and validation only.

    Raises:
        LocalizerTrainingError: either partition ends up empty, which would make the
            run meaningless rather than merely small.
    """
    dataset = LocalizerDataset()
    by_partition: Dict[str, List[LocalizerSample]] = {"train": [], "validation": []}
    unannotated: Dict[str, List[str]] = {}

    for record in records:
        partition = manifest.image_to_partition.get(record.image_id)
        if partition is None:
            continue
        box = _lesion_box_for(record)
        if box is None:
            unannotated.setdefault(partition, []).append(record.image_id)
            continue
        if partition not in TRAINABLE_PARTITIONS:
            # A test-partition lesion box exists but is not read. Counted only as
            # "has an annotation", never loaded, never used for selection.
            continue

        if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
            dataset.unusable_annotations[record.image_id] = f"degenerate_box:{box}"
            continue
        try:
            with Image.open(record.path) as handle:
                source_size = (int(handle.width), int(handle.height))
        except (OSError, ValueError) as exc:
            dataset.unusable_annotations[record.image_id] = f"{type(exc).__name__}: {exc}"
            continue

        normalised = normalised_box_from_pixels(box, source_size[0], source_size[1])
        overshoot = box_overshoot(box, source_size[0], source_size[1])
        if overshoot > OUT_OF_FRAME_TOLERANCE:
            # The polygon and the image disagree about their coordinate system.
            # Clipping would fabricate a label -- see OUT_OF_FRAME_TOLERANCE.
            dataset.unusable_annotations[record.image_id] = (
                f"annotation_outside_frame:box={box} image={source_size[0]}x"
                f"{source_size[1]} overshoot={overshoot:.3f}"
            )
            continue
        clipped = tuple(float(min(1.0, max(0.0, v))) for v in normalised)
        if clipped[2] - clipped[0] <= 0.0 or clipped[3] - clipped[1] <= 0.0:
            dataset.unusable_annotations[record.image_id] = (
                f"box_collapsed_after_clipping:{clipped}"
            )
            continue

        by_partition[partition].append(
            LocalizerSample(
                image_id=record.image_id,
                patient_id=record.patient_id,
                path=Path(record.path),
                diagnostic_class=record.diagnostic_class,
                box_normalised=clipped,  # type: ignore[arg-type]
                source_size=source_size,
            )
        )

    # Sorted by image id so the sample order does not depend on filesystem walk
    # order; the shuffle that follows is seeded, and it can only be reproducible if
    # what it shuffles is deterministic to begin with.
    dataset.train = sorted(by_partition["train"], key=lambda s: s.image_id)
    dataset.validation = sorted(by_partition["validation"], key=lambda s: s.image_id)
    dataset.unannotated = {k: sorted(v) for k, v in unannotated.items()}

    if not dataset.train:
        raise LocalizerTrainingError(
            "No lesion-annotated images in the training partition. The localiser "
            "target is the lesion bounding box (DEC-020), so there is nothing to fit."
        )
    if not dataset.validation:
        raise LocalizerTrainingError(
            "No lesion-annotated images in the validation partition; the epoch and "
            "confidence threshold could not be selected without touching test."
        )

    overlap = {s.patient_id for s in dataset.train} & {
        s.patient_id for s in dataset.validation
    }
    if overlap:  # pragma: no cover - the manifest guard should prevent this
        raise LocalizerTrainingError(
            f"LEAKAGE: {len(overlap)} patients appear in both the localiser train and "
            f"validation sets (first: {sorted(overlap)[0]})."
        )
    return dataset


def load_batch(
    samples: Sequence[LocalizerSample],
    *,
    augment: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(inputs, targets)`` as ``(n, 3, 224, 224)`` and ``(n, 4)``.

    Augmentation is horizontal flip plus brightness jitter only. No rotation or
    crop: a crop would move the lesion out of frame for some draws and there is no
    principled label for "the box is no longer visible", and rotation would need the
    box rotated too, which for an axis-aligned target means growing it -- changing
    the thing being learnt.
    """
    inputs: List[np.ndarray] = []
    targets: List[Tuple[float, float, float, float]] = []
    for sample in samples:
        with Image.open(sample.path) as handle:
            handle.load()
            image = handle.convert("RGB")
        box = sample.box_normalised
        if augment and rng is not None:
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                box = (1.0 - box[2], box[1], 1.0 - box[0], box[3])
            factor = float(rng.uniform(0.85, 1.15))
            if abs(factor - 1.0) > 1e-6:
                array = np.asarray(image, dtype=np.float32) * factor
                image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        inputs.append(preprocess_for_localizer(image))
        targets.append(box)
    return (
        np.stack(inputs).astype(np.float32),
        np.asarray(targets, dtype=np.float32).reshape(-1, 4),
    )


# -------------------------------------------------------------------- training
def _set_all_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _evaluate(network, samples: Sequence[LocalizerSample], batch_size: int) -> Dict[str, Any]:
    """Box IoU and confidence on a held-out set, without acceptance thresholds.

    Threshold-free on purpose: the threshold is chosen afterwards from these same
    numbers, and mixing the two would select an operating point using a metric that
    already assumed one.
    """
    import torch

    network.eval()
    ious: List[float] = []
    confidences: List[float] = []
    predictions: List[Tuple[float, float, float, float]] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            inputs, targets = load_batch(chunk)
            output = torch.sigmoid(network(torch.from_numpy(inputs))).numpy()
            for row, target in zip(output, targets):
                box = tuple(float(v) for v in row[:4])
                predictions.append(box)  # type: ignore[arg-type]
                ious.append(box_iou(box, target))
                confidences.append(float(row[4]))

    iou_array = np.asarray(ious, dtype=np.float64)
    confidence_array = np.asarray(confidences, dtype=np.float64)
    # Correlation between predicted confidence and achieved IoU: the confidence
    # head's actual job. Reported because a confidence that does not track quality
    # is worse than none -- it would authorise bad crops.
    if iou_array.size > 1 and iou_array.std() > 0 and confidence_array.std() > 0:
        correlation = float(np.corrcoef(confidence_array, iou_array)[0, 1])
    else:
        correlation = 0.0
    return {
        "n": int(iou_array.size),
        "mean_iou": float(iou_array.mean()) if iou_array.size else 0.0,
        "median_iou": float(np.median(iou_array)) if iou_array.size else 0.0,
        "iou_at_least_0.25": float((iou_array >= 0.25).mean()) if iou_array.size else 0.0,
        "iou_at_least_0.5": float((iou_array >= 0.5).mean()) if iou_array.size else 0.0,
        "mean_confidence": float(confidence_array.mean()) if confidence_array.size else 0.0,
        "confidence_iou_correlation": correlation,
        "ious": ious,
        "confidences": confidences,
        "predictions": predictions,
    }


def select_confidence_threshold(
    evaluation: Dict[str, Any], *, min_iou: float = 0.25
) -> Dict[str, Any]:
    """Choose the acceptance threshold on the **validation** partition.

    The objective is the F1 of "accepted" against "actually localised well enough"
    (IoU >= ``min_iou``), not raw accept rate: a threshold of zero accepts
    everything and would look perfect on recall alone. 0.25 rather than the usual
    0.5 because the crop is fed to a 256x256 resize -- a box overlapping a quarter
    of the lesion still delivers lesion pixels, whereas a detection benchmark would
    not count it.
    """
    ious = np.asarray(evaluation["ious"], dtype=np.float64)
    confidences = np.asarray(evaluation["confidences"], dtype=np.float64)
    good = ious >= min_iou

    rows: List[Dict[str, Any]] = []
    for threshold in THRESHOLD_GRID:
        accepted = confidences >= threshold
        true_positive = int(np.sum(accepted & good))
        false_positive = int(np.sum(accepted & ~good))
        false_negative = int(np.sum(~accepted & good))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        rows.append(
            {
                "threshold": threshold,
                "accept_rate": round(float(accepted.mean()), 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "mean_iou_of_accepted": (
                    round(float(ious[accepted].mean()), 4) if accepted.any() else 0.0
                ),
            }
        )

    # Ties broken toward the higher threshold: with 52 validation images several
    # thresholds routinely share an F1, and the stricter one converts ambiguous
    # cases into explicit fallbacks instead of silent bad crops.
    best = max(rows, key=lambda r: (r["f1"], r["threshold"]))
    return {
        "min_iou_for_good": min_iou,
        "grid": rows,
        "selected_threshold": best["threshold"],
        "selected": best,
        "n_good": int(good.sum()),
    }


def train(
    *,
    config: Optional[Settings] = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
    index: Optional[SmartOmIndex] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """Fit the localiser and record everything needed to reproduce the run.

    Model selection is by best validation mean IoU, evaluated after every epoch;
    the best state is what gets persisted, not the last one. The test partition is
    not read.
    """
    import torch
    import torch.nn.functional as functional

    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)
    _set_all_seeds(seed)
    torch.set_num_threads(8)

    dataset_index = index if index is not None else build_index(cfg.DATASET_ROOT)
    manifest = SplitManifest.load(store.split_manifest_path)
    manifest.assert_no_patient_leakage(dataset_index.records)
    dataset = build_localizer_dataset(dataset_index.records, manifest)

    logger.info(
        "Localiser data: %d train / %d validation lesion-annotated images "
        "(%d / %d patients)",
        len(dataset.train),
        len(dataset.validation),
        len({s.patient_id for s in dataset.train}),
        len({s.patient_id for s in dataset.validation}),
    )

    network = build_localizer_network(pretrained=True)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    rng = np.random.default_rng(seed)

    order = list(range(len(dataset.train)))
    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, Any]] = None
    best_iou = -1.0
    best_epoch = -1
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        network.train()
        rng.shuffle(order)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            chunk = [dataset.train[i] for i in rows]
            inputs, targets = load_batch(chunk, augment=True, rng=rng)
            input_tensor = torch.from_numpy(inputs)
            target_tensor = torch.from_numpy(targets)

            output = torch.sigmoid(network(input_tensor))
            box_loss = functional.smooth_l1_loss(output[:, :4], target_tensor, beta=0.05)

            # Confidence target = the IoU this prediction actually achieved.
            # Detached so the head cannot lower the loss by making the box worse.
            with torch.no_grad():
                achieved = torch.tensor(
                    [
                        box_iou(box, target)
                        for box, target in zip(
                            output[:, :4].numpy(), target_tensor.numpy()
                        )
                    ],
                    dtype=torch.float32,
                )
            confidence_loss = functional.binary_cross_entropy(output[:, 4], achieved)

            loss = box_loss + CONFIDENCE_LOSS_WEIGHT * confidence_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        scheduler.step()

        evaluation = _evaluate(network, dataset.validation, batch_size)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(epoch_loss / max(1, n_batches), 6),
                "val_mean_iou": round(evaluation["mean_iou"], 4),
                "val_iou_at_least_0.25": round(evaluation["iou_at_least_0.25"], 4),
                "val_iou_at_least_0.5": round(evaluation["iou_at_least_0.5"], 4),
                "val_confidence_iou_correlation": round(
                    evaluation["confidence_iou_correlation"], 4
                ),
                "learning_rate": round(float(scheduler.get_last_lr()[0]), 8),
            }
        )
        logger.info(
            "  epoch %2d  loss %.4f  val IoU %.4f  (>=0.25: %.2f, >=0.5: %.2f)",
            epoch,
            epoch_loss / max(1, n_batches),
            evaluation["mean_iou"],
            evaluation["iou_at_least_0.25"],
            evaluation["iou_at_least_0.5"],
        )

        if evaluation["mean_iou"] > best_iou:
            best_iou = evaluation["mean_iou"]
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in network.state_dict().items()
            }

    if best_state is None:  # pragma: no cover - epochs >= 1 guarantees a state
        raise LocalizerTrainingError("Training produced no model state.")
    network.load_state_dict(best_state)
    network.eval()

    final = _evaluate(network, dataset.validation, batch_size)
    threshold = select_confidence_threshold(final)
    duration = time.perf_counter() - started

    training_config = {
        "localizer_version": LOCALIZER_VERSION,
        "target": "lesion_bounding_box_normalised",
        "target_decision": "DEC-020",
        "architecture": "mobilenet_v3_small",
        "pretrained_weights": "IMAGENET1K_V1",
        "head": "Linear(576,256)->Hardswish->Dropout(0.2)->Linear(256,5)",
        "output_activation": "sigmoid",
        "outputs": "4 normalised box coordinates + 1 confidence",
        "input_size": list(LOCALIZER_INPUT_SIZE),
        "input_normalisation": "imagenet_mean_std",
        "resample_filter": "bilinear",
        "augmentation": ["horizontal_flip_p0.5", "brightness_jitter_0.85_1.15"],
        "loss": "smooth_l1(beta=0.05) + 0.5 * bce(confidence, detached_iou)",
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "lr_schedule": "cosine_annealing",
        "grad_clip_norm": 5.0,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "torch_version": torch.__version__,
        "device": "cpu",
        "python_version": platform.python_version(),
        "min_confidence": threshold["selected_threshold"],
        "min_box_edge_fraction": MIN_BOX_EDGE_FRACTION,
        "max_box_area_fraction": MAX_BOX_AREA_FRACTION,
        "default_min_confidence_constant": MIN_CONFIDENCE,
        "split_manifest_seed": manifest.seed,
        "index_fingerprint": manifest.index_fingerprint,
        "partitions_used": list(TRAINABLE_PARTITIONS),
        "test_partition_touched": False,
    }

    summary: Dict[str, Any] = {
        "localizer_version": LOCALIZER_VERSION,
        "training_config": training_config,
        "dataset": dataset.counts(),
        "best_epoch": best_epoch,
        "best_val_mean_iou": round(best_iou, 4),
        "final_validation": {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in final.items()
            if key not in ("ious", "confidences", "predictions")
        },
        "threshold_selection": threshold,
        "history": history,
        "duration_seconds": round(duration, 1),
    }

    if save:
        directory = store.localizer_dir(LOCALIZER_VERSION)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(network.state_dict(), directory / "localizer.pt")
        (directory / "localizer.json").write_text(
            json.dumps(training_config, indent=2), encoding="utf-8"
        )
        (directory / "training_report.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        summary["artifact_dir"] = str(directory)
        logger.info("Saved localiser to %s", directory)

    return summary


# ------------------------------------------------------------------- reporting
def report_localization_rates(
    *,
    config: Optional[Settings] = None,
    index: Optional[SmartOmIndex] = None,
    partitions: Sequence[str] = TRAINABLE_PARTITIONS,
    limit_per_partition: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the trained localiser over whole images and tally every outcome.

    Reports per partition: localised / fallback / rejected counts, the confidence
    distribution, and -- for the lesion-annotated subset only -- the achieved IoU.
    The annotated and unannotated populations are reported separately and never
    merged into one IoU, because a high localisation rate on unannotated images is
    unverifiable by construction: there is no box to compare against, so "localised"
    there means only "the network was confident", which is a weaker claim.

    Images whose annotation lies outside the frame are counted in their own group
    rather than scored: they were excluded from training for the same reason
    (:data:`OUT_OF_FRAME_TOLERANCE`), and measuring IoU against a label the model
    was never shown would report the annotation's error as the localiser's.

    ``partitions`` defaults to train and validation. Passing ``"test"`` is possible
    and is the intended route for final evaluation, but it is never the default, so
    reporting on test is an explicit act.
    """
    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)
    localizer = LesionLocalizer.load(store.localizer_dir(LOCALIZER_VERSION))

    dataset_index = index if index is not None else build_index(cfg.DATASET_ROOT)
    manifest = SplitManifest.load(store.split_manifest_path)
    records = {r.image_id: r for r in dataset_index.records}

    report: Dict[str, Any] = {
        "localizer_version": localizer.version,
        "min_confidence": localizer.min_confidence,
        "partitions": {},
    }
    for name in partitions:
        image_ids = sorted(manifest.images_in(name))
        if limit_per_partition is not None:
            image_ids = image_ids[:limit_per_partition]
        annotated = LocalizationTally()
        unannotated = LocalizationTally()
        unusable = LocalizationTally()
        unreadable: Dict[str, str] = {}
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
                # Recorded, not skipped: an image that could not be read is an
                # image with no ROI, which is a rejection the rates must show.
                unreadable[image_id] = f"{type(exc).__name__}: {exc}"
                continue
            if truth is None:
                unannotated.record(result)
            elif overshoot > OUT_OF_FRAME_TOLERANCE:
                unusable.record(result)
            else:
                iou = (
                    box_iou(result.roi.box, truth)
                    if result.is_localized and result.roi is not None
                    else None
                )
                annotated.record(result, iou=iou)

        groups = (annotated, unannotated, unusable)
        combined = LocalizationTally(
            localized=sum(g.localized for g in groups),
            fallback_used=sum(g.fallback_used for g in groups),
            # Unreadable images have no prediction at all, so they are rejections
            # of the localisation attempt and belong in the denominator.
            rejected=sum(g.rejected for g in groups) + len(unreadable),
            reasons={
                key: sum(g.reasons.get(key, 0) for g in groups)
                for key in set().union(*(g.reasons for g in groups))
            },
            confidences=[c for g in groups for c in g.confidences],
            ious=annotated.ious,
        )
        if unreadable:
            combined.reasons["unreadable_image"] = (
                combined.reasons.get("unreadable_image", 0) + len(unreadable)
            )
        report["partitions"][name] = {
            "all_images": combined.summary(),
            "lesion_annotated": annotated.summary(),
            "not_annotated": unannotated.summary(),
            "unusable_annotation": unusable.summary(),
            "unreadable_images": unreadable,
        }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the MobileNet lesion-ROI localiser (train partition only)."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--no-save", action="store_true", help="Train without writing artifacts."
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip training; tally localisation rates for an existing localiser.",
    )
    parser.add_argument(
        "--report-partitions",
        nargs="*",
        default=list(TRAINABLE_PARTITIONS),
        help="Partitions to tally. 'test' is allowed but never the default.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.report_only:
        payload: Dict[str, Any] = report_localization_rates(
            partitions=args.report_partitions
        )
    else:
        payload = train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            save=not args.no_save,
        )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
