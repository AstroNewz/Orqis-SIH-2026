"""The MobileNet ROI localiser: outcome accounting, determinism, split discipline.

Two groups, separable:

* geometry, acceptance checks and tallying -- pure functions and a randomly
  initialised network, so these always run;
* the trained artifact and the dataset subset -- skipped without the dataset or a
  trained localiser.

The emphasis is on the requirement that a localisation *failure* never becomes a
silent success. That is not a numerical property, so it cannot be caught by an IoU
metric; it has to be asserted structurally, on the status the localiser returns and
on the :class:`RoiSource` it attaches.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from backend.dataset.smartom import discover_dataset_root
from backend.ml.artifacts import ArtifactError, ArtifactStore
from backend.ml.localizer import (
    LOCALIZER_VERSION,
    MAX_BOX_AREA_FRACTION,
    MIN_BOX_EDGE_FRACTION,
    LesionLocalizer,
    LocalizationResult,
    LocalizationStatus,
    LocalizationTally,
    box_iou,
    build_localizer_network,
    check_predicted_box,
    normalised_box_from_pixels,
    pixel_box_from_normalised,
    preprocess_for_localizer,
)
from backend.ml.types import RoiSource
from backend.training.train_localizer import (
    OUT_OF_FRAME_TOLERANCE,
    TRAINABLE_PARTITIONS,
    select_confidence_threshold,
)


requires_dataset = pytest.mark.skipif(
    discover_dataset_root() is None, reason="Clinical dataset not present."
)


@pytest.fixture(scope="module")
def photo() -> Image.Image:
    """A deterministic non-uniform colour image at a non-square size."""
    rng = np.random.default_rng(11)
    return Image.fromarray(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))


@pytest.fixture(scope="module")
def untrained() -> LesionLocalizer:
    """A randomly initialised localiser.

    Its boxes are meaningless, which is exactly what is wanted for testing the
    plumbing: every assertion here is about outcome accounting, not accuracy.
    """
    return LesionLocalizer(
        build_localizer_network(pretrained=False), metadata={"min_confidence": 0.35}
    )


# ------------------------------------------------------------------- geometry
def test_normalised_and_pixel_coordinates_round_trip():
    box = (100, 50, 400, 300)
    normalised = normalised_box_from_pixels(box, 640, 480)
    assert pixel_box_from_normalised(normalised, 640, 480) == box


def test_normalised_coordinates_are_resolution_independent():
    """The same lesion in two resolutions of one photo must give one target.

    This is why the head regresses fractions: otherwise a 4000x3000 capture and a
    640x480 capture of the same lesion would be different labels.
    """
    small = normalised_box_from_pixels((64, 48, 320, 240), 640, 480)
    large = normalised_box_from_pixels((400, 300, 2000, 1500), 4000, 3000)
    assert small == pytest.approx(large)


def test_inverted_predictions_are_ordered_not_accepted_as_negative_width():
    """An unconstrained head can predict x1 < x0; a negative width must not survive.

    Left unsorted this produces an empty PIL crop several stages later, where the
    cause is no longer visible.
    """
    x0, y0, x1, y1 = pixel_box_from_normalised((0.8, 0.9, 0.2, 0.3), 640, 480)
    assert x1 > x0 and y1 > y0
    assert (x0, y0, x1, y1) == (128, 144, 512, 432)


def test_predictions_outside_the_frame_are_clamped_to_it():
    assert pixel_box_from_normalised((-0.5, -0.5, 1.5, 1.5), 640, 480) == (0, 0, 640, 480)


def test_iou_is_zero_for_disjoint_and_one_for_identical():
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(5 / 15)


def test_iou_of_a_degenerate_box_is_zero_not_a_division_error():
    assert box_iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0
    assert box_iou((5, 5, 5, 5), (5, 5, 5, 5)) == 0.0


# ----------------------------------------------------------- acceptance checks
def test_a_good_box_passes_every_check():
    assert check_predicted_box((0.2, 0.2, 0.6, 0.6), 0.9) == []


def test_low_confidence_is_named_with_its_value():
    reasons = check_predicted_box((0.2, 0.2, 0.6, 0.6), 0.1, min_confidence=0.35)
    assert len(reasons) == 1 and reasons[0].startswith("low_confidence:")


def test_a_collapsed_box_is_rejected_rather_than_upsampled():
    """A sliver stretched to 256x256 would invent detail that was never captured."""
    edge = MIN_BOX_EDGE_FRACTION / 2
    reasons = check_predicted_box((0.5, 0.5, 0.5 + edge, 0.5 + edge), 0.99)
    assert any(r.startswith("degenerate_box") for r in reasons)


def test_a_full_frame_box_does_not_count_as_a_localisation():
    """Predicting the whole image is doing nothing, and must not score as success.

    This is the specific failure mode condition C exists to expose: a localiser that
    returns the frame would show a 100% localisation rate while never localising.
    """
    reasons = check_predicted_box((0.0, 0.0, 1.0, 1.0), 0.99)
    assert any(r.startswith("box_covers_whole_frame") for r in reasons)
    assert MAX_BOX_AREA_FRACTION < 1.0


def test_a_box_entirely_off_frame_is_its_own_reason():
    reasons = check_predicted_box((1.2, 1.2, 1.6, 1.6), 0.99)
    assert "box_outside_frame" in reasons


def test_non_finite_predictions_are_refused_immediately():
    assert check_predicted_box((0.2, 0.2, np.nan, 0.6), 0.9) == ["non_finite_prediction"]
    assert check_predicted_box((0.2, 0.2, 0.6, 0.6), np.inf) == ["non_finite_prediction"]


# ------------------------------------------------------- the three outcomes
def test_a_missing_model_is_rejected_and_yields_no_roi(photo):
    """MANDATORY: with no localiser there is no ROI, and that is reported as such.

    Not a fallback: a fallback means "we ran and could not use the answer". Silently
    centre-cropping here would make an unconfigured deployment look like a working
    one.
    """
    result = LesionLocalizer(None).localize(photo)
    assert result.status is LocalizationStatus.REJECTED
    assert result.roi is None
    assert result.reasons == ("localizer_unavailable",)
    assert not result.is_localized


def test_a_rejected_prediction_falls_back_but_is_never_labelled_predicted(
    untrained, photo
):
    """MANDATORY (requirement 9): a failure must stay distinguishable downstream.

    The fallback ROI is returned so the pipeline can proceed, but it carries
    ``PREDICTED_REJECTED``, which is a different value from ``CENTER_CROP``. That
    distinction is what keeps the 610 never-localised centre crops separable from
    localisation failures in the statistics.
    """
    forced = LesionLocalizer(untrained.network, metadata={"min_confidence": 1.01})
    result = forced.localize(photo)
    assert result.status is LocalizationStatus.FALLBACK_USED
    assert result.roi is not None
    assert result.roi.source is RoiSource.PREDICTED_REJECTED
    assert result.roi.source is not RoiSource.CENTER_CROP
    assert result.reasons
    assert not result.is_localized and result.used_fallback


def test_an_accepted_prediction_is_labelled_predicted(untrained, photo):
    accepting = LesionLocalizer(untrained.network, metadata={"min_confidence": 0.0})
    result = accepting.localize(photo)
    if result.status is LocalizationStatus.LOCALIZED:
        assert result.roi is not None
        assert result.roi.source is RoiSource.PREDICTED
        assert result.reasons == ()
    else:
        # A random network can legitimately emit a degenerate box; the point is
        # that it is then not called a localisation.
        assert result.roi is not None
        assert result.roi.source is RoiSource.PREDICTED_REJECTED


def test_the_raw_prediction_is_recorded_even_when_rejected(untrained, photo):
    """A rejected box is still evidence; discarding it would make failures unanalysable."""
    forced = LesionLocalizer(untrained.network, metadata={"min_confidence": 1.01})
    result = forced.localize(photo)
    assert result.predicted_box_normalised is not None
    assert len(result.predicted_box_normalised) == 4
    described = result.describe()
    assert described["localization_status"] == "fallback_used"
    assert described["roi_source"] == "predicted_rejected"
    assert described["localization_reasons"]


def test_the_fallback_roi_lies_inside_the_source_image(untrained, photo):
    forced = LesionLocalizer(untrained.network, metadata={"min_confidence": 1.01})
    roi = forced.localize(photo).roi
    assert roi is not None
    assert 0 <= roi.x0 < roi.x1 <= photo.width
    assert 0 <= roi.y0 < roi.y1 <= photo.height
    assert roi.source_width == photo.width and roi.source_height == photo.height


# ------------------------------------------------------------- determinism
def test_the_same_image_gives_the_same_box_every_time(untrained, photo):
    """MANDATORY (requirement 7): inference is reproducible.

    Dropout in the head would break this if the network were left in training mode,
    which is the realistic way it would regress.
    """
    first = untrained.localize(photo)
    for _ in range(3):
        again = untrained.localize(photo)
        assert again.predicted_box_normalised == first.predicted_box_normalised
        assert again.confidence == first.confidence
        assert again.status is first.status


def test_batched_and_single_predictions_agree(untrained, photo):
    """Batching must not change the answer, or a cache would differ from live inference."""
    other = photo.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    boxes, confidences = untrained.predict_normalised([photo, other])
    single_a, conf_a = untrained.predict_normalised([photo])
    single_b, conf_b = untrained.predict_normalised([other])
    assert np.allclose(boxes[0], single_a[0], atol=1e-6)
    assert np.allclose(boxes[1], single_b[0], atol=1e-6)
    assert confidences[0] == pytest.approx(conf_a[0], abs=1e-6)
    assert confidences[1] == pytest.approx(conf_b[0], abs=1e-6)


@requires_dataset
def test_two_training_runs_with_the_same_seed_agree_exactly():
    """MANDATORY (requirement 7): the *training* run is reproducible, not just inference.

    Two epochs is enough: what could break determinism -- unseeded shuffling,
    unseeded augmentation, a global RNG shared with another import -- diverges on the
    first batch, not gradually. A longer run would cost minutes to test the same
    property.
    """
    from backend.dataset.smartom import cached_index
    from backend.training.train_localizer import train

    index = cached_index()
    first = train(epochs=2, seed=7, index=index, save=False)
    second = train(epochs=2, seed=7, index=index, save=False)
    assert first["history"] == second["history"]
    assert first["final_validation"] == second["final_validation"]
    assert (
        first["threshold_selection"]["selected_threshold"]
        == second["threshold_selection"]["selected_threshold"]
    )


def test_two_networks_built_with_the_same_seed_are_identical():
    import torch

    torch.manual_seed(7)
    first = build_localizer_network(pretrained=False)
    torch.manual_seed(7)
    second = build_localizer_network(pretrained=False)
    for (name, a), (_, b) in zip(
        first.state_dict().items(), second.state_dict().items()
    ):
        assert torch.equal(a, b), name


def test_an_empty_batch_returns_empty_arrays_not_an_error(untrained):
    boxes, confidences = untrained.predict_normalised([])
    assert boxes.shape == (0, 4) and confidences.shape == (0,)


# ------------------------------------------------------------- preprocessing
def test_preprocessing_produces_the_pretraining_shape(photo):
    array = preprocess_for_localizer(photo)
    assert array.shape == (3, 224, 224)
    assert array.dtype == np.float32


def test_preprocessing_uses_the_whole_frame_not_a_crop():
    """The localiser must see the entire image; pre-cropping would beg its question.

    Checked by putting a marker in a corner that a centre crop would remove and
    confirming the standardised tensor still differs there from a blank image.
    """
    blank = Image.new("RGB", (640, 480), (10, 10, 10))
    marked = blank.copy()
    marked.paste(Image.new("RGB", (60, 60), (250, 250, 250)), (0, 0))
    assert not np.allclose(
        preprocess_for_localizer(blank)[:, :20, :20],
        preprocess_for_localizer(marked)[:, :20, :20],
    )


def test_every_image_mode_reaches_three_channels():
    for mode, size in (("L", (100, 80)), ("RGBA", (100, 80)), ("P", (100, 80))):
        assert preprocess_for_localizer(Image.new(mode, size)).shape == (3, 224, 224)


# ------------------------------------------------------------------ tallying
def _result(status: LocalizationStatus, confidence: float, *reasons: str):
    return LocalizationResult(
        status=status,
        roi=None,
        confidence=confidence,
        predicted_box_normalised=None,
        reasons=tuple(reasons),
    )


def test_the_tally_accounts_for_every_image():
    """MANDATORY: localised + fallback + rejected must equal the total.

    An image that fell out of all three buckets would be an image whose outcome was
    never reported, which is the accounting failure requirement 8 forbids.
    """
    tally = LocalizationTally()
    tally.record(_result(LocalizationStatus.LOCALIZED, 0.9))
    tally.record(_result(LocalizationStatus.LOCALIZED, 0.7))
    tally.record(_result(LocalizationStatus.FALLBACK_USED, 0.1, "low_confidence:0.1<0.35"))
    tally.record(_result(LocalizationStatus.REJECTED, 0.0, "localizer_unavailable"))
    assert tally.total == 4
    summary = tally.summary()
    assert summary["localized"] + summary["fallback_used"] + summary["rejected"] == 4
    assert summary["localization_rate"] == 0.5
    assert summary["fallback_rate"] == 0.25
    assert summary["rejection_rate"] == 0.25


def test_failure_reasons_aggregate_by_kind_not_by_value():
    """Two low-confidence failures at different values are one reason, counted twice."""
    tally = LocalizationTally()
    tally.record(_result(LocalizationStatus.FALLBACK_USED, 0.1, "low_confidence:0.10<0.35"))
    tally.record(_result(LocalizationStatus.FALLBACK_USED, 0.3, "low_confidence:0.30<0.35"))
    tally.record(_result(LocalizationStatus.FALLBACK_USED, 0.9, "degenerate_box:0.01x0.01"))
    assert tally.summary()["failure_reasons"] == {
        "low_confidence": 2,
        "degenerate_box": 1,
    }


def test_a_rejected_image_contributes_no_confidence():
    """There is no prediction to have confidence in, so the mean must not include it."""
    tally = LocalizationTally()
    tally.record(_result(LocalizationStatus.LOCALIZED, 0.8))
    tally.record(_result(LocalizationStatus.REJECTED, 0.0, "localizer_unavailable"))
    assert tally.summary()["confidence"]["mean"] == 0.8


def test_an_empty_tally_reports_zeroes_rather_than_dividing_by_zero():
    summary = LocalizationTally().summary()
    assert summary["n_images"] == 0
    assert summary["localization_rate"] == 0.0
    assert "confidence" not in summary and "iou" not in summary


def test_iou_is_only_tallied_where_a_ground_truth_box_exists():
    tally = LocalizationTally()
    tally.record(_result(LocalizationStatus.LOCALIZED, 0.9), iou=0.6)
    tally.record(_result(LocalizationStatus.LOCALIZED, 0.9), iou=None)
    summary = tally.summary()
    assert summary["localized"] == 2
    assert summary["iou"]["mean"] == 0.6
    assert summary["iou"]["at_least_0.5"] == 1.0


# --------------------------------------------------- threshold selection
def test_the_threshold_is_chosen_on_f1_not_on_accept_rate():
    """A threshold of zero accepts everything; recall alone would always prefer it."""
    selection = select_confidence_threshold(
        {
            # Confident-and-good, plus confident-and-bad: a pure accept-rate
            # objective cannot tell these apart.
            "ious": [0.8, 0.7, 0.6, 0.05, 0.02, 0.01],
            "confidences": [0.9, 0.8, 0.7, 0.3, 0.25, 0.2],
        }
    )
    assert selection["selected_threshold"] >= 0.35
    assert selection["selected"]["precision"] == 1.0
    assert selection["n_good"] == 3


def test_ties_break_toward_the_stricter_threshold():
    """With a tie, prefer the threshold that converts doubt into an explicit fallback."""
    selection = select_confidence_threshold(
        {"ious": [0.9, 0.9], "confidences": [0.95, 0.95]}
    )
    tied = [r["threshold"] for r in selection["grid"] if r["f1"] == 1.0]
    assert selection["selected_threshold"] == max(tied)


# ------------------------------------------------------------ split discipline
def test_the_trainer_can_only_read_train_and_validation():
    """MANDATORY (requirements 1-3): test is unreachable from the training module."""
    assert TRAINABLE_PARTITIONS == ("train", "validation")
    assert "test" not in TRAINABLE_PARTITIONS


def test_a_localiser_version_cannot_escape_the_artifact_root():
    store = ArtifactStore.from_settings()
    for bad in ("../escape", "a/b", "..", ""):
        with pytest.raises(ArtifactError):
            store.localizer_dir(bad)


def test_a_missing_artifact_names_the_command_that_trains_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_localizer"):
        LesionLocalizer.load(tmp_path / "absent")


# ============================ THE TRAINED LOCALISER ==========================
_store = ArtifactStore.from_settings()
requires_localizer = pytest.mark.skipif(
    discover_dataset_root() is None
    or not (_store.localizer_dir(LOCALIZER_VERSION) / "localizer.pt").exists(),
    reason="Localiser not trained (run `python -m backend.training.train_localizer`).",
)


@pytest.fixture(scope="module")
def trained() -> LesionLocalizer:
    return LesionLocalizer.load(_store.localizer_dir(LOCALIZER_VERSION))


@requires_dataset
def test_no_patient_crosses_the_localiser_train_validation_boundary():
    """MANDATORY (requirements 4-5): patient separation holds on the localiser subset.

    The localiser reads a *subset* of the manifest -- only lesion-annotated images --
    which is a second, independent join against it. A manifest with no leakage can
    still be joined incorrectly, so the guarantee is re-asserted on the data the
    localiser actually fits, not inferred from the manifest being clean.
    """
    from backend.dataset.smartom import cached_index
    from backend.dataset.split import SplitManifest
    from backend.training.train_localizer import build_localizer_dataset

    index = cached_index()
    manifest = SplitManifest.load(_store.split_manifest_path)
    dataset = build_localizer_dataset(index.records, manifest)

    train_patients = {s.patient_id for s in dataset.train}
    validation_patients = {s.patient_id for s in dataset.validation}
    assert not (train_patients & validation_patients)
    assert len(train_patients) > 50 and len(validation_patients) > 10

    # And no image is in both, which a patient-level check alone would not catch if
    # the same image id were emitted twice.
    train_ids = {s.image_id for s in dataset.train}
    validation_ids = {s.image_id for s in dataset.validation}
    assert not (train_ids & validation_ids)
    assert len(train_ids) == len(dataset.train)


@requires_dataset
def test_the_localiser_never_loads_a_test_partition_image():
    """MANDATORY (requirement 3): test images are absent from what the trainer builds."""
    from backend.dataset.smartom import cached_index
    from backend.dataset.split import SplitManifest
    from backend.training.train_localizer import build_localizer_dataset

    index = cached_index()
    manifest = SplitManifest.load(_store.split_manifest_path)
    dataset = build_localizer_dataset(index.records, manifest)
    test_images = set(manifest.images_in("test"))
    test_patients = set(manifest.patients_in("test"))
    assert test_images
    for sample in list(dataset.train) + list(dataset.validation):
        assert sample.image_id not in test_images
        assert sample.patient_id not in test_patients


@requires_dataset
def test_unusable_annotations_are_named_rather_than_clipped_away():
    """DEC-021: an annotation that overshoots its frame is excluded and reported.

    Clipping a 97%-overshoot box would turn "this annotation is broken" into "the
    lesion fills the frame", which is the DEC-020 leak arriving through a rounding
    convention. The exclusion has to be visible in the counts or the training-set
    size stops accounting for every annotated image.
    """
    from backend.dataset.smartom import cached_index
    from backend.dataset.split import SplitManifest
    from backend.training.train_localizer import build_localizer_dataset

    index = cached_index()
    manifest = SplitManifest.load(_store.split_manifest_path)
    dataset = build_localizer_dataset(index.records, manifest)
    excluded = dataset.unusable_annotations
    assert excluded, "expected the known out-of-frame annotations to be excluded"
    for image_id, reason in excluded.items():
        assert "overshoot=" in reason or "degenerate" in reason, (image_id, reason)
    counts = dataset.counts()
    assert counts["excluded_unusable_annotations"] == excluded


@requires_localizer
def test_the_trained_localiser_records_its_configuration(trained):
    """MANDATORY (requirement 6): preprocessing and training config are persisted."""
    metadata = trained.metadata
    for key in (
        "architecture",
        "pretrained_weights",
        "input_size",
        "input_normalisation",
        "augmentation",
        "loss",
        "optimizer",
        "learning_rate",
        "epochs",
        "batch_size",
        "seed",
        "torch_version",
        "min_confidence",
        "split_manifest_seed",
        "index_fingerprint",
    ):
        assert key in metadata, key
    assert metadata["target"] == "lesion_bounding_box_normalised"
    assert metadata["target_decision"] == "DEC-020"
    assert metadata["partitions_used"] == list(TRAINABLE_PARTITIONS)
    assert metadata["test_partition_touched"] is False


@requires_localizer
def test_the_selected_threshold_came_from_validation(trained):
    import json

    report = json.loads(
        (_store.localizer_dir(LOCALIZER_VERSION) / "training_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["threshold_selection"]["selected_threshold"] == trained.min_confidence
    assert report["final_validation"]["n"] == report["dataset"]["validation"]["n_images"]
    # The report must not contain a test-partition entry at all.
    assert "test" not in report["dataset"]


@requires_localizer
def test_the_trained_localiser_is_deterministic_on_a_real_image(trained):
    from backend.dataset.smartom import cached_index

    record = next(r for r in cached_index().records if r.lesion_polygons)
    with Image.open(record.path) as handle:
        handle.load()
        image = handle.convert("RGB")
    first = trained.localize(image)
    for _ in range(3):
        again = trained.localize(image)
        assert again.predicted_box_normalised == first.predicted_box_normalised
        assert again.confidence == first.confidence


@requires_localizer
def test_the_localiser_beats_a_centre_crop_on_held_out_lesions(trained):
    """The localiser has to be better than the fallback it replaces.

    Otherwise condition B is not worth reporting: the honest thing would be to keep
    centre-cropping. Measured on validation only -- test stays untouched until final
    evaluation.
    """
    from backend.dataset.smartom import cached_index
    from backend.dataset.split import SplitManifest
    from backend.ml.roi import center_crop_box, polygons_bounding_box

    index = cached_index()
    manifest = SplitManifest.load(_store.split_manifest_path)
    predicted, centred = [], []
    for record in index.records:
        if manifest.image_to_partition.get(record.image_id) != "validation":
            continue
        truth = polygons_bounding_box(record.lesion_polygons)
        if truth is None:
            continue
        with Image.open(record.path) as handle:
            handle.load()
            if max(truth[2] / handle.width, truth[3] / handle.height) - 1.0 > (
                OUT_OF_FRAME_TOLERANCE
            ):
                continue  # unusable annotation, excluded from training too
            result = trained.localize(handle.convert("RGB"))
            centred.append(
                box_iou(center_crop_box(handle.width, handle.height), truth)
            )
        predicted.append(
            box_iou(result.roi.box, truth) if result.roi is not None else 0.0
        )

    assert len(predicted) > 30
    assert float(np.mean(predicted)) > float(np.mean(centred)), (
        f"Predicted ROI mean IoU {np.mean(predicted):.4f} does not beat the centre "
        f"crop's {np.mean(centred):.4f}; the localiser adds nothing."
    )


# ====================== THE FROZEN TEST EVALUATION ===========================
# These protect the invariants of the *final* evaluation: that the artifact under
# test is the one development produced, that nothing test-derived can flow back
# into a fitting decision, and that "accepted" is never published as "correct".


def test_the_evaluation_module_carries_no_fitting_machinery():
    """Structural: there is nothing in this module that *could* tune on test.

    A docstring promising not to tune is worth little. The evaluator imports no
    optimiser, defines no threshold grid and has no training entry point, so a
    future change that reintroduces one has to add the import -- and fail here.
    """
    import inspect

    from backend.training import evaluate_localizer

    source = inspect.getsource(evaluate_localizer)
    for forbidden in ("optim.", "AdamW", "backward(", "THRESHOLD_GRID", "loss"):
        assert forbidden not in source, forbidden
    assert not hasattr(evaluate_localizer, "train")
    assert not hasattr(evaluate_localizer, "select_confidence_threshold")


def test_the_evaluator_reads_the_threshold_rather_than_choosing_one(tmp_path, monkeypatch):
    """Requirement 11: the applied threshold comes from disk, not from test results.

    Asserted by mutating the persisted config: if the evaluator recomputed or
    re-selected a threshold, a config that disagrees with the validation sweep would
    pass unnoticed. Instead it must refuse to treat the artifact as frozen.
    """
    import json

    from backend.training.evaluate_localizer import (
        LocalizerEvaluationError,
        frozen_artifact_description,
    )

    directory = tmp_path / "carescan-localizer-1"
    directory.mkdir()
    (directory / "localizer.pt").write_bytes(b"weights")
    report = {
        "threshold_selection": {"selected_threshold": 0.30, "min_iou_for_good": 0.25},
        "best_epoch": 17,
    }
    (directory / "localizer.json").write_text(json.dumps({"min_confidence": 0.30}))
    frozen = frozen_artifact_description(directory, training_report=report)
    assert frozen["min_confidence"] == 0.30
    assert frozen["threshold_selected_on"] == "validation"

    # Retuned after development closed -- exactly what requirement 11 forbids.
    (directory / "localizer.json").write_text(json.dumps({"min_confidence": 0.45}))
    with pytest.raises(LocalizerEvaluationError, match="modified since"):
        frozen_artifact_description(directory, training_report=report)


def test_an_unannotated_population_is_never_reported_as_verified():
    """Requirement 4: acceptance without a ground-truth box is labelled as such."""
    from backend.training.evaluate_localizer import _unannotated_section

    tally = LocalizationTally()
    for _ in range(7):
        tally.record(_result(LocalizationStatus.LOCALIZED, 0.9))
    tally.record(_result(LocalizationStatus.FALLBACK_USED, 0.1, "low_confidence:0.1<0.3"))

    section = _unannotated_section(tally)
    assert section["verified_against_ground_truth"] is False
    assert section["unverified_localization"] is True
    assert section["accepted"] == 7
    assert "iou" not in section, "an unannotated population has no IoU to report"
    assert "acceptance rate, not an accuracy" in section["note"]


def test_confidence_above_the_threshold_is_not_reported_as_correctness():
    """Requirement 5, as a number: accepted-but-wrong is counted, not absorbed.

    Two of the three rows clear the threshold; one of those has an IoU below the
    floor. The section must show one verified and one accepted-but-below-floor rather
    than two successes.
    """
    from backend.training.evaluate_localizer import _annotated_section

    tally = LocalizationTally()
    tally.record(_result(LocalizationStatus.LOCALIZED, 0.80), iou=0.61)
    tally.record(_result(LocalizationStatus.LOCALIZED, 0.75), iou=0.04)
    tally.record(
        _result(LocalizationStatus.FALLBACK_USED, 0.10, "low_confidence:0.1<0.3")
    )
    rows = [
        {
            "image_id": "a",
            "diagnostic_class": "opmd",
            "status": "localized",
            "confidence": 0.80,
            "iou_of_used_roi": 0.61,
            "accepted": True,
            "reasons": [],
        },
        {
            "image_id": "b",
            "diagnostic_class": "opmd",
            "status": "localized",
            "confidence": 0.75,
            "iou_of_used_roi": 0.04,
            "accepted": True,
            "reasons": [],
        },
        {
            "image_id": "c",
            "diagnostic_class": "oral_cancer",
            "status": "fallback_used",
            "confidence": 0.10,
            "iou_of_used_roi": 0.30,
            "accepted": False,
            "reasons": ["low_confidence:0.1<0.3"],
        },
    ]

    section = _annotated_section(tally, rows)
    assert section["accepted"] == 2
    assert section["accepted_and_verified"] == 1
    assert section["accepted_but_below_floor"] == 1
    assert section["verified_share_of_accepted"] == 0.5
    assert section["verified_against_ground_truth"] is True
    # The ROI the pipeline receives includes the fallback, so the two IoU summaries
    # must not be the same number.
    assert section["iou_of_accepted"]["mean"] != (
        section["iou_of_used_roi_including_fallbacks"]["mean"]
    )


def test_a_correlation_that_cannot_be_computed_is_not_reported_as_zero():
    """A zero correlation is a finding; "not enough data" is a different one."""
    from backend.training.evaluate_localizer import _correlation

    assert _correlation([0.4, 0.6], [0.2, 0.8])["reason"] == "fewer_than_3_pairs"
    assert _correlation([0.5] * 5, [0.1, 0.2, 0.3, 0.4, 0.5])["reason"] == "zero_variance"
    assert _correlation([0.5] * 5, [0.1, 0.2, 0.3, 0.4, 0.5])["value"] is None
    computed = _correlation([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4])
    assert computed["value"] == 1.0 and computed["reason"] is None


def test_no_numeric_go_no_go_criteria_are_invented():
    """The evaluator must not assert a pass it has no documented basis for.

    If the project later records numeric criteria, this test is the reminder to wire
    them in rather than leaving the verdict to prose.
    """
    import inspect

    from backend.training import evaluate_localizer

    source = inspect.getsource(evaluate_localizer)
    assert '"documented_numeric_criteria": None' in source
    assert "requires a project" in source


@requires_localizer
def test_the_frozen_evaluation_touched_only_the_test_partition(trained):
    """Requirement 8, read back off the persisted report.

    The audit is recomputed at evaluation time from the manifest and the training
    record; this asserts it ran, passed, and found no test image or patient in
    anything that was fitted or selected.
    """
    import json

    path = _store.localizer_dir(LOCALIZER_VERSION) / "test_evaluation_report.json"
    if not path.exists():
        pytest.skip("Frozen evaluation not run (`python -m backend.training.evaluate_localizer`).")
    report = json.loads(path.read_text(encoding="utf-8"))

    audit = report["test_isolation_audit"]
    assert audit["passed"] is True and audit["failures"] == []
    assert audit["n_test_images_in_fitted_set"] == 0
    assert audit["n_test_patients_in_development_patients"] == 0
    assert audit["trainer_partitions_exclude_test"] is True
    assert audit["test_partition_touched_flag"] is False
    assert audit["normalisation_fitted_on_dataset"] is False
    assert audit["checkpoint_selected_on"] == "validation_mean_iou"
    assert audit["threshold_selected_on"] == "validation"

    # The threshold that produced the final numbers is the one validation chose.
    assert report["frozen_artifact"]["min_confidence"] == trained.min_confidence
    assert report["final_test"]["applied_threshold"] == trained.min_confidence
    assert report["final_test"]["threshold_changed_for_this_evaluation"] is False
    # Development metrics are carried from the training record, not recomputed on
    # test, so the two sections cannot be confused for one another.
    assert report["development"]["partitions"] == list(TRAINABLE_PARTITIONS)
    assert "test" not in report["development"]["localization_rates"]


@requires_localizer
def test_the_frozen_report_keeps_annotated_and_unannotated_apart(trained):
    """Requirement 3 and 4: two populations, two claims, never one average."""
    import json

    path = _store.localizer_dir(LOCALIZER_VERSION) / "test_evaluation_report.json"
    if not path.exists():
        pytest.skip("Frozen evaluation not run.")
    report = json.loads(path.read_text(encoding="utf-8"))
    final = report["final_test"]

    annotated = final["lesion_annotated"]
    unannotated = final["unannotated"]
    assert annotated["verified_against_ground_truth"] is True
    assert unannotated["verified_against_ground_truth"] is False
    assert unannotated["unverified_localization"] is True
    assert "iou" not in unannotated

    # Every image is accounted for exactly once across the populations.
    counted = (
        annotated["n_images"]
        + unannotated["n_images"]
        + final["unusable_annotation"]["n_images"]
        + len(final["unreadable_images"])
    )
    assert counted == final["n_images_in_partition"]

    # The whole-partition IoU block is labelled with the subset it came from, so a
    # reader cannot lift a 381-image IoU out of a section headed "all_images".
    iou_block = final["all_images"]["iou"]
    assert iou_block["measured_on"] == "lesion_annotated_subset_only"
    assert iou_block["n"] == annotated["accepted"]
    assert iou_block["n"] < final["n_images_in_partition"]


@requires_localizer
def test_the_three_conditions_survive_into_the_frozen_report(trained):
    """Requirement 6: A, B and C are still distinguishable after final evaluation."""
    import json

    path = _store.localizer_dir(LOCALIZER_VERSION) / "test_evaluation_report.json"
    if not path.exists():
        pytest.skip("Frozen evaluation not run.")
    report = json.loads(path.read_text(encoding="utf-8"))
    conditions = report["conditions"]
    final = report["final_test"]

    assert conditions["B_predicted_roi"]["roi_source"] == RoiSource.PREDICTED.value
    assert (
        conditions["C_fallback_or_rejected"]["roi_source"]
        == RoiSource.PREDICTED_REJECTED.value
    )
    # B and C partition the images that received an ROI, and C is not empty --
    # otherwise the fallback subset would have quietly disappeared.
    assert conditions["C_fallback_or_rejected"]["n"] > 0
    assert (
        conditions["B_predicted_roi"]["n"] + conditions["C_fallback_or_rejected"]["n"]
        == final["n_images_in_partition"] - final["all_images"]["rejected"]
    )
    assert conditions["A_oracle_lesion_roi"]["verifiable"] is True
    # Accepted-but-wrong stays visible at condition level, not just in the tally.
    assert "n_accepted_but_below_iou_floor" in conditions["B_predicted_roi"]


@requires_localizer
def test_the_frozen_localiser_is_deterministic_across_a_reload(trained):
    """Requirement 7: the same image gives the same box, in-process and after reload.

    The reload matters separately from the repeat: a prediction that depends on
    process state would repeat consistently within one run and still differ from what
    the saved checkpoint produces in deployment.
    """
    import json

    path = _store.localizer_dir(LOCALIZER_VERSION) / "test_evaluation_report.json"
    if not path.exists():
        pytest.skip("Frozen evaluation not run.")
    determinism = json.loads(path.read_text(encoding="utf-8"))["determinism"]
    assert determinism["identical_on_repeat"] is True
    assert determinism["identical_after_reload"] is True
    assert determinism["reloaded_checkpoint_compared"] is True
    assert determinism["n_images_checked"] >= 5
    assert determinism["mismatches"] == [] and determinism["reload_mismatches"] == []
