"""The V1 pixel path: localised ROI -> 65,536 pixels -> 16-qubit amplitudes.

Every assertion here is on a property the V1 architecture decision states
explicitly, so a regression reads as a named violation rather than as a numeric
drift somebody has to interpret:

* the register is 16 qubits and the input is 65,536 pixels, fixed;
* the raw 0-255 representation survives alongside the normalised one;
* an all-black region is a controlled error, never a fabricated state;
* a wrong-length vector is refused, never padded or truncated.

The real-dataset tests skip on a checkout without the clinical images. The rest
build their input in-process and always run.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from backend.dataset.smartom import discover_dataset_root
from backend.ml.pixel_pipeline import (
    NORM_TOLERANCE,
    V1_PIXEL_COUNT,
    V1_PIXEL_PREPROCESSING_VERSION,
    V1_QUBIT_COUNT,
    V1_RESAMPLE,
    V1_ROI_EDGE_PX,
    PixelPreprocessingError,
    V1PixelVector,
    ZeroImageError,
    amplitudes_from_grayscale,
    grayscale_roi,
    preprocess_v1_pixels,
)
from backend.ml.roi import extract_roi
from backend.ml.types import RoiSource
from tests.fixtures import synthetic_capture


@pytest.fixture(scope="module")
def capture() -> Image.Image:
    return synthetic_capture()


@pytest.fixture(scope="module")
def encoded(capture) -> V1PixelVector:
    return preprocess_v1_pixels(capture)


# ------------------------------------------------------------- the fixed geometry
def test_v1_is_fixed_at_sixteen_qubits():
    """MANDATORY: V1 is 16 qubits, and the dimensions must actually agree.

    Asserted as three separate facts rather than one, because the failure this
    guards against is a mismatch *between* them -- a pipeline that resizes to
    128x128 while still calling itself 16-qubit would satisfy any one of these
    checks alone.
    """
    assert V1_QUBIT_COUNT == 16
    assert V1_PIXEL_COUNT == 65_536
    assert V1_PIXEL_COUNT == 2**V1_QUBIT_COUNT
    assert V1_ROI_EDGE_PX == 256
    assert V1_ROI_EDGE_PX**2 == V1_PIXEL_COUNT


def test_the_qubit_count_is_a_constant_not_a_setting():
    """The width must not be reachable through configuration.

    A settings field would make it possible to run 12 qubits while the response
    still says 16, which is the specific outcome the decision forbids. So the
    constant lives in the module, and application settings must not shadow it.
    """
    from backend.core.config import Settings

    assert not any(
        "V1_QUBIT" in name or name == "V1_PIXEL_COUNT" for name in Settings.model_fields
    ), "V1 qubit count must not be an environment-overridable setting."


def test_reported_qubit_count_matches_the_vector_that_was_built(encoded):
    """The claim and the artifact must agree, which is the whole point of the rule."""
    assert encoded.qubit_count == 16
    assert encoded.n_pixels == V1_PIXEL_COUNT
    assert encoded.amplitudes.shape == (V1_PIXEL_COUNT,)
    assert encoded.describe()["qubit_count"] == 16
    assert encoded.describe()["n_pixels"] == V1_PIXEL_COUNT


# --------------------------------------------------------------- the two encodings
def test_raw_grayscale_representation_is_preserved(encoded):
    """The 0-255 representation must survive, not be overwritten by the normalised one."""
    raw = encoded.raw_grayscale
    assert raw.dtype == np.uint8
    assert raw.shape == (V1_ROI_EDGE_PX, V1_ROI_EDGE_PX)
    assert 0 <= int(raw.min()) and int(raw.max()) <= 255
    # A real capture must not have collapsed to a constant, or the "image" carries
    # no information and the rest of the pipeline is measuring nothing.
    assert int(raw.max()) > int(raw.min())


def test_scaled_and_amplitude_vectors_derive_from_the_raw_one_exactly(encoded):
    """Each representation must be reconstructible from the previous one.

    Equality is exact rather than approximate: these are two divisions, not a
    fitted transform, so any discrepancy means a stage is doing something it does
    not document.
    """
    expected_scaled = encoded.raw_grayscale.astype(np.float64).ravel(order="C") / 255.0
    assert np.array_equal(encoded.scaled, expected_scaled)
    assert np.array_equal(
        encoded.amplitudes, expected_scaled / np.linalg.norm(expected_scaled)
    )
    assert encoded.scaled_l2_norm == pytest.approx(
        float(np.linalg.norm(expected_scaled)), abs=1e-12
    )


def test_amplitudes_are_unit_norm(encoded):
    """The defining property of an amplitude encoding."""
    total = float(np.sum(encoded.amplitudes**2))
    assert abs(total - 1.0) < NORM_TOLERANCE
    assert np.all(np.isfinite(encoded.amplitudes))
    # Grayscale is non-negative, so every amplitude is too -- a negative one would
    # mean a sign was introduced somewhere it should not have been.
    assert np.all(encoded.amplitudes >= 0.0)


def test_normalisation_does_not_write_over_the_raw_pixels(encoded):
    """In-place normalisation would destroy the traceable representation.

    The arrays are read-only for exactly this reason, so the attempt fails loudly
    rather than silently corrupting the provenance the response reports.
    """
    for array in (encoded.raw_grayscale, encoded.scaled, encoded.amplitudes):
        with pytest.raises(ValueError):
            array[0] = 0


def test_brightness_is_recoverable_after_normalisation(capture):
    """L2 normalisation discards overall brightness, so it must be recorded.

    Two ROIs differing only by a scale factor encode to the *same* amplitudes.
    That is correct behaviour for amplitude encoding, but it means the response
    could not otherwise distinguish a dim capture from a bright one.
    """
    bright = preprocess_v1_pixels(capture)
    dimmed = Image.fromarray((np.asarray(capture.convert("RGB")) // 2).astype(np.uint8))
    dim = preprocess_v1_pixels(dimmed)

    # Halving every pixel roughly halves the norm but leaves the direction alone.
    assert dim.scaled_l2_norm < bright.scaled_l2_norm * 0.75
    assert np.allclose(dim.amplitudes, bright.amplitudes, atol=2e-3)


# ------------------------------------------------------------------ the zero vector
def test_all_black_region_raises_a_controlled_error():
    """MANDATORY: an all-zero image must not produce a fabricated normalised vector.

    A uniform superposition would be a legal quantum state that encodes nothing
    about the patient, and it would flow through the classifier to a
    confident-looking probability. The pipeline must stop instead.
    """
    black = Image.new("RGB", (600, 600), color=(0, 0, 0))
    with pytest.raises(ZeroImageError) as excinfo:
        preprocess_v1_pixels(black)
    message = str(excinfo.value)
    # The message has to be actionable for whoever is holding the phone.
    assert "re-acquire" in message.lower()
    assert str(V1_PIXEL_COUNT) in message


def test_zero_vector_error_is_raised_by_the_vector_stage_directly():
    """The guard lives in the encoding stage, not only in the image-reading path."""
    with pytest.raises(ZeroImageError):
        amplitudes_from_grayscale(
            np.zeros((V1_ROI_EDGE_PX, V1_ROI_EDGE_PX), dtype=np.uint8)
        )


def test_a_single_nonzero_pixel_still_encodes():
    """Near-zero is not zero. The error must be exact, not a brightness threshold.

    Otherwise a legitimately very dark capture would be rejected as a failure, and
    the boundary of that rejection would be an undocumented magic number.
    """
    raw = np.zeros((V1_ROI_EDGE_PX, V1_ROI_EDGE_PX), dtype=np.uint8)
    raw[100, 100] = 1
    _, amplitudes, norm = amplitudes_from_grayscale(raw)
    assert norm > 0.0
    assert float(np.sum(amplitudes**2)) == pytest.approx(1.0, abs=NORM_TOLERANCE)
    # All the weight sits on the one lit pixel: a basis state.
    assert amplitudes[100 * V1_ROI_EDGE_PX + 100] == pytest.approx(1.0)


# -------------------------------------------------------------- shape enforcement
@pytest.mark.parametrize("size", [1024, V1_PIXEL_COUNT - 1, V1_PIXEL_COUNT + 1])
def test_wrong_length_is_refused_never_padded_or_truncated(size):
    """A dimension mismatch must fail, because both repairs are silently wrong.

    Padding appends amplitudes that correspond to no pixel; truncating drops part
    of the image. Either would let a misconfigured pipeline keep running and report
    a result computed on something other than the stated input.
    """
    with pytest.raises(PixelPreprocessingError) as excinfo:
        amplitudes_from_grayscale(np.full(size, 128, dtype=np.uint8))
    message = str(excinfo.value)
    assert str(V1_PIXEL_COUNT) in message and str(size) in message
    assert "pad" in message.lower() and "truncat" in message.lower()


def test_flattening_order_is_row_major_and_stated(encoded):
    """The pixel-to-amplitude map must be fixed, or trained weights address the wrong pixels.

    Checked with a gradient whose value identifies its own position, so an
    off-by-one or a transposition is unambiguous rather than merely "different".
    """
    raw = np.arange(V1_PIXEL_COUNT, dtype=np.int64).reshape(
        V1_ROI_EDGE_PX, V1_ROI_EDGE_PX
    ) % 256
    raw = raw.astype(np.uint8)
    scaled, _, _ = amplitudes_from_grayscale(raw)
    for row, col in ((0, 0), (0, 255), (1, 0), (137, 42), (255, 255)):
        assert scaled[row * V1_ROI_EDGE_PX + col] == pytest.approx(raw[row, col] / 255.0)


# ------------------------------------------------------- determinism and stage order
def test_batch_and_single_amplitudes_agree_exactly(capture):
    """The vectorised path must not be a second implementation with its own rounding.

    Bit-identical, not merely close: both do the same two divisions in the same
    order, so any difference would mean one of them is doing something else.
    """
    from backend.ml.pixel_pipeline import amplitudes_from_grayscale_batch

    rng = np.random.default_rng(4)
    rows = np.stack(
        [
            grayscale_roi(capture, extract_roi(capture.width, capture.height)).ravel(),
            rng.integers(0, 256, V1_PIXEL_COUNT, dtype=np.uint8),
            np.full(V1_PIXEL_COUNT, 200, dtype=np.uint8),
        ]
    )
    batched = amplitudes_from_grayscale_batch(rows)
    for index, row in enumerate(rows):
        _, single, _ = amplitudes_from_grayscale(row)
        assert batched[index].tobytes() == single.tobytes()


def test_batch_names_the_zero_rows_rather_than_failing_anonymously():
    """A batch failure must identify which capture is at fault."""
    from backend.ml.pixel_pipeline import amplitudes_from_grayscale_batch

    rows = np.full((3, V1_PIXEL_COUNT), 120, dtype=np.uint8)
    rows[1] = 0
    with pytest.raises(ZeroImageError) as excinfo:
        amplitudes_from_grayscale_batch(rows)
    assert "index 1" in str(excinfo.value)


def test_the_pixel_path_is_deterministic(capture):
    """The same capture must give bit-identical amplitudes, not merely close ones."""
    first = preprocess_v1_pixels(capture)
    second = preprocess_v1_pixels(capture)
    assert np.array_equal(first.raw_grayscale, second.raw_grayscale)
    assert first.amplitudes.tobytes() == second.amplitudes.tobytes()


def test_grayscale_and_resize_commute_within_rounding(capture):
    """The two stage orders in the specification must agree, and by how much is measured.

    The primary V1 pipeline lists GRAYSCALE then RESIZE; the acceptance-test listing
    lists RESIZE then GRAYSCALE. Both PIL operations are linear, so in real
    arithmetic they commute exactly and any difference comes from re-quantising the
    intermediate to 8 bits. This test pins that claim to a number instead of leaving
    it as an argument -- if a future change makes the orders genuinely diverge (a
    non-linear tone curve, say), it fails.
    """
    roi = extract_roi(capture.width, capture.height)
    grayscale_first = grayscale_roi(capture, roi).astype(np.int16)

    cropped = capture.crop(roi.box).convert("RGB")
    resize_first = np.asarray(
        cropped.resize((V1_ROI_EDGE_PX, V1_ROI_EDGE_PX), V1_RESAMPLE).convert("L"),
        dtype=np.int16,
    )

    difference = np.abs(grayscale_first - resize_first)
    # A couple of levels out of 256 is 8-bit rounding; a real divergence is not.
    assert difference.max() <= 4, f"orders diverge by up to {difference.max()} levels"
    assert difference.mean() < 1.0


# ---------------------------------------------------------------------- provenance
def test_describe_reports_the_actual_input_not_restated_constants(encoded, capture):
    """Provenance must be sufficient to reproduce the input, per the response contract."""
    described = encoded.describe()
    assert described["preprocessing_version"] == V1_PIXEL_PREPROCESSING_VERSION
    assert described["source_width"] == capture.width
    assert described["source_height"] == capture.height
    assert described["roi_source"] in {source.value for source in RoiSource}
    assert len(described["roi_box"]) == 4
    # Measured from the pixels that were actually built, not from the constants.
    assert described["raw_grayscale_mean"] == pytest.approx(
        float(encoded.raw_grayscale.mean()), abs=1e-6
    )
    assert described["amplitude_l2_norm"] == pytest.approx(1.0, abs=1e-9)
    assert described["resample_filter"] == "lanczos"


def test_an_explicit_roi_overrides_the_fallback(capture):
    """Inference supplies a predicted ROI; it must be used verbatim.

    If the pipeline silently re-derived its own ROI, the localiser would have no
    effect and the predicted-ROI evaluation would secretly be the centre-crop one.
    """
    from backend.ml.types import RoiResult

    explicit = RoiResult(
        source=RoiSource.CENTER_CROP,
        x0=10,
        y0=20,
        x1=210,
        y1=180,
        source_width=capture.width,
        source_height=capture.height,
    )
    result = preprocess_v1_pixels(capture, roi=explicit)
    assert result.roi.box == (10, 20, 210, 180)
    assert result.roi_aspect_ratio == pytest.approx(200 / 160)
    # And it differs from the default: the ROI genuinely changes the encoding.
    assert not np.array_equal(
        result.raw_grayscale, preprocess_v1_pixels(capture).raw_grayscale
    )


# ------------------------------------------------------- the quantum encoder contract
def test_the_existing_encoder_accepts_the_v1_vector(encoded):
    """The vector must satisfy the encoder's own validation, unchanged.

    The encoder is reused rather than reimplemented, so this is the real contract
    between the two stages: if it holds, no padding, truncation or renormalisation
    happens between preprocessing and the circuit.
    """
    from quantum_ml.quantum_encoder import QuantumEncoder

    encoder = QuantumEncoder(num_qubits=V1_QUBIT_COUNT, use_pca=False)
    assert encoder.dim == V1_PIXEL_COUNT
    encoder.validate_state(encoded.amplitudes)  # raises if not a legal state

    # And going through transform() must be a no-op on an already-normalised vector.
    state = encoder.transform(encoded.amplitudes)
    assert encoder.last_padded_slots == 0
    assert encoder.last_truncated_features == 0
    assert np.allclose(state, encoded.amplitudes, atol=1e-15)


def test_the_encoder_rejects_a_zero_vector_too(capture):
    """Defence in depth: both stages refuse, so neither can be bypassed."""
    from quantum_ml.quantum_encoder import QuantumEncoder, ZeroVectorError

    encoder = QuantumEncoder(num_qubits=V1_QUBIT_COUNT, use_pca=False)
    with pytest.raises(ZeroVectorError):
        encoder.transform(np.zeros(V1_PIXEL_COUNT))


# ============================= THE REAL DATASET ==============================
requires_dataset = pytest.mark.skipif(
    discover_dataset_root() is None,
    reason="Clinical dataset not present (it is not committed).",
)


@requires_dataset
def test_real_images_encode_to_valid_states():
    """A sample of real captures must all reach a legal 65,536-D unit vector.

    Sampled deterministically across classes rather than taking the first N, which
    would only ever exercise one diagnostic class and one site directory.
    """
    from backend.dataset.smartom import cached_index
    from backend.ml.pixel_pipeline import preprocess_v1_pixels_for_record

    records = cached_index().records
    sample = records[:: max(1, len(records) // 24)][:24]
    assert len(sample) >= 8

    classes = set()
    for record in sample:
        vector = preprocess_v1_pixels_for_record(record)
        assert vector.amplitudes.shape == (V1_PIXEL_COUNT,)
        assert abs(float(np.sum(vector.amplitudes**2)) - 1.0) < NORM_TOLERANCE
        assert vector.raw_grayscale.dtype == np.uint8
        assert int(vector.raw_grayscale.max()) > 0
        classes.add(record.diagnostic_class)

    assert len(classes) > 1, "The sample must span more than one diagnostic class."


@requires_dataset
def test_real_annotated_image_uses_the_annotated_region():
    """With a lesion outline present, the ROI must come from it, not the centre crop.

    This is the oracle-ROI path used for the upper-bound evaluation condition, so
    it has to be demonstrably reading the annotation.
    """
    from backend.dataset.smartom import cached_index
    from backend.ml.pixel_pipeline import preprocess_v1_pixels_for_record

    annotated = next(
        (record for record in cached_index().records if record.lesion_polygons), None
    )
    if annotated is None:
        pytest.skip("No lesion-annotated record in the index.")

    vector = preprocess_v1_pixels_for_record(annotated)
    assert vector.roi.source == RoiSource.LESION_POLYGON
    assert vector.describe()["roi_source"] == "lesion_polygon"
