"""Tests for the fold-honest ECG feature extractor (Phase E4).

Two kinds of test here:

* **Synthetic**, on a beat train with known rate, amplitude and lead geometry, so
  the detector, template, delineator and per-block feature maths can be checked
  against values that are true by construction.
* **Real-data regression locks**, skipped when the PTB-XL sample is absent. These
  read only ``records100/00000/00001_lr``, which is ``strat_fold`` 3 — a TRAIN
  fold. The validation and frozen-test folds are deliberately never touched by
  the test suite.

The extractor is a deterministic function of one record, so every assertion is
exact or bounded rather than statistical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pytest

from backend.dataset.wfdb_reader import EcgSignal, read_record
from backend.ml.features_ecg import (
    FEATURE_SET_VERSION,
    STANDARD_LEADS,
    EcgFeatureConfig,
    EcgFeatures,
    _rhythm_features,
    build_beat_template,
    condition_signal,
    delineate_template,
    detect_r_peaks,
    extract_features,
    feature_names,
)

_REAL_STEM = (
    Path(__file__).resolve().parents[1]
    / "backend" / "artifacts" / "dataset" / "ptbxl" / "records100" / "00000" / "00001_lr"
)
_HAVE_REAL = _REAL_STEM.with_suffix(".hea").is_file() and _REAL_STEM.with_suffix(".dat").is_file()

EXPECTED_DIMENSION = 97
"""9 rhythm + 4 global morphology + 72 per-lead + 2 axis + 5 spectral + 5 quality."""


def _synth_ecg(
    hr_bpm: float = 60.0,
    fs: float = 100.0,
    duration_s: float = 10.0,
    amplitude: float = 1.0,
    qrs_width_s: float = 0.012,
    lead_scales: Optional[Dict[str, float]] = None,
    default_scale: float = 1.0,
    lead_names: Sequence[str] = STANDARD_LEADS,
    noise_mv: float = 0.0,
    seed: int = 7,
) -> Tuple[EcgSignal, np.ndarray]:
    """Build a synthetic multi-lead beat train and the true R-peak indices.

    Each beat is Q/R/S/T Gaussians at fixed offsets from R, so the beat count,
    rate and per-lead amplitude ratios are known exactly. Leads are scaled copies
    of one shape, which makes the frontal-plane axis analytically predictable.
    """
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    rr = 60.0 / hr_bpm

    shape = np.zeros(n, dtype=np.float64)
    r_times = []
    k = 0
    while True:
        r_t = 0.4 + k * rr
        if r_t > duration_s - 0.5:
            break
        r_times.append(r_t)
        shape += amplitude * np.exp(-0.5 * ((t - r_t) / qrs_width_s) ** 2)
        shape += -0.15 * amplitude * np.exp(-0.5 * ((t - (r_t - 0.03)) / 0.008) ** 2)
        shape += -0.25 * amplitude * np.exp(-0.5 * ((t - (r_t + 0.03)) / 0.010) ** 2)
        shape += 0.30 * amplitude * np.exp(-0.5 * ((t - (r_t + 0.25)) / 0.045) ** 2)
        k += 1

    scales = np.array(
        [(lead_scales or {}).get(name, default_scale) for name in lead_names],
        dtype=np.float64,
    )
    samples = shape[:, None] * scales[None, :]
    if noise_mv > 0.0:
        rng = np.random.default_rng(seed)
        samples = samples + rng.normal(0.0, noise_mv, size=samples.shape)

    signal = EcgSignal(
        samples=samples,
        sampling_frequency=fs,
        lead_names=tuple(lead_names),
        units=tuple("mV" for _ in lead_names),
        record_name=f"synth_{hr_bpm:g}bpm",
    )
    return signal, np.round(np.asarray(r_times) * fs).astype(int)


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


def test_feature_names_layout_is_stable_and_unique() -> None:
    names = feature_names()

    assert len(names) == EXPECTED_DIMENSION
    assert len(set(names)) == len(names), "duplicate feature name would corrupt the layout"
    # Spot-check one name from each block so a reordering is caught.
    for expected in (
        "hr_bpm",
        "qrs_duration_ms",
        "ii_r_amp_mv",
        "v6_t_amp_mv",
        "qrs_axis_deg",
        "spectral_entropy",
        "rr_regularity",
    ):
        assert expected in names
    # All 12 standard leads must appear, in the canonical order.
    lead_order = [name.split("_")[0] for name in names if name.endswith("_r_amp_mv")]
    assert lead_order == [lead.lower() for lead in STANDARD_LEADS]


def test_extracted_names_match_declared_layout() -> None:
    ecg, _ = _synth_ecg()

    features = extract_features(ecg)

    assert features.names == feature_names()
    assert features.dimension == EXPECTED_DIMENSION
    assert len(features.values) == len(features.names)
    assert features.feature_set_version == FEATURE_SET_VERSION


def test_layout_is_identical_for_a_record_missing_leads() -> None:
    """A record carrying fewer leads must still emit the full 12-lead layout."""
    ecg, _ = _synth_ecg(lead_names=("I", "II", "V2"))

    features = extract_features(ecg)
    as_dict = features.as_dict()

    assert features.names == feature_names()
    assert np.isfinite(as_dict["ii_r_amp_mv"])
    # Leads the record does not carry are NaN, never zero-filled.
    assert np.isnan(as_dict["v5_r_amp_mv"])
    assert np.isnan(as_dict["avf_t_amp_mv"])


# --------------------------------------------------------------------------
# R-peak detection and rhythm
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hr_bpm", [50.0, 60.0, 75.0, 100.0])
def test_detects_every_beat_at_a_known_rate(hr_bpm: float) -> None:
    ecg, true_peaks = _synth_ecg(hr_bpm=hr_bpm)

    conditioned = condition_signal(ecg.samples, ecg.sampling_frequency, EcgFeatureConfig())
    peaks = detect_r_peaks(conditioned, ecg.sampling_frequency)

    assert peaks.size == true_peaks.size
    # Each detection lands on its beat, within one 10 ms sample of the true apex.
    assert np.all(np.abs(peaks - true_peaks) <= 1)


@pytest.mark.parametrize("hr_bpm", [50.0, 60.0, 75.0, 100.0])
def test_rhythm_features_recover_the_known_rate(hr_bpm: float) -> None:
    ecg, true_peaks = _synth_ecg(hr_bpm=hr_bpm)

    values = extract_features(ecg).as_dict()

    assert values["hr_bpm"] == pytest.approx(hr_bpm, abs=1.0)
    assert values["rr_mean_ms"] == pytest.approx(60000.0 / hr_bpm, abs=10.0)
    assert values["n_beats"] == float(true_peaks.size)
    # A perfectly regular synthetic rhythm has no beat-to-beat variability.
    assert values["rr_sdnn_ms"] == pytest.approx(0.0, abs=1.0)
    assert values["pnn50"] == 0.0
    assert values["rr_regularity"] == pytest.approx(1.0, abs=0.01)


def test_detection_survives_noise_on_top_of_the_beat_train() -> None:
    ecg, true_peaks = _synth_ecg(hr_bpm=72.0, noise_mv=0.02)

    features = extract_features(ecg)

    assert features.n_beats == true_peaks.size
    assert features.template_beats == true_peaks.size
    assert features.usable


def test_rhythm_features_are_nan_for_a_single_beat() -> None:
    """One beat gives a count but no interval; NaN must not be filled here."""
    names, values = _rhythm_features(np.array([50], dtype=int), 100.0, 1000)
    as_dict = dict(zip(names, values))

    assert as_dict["n_beats"] == 1.0
    for key in ("hr_bpm", "rr_mean_ms", "rr_sdnn_ms", "rr_rmssd_ms", "pnn50"):
        assert np.isnan(as_dict[key]), key


def test_flat_record_yields_no_beats_and_is_flagged() -> None:
    flat = EcgSignal(
        samples=np.zeros((1000, len(STANDARD_LEADS))),
        sampling_frequency=100.0,
        lead_names=STANDARD_LEADS,
        units=tuple("mV" for _ in STANDARD_LEADS),
        record_name="flat",
    )

    features = extract_features(flat)
    values = features.as_dict()

    assert features.n_beats == 0
    assert features.template_beats == 0
    assert features.usable is False
    assert "no_beat_template" in features.quality_flags
    assert any(flag.startswith("flatline_leads=") for flag in features.quality_flags)
    assert values["flatline_lead_fraction"] == 1.0
    assert np.isnan(values["hr_bpm"])
    assert np.isnan(values["qrs_duration_ms"])
    assert np.isnan(values["ii_st60_mv"])


def test_saturated_record_is_flagged() -> None:
    ecg, _ = _synth_ecg(amplitude=20.0)

    features = extract_features(ecg)

    assert "saturated_samples" in features.quality_flags
    assert features.as_dict()["saturated_sample_fraction"] > 0.0


def test_short_record_does_not_raise() -> None:
    """A pathologically short record must degrade, not abort a cohort run."""
    tiny = EcgSignal(
        samples=np.zeros((5, 2)),
        sampling_frequency=100.0,
        lead_names=("I", "II"),
        units=("mV", "mV"),
        record_name="tiny",
    )

    features = extract_features(tiny)

    assert features.dimension == EXPECTED_DIMENSION
    assert features.n_beats == 0


# --------------------------------------------------------------------------
# Template and delineation
# --------------------------------------------------------------------------


def test_template_uses_only_whole_beats() -> None:
    ecg, true_peaks = _synth_ecg(hr_bpm=60.0)
    config = EcgFeatureConfig()
    conditioned = condition_signal(ecg.samples, ecg.sampling_frequency, config)

    # A "peak" at index 2 cannot supply a 0.25 s pre-window and must be dropped.
    peaks = np.concatenate(([2], true_peaks))
    template = build_beat_template(conditioned, peaks, ecg.sampling_frequency, config)

    assert template is not None
    assert template.n_beats == true_peaks.size
    assert template.r_index == int(round(config.template_pre_s * ecg.sampling_frequency))
    assert template.waveform.shape[0] == int(
        round((config.template_pre_s + config.template_post_s) * ecg.sampling_frequency)
    )


def test_template_median_rejects_a_single_artefact_beat() -> None:
    ecg, true_peaks = _synth_ecg(hr_bpm=60.0)
    config = EcgFeatureConfig()
    conditioned = condition_signal(ecg.samples, ecg.sampling_frequency, config)

    clean = build_beat_template(conditioned, true_peaks, ecg.sampling_frequency, config)
    corrupted = conditioned.copy()
    mid = int(true_peaks[len(true_peaks) // 2])
    corrupted[mid - 10 : mid + 10, :] += 25.0  # one grossly corrupted beat
    with_artefact = build_beat_template(corrupted, true_peaks, ecg.sampling_frequency, config)

    assert clean is not None and with_artefact is not None
    # The median over 10 beats is essentially untouched by one outlier.
    assert np.max(np.abs(with_artefact.waveform - clean.waveform)) < 0.05


def test_build_beat_template_returns_none_without_usable_beats() -> None:
    config = EcgFeatureConfig()
    data = np.zeros((1000, 2))

    assert build_beat_template(data, [], 100.0, config) is None
    assert build_beat_template(data, [1, 2], 100.0, config) is None  # all edge beats


def test_delineation_orders_fiducials_and_is_physiological() -> None:
    ecg, true_peaks = _synth_ecg(hr_bpm=60.0)
    config = EcgFeatureConfig()
    conditioned = condition_signal(ecg.samples, ecg.sampling_frequency, config)
    template = build_beat_template(conditioned, true_peaks, ecg.sampling_frequency, config)
    assert template is not None

    delineation = delineate_template(template, config)

    assert delineation.qrs_onset < delineation.qrs_peak < delineation.qrs_offset
    assert delineation.t_peak is not None and delineation.t_peak > delineation.qrs_offset
    assert delineation.t_offset is not None and delineation.t_offset >= delineation.t_peak
    assert config.min_qrs_duration_s <= delineation.qrs_duration_s <= config.max_qrs_duration_s
    # The synthetic T apex sits 250 ms after R by construction.
    t_delay_s = (delineation.t_peak - template.r_index) / template.fs
    assert t_delay_s == pytest.approx(0.25, abs=0.04)
    assert delineation.qt_s is not None and 0.20 <= delineation.qt_s <= 0.55


@pytest.mark.parametrize("max_qrs_s", [0.20, 0.12, 0.08])
def test_qrs_duration_never_exceeds_the_configured_maximum(max_qrs_s: float) -> None:
    """Regression: the per-side search bounds once admitted 2x the configured cap.

    ``max_qrs_duration_s`` bounded the search on each side of the apex
    independently, so a broad complex could report up to twice the documented
    physiological maximum. The total duration is now capped explicitly.
    """
    config = EcgFeatureConfig(max_qrs_duration_s=max_qrs_s)
    # A deliberately very broad QRS, which is what made the old bound overshoot.
    ecg, true_peaks = _synth_ecg(hr_bpm=60.0, qrs_width_s=0.06)
    conditioned = condition_signal(ecg.samples, ecg.sampling_frequency, config)
    template = build_beat_template(conditioned, true_peaks, ecg.sampling_frequency, config)
    assert template is not None

    delineation = delineate_template(template, config)

    assert delineation.qrs_duration_s <= max_qrs_s + 1e-9
    assert delineation.qrs_duration_s >= config.min_qrs_duration_s
    assert delineation.qrs_onset < delineation.qrs_peak < delineation.qrs_offset


def test_wide_complex_reports_a_wider_qrs_than_a_narrow_one() -> None:
    """The cap must not flatten the feature: width still has to be informative."""
    config = EcgFeatureConfig()
    durations = []
    for width in (0.012, 0.045):
        ecg, peaks = _synth_ecg(hr_bpm=60.0, qrs_width_s=width)
        conditioned = condition_signal(ecg.samples, ecg.sampling_frequency, config)
        template = build_beat_template(conditioned, peaks, ecg.sampling_frequency, config)
        assert template is not None
        durations.append(delineate_template(template, config).qrs_duration_s)

    assert durations[1] > durations[0]


# --------------------------------------------------------------------------
# Per-lead, axis and spectral blocks
# --------------------------------------------------------------------------


def test_per_lead_amplitudes_track_the_synthetic_lead_scaling() -> None:
    ecg, _ = _synth_ecg(lead_scales={"II": 1.0, "V1": 0.5}, default_scale=1.0)

    values = extract_features(ecg).as_dict()

    assert values["ii_r_amp_mv"] > 0.0
    assert values["ii_s_amp_mv"] < 0.0
    assert values["v1_r_amp_mv"] == pytest.approx(0.5 * values["ii_r_amp_mv"], rel=1e-6)
    assert values["v1_s_amp_mv"] == pytest.approx(0.5 * values["ii_s_amp_mv"], rel=1e-6)
    assert values["v1_t_amp_mv"] == pytest.approx(0.5 * values["ii_t_amp_mv"], rel=1e-6)


@pytest.mark.parametrize(
    "avf_over_i, expected_deg",
    [(np.tan(np.deg2rad(30.0)), 30.0), (1.0, 45.0), (np.tan(np.deg2rad(60.0)), 60.0)],
)
def test_frontal_axis_matches_the_constructed_lead_ratio(
    avf_over_i: float, expected_deg: float
) -> None:
    """Leads are scaled copies of one shape, so the axis is analytically known.

    Only same-polarity ratios are predictable in closed form: the net deflection
    is ``max(QRS) + min(post-apex)``, and that convention is deliberately not
    linear across a polarity flip (an inverted complex measures its own R and S,
    not the negated originals). Inverted leads are covered by the next test.
    """
    ecg, _ = _synth_ecg(lead_scales={"I": 1.0, "AVF": float(avf_over_i)}, default_scale=0.7)

    values = extract_features(ecg).as_dict()

    assert values["qrs_axis_deg"] == pytest.approx(expected_deg, abs=1.0)


def test_frontal_axis_sign_and_monotonicity_follow_the_avf_deflection() -> None:
    """An inverted aVF must give a negative (leftward) axis, monotone in the ratio."""
    axes = []
    for scale in (-1.0, -0.4, 0.4, 1.0):
        ecg, _ = _synth_ecg(lead_scales={"I": 1.0, "AVF": scale}, default_scale=0.7)
        axes.append(extract_features(ecg).as_dict()["qrs_axis_deg"])

    assert axes[0] < 0.0 and axes[1] < 0.0, "inverted aVF should read as a negative axis"
    assert axes[2] > 0.0 and axes[3] > 0.0
    assert axes == sorted(axes), f"axis must increase with the aVF/I ratio, got {axes}"
    assert all(-180.0 <= axis <= 180.0 for axis in axes)


def test_spectral_features_are_normalised() -> None:
    ecg, _ = _synth_ecg(hr_bpm=72.0)

    values = extract_features(ecg).as_dict()
    band_keys = [name for name in values if name.startswith("relpow_")]

    assert len(band_keys) == 3
    for key in band_keys:
        assert 0.0 <= values[key] <= 1.0, key
    assert sum(values[key] for key in band_keys) <= 1.0 + 1e-9
    assert 0.0 <= values["spectral_entropy"] <= 1.0
    assert 0.0 <= values["dominant_freq_hz"] <= 50.0


# --------------------------------------------------------------------------
# Contract guarantees
# --------------------------------------------------------------------------


def test_extraction_is_deterministic() -> None:
    ecg, _ = _synth_ecg(hr_bpm=68.0, noise_mv=0.01)

    first = extract_features(ecg)
    second = extract_features(ecg)

    assert first.names == second.names
    np.testing.assert_array_equal(first.vector, second.vector)


def test_extraction_is_record_local() -> None:
    """Fold honesty by construction: one record's features cannot depend on another.

    Extracting record A alone must give the same vector as extracting it after a
    very different record B, which is what guarantees no statistic is pooled.
    """
    ecg_a, _ = _synth_ecg(hr_bpm=55.0)
    ecg_b, _ = _synth_ecg(hr_bpm=110.0, amplitude=3.0, noise_mv=0.05)

    alone = extract_features(ecg_a).vector
    extract_features(ecg_b)
    after_other = extract_features(ecg_a).vector

    np.testing.assert_array_equal(alone, after_other)


def test_features_container_reports_missingness() -> None:
    features = EcgFeatures(
        record_name="r",
        names=("a", "b", "c", "d"),
        values=(1.0, float("nan"), 3.0, float("nan")),
        n_beats=5,
        template_beats=4,
    )

    assert features.dimension == 4
    assert features.n_missing == 2
    assert features.usable is True  # exactly half missing, template present
    assert features.as_dict()["a"] == 1.0

    no_template = EcgFeatures(
        record_name="r", names=("a",), values=(1.0,), n_beats=0, template_beats=0
    )
    assert no_template.usable is False


def test_config_is_frozen_so_artifacts_stay_reproducible() -> None:
    config = EcgFeatureConfig()

    with pytest.raises(Exception):
        config.highpass_hz = 1.0  # type: ignore[misc]


# --------------------------------------------------------------------------
# Real PTB-XL regression lock (TRAIN fold only)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_REAL, reason="PTB-XL signal sample not downloaded")
def test_real_train_fold_record_extracts_plausible_physiology() -> None:
    """Lock the extractor's behaviour on a genuine record from TRAIN fold 3.

    Bounds are published adult reference ranges, not tuned targets: this test
    fails if the extractor starts emitting physiologically impossible values.
    """
    ecg = read_record(_REAL_STEM)

    features = extract_features(ecg)
    values = features.as_dict()

    assert features.dimension == EXPECTED_DIMENSION
    assert features.quality_flags == ()
    assert features.usable
    assert features.n_missing == 0
    assert np.all(np.isfinite(features.vector))

    assert 30.0 <= values["hr_bpm"] <= 200.0
    assert 300.0 <= values["rr_mean_ms"] <= 2000.0
    assert 0.04 <= values["qrs_duration_ms"] / 1000.0 <= 0.20
    assert 200.0 <= values["qt_approx_ms"] <= 600.0
    assert 250.0 <= values["qtc_bazett_approx_ms"] <= 650.0
    assert -180.0 <= values["qrs_axis_deg"] <= 180.0
    assert abs(values["ii_r_amp_mv"]) <= 10.0
    assert abs(values["ii_st60_mv"]) <= 2.0
    assert 0.0 <= values["rr_regularity"] <= 1.0
    # A clean published record should carry no dead or clipped leads.
    assert values["flatline_lead_fraction"] == 0.0
    assert values["saturated_sample_fraction"] == 0.0
