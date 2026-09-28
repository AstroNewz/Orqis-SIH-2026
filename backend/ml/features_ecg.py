"""Fold-honest ECG feature extraction for the PTB-XL arena (Phase E4).

This is the representation stage the E4 thesis rests on. E3's four quantum nulls
were obtained after crushing megapixel images into 8 amplitudes; the argument for
moving to ECG (``docs/PHASE_E4_DATA_EXPANSION.md``) is that a 12-lead ECG reduces
*naturally* to a compact vector of clinically meaningful measurements — rhythm,
QRS/ST/T morphology, axis, spectral content — which is the dimensionality a
near-term quantum kernel or feature map actually operates in. This module produces
that vector.

**Fold honesty is structural, not a convention here.** Every function in this
module is a deterministic function of *one record*: no statistic is pooled across
records, so extraction cannot leak information between train, validation and test.
The steps that genuinely must learn from data — imputation, scaling, feature
selection, PCA — are deliberately *not* in this module; they belong to a
TRAIN-only fitting stage. A feature that cannot be measured on a record is emitted
as ``NaN`` rather than silently filled, precisely so the fill value has to come
from a fold-aware imputer instead of from this function.

Honest limitations, stated up front:

* The 100 Hz ``records100`` signals give a 10 ms sampling grid, so interval
  features (QRS duration, QT) are quantised to ±10 ms. They are useful as
  *relative* features across a cohort; they are not clinical-grade measurements,
  and the names carry ``_approx`` where the estimate is coarse.
* Delineation is a threshold-and-template method (median beat, energy-envelope
  onset/offset), not a validated clinical delineator. It is reproducible and
  deterministic, which is what a benchmark needs, but no clinical claim is made.
* No feature here is a diagnosis. They are inputs to a classifier that is itself
  evaluated under the patient-level protocol.

Only numpy and scipy are used: no ``wfdb``/``neurokit2`` dependency, matching the
constrained environment documented in the phase notes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sp_signal

from backend.dataset.wfdb_reader import EcgSignal

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

FEATURE_SET_VERSION = "ecg-v1"
"""Bumped whenever the emitted feature names or semantics change."""

STANDARD_LEADS: Tuple[str, ...] = (
    "I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6",
)


@dataclass(frozen=True)
class EcgFeatureConfig:
    """Deterministic parameters of the extractor.

    Persisted alongside any artifact so a feature vector can be reproduced. These
    are signal-processing constants chosen from standard ECG practice, *not*
    hyperparameters tuned against a label — nothing in this module is fitted.
    """

    # Baseline / noise conditioning
    highpass_hz: float = 0.5
    lowpass_hz: float = 40.0
    filter_order: int = 2

    # QRS energy band for R-peak detection
    qrs_band_low_hz: float = 5.0
    qrs_band_high_hz: float = 18.0
    integration_window_s: float = 0.12
    refractory_s: float = 0.24
    peak_threshold_fraction: float = 0.35
    refine_window_s: float = 0.05

    # Beat template window around R
    template_pre_s: float = 0.25
    template_post_s: float = 0.45
    baseline_window_s: float = 0.10
    """Isoelectric estimate is the median of the first ``baseline_window_s`` of the
    template window (i.e. the PR segment, well before R)."""

    # Delineation thresholds on the template energy envelope.
    # The onset/offset fractions are the excursion above the PR-segment noise floor
    # at which the QRS is considered to begin/end. 0.08 was chosen by a rule fixed
    # in advance and evaluated on TRAIN folds only, against *published physiology*
    # and never against a label: the largest fraction whose median QRS duration
    # falls in the normal adult window (80-100 ms) while clamping fewer than 5% of
    # records at the 200 ms hard cap. It yields median QRS 80 ms, median QT 370 ms,
    # median QTc 397 ms, and 1.6% clamped. This is instrument calibration, not
    # model selection -- no classification metric was consulted.
    qrs_onset_fraction: float = 0.08
    qrs_offset_fraction: float = 0.08
    min_qrs_duration_s: float = 0.04
    max_qrs_duration_s: float = 0.20
    st_measure_delay_s: float = 0.06
    st_slope_early_s: float = 0.02
    st_slope_late_s: float = 0.08
    t_search_start_s: float = 0.06
    t_search_end_s: float = 0.40
    t_offset_fraction: float = 0.15

    # Spectral analysis
    spectral_bands_hz: Tuple[Tuple[float, float], ...] = (
        (0.5, 4.0), (4.0, 15.0), (15.0, 40.0),
    )

    # Quality thresholds
    flatline_std_mv: float = 1e-4
    saturation_mv: float = 9.0
    min_beats_for_template: int = 3


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EcgFeatures:
    """Named feature vector for one ECG record.

    ``values`` may contain ``NaN`` for genuinely unmeasurable features (e.g. ST
    level when no beat could be delineated). Imputation is a TRAIN-only step
    performed downstream, never here.
    """

    record_name: str
    names: Tuple[str, ...]
    values: Tuple[float, ...]
    n_beats: int
    template_beats: int
    quality_flags: Tuple[str, ...] = ()
    feature_set_version: str = FEATURE_SET_VERSION

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)

    @property
    def dimension(self) -> int:
        return len(self.values)

    @property
    def n_missing(self) -> int:
        return int(np.count_nonzero(~np.isfinite(self.vector)))

    @property
    def usable(self) -> bool:
        """True when a beat template was built and most features are finite."""
        if self.template_beats < 1:
            return False
        return self.n_missing <= 0.5 * self.dimension

    def as_dict(self) -> Dict[str, float]:
        return dict(zip(self.names, self.values))


# --------------------------------------------------------------------------
# Signal conditioning
# --------------------------------------------------------------------------


def _safe_bandpass(
    data: np.ndarray, fs: float, low: float, high: float, order: int
) -> np.ndarray:
    """Zero-phase bandpass, degrading gracefully when the band is unrealisable.

    A short record or a band that reaches the Nyquist limit would make
    ``butter`` raise; returning the mean-removed signal instead keeps a single
    unusual record from aborting a whole cohort extraction.
    """
    nyquist = 0.5 * fs
    if nyquist <= 0 or data.size < 4 * (order + 1):
        return data - np.mean(data) if data.size else data
    low_norm = max(low / nyquist, 1e-6)
    high_norm = min(high / nyquist, 0.99)
    if low_norm >= high_norm:
        return data - np.mean(data) if data.size else data
    try:
        b, a = sp_signal.butter(order, [low_norm, high_norm], btype="band")
        padlen = 3 * (max(len(a), len(b)) - 1)
        if data.shape[0] <= padlen:
            return data - np.mean(data)
        return sp_signal.filtfilt(b, a, data, axis=0)
    except ValueError:
        return data - np.mean(data)


def condition_signal(samples: np.ndarray, fs: float, config: EcgFeatureConfig) -> np.ndarray:
    """Remove baseline wander and high-frequency noise from every lead."""
    return _safe_bandpass(
        np.asarray(samples, dtype=np.float64),
        fs,
        config.highpass_hz,
        config.lowpass_hz,
        config.filter_order,
    )


# --------------------------------------------------------------------------
# R-peak detection
# --------------------------------------------------------------------------


def detect_r_peaks(
    samples: np.ndarray, fs: float, config: Optional[EcgFeatureConfig] = None
) -> np.ndarray:
    """Locate R-peaks with a Pan-Tompkins-style energy detector.

    Detection runs on the RMS across all available leads rather than a single
    lead, so one disconnected or saturated lead cannot destroy the beat series.

    Returns:
        Sorted sample indices of detected R-peaks (possibly empty).
    """
    config = config or EcgFeatureConfig()
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[0] < 8:
        return np.empty(0, dtype=int)

    qrs_band = _safe_bandpass(
        data, fs, config.qrs_band_low_hz, config.qrs_band_high_hz, config.filter_order
    )
    composite = np.sqrt(np.mean(np.square(qrs_band), axis=1))
    if not np.any(np.isfinite(composite)) or np.allclose(composite, 0.0):
        return np.empty(0, dtype=int)

    # Differentiate -> square -> moving-window integrate (classic Pan-Tompkins).
    derivative = np.gradient(composite)
    energy = np.square(derivative)
    window = max(1, int(round(config.integration_window_s * fs)))
    kernel = np.ones(window, dtype=np.float64) / window
    integrated = np.convolve(energy, kernel, mode="same")

    refractory = max(1, int(round(config.refractory_s * fs)))

    # Pass 1: candidate peaks with only the refractory constraint, so the
    # amplitude threshold can be derived from the record's own beat energies
    # instead of an absolute constant that would not transfer across records.
    candidates, _ = sp_signal.find_peaks(integrated, distance=refractory)
    if candidates.size == 0:
        return np.empty(0, dtype=int)

    heights = integrated[candidates]
    strong = heights[heights >= np.percentile(heights, 70)]
    reference = float(np.median(strong)) if strong.size else float(np.max(heights))
    threshold = config.peak_threshold_fraction * reference

    # Pass 2: keep candidates whose energy is a real fraction of a typical beat.
    peaks = candidates[integrated[candidates] >= threshold]
    if peaks.size == 0:
        return np.empty(0, dtype=int)

    # The integration window delays the energy peak; snap each detection to the
    # nearby maximum of the band-passed composite, which is the true QRS apex.
    refine = max(1, int(round(config.refine_window_s * fs)))
    refined: List[int] = []
    n = composite.shape[0]
    for peak in peaks:
        lo = max(0, int(peak) - refine)
        hi = min(n, int(peak) + refine + 1)
        refined.append(lo + int(np.argmax(composite[lo:hi])))

    unique = np.unique(np.asarray(refined, dtype=int))
    # Refinement can collapse two detections onto one apex; re-impose refractoriness.
    kept: List[int] = []
    for index in unique:
        if not kept or index - kept[-1] >= refractory:
            kept.append(int(index))
        elif composite[index] > composite[kept[-1]]:
            kept[-1] = int(index)
    return np.asarray(kept, dtype=int)


# --------------------------------------------------------------------------
# Beat template and delineation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BeatTemplate:
    """Median beat across detected R-peaks, per lead, baseline corrected."""

    waveform: np.ndarray
    """``(n_template_samples, n_leads)`` in mV, isoelectric baseline removed."""
    r_index: int
    """Index of the R reference within the template window."""
    n_beats: int
    fs: float

    @property
    def n_leads(self) -> int:
        return int(self.waveform.shape[1])


def build_beat_template(
    samples: np.ndarray,
    r_peaks: Sequence[int],
    fs: float,
    config: Optional[EcgFeatureConfig] = None,
) -> Optional[BeatTemplate]:
    """Average beats into a median template, or return None if not possible.

    The median (not the mean) is used so a single ectopic or artefact-corrupted
    beat cannot drag the template.
    """
    config = config or EcgFeatureConfig()
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]

    pre = int(round(config.template_pre_s * fs))
    post = int(round(config.template_post_s * fs))
    if pre <= 0 or post <= 0:
        return None

    beats: List[np.ndarray] = []
    for peak in r_peaks:
        start = int(peak) - pre
        stop = int(peak) + post
        if start < 0 or stop > data.shape[0]:
            continue  # only whole beats; edge beats would bias the template
        beats.append(data[start:stop, :])
    if not beats:
        return None

    stacked = np.stack(beats, axis=0)
    template = np.median(stacked, axis=0)

    baseline_len = max(1, int(round(config.baseline_window_s * fs)))
    baseline_len = min(baseline_len, template.shape[0])
    baseline = np.median(template[:baseline_len, :], axis=0)
    template = template - baseline[None, :]

    return BeatTemplate(waveform=template, r_index=pre, n_beats=len(beats), fs=fs)


@dataclass(frozen=True)
class Delineation:
    """Fiducial points located on the beat template (all indices into it)."""

    qrs_onset: int
    qrs_peak: int
    qrs_offset: int
    t_peak: Optional[int]
    t_offset: Optional[int]
    fs: float

    @property
    def qrs_duration_s(self) -> float:
        return max(0.0, (self.qrs_offset - self.qrs_onset) / self.fs)

    @property
    def qt_s(self) -> Optional[float]:
        if self.t_offset is None:
            return None
        return max(0.0, (self.t_offset - self.qrs_onset) / self.fs)


def delineate_template(
    template: BeatTemplate, config: Optional[EcgFeatureConfig] = None
) -> Delineation:
    """Locate QRS onset/offset and the T wave on the template energy envelope.

    Threshold crossings on the across-lead RMS envelope, bounded by physiological
    limits. Coarse but deterministic; see the module docstring.
    """
    config = config or EcgFeatureConfig()
    waveform = template.waveform
    fs = template.fs
    envelope = np.sqrt(np.mean(np.square(waveform), axis=1))
    n = envelope.shape[0]

    # Search for the QRS apex near the R reference rather than globally, so a tall
    # T wave cannot be mistaken for the QRS complex.
    guard = max(1, int(round(0.08 * fs)))
    lo = max(0, template.r_index - guard)
    hi = min(n, template.r_index + guard + 1)
    qrs_peak = lo + int(np.argmax(envelope[lo:hi])) if hi > lo else int(template.r_index)

    peak_energy = float(envelope[qrs_peak])

    # Anchor the crossing level to the PR-segment noise floor rather than to zero.
    # A bare fraction of the peak systematically truncates the low-amplitude onset
    # and offset slurs, which at 100 Hz cost a whole 10 ms bin each; measuring the
    # excursion *above the isoelectric noise* recovers them without becoming a
    # noise detector.
    baseline_len = max(1, min(int(round(config.baseline_window_s * fs)), n))
    pr_noise = float(np.median(envelope[:baseline_len]))
    span = max(peak_energy - pr_noise, 1e-12)
    onset_level = pr_noise + config.qrs_onset_fraction * span
    offset_level = pr_noise + config.qrs_offset_fraction * span

    min_half = max(1, int(round(0.5 * config.min_qrs_duration_s * fs)))
    # Bound each side's *search* by the full maximum duration so a genuinely wide
    # complex can still be found; the total is capped separately below.
    max_span = max(min_half + 1, int(round(config.max_qrs_duration_s * fs)))

    onset = qrs_peak - min_half
    lower_bound = max(0, qrs_peak - max_span)
    for index in range(qrs_peak, lower_bound - 1, -1):
        if envelope[index] <= onset_level:
            onset = index
            break
    else:
        onset = lower_bound
    onset = int(min(max(0, onset), max(0, qrs_peak - min_half)))

    offset = qrs_peak + min_half
    upper_bound = min(n - 1, qrs_peak + max_span)
    for index in range(qrs_peak, upper_bound + 1):
        if envelope[index] <= offset_level:
            offset = index
            break
    else:
        offset = upper_bound
    offset = int(max(min(n - 1, offset), min(n - 1, qrs_peak + min_half)))

    # Cap the *total* duration at the configured physiological maximum. Without
    # this the two per-side bounds admit up to 2x max_qrs_duration_s, which would
    # emit durations no QRS complex can have. Trim whichever side reaches further
    # from the apex, so a genuinely wide complex stays wide (informative for
    # conduction disease) instead of being discarded.
    max_total = max(2 * min_half, int(round(config.max_qrs_duration_s * fs)))
    while (offset - onset) > max_total:
        if (qrs_peak - onset) >= (offset - qrs_peak):
            if (qrs_peak - onset) <= min_half:
                break
            onset += 1
        else:
            if (offset - qrs_peak) <= min_half:
                break
            offset -= 1

    # T wave: search a window that starts clear of the J point.
    t_start = offset + max(1, int(round(config.t_search_start_s * fs)))
    t_end = min(n, offset + int(round(config.t_search_end_s * fs)))
    t_peak: Optional[int] = None
    t_offset: Optional[int] = None
    if t_end - t_start >= 2:
        segment = envelope[t_start:t_end]
        t_peak = t_start + int(np.argmax(segment))
        t_level = config.t_offset_fraction * float(envelope[t_peak])
        for index in range(t_peak, n):
            if envelope[index] <= t_level:
                t_offset = index
                break
        if t_offset is None:
            t_offset = n - 1

    return Delineation(
        qrs_onset=onset,
        qrs_peak=qrs_peak,
        qrs_offset=offset,
        t_peak=t_peak,
        t_offset=t_offset,
        fs=fs,
    )


# --------------------------------------------------------------------------
# Feature blocks
# --------------------------------------------------------------------------


def _rhythm_features(
    r_peaks: np.ndarray, fs: float, n_samples: int
) -> Tuple[List[str], List[float]]:
    names = [
        "hr_bpm",
        "rr_mean_ms",
        "rr_sdnn_ms",
        "rr_rmssd_ms",
        "rr_cv",
        "rr_min_ms",
        "rr_max_ms",
        "pnn50",
        "n_beats",
    ]
    nan = float("nan")
    if r_peaks.size < 2 or fs <= 0:
        # A single beat still tells us the count; every interval feature is undefined.
        return names, [nan, nan, nan, nan, nan, nan, nan, nan, float(r_peaks.size)]

    rr_ms = np.diff(r_peaks.astype(np.float64)) / fs * 1000.0
    rr_mean = float(np.mean(rr_ms))
    sdnn = float(np.std(rr_ms, ddof=1)) if rr_ms.size >= 2 else nan
    if rr_ms.size >= 2:
        successive = np.diff(rr_ms)
        rmssd = float(np.sqrt(np.mean(np.square(successive))))
        pnn50 = float(np.mean(np.abs(successive) > 50.0))
    else:
        rmssd = nan
        pnn50 = nan
    hr = 60000.0 / rr_mean if rr_mean > 0 else nan
    cv = (sdnn / rr_mean) if (rr_mean > 0 and math.isfinite(sdnn)) else nan

    return names, [
        hr,
        rr_mean,
        sdnn,
        rmssd,
        cv,
        float(np.min(rr_ms)),
        float(np.max(rr_ms)),
        pnn50,
        float(r_peaks.size),
    ]


def _global_morphology_features(
    delineation: Optional[Delineation], rr_mean_ms: float
) -> Tuple[List[str], List[float]]:
    names = ["qrs_duration_ms", "qt_approx_ms", "qtc_bazett_approx_ms", "jt_approx_ms"]
    nan = float("nan")
    if delineation is None:
        return names, [nan, nan, nan, nan]

    qrs_ms = delineation.qrs_duration_s * 1000.0
    qt_s = delineation.qt_s
    qt_ms = qt_s * 1000.0 if qt_s is not None else nan
    if qt_s is not None and math.isfinite(rr_mean_ms) and rr_mean_ms > 0:
        qtc_ms = qt_ms / math.sqrt(rr_mean_ms / 1000.0)
    else:
        qtc_ms = nan
    jt_ms = (qt_ms - qrs_ms) if math.isfinite(qt_ms) else nan
    return names, [qrs_ms, qt_ms, qtc_ms, jt_ms]


def _lead_index(lead_names: Sequence[str], target: str) -> Optional[int]:
    upper = target.strip().upper()
    for index, name in enumerate(lead_names):
        if name.strip().upper() == upper:
            return index
    return None


def _per_lead_features(
    template: Optional[BeatTemplate],
    delineation: Optional[Delineation],
    lead_names: Sequence[str],
    config: EcgFeatureConfig,
) -> Tuple[List[str], List[float], Dict[str, float]]:
    """Amplitude/ST/T measurements per standard lead.

    Always emits the 12 standard leads in a fixed order (NaN for any lead the
    record does not carry) so the vector layout is identical for every record.
    """
    suffixes = ("q_amp_mv", "r_amp_mv", "s_amp_mv", "st60_mv", "st_slope_mv_s", "t_amp_mv")
    names: List[str] = [f"{lead.lower()}_{suffix}" for lead in STANDARD_LEADS for suffix in suffixes]
    nan = float("nan")

    if template is None or delineation is None:
        return names, [nan] * len(names), {}

    waveform = template.waveform
    fs = template.fs
    n = waveform.shape[0]

    st_index = delineation.qrs_offset + int(round(config.st_measure_delay_s * fs))
    st_early = delineation.qrs_offset + int(round(config.st_slope_early_s * fs))
    st_late = delineation.qrs_offset + int(round(config.st_slope_late_s * fs))
    slope_dt = max((st_late - st_early) / fs, 1e-6)

    values: List[float] = []
    net_qrs: Dict[str, float] = {}
    for lead in STANDARD_LEADS:
        index = _lead_index(lead_names, lead)
        if index is None or index >= waveform.shape[1]:
            values.extend([nan] * len(suffixes))
            continue
        trace = waveform[:, index]

        onset = max(0, min(delineation.qrs_onset, n - 1))
        offset = max(onset + 1, min(delineation.qrs_offset + 1, n))
        qrs_segment = trace[onset:offset]
        peak = max(0, min(delineation.qrs_peak, n - 1))

        # Q is the most negative deflection before the apex, S the one after it.
        pre_apex = trace[onset : max(onset + 1, peak)]
        post_apex = trace[min(peak + 1, n - 1) : offset]
        q_amp = float(np.min(pre_apex)) if pre_apex.size else nan
        s_amp = float(np.min(post_apex)) if post_apex.size else nan
        r_amp = float(np.max(qrs_segment)) if qrs_segment.size else nan

        st60 = float(trace[st_index]) if 0 <= st_index < n else nan
        if 0 <= st_early < n and 0 <= st_late < n:
            st_slope = float((trace[st_late] - trace[st_early]) / slope_dt)
        else:
            st_slope = nan

        if delineation.t_peak is not None and 0 <= delineation.t_peak < n:
            t_amp = float(trace[delineation.t_peak])
        else:
            t_amp = nan

        values.extend([q_amp, r_amp, s_amp, st60, st_slope, t_amp])

        if math.isfinite(r_amp) and math.isfinite(s_amp):
            net_qrs[lead] = r_amp + s_amp  # s_amp is negative; this is the net deflection

    return names, values, net_qrs


def _axis_features(
    net_qrs: Dict[str, float],
    template: Optional[BeatTemplate],
    delineation: Optional[Delineation],
    lead_names: Sequence[str],
) -> Tuple[List[str], List[float]]:
    """Frontal-plane QRS and T axes, from leads I and aVF."""
    names = ["qrs_axis_deg", "t_axis_deg"]
    nan = float("nan")

    lead_i = net_qrs.get("I")
    lead_avf = net_qrs.get("AVF")
    if lead_i is None or lead_avf is None:
        qrs_axis = nan
    else:
        qrs_axis = math.degrees(math.atan2(lead_avf, lead_i))

    t_axis = nan
    if template is not None and delineation is not None and delineation.t_peak is not None:
        i_index = _lead_index(lead_names, "I")
        avf_index = _lead_index(lead_names, "AVF")
        t_peak = delineation.t_peak
        if (
            i_index is not None
            and avf_index is not None
            and 0 <= t_peak < template.waveform.shape[0]
        ):
            t_i = float(template.waveform[t_peak, i_index])
            t_avf = float(template.waveform[t_peak, avf_index])
            t_axis = math.degrees(math.atan2(t_avf, t_i))

    return names, [qrs_axis, t_axis]


def _spectral_features(
    conditioned: np.ndarray, fs: float, lead_names: Sequence[str], config: EcgFeatureConfig
) -> Tuple[List[str], List[float]]:
    """Relative band powers and spectral entropy on lead II (or lead 0)."""
    band_names = [
        f"relpow_{int(lo * 10)}_{int(hi * 10)}dHz" for lo, hi in config.spectral_bands_hz
    ]
    names = band_names + ["spectral_entropy", "dominant_freq_hz"]
    nan = float("nan")

    index = _lead_index(lead_names, "II")
    if index is None:
        index = 0
    if conditioned.ndim == 1:
        trace = conditioned
    elif index < conditioned.shape[1]:
        trace = conditioned[:, index]
    else:
        return names, [nan] * len(names)

    if trace.size < 16 or fs <= 0 or not np.any(np.isfinite(trace)):
        return names, [nan] * len(names)

    nperseg = int(min(256, trace.size))
    try:
        freqs, psd = sp_signal.welch(trace, fs=fs, nperseg=nperseg)
    except ValueError:
        return names, [nan] * len(names)

    total = float(np.sum(psd))
    if not math.isfinite(total) or total <= 0:
        return names, [nan] * len(names)

    values: List[float] = []
    for low, high in config.spectral_bands_hz:
        mask = (freqs >= low) & (freqs < high)
        values.append(float(np.sum(psd[mask]) / total))

    normalised = psd / total
    nonzero = normalised[normalised > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    max_entropy = math.log(nonzero.size) if nonzero.size > 1 else 1.0
    values.append(entropy / max_entropy if max_entropy > 0 else nan)
    values.append(float(freqs[int(np.argmax(psd))]))

    return names, values


def _quality_features(
    raw: np.ndarray,
    conditioned: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
    config: EcgFeatureConfig,
) -> Tuple[List[str], List[float], List[str]]:
    """Acquisition-quality descriptors, plus human-readable flags."""
    names = [
        "flatline_lead_fraction",
        "saturated_sample_fraction",
        "baseline_wander_mv",
        "hf_noise_mv",
        "rr_regularity",
    ]
    flags: List[str] = []
    nan = float("nan")

    data = raw if raw.ndim == 2 else raw[:, None]
    n_leads = data.shape[1]

    flat = 0
    for lead in range(n_leads):
        if float(np.std(data[:, lead])) < config.flatline_std_mv:
            flat += 1
    flatline_fraction = flat / n_leads if n_leads else nan
    if flat:
        flags.append(f"flatline_leads={flat}")

    saturated = float(np.mean(np.abs(data) >= config.saturation_mv)) if data.size else nan
    if math.isfinite(saturated) and saturated > 0.0:
        flags.append("saturated_samples")

    # Baseline wander: residual after the conditioning band removed <0.5 Hz drift.
    residual = data - (conditioned if conditioned.shape == data.shape else data)
    wander = float(np.mean(np.std(residual, axis=0))) if residual.size else nan

    # High-frequency noise proxy: mean absolute second difference of the conditioned signal.
    if conditioned.shape[0] >= 3:
        hf = float(np.mean(np.abs(np.diff(conditioned, n=2, axis=0))))
    else:
        hf = nan

    if r_peaks.size >= 3:
        rr = np.diff(r_peaks.astype(np.float64)) / fs
        regularity = float(1.0 - min(1.0, np.std(rr) / max(np.mean(rr), 1e-9)))
    else:
        regularity = nan

    if r_peaks.size < config.min_beats_for_template:
        flags.append(f"few_beats={int(r_peaks.size)}")

    return names, [flatline_fraction, saturated, wander, hf, regularity], flags


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def feature_names(config: Optional[EcgFeatureConfig] = None) -> Tuple[str, ...]:
    """Feature layout, independent of any record.

    Deriving the names from the same helpers that produce the values (run on a
    trivial synthetic record) guarantees names and values cannot drift apart.
    """
    config = config or EcgFeatureConfig()
    fs = 100.0
    blank = np.zeros((int(2 * fs), len(STANDARD_LEADS)), dtype=np.float64)
    probe = EcgSignal(
        samples=blank,
        sampling_frequency=fs,
        lead_names=STANDARD_LEADS,
        units=tuple("mV" for _ in STANDARD_LEADS),
        record_name="__layout__",
    )
    return extract_features(probe, config).names


def extract_features(
    ecg: EcgSignal, config: Optional[EcgFeatureConfig] = None
) -> EcgFeatures:
    """Extract the full named feature vector for one ECG record.

    Deterministic and record-local: the returned vector depends only on this
    record, which is what makes the pipeline fold-honest by construction.
    Unmeasurable features come back as ``NaN`` for a TRAIN-only imputer to fill.
    """
    config = config or EcgFeatureConfig()
    raw = np.asarray(ecg.samples, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[:, None]
    fs = float(ecg.sampling_frequency)

    conditioned = condition_signal(raw, fs, config)
    r_peaks = detect_r_peaks(conditioned, fs, config)
    template = build_beat_template(conditioned, r_peaks, fs, config)
    delineation = delineate_template(template, config) if template is not None else None

    names: List[str] = []
    values: List[float] = []

    rhythm_names, rhythm_values = _rhythm_features(r_peaks, fs, raw.shape[0])
    names += rhythm_names
    values += rhythm_values
    rr_mean_ms = rhythm_values[rhythm_names.index("rr_mean_ms")]

    morph_names, morph_values = _global_morphology_features(delineation, rr_mean_ms)
    names += morph_names
    values += morph_values

    lead_names_out, lead_values, net_qrs = _per_lead_features(
        template, delineation, ecg.lead_names, config
    )
    names += lead_names_out
    values += lead_values

    axis_names, axis_values = _axis_features(net_qrs, template, delineation, ecg.lead_names)
    names += axis_names
    values += axis_values

    spec_names, spec_values = _spectral_features(conditioned, fs, ecg.lead_names, config)
    names += spec_names
    values += spec_values

    qual_names, qual_values, flags = _quality_features(raw, conditioned, r_peaks, fs, config)
    names += qual_names
    values += qual_values

    if template is None:
        flags.append("no_beat_template")

    return EcgFeatures(
        record_name=ecg.record_name,
        names=tuple(names),
        values=tuple(float(v) for v in values),
        n_beats=int(r_peaks.size),
        template_beats=int(template.n_beats) if template is not None else 0,
        quality_flags=tuple(flags),
    )
