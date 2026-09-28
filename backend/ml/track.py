"""The platform spine: a disease track, and the registry that serves them.

CareScan began as one model for one condition, and every layer says so -- the routes take
an image, the service loads one artifact version, and :class:`InferenceResult` describes a
variational quantum classifier looking at an oral photograph. That was the right shape for
one model. It is the wrong shape for the thing the project is now: a *platform* for early
disease detection, with an oral-lesion track and a 12-lead ECG track that share almost
everything except their front end.

What they share, once you strip the modality away, is one pipeline:

    artifacts -> input validation -> features -> representation
              -> classical reference + optional quantum head -> calibration -> banded verdict

Only the first two stages are modality-specific. This module names the rest as a contract,
so adding a third condition is a registration rather than a rewrite.

**The contract carries the methodology, not just the plumbing.** A track cannot be
registered without declaring a :class:`ValidationSummary`: which dataset it was fitted on,
which partition its headline number came from, how many *patients* that was, and -- the
field that matters most -- whether a frozen test partition has ever been evaluated. That
last flag is deliberately not optional and deliberately not defaulted to ``True``. A model
whose only number comes from a partition it was selected on is a legitimate research result
and an illegitimate clinical claim, and the platform should be able to tell a caller which
one it is holding without anybody having to remember to mention it.

**Nothing here changes the oral-lesion path.** :class:`DiseaseTrack` is a protocol, not a
base class, and the existing :class:`~backend.services.inference_service.InferenceService`
is adapted to it from the outside. Its behaviour, its schema and its routes are untouched;
this module only gives the platform a second way to reach it, alongside a way to reach
anything else.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Modality",
    "RiskBand",
    "TrackError",
    "TrackNotFoundError",
    "TrackNotReadyError",
    "TrackInputError",
    "InputSpec",
    "ValidationSummary",
    "TrackDescriptor",
    "TrackAssessment",
    "DiseaseTrack",
    "TrackRegistry",
    "registry",
]


class Modality(str, Enum):
    """What a track consumes. Determines the transport, not the science."""

    IMAGE = "image"
    SIGNAL = "signal"
    TABULAR = "tabular"


class RiskBand(str, Enum):
    """The patient-facing verdict vocabulary, shared so two tracks cannot disagree.

    ``INDETERMINATE`` exists because refusing to band is sometimes the honest answer --
    an unreadable image, a lead-off ECG -- and a platform that can only say LOW or HIGH
    will say one of them anyway.
    """

    LOW = "LOW RISK"
    MODERATE = "MODERATE RISK"
    HIGH = "HIGH RISK"
    INDETERMINATE = "INDETERMINATE"


class TrackError(RuntimeError):
    """Base class for every failure a track can raise."""


class TrackNotFoundError(TrackError):
    """No track is registered under the requested identifier."""


class TrackNotReadyError(TrackError):
    """The track exists but its artifacts are missing, unloadable or untrained."""


class TrackInputError(TrackError):
    """The submitted input does not satisfy the track's :class:`InputSpec`."""


class InputSpec(BaseModel):
    """What a caller must send, in enough detail that a client can be built from it alone."""

    modality: Modality
    content_types: List[str] = Field(default_factory=list)
    """Acceptable MIME types, e.g. ``["image/jpeg"]`` or ``["application/json"]``."""

    description: str
    """One sentence a UI can show: "a well-lit photograph of the lesion", "a 10-second
    12-lead ECG at 100 Hz, in millivolts"."""

    shape: Optional[List[int]] = None
    """Expected array shape for signal or tabular tracks, e.g. ``[12, 1000]``. ``None``
    for tracks whose input has no fixed shape, such as an image."""

    units: Optional[str] = None
    sampling_frequency_hz: Optional[float] = None
    channel_names: Optional[List[str]] = None
    """Channel order for multi-channel signals. A caller that sends leads in a different
    order gets a different answer, so the order is part of the published contract rather
    than something to infer."""

    max_bytes: Optional[int] = None


class ValidationSummary(BaseModel):
    """How a track's headline number was obtained. Required, and deliberately blunt.

    Every field here exists because omitting it is how screening models get oversold.
    A number without its partition, its patient count and its test-set status is not a
    result; it is a number.
    """

    dataset: str
    """Human-readable provenance, e.g. ``"PTB-XL v1.0.3 (PhysioNet, CC BY 4.0)"``."""

    task: str
    """What the reported metric measures, e.g. ``"NORM vs abnormal"``."""

    primary_metric: str
    primary_metric_value: Optional[float] = None
    confidence_interval: Optional[List[float]] = None
    """``[low, high]``, bootstrapped over **patients** rather than records wherever a
    patient can contribute more than one record. Resampling records would treat two
    ECGs from the same person as independent evidence and narrow the interval for free."""

    partition_scored: str
    """The partition the headline number came from, named exactly, e.g.
    ``"strat_fold 9 (validation)"``."""

    n_records: Optional[int] = None
    n_patients: Optional[int] = None

    frozen_test_evaluated: bool
    """``True`` only if a partition untouched by every selection decision has been
    scored. When ``False`` the headline number is a *development* estimate: it comes
    from data that model selection has already seen, so it is optimistically biased by
    an unknown amount, and the track must not be described as validated."""

    partition_reused_for_selection: bool = True
    """``True`` when :attr:`partition_scored` also informed model selection. Paired with
    :attr:`frozen_test_evaluated` this distinguishes "not yet tested" from "tested"."""

    notes: Optional[str] = None

    @property
    def is_clinically_claimable(self) -> bool:
        """Whether this track may be described as validated. Never true without a
        frozen-test evaluation, regardless of how good the development number looks."""
        return bool(self.frozen_test_evaluated)


class TrackDescriptor(BaseModel):
    """Everything a client needs to decide whether to offer a track, and how."""

    track_id: str
    display_name: str
    condition: str
    """The condition screened for, in patient-facing words."""

    modality: Modality
    input_spec: InputSpec
    validation: ValidationSummary

    model_version: str
    ready: bool
    unready_reason: Optional[str] = None
    """Why :attr:`ready` is ``False`` -- missing artifacts, a failed load, an untrained
    model. Returned rather than raised so a client can grey out one track and still show
    the rest of the platform."""

    primary_model: str
    """Which member headlines the verdict. Not assumed to be the quantum one: on the
    oral-lesion track the validated classical baseline ranks above the VQC and therefore
    headlines it (DEC-034)."""

    uses_quantum: bool
    quantum_role: Optional[str] = None
    """What the quantum component actually does here, in one phrase, or ``None`` if
    there is none. A platform that advertises "quantum" without saying where should not
    be believed, including by its own authors."""

    disclaimer: str

    model_config = ConfigDict(protected_namespaces=())


class TrackAssessment(BaseModel):
    """One track's answer, in the vocabulary every track shares.

    The common core is small on purpose. Anything a track knows that the others do not --
    image quality, circuit depth, per-lead findings -- belongs in :attr:`detail`, where it
    can be typed by that track without every other track having to grow a field for it.
    """

    track_id: str
    model_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    risk_band: RiskBand
    probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    probability_is_calibrated: bool
    """``False`` when :attr:`probability` is a ranking score rather than a risk estimate.
    A client must not render an uncalibrated value as a percentage; the band is the
    verdict in that case."""

    threshold: Optional[float] = None
    threshold_selected_on: Optional[str] = None
    """Where the operating point came from. A threshold chosen on the partition it is
    then reported against is self-selected, and naming its origin makes that visible."""

    primary_model: str
    classical_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    quantum_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    """Reported side by side, never silently blended. A blended number would correspond
    to no model in any evaluation report."""

    detail: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float = Field(default=0.0, ge=0.0)
    is_mock: bool = False
    disclaimer: str = ""

    model_config = ConfigDict(protected_namespaces=())


@runtime_checkable
class DiseaseTrack(Protocol):
    """One condition the platform can screen for.

    A :class:`~typing.Protocol` rather than a base class, so an existing service can be
    adapted from the outside without inheriting from anything -- which is exactly how the
    oral-lesion track is wired, leaving its code untouched.
    """

    track_id: str

    def describe(self) -> TrackDescriptor:
        """Static capability and provenance. Must not raise when artifacts are missing:
        report ``ready=False`` with a reason instead, so an unready track is still
        listable."""

    def analyze(self, payload: Any, **options: Any) -> TrackAssessment:
        """Run one assessment.

        Raises:
            TrackNotReadyError: artifacts unavailable.
            TrackInputError: payload violates the published :class:`InputSpec`.
        """


class TrackRegistry:
    """The set of tracks this deployment serves.

    Thread-safe because FastAPI serves requests from a pool and registration happens at
    import time; a half-registered registry read concurrently is the kind of bug that
    shows up once, in a demo.
    """

    def __init__(self) -> None:
        self._tracks: Dict[str, DiseaseTrack] = {}
        self._lock = threading.RLock()

    def register(self, track: DiseaseTrack, *, replace: bool = False) -> DiseaseTrack:
        """Add a track. Refuses a silent overwrite.

        Two tracks claiming one id is a configuration error, and resolving it by
        last-write-wins means the served model depends on import order.
        """
        track_id = getattr(track, "track_id", "")
        if not track_id:
            raise TrackError(f"{type(track).__name__} declares no track_id.")
        with self._lock:
            if track_id in self._tracks and not replace:
                raise TrackError(
                    f"A track is already registered as {track_id!r}. Pass replace=True "
                    "to substitute it deliberately."
                )
            self._tracks[track_id] = track
        return track

    def unregister(self, track_id: str) -> None:
        with self._lock:
            self._tracks.pop(track_id, None)

    def get(self, track_id: str) -> DiseaseTrack:
        with self._lock:
            try:
                return self._tracks[track_id]
            except KeyError:
                raise TrackNotFoundError(
                    f"No track {track_id!r}. Registered: {sorted(self._tracks)}."
                ) from None

    def ids(self) -> List[str]:
        with self._lock:
            return sorted(self._tracks)

    def describe_all(self) -> List[TrackDescriptor]:
        """Describe every track, including the ones that cannot currently run.

        A track whose ``describe`` raises is reported as unready rather than being
        allowed to take the whole listing down with it: one broken artifact directory
        should cost one card in the UI, not the platform.
        """
        descriptors: List[TrackDescriptor] = []
        for track_id in self.ids():
            track = self.get(track_id)
            try:
                descriptors.append(track.describe())
            except Exception as error:  # noqa: BLE001 - deliberately broad; see docstring
                descriptors.append(_unready_descriptor(track_id, error))
        return descriptors

    def ready_ids(self) -> List[str]:
        return [d.track_id for d in self.describe_all() if d.ready]

    def clear(self) -> None:
        """Drop every registration. For tests; production registers once at import."""
        with self._lock:
            self._tracks.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._tracks)

    def __contains__(self, track_id: object) -> bool:
        with self._lock:
            return track_id in self._tracks


def _unready_descriptor(track_id: str, error: Exception) -> TrackDescriptor:
    """A placeholder for a track that could not even describe itself."""
    return TrackDescriptor(
        track_id=track_id,
        display_name=track_id,
        condition="unknown",
        modality=Modality.TABULAR,
        input_spec=InputSpec(modality=Modality.TABULAR, description="unavailable"),
        validation=ValidationSummary(
            dataset="unknown",
            task="unknown",
            primary_metric="unknown",
            partition_scored="unknown",
            frozen_test_evaluated=False,
            notes="This track failed to describe itself; treat every claim as unverified.",
        ),
        model_version="",
        ready=False,
        unready_reason=f"{type(error).__name__}: {error}",
        primary_model="unknown",
        uses_quantum=False,
        disclaimer="",
    )


registry = TrackRegistry()
"""The process-wide registry. Tracks register themselves where they are wired up, not
here, so importing this module does not drag in torch, qiskit and every artifact store."""
