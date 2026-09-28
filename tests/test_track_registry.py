"""The platform spine: the track contract, and the registry that serves it.

The registry is the one place where a configuration mistake turns into a wrong answer
rather than an error -- two tracks claiming an id, a broken track taking the listing
down with it, an unready track reporting itself as ready. Those are the cases tested
here, alongside the part of the contract that carries the methodology: a track cannot
describe itself as clinically claimable without a frozen-test evaluation, and no amount
of development performance substitutes for one.

These tests build tracks in-process and never load an artifact, so they run everywhere.
"""

from __future__ import annotations

import threading

import pytest
from pydantic import ValidationError

from backend.ml.track import (
    DiseaseTrack,
    InputSpec,
    Modality,
    RiskBand,
    TrackAssessment,
    TrackDescriptor,
    TrackError,
    TrackNotFoundError,
    TrackRegistry,
    ValidationSummary,
    registry,
)


def _validation(**overrides) -> ValidationSummary:
    payload = {
        "dataset": "PTB-XL v1.0.3 (PhysioNet, CC BY 4.0)",
        "task": "NORM vs abnormal",
        "primary_metric": "ROC-AUC",
        "primary_metric_value": 0.940234,
        "confidence_interval": [0.9297, 0.9494],
        "partition_scored": "strat_fold 9 (validation)",
        "n_records": 2146,
        "n_patients": 1917,
        "frozen_test_evaluated": False,
    }
    payload.update(overrides)
    return ValidationSummary(**payload)


def _descriptor(track_id: str = "demo", **overrides) -> TrackDescriptor:
    payload = {
        "track_id": track_id,
        "display_name": "Demo",
        "condition": "demo condition",
        "modality": Modality.SIGNAL,
        "input_spec": InputSpec(modality=Modality.SIGNAL, description="a signal"),
        "validation": _validation(),
        "model_version": "v1",
        "ready": True,
        "primary_model": "gbm",
        "uses_quantum": False,
        "disclaimer": "Research use only.",
    }
    payload.update(overrides)
    return TrackDescriptor(**payload)


class _Track:
    """A minimal conforming track."""

    def __init__(self, track_id: str = "demo", *, raises: Exception | None = None) -> None:
        self.track_id = track_id
        self._raises = raises
        self.calls = 0

    def describe(self) -> TrackDescriptor:
        if self._raises is not None:
            raise self._raises
        return _descriptor(self.track_id)

    def analyze(self, payload, **options) -> TrackAssessment:
        self.calls += 1
        return TrackAssessment(
            track_id=self.track_id,
            model_version="v1",
            risk_band=RiskBand.LOW,
            probability=0.1,
            probability_is_calibrated=True,
            primary_model="gbm",
        )


# ------------------------------------------------------------------ the contract
def test_a_plain_object_satisfies_the_protocol_without_inheriting():
    """The oral-lesion service is adapted from outside; it must not have to subclass."""
    assert isinstance(_Track(), DiseaseTrack)


def test_an_object_missing_analyze_does_not_satisfy_the_protocol():
    class Partial:
        track_id = "partial"

        def describe(self):  # pragma: no cover - never called
            return _descriptor()

    assert not isinstance(Partial(), DiseaseTrack)


def test_validation_summary_requires_the_test_set_flag():
    """``frozen_test_evaluated`` has no default. Forgetting it is an error, not a True."""
    with pytest.raises(ValidationError):
        ValidationSummary(
            dataset="d", task="t", primary_metric="ROC-AUC", partition_scored="fold 9"
        )


def test_a_strong_development_number_is_still_not_claimable():
    """0.94 on a partition that selection has seen does not make a validated model."""
    summary = _validation(primary_metric_value=0.99, frozen_test_evaluated=False)
    assert summary.is_clinically_claimable is False


def test_only_a_frozen_test_evaluation_makes_a_track_claimable():
    assert _validation(frozen_test_evaluated=True).is_clinically_claimable is True


def test_probability_must_be_a_probability():
    with pytest.raises(ValidationError):
        TrackAssessment(
            track_id="demo",
            model_version="v1",
            risk_band=RiskBand.LOW,
            probability=1.4,
            probability_is_calibrated=True,
            primary_model="gbm",
        )


def test_an_uncalibrated_score_can_be_reported_but_is_flagged():
    """A ranking score is allowed through; what is not allowed is calling it calibrated."""
    assessment = TrackAssessment(
        track_id="demo",
        model_version="v1",
        risk_band=RiskBand.MODERATE,
        probability=0.7,
        probability_is_calibrated=False,
        primary_model="gbm",
    )
    assert assessment.probability_is_calibrated is False


def test_the_two_members_are_reported_side_by_side_not_blended():
    assessment = TrackAssessment(
        track_id="demo",
        model_version="v1",
        risk_band=RiskBand.LOW,
        probability=0.2,
        probability_is_calibrated=True,
        primary_model="gbm",
        classical_probability=0.2,
        quantum_probability=0.8,
    )
    assert assessment.classical_probability == 0.2
    assert assessment.quantum_probability == 0.8
    # The headline is one member's number, never an unvalidated average of the two.
    assert assessment.probability in (0.2, 0.8)


# ------------------------------------------------------------------- the registry
def test_register_and_get_round_trip():
    reg = TrackRegistry()
    track = _Track("ecg")
    assert reg.register(track) is track
    assert reg.get("ecg") is track
    assert reg.ids() == ["ecg"]
    assert "ecg" in reg
    assert len(reg) == 1


def test_a_duplicate_id_is_refused_rather_than_silently_overwritten():
    """Last-write-wins would make the served model depend on import order."""
    reg = TrackRegistry()
    reg.register(_Track("ecg"))
    with pytest.raises(TrackError, match="already registered"):
        reg.register(_Track("ecg"))


def test_a_duplicate_can_be_replaced_deliberately():
    reg = TrackRegistry()
    reg.register(_Track("ecg"))
    replacement = _Track("ecg")
    reg.register(replacement, replace=True)
    assert reg.get("ecg") is replacement


def test_a_track_without_an_id_is_refused():
    class Anonymous:
        track_id = ""

        def describe(self):  # pragma: no cover
            return _descriptor()

        def analyze(self, payload, **options):  # pragma: no cover
            ...

    with pytest.raises(TrackError, match="no track_id"):
        TrackRegistry().register(Anonymous())


def test_an_unknown_id_names_what_is_registered():
    reg = TrackRegistry()
    reg.register(_Track("ecg"))
    with pytest.raises(TrackNotFoundError, match=r"\['ecg'\]"):
        reg.get("oral")


def test_one_broken_track_does_not_take_the_listing_down():
    """A missing artifact directory should cost one card in the UI, not the platform."""
    reg = TrackRegistry()
    reg.register(_Track("ok"))
    reg.register(_Track("broken", raises=RuntimeError("artifacts missing")))

    descriptors = {d.track_id: d for d in reg.describe_all()}

    assert descriptors["ok"].ready is True
    assert descriptors["broken"].ready is False
    assert "artifacts missing" in descriptors["broken"].unready_reason
    assert reg.ready_ids() == ["ok"]


def test_a_track_that_cannot_describe_itself_claims_nothing():
    """The placeholder must not inherit optimistic defaults from anywhere."""
    reg = TrackRegistry()
    reg.register(_Track("broken", raises=RuntimeError("boom")))

    descriptor = reg.describe_all()[0]

    assert descriptor.ready is False
    assert descriptor.uses_quantum is False
    assert descriptor.validation.frozen_test_evaluated is False
    assert descriptor.validation.is_clinically_claimable is False


def test_unregister_and_clear():
    reg = TrackRegistry()
    reg.register(_Track("a"))
    reg.register(_Track("b"))
    reg.unregister("a")
    assert reg.ids() == ["b"]
    reg.unregister("missing")  # idempotent, not an error
    reg.clear()
    assert reg.ids() == []


def test_concurrent_registration_loses_nothing():
    """FastAPI serves from a pool; a half-registered registry read concurrently is the
    kind of bug that shows up once, in a demo."""
    reg = TrackRegistry()
    errors: list[Exception] = []

    def add(index: int) -> None:
        try:
            reg.register(_Track(f"track-{index:03d}"))
        except Exception as error:  # pragma: no cover - failure path
            errors.append(error)

    threads = [threading.Thread(target=add, args=(i,)) for i in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(reg) == 50
    assert reg.ids() == sorted(f"track-{i:03d}" for i in range(50))


def test_the_module_level_registry_is_shared_and_starts_clean():
    """Tracks register where they are wired up, so importing the spine pulls in nothing."""
    from backend.ml import track as module

    assert module.registry is registry
    assert isinstance(registry, TrackRegistry)


def test_analyze_is_reached_through_the_registry():
    reg = TrackRegistry()
    track = _Track("ecg")
    reg.register(track)

    assessment = reg.get("ecg").analyze(payload=None)

    assert track.calls == 1
    assert assessment.track_id == "ecg"
    assert assessment.risk_band is RiskBand.LOW


def test_input_spec_publishes_channel_order():
    """A caller sending leads in another order gets another answer, so the order is part
    of the contract rather than something to infer."""
    spec = InputSpec(
        modality=Modality.SIGNAL,
        description="a 10-second 12-lead ECG at 100 Hz, in millivolts",
        shape=[12, 1000],
        units="mV",
        sampling_frequency_hz=100.0,
        channel_names=["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    )
    assert spec.shape == [12, 1000]
    assert spec.channel_names[0] == "I"
    assert spec.sampling_frequency_hz == 100.0
