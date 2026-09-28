"""Concrete disease tracks, and the wiring that registers them.

Importing this package registers every track that can be constructed without loading an
artifact. Construction is cheap and deliberately lazy: a track's ``__init__`` must not
touch the filesystem, so a deployment missing one model still starts, serves the rest of
the platform, and reports the missing one as ``ready=False`` with a reason.
"""

from __future__ import annotations

from typing import List

from backend.ml.track import DiseaseTrack, TrackRegistry, registry
from backend.ml.tracks.ecg import EcgTrack
from backend.ml.tracks.oral_lesion import OralLesionTrack

__all__ = ["EcgTrack", "OralLesionTrack", "install_default_tracks"]


def install_default_tracks(into: TrackRegistry | None = None) -> List[DiseaseTrack]:
    """Register the tracks this deployment serves.

    Idempotent: re-registers in place rather than raising, so calling it from both
    application startup and a test fixture is safe. That is the one case where replacing
    a registration is not a configuration mistake.
    """
    target = registry if into is None else into
    tracks: List[DiseaseTrack] = [OralLesionTrack(), EcgTrack()]
    for track in tracks:
        target.register(track, replace=True)
    return tracks
