"""Response schema for the ROI localisation endpoint.

Kept separate from :mod:`backend.schemas.result` on purpose: localisation is a
*non-gating* visual aid (DEC-034). ``POST /api/localize`` runs only the MobileNet ROI
localiser and never the screening pipeline, so its frozen centre-crop calibration is
untouched. The client draws an overlay box when :attr:`LocalizationResponse.localized`
is true and shows nothing otherwise, so an unavailable or unsure localiser can never
block or alter a screening verdict.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class LocalizationResponse(BaseModel):
    """One localisation attempt over a single image, for a client-side overlay.

    Coordinates are reported two ways so the client never has to guess the frame:
    :attr:`box_normalised` is resolution-independent (fractions of width/height) and
    is what the overlay draws; :attr:`roi_box_pixels` is the concrete ROI in the
    source image's own pixels, alongside :attr:`source_width` / :attr:`source_height`.
    """

    status: str = Field(description="localized | fallback_used | rejected")
    localized: bool = Field(
        description="True only when a box passed every acceptance check. The client "
        "draws the overlay iff this is true."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The localiser's learned self-estimate of how well it localised "
        "this image -- not a lesion-presence probability.",
    )
    box_normalised: Optional[List[float]] = Field(
        default=None,
        serialization_alias="boxNormalised",
        description="Predicted box [x0, y0, x1, y1] as fractions of the frame in "
        "[0, 1]. Present even when rejected, so a weak prediction is inspectable.",
    )
    roi_box_pixels: Optional[List[int]] = Field(
        default=None,
        serialization_alias="roiBoxPixels",
        description="The accepted (or fallback) ROI in source pixels [x0, y0, x1, y1].",
    )
    source_width: int = Field(serialization_alias="sourceWidth")
    source_height: int = Field(serialization_alias="sourceHeight")
    roi_source: Optional[str] = Field(
        default=None,
        serialization_alias="roiSource",
        description="predicted | predicted_rejected | center_crop | ... -- the "
        "provenance of the ROI, kept distinguishable downstream.",
    )
    reasons: List[str] = Field(
        default_factory=list,
        description="Why a prediction was not accepted, when it was not. Empty on a "
        "clean localisation.",
    )
    localizer_version: str = Field(serialization_alias="localizerVersion")
