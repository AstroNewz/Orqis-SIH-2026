"""Clinical risk-factor encoding.

Processed strictly separately from image features (PART 7): this module never
sees pixels, and the fusion stage is the only place the two meet. Encoders are
fitted on **training patients only** and persisted, so validation/test/production
rows are transformed with train-derived statistics and never contribute to them.

Column mapping onto the three named risk factors
------------------------------------------------
The demographics sheet does not use the words "betel quid". Its habit columns are:

    Habit_history                     Yes / No          <- master gate
    Type_of_habit                     Tobacco | Alcohol | Arecanut (free text)
    Form_of_tobacco                   Smoking | Chewing (free text)
    Smoking  :: Habit_status / Frequency (per day)  / Duration of Habit (years)
    Chewing  :: Habit_status / Frequency (per day)  / Duration of Habit (years)
    Arecanut :: Habit_status / Frequency (per day)  / Duration of Habit (years)
    Alcohol  :: Habit status / Frequency (per week) / Duration of Habit (years)

Mapping decision, recorded in DECISIONS.md:

- **smoking**   -> ``Smoking :: *``
- **alcohol**   -> ``Alcohol :: *``
- **betel_quid** -> union of ``Arecanut :: *`` and ``Chewing :: *``. Betel quid is
  areca nut wrapped in betel leaf, chewed with or without tobacco; this dataset
  records the areca-nut component and the smokeless-tobacco-chewing component in
  separate column families. Taking their union is the closest faithful
  representation of betel-quid exposure available here. It is an explicit
  interpretation of the source columns, not a clinical claim.

The ``Habit_history`` gate matters for missingness
-------------------------------------------------
214 of 304 patients record ``Habit_history = No``. Their habit-specific cells
hold ``-``. For those patients ``-`` means *known absent*, not *unknown*. Only
patients with ``Habit_history = Yes`` (or blank) and an empty habit cell are
encoded as missing. Collapsing the two would throw away 214 genuine negatives
and inflate the missing-data rate roughly sevenfold.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from backend.ml.types import ClinicalFeatures

ENCODER_VERSION = "clinical-encoder-1"

# Cells that carry no information. "-" and "_" are the sheet's placeholder glyphs.
_BLANK_TOKENS = frozenset({"", "-", "_", "--", "nan", "none", "null", "na", "n/a"})

# Values of a ``:: Habit_status`` cell that mean the patient currently has the habit.
_AFFIRMATIVE_STATUS = ("current user", "current", "former", "past user", "past", "yes")

RISK_FACTORS: Tuple[str, ...] = ("smoking", "alcohol", "betel_quid")

# Column families backing each risk factor. Several spellings are listed because
# the sheet writes "Alcohol :: Habit status" without the underscore.
_FACTOR_COLUMNS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "smoking": {
        "status": ("Smoking :: Habit_status", "Smoking :: Habit status"),
        "frequency": ("Smoking :: Frequency (No of times per day)",),
        "duration": ("Smoking :: Duration of Habit (in years)",),
    },
    "alcohol": {
        "status": ("Alcohol :: Habit status", "Alcohol :: Habit_status"),
        "frequency": ("Alcohol :: Frequency (No of times per week)",),
        "duration": ("Alcohol :: Duration of Habit (in years)",),
    },
    # Two families are merged for betel quid; see the module docstring.
    "betel_quid": {
        "status": (
            "Arecanut :: Habit_status",
            "Arecanut :: Habit status",
            "Chewing :: Habit_status",
            "Chewing :: Habit status",
        ),
        "frequency": (
            "Arecanut :: Frequency (No of times per day)",
            "Chewing :: Frequency (No of times per day)",
        ),
        "duration": (
            "Arecanut :: Duration of Habit (in years)",
            "Chewing :: Duration of Habit (in years)",
        ),
    },
}

_HABIT_GATE_COLUMNS = ("Habit_history",)
_TYPE_COLUMNS = ("Type_of_habit",)
_FORM_COLUMNS = ("Form_of_tobacco",)
_AGE_COLUMNS = ("Age",)
_SEX_COLUMNS = ("Sex",)

# Free-text tokens in ``Type_of_habit`` / ``Form_of_tobacco`` that corroborate a
# factor when the ``::`` family is blank.
_TYPE_TOKENS: Dict[str, Tuple[str, ...]] = {
    "smoking": ("smoking", "cigarette", "beedi", "bidi"),
    "alcohol": ("alcohol",),
    "betel_quid": ("arecanut", "areca", "betel", "quid", "pan", "paan", "chewing", "gutkha"),
}

_SEX_NORMALISATION = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female",
}


def is_blank(value: Any) -> bool:
    """True when a metadata cell carries no information."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in _BLANK_TOKENS


def _first_present(row: Mapping[str, Any], columns: Sequence[str]) -> Optional[Any]:
    """First non-blank value among ``columns``, or ``None``."""
    for column in columns:
        if column in row and not is_blank(row[column]):
            return row[column]
    return None


def parse_numeric(value: Any) -> Optional[float]:
    """Extract the leading number from a free-text cell, or ``None``.

    Cells such as ``"2"``, ``"2-3"``, ``"10 years"`` and ``"2/day"`` all occur.
    The first number is taken; a range yields its lower bound, which is the
    conservative reading for an exposure measure.
    """
    if is_blank(value):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", "."))
    if match is None:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


@dataclass(frozen=True)
class FactorObservation:
    """Interpretation of one risk factor for one patient."""

    present: Optional[bool]
    """``True`` current/past user, ``False`` known absent, ``None`` unknown."""

    frequency: Optional[float]
    duration_years: Optional[float]
    evidence: str

    @property
    def known(self) -> bool:
        return self.present is not None


def read_factor(row: Mapping[str, Any], factor: str) -> FactorObservation:
    """Interpret one risk factor from a demographics row."""
    columns = _FACTOR_COLUMNS[factor]
    status_raw = _first_present(row, columns["status"])
    frequency = parse_numeric(_first_present(row, columns["frequency"]))
    duration = parse_numeric(_first_present(row, columns["duration"]))

    if status_raw is not None:
        affirmative = any(token in _text(status_raw) for token in _AFFIRMATIVE_STATUS)
        return FactorObservation(
            present=affirmative,
            frequency=frequency,
            duration_years=duration,
            evidence=f"status={status_raw!r}",
        )

    # No explicit status. Fall back to the free-text habit summary.
    free_text = " ".join(
        _text(_first_present(row, cols) or "")
        for cols in (_TYPE_COLUMNS, _FORM_COLUMNS)
    )
    if any(token in free_text for token in _TYPE_TOKENS[factor]):
        return FactorObservation(
            present=True,
            frequency=frequency,
            duration_years=duration,
            evidence=f"free_text={free_text.strip()!r}",
        )

    # Still nothing. The master gate decides whether blank means absent or unknown.
    gate = _first_present(row, _HABIT_GATE_COLUMNS)
    if gate is not None:
        gate_text = _text(gate)
        if gate_text.startswith("n"):  # "No"
            return FactorObservation(
                present=False,
                frequency=0.0,
                duration_years=0.0,
                evidence="Habit_history=No",
            )
        if gate_text.startswith("y") and free_text.strip():
            # Habits recorded, but none of them this one.
            return FactorObservation(
                present=False,
                frequency=0.0,
                duration_years=0.0,
                evidence=f"Habit_history=Yes, other habits only ({free_text.strip()!r})",
            )

    return FactorObservation(
        present=None, frequency=frequency, duration_years=duration, evidence="unknown"
    )


def row_from_flags(
    *,
    smoking: Optional[bool] = None,
    alcohol: Optional[bool] = None,
    betel_quid: Optional[bool] = None,
    age: Optional[float] = None,
    sex: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a demographics-shaped row from the API's boolean risk-factor flags.

    The encoder is fitted on the SMART-OM demographics sheet and reads its column
    names. A mobile client sends three booleans instead. This function performs that
    translation in the one file that owns the column names, so the two cannot drift.

    ``False`` is written as an explicit ``"No"`` rather than left blank. Blank would
    be encoded as *unknown*, and there is a real difference between a patient who
    stated they do not smoke and a patient who was never asked -- the encoder carries
    a separate ``*_unknown`` channel precisely to keep them apart.

    ``None`` leaves the column absent, which the encoder reads as unknown. Frequency
    and duration are never synthesised: the API does not collect them, so those
    channels fall back to the train-derived mean, which is what an absent cell does
    for a dataset patient too.

    Args:
        smoking: tobacco smoking, ``None`` if not asked.
        alcohol: alcohol consumption, ``None`` if not asked.
        betel_quid: betel quid / areca nut / chewing tobacco, ``None`` if not asked.
        age: years, ``None`` if not collected.
        sex: ``"male"``, ``"female"``, or any spelling the sheet uses; ``None`` if not
            collected. Unrecognised values are encoded as unknown rather than guessed.

    Returns:
        A mapping accepted by :meth:`ClinicalFeatureEncoder.transform`.
    """
    row: Dict[str, Any] = {}
    for factor, flag in (("smoking", smoking), ("alcohol", alcohol), ("betel_quid", betel_quid)):
        if flag is None:
            continue
        # The first spelling in each family is canonical; the alternates exist to
        # read the sheet, not to write it.
        row[_FACTOR_COLUMNS[factor]["status"][0]] = "Yes" if flag else "No"
    if age is not None:
        row[_AGE_COLUMNS[0]] = age
    if sex is not None:
        row[_SEX_COLUMNS[0]] = sex
    return row


def _feature_names(include_demographics: bool, sex_categories: Sequence[str]) -> List[str]:
    names: List[str] = []
    for factor in RISK_FACTORS:
        names.extend(
            [
                f"{factor}_present",
                f"{factor}_unknown",
                f"{factor}_frequency_z",
                f"{factor}_duration_z",
            ]
        )
    names.append("n_risk_factors_present")
    if include_demographics:
        names.extend(["age_z", "age_unknown"])
        names.extend([f"sex_{c}" for c in sex_categories])
        names.append("sex_unknown")
    return names


@dataclass
class ClinicalFeatureEncoder:
    """Deterministic encoder for clinical risk factors.

    Fitted statistics are the per-factor mean/std of frequency and duration among
    patients who *have* the habit, plus the mean/std of age and the observed sex
    categories. Nothing about the target is ever consulted, so no target
    information can leak into preprocessing.
    """

    include_demographics: bool = True
    version: str = ENCODER_VERSION

    fitted: bool = False
    frequency_stats: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    duration_stats: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    age_stats: Tuple[float, float] = (0.0, 1.0)
    sex_categories: List[str] = field(default_factory=list)
    n_fit_rows: int = 0

    # ------------------------------------------------------------------ fit
    def fit(self, rows: Iterable[Mapping[str, Any]]) -> "ClinicalFeatureEncoder":
        """Fit on training-patient rows only."""
        rows = list(rows)
        if not rows:
            raise ValueError("ClinicalFeatureEncoder.fit received no rows.")

        for factor in RISK_FACTORS:
            observations = [read_factor(row, factor) for row in rows]
            # Standardise using only patients who have the habit; including the
            # zeros of the 214 known-absent patients would collapse the scale.
            freqs = [
                o.frequency for o in observations if o.present and o.frequency is not None
            ]
            durs = [
                o.duration_years
                for o in observations
                if o.present and o.duration_years is not None
            ]
            self.frequency_stats[factor] = _mean_std(freqs)
            self.duration_stats[factor] = _mean_std(durs)

        ages = [a for a in (parse_numeric(_first_present(r, _AGE_COLUMNS)) for r in rows) if a is not None]
        self.age_stats = _mean_std(ages)

        seen: List[str] = []
        for row in rows:
            category = self._sex_category(row)
            if category and category not in seen:
                seen.append(category)
        self.sex_categories = sorted(seen)

        self.n_fit_rows = len(rows)
        self.fitted = True
        return self

    @staticmethod
    def _sex_category(row: Mapping[str, Any]) -> Optional[str]:
        raw = _first_present(row, _SEX_COLUMNS)
        if raw is None:
            return None
        return _SEX_NORMALISATION.get(_text(raw), _text(raw) or None)

    # ------------------------------------------------------------ transform
    @property
    def feature_names(self) -> List[str]:
        return _feature_names(self.include_demographics, self.sex_categories)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def transform(self, row: Optional[Mapping[str, Any]]) -> ClinicalFeatures:
        """Encode one patient. ``None`` yields the all-unknown encoding.

        A patient with no demographics row is a legitimate production case (27 of
        329 dataset patients, and any walk-in user of the app), so it is encoded
        explicitly rather than treated as an error.
        """
        if not self.fitted:
            raise RuntimeError("ClinicalFeatureEncoder must be fitted before transform().")

        row = dict(row or {})
        values: List[float] = []
        raw: Dict[str, Optional[str]] = {}
        n_missing = 0
        n_present = 0

        for factor in RISK_FACTORS:
            observation = read_factor(row, factor) if row else FactorObservation(
                None, None, None, "no metadata row"
            )
            unknown = 0.0 if observation.known else 1.0
            present = 1.0 if observation.present else 0.0
            if not observation.known:
                n_missing += 1
            if observation.present:
                n_present += 1

            freq_mean, freq_std = self.frequency_stats.get(factor, (0.0, 1.0))
            dur_mean, dur_std = self.duration_stats.get(factor, (0.0, 1.0))

            # An absent or unknown habit contributes 0 on the standardised axes:
            # the dedicated present/unknown flags already carry that information,
            # so imputing a mean here would fabricate exposure.
            freq_z = (
                (observation.frequency - freq_mean) / freq_std
                if observation.present and observation.frequency is not None
                else 0.0
            )
            dur_z = (
                (observation.duration_years - dur_mean) / dur_std
                if observation.present and observation.duration_years is not None
                else 0.0
            )

            values.extend([present, unknown, freq_z, dur_z])
            raw[factor] = observation.evidence

        values.append(float(n_present))

        if self.include_demographics:
            age = parse_numeric(_first_present(row, _AGE_COLUMNS)) if row else None
            age_mean, age_std = self.age_stats
            values.append((age - age_mean) / age_std if age is not None else 0.0)
            values.append(0.0 if age is not None else 1.0)
            if age is None:
                n_missing += 1
            raw["age"] = None if age is None else str(age)

            category = self._sex_category(row) if row else None
            for known_category in self.sex_categories:
                values.append(1.0 if category == known_category else 0.0)
            # Unseen or absent category -> explicit unknown flag rather than a
            # silent all-zeros row, which would be indistinguishable from a
            # category the encoder simply never saw during fitting.
            values.append(0.0 if category in self.sex_categories else 1.0)
            if category not in self.sex_categories:
                n_missing += 1
            raw["sex"] = category

        return ClinicalFeatures(
            values=[float(v) for v in values],
            names=self.feature_names,
            raw=raw,
            n_missing=n_missing,
        )

    def transform_many(
        self, rows: Sequence[Optional[Mapping[str, Any]]]
    ) -> np.ndarray:
        """Encode many patients into a ``(n_rows, n_features)`` matrix."""
        return np.vstack([self.transform(row).vector for row in rows]) if rows else np.empty((0, self.n_features))

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "include_demographics": self.include_demographics,
            "frequency_stats": {k: list(v) for k, v in self.frequency_stats.items()},
            "duration_stats": {k: list(v) for k, v in self.duration_stats.items()},
            "age_stats": list(self.age_stats),
            "sex_categories": list(self.sex_categories),
            "n_fit_rows": self.n_fit_rows,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClinicalFeatureEncoder":
        encoder = cls(
            include_demographics=bool(payload.get("include_demographics", True)),
            version=str(payload.get("version", ENCODER_VERSION)),
        )
        encoder.frequency_stats = {
            k: (float(v[0]), float(v[1])) for k, v in payload.get("frequency_stats", {}).items()
        }
        encoder.duration_stats = {
            k: (float(v[0]), float(v[1])) for k, v in payload.get("duration_stats", {}).items()
        }
        age = payload.get("age_stats", [0.0, 1.0])
        encoder.age_stats = (float(age[0]), float(age[1]))
        encoder.sex_categories = [str(c) for c in payload.get("sex_categories", [])]
        encoder.n_fit_rows = int(payload.get("n_fit_rows", 0))
        encoder.fitted = True
        return encoder


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and a numerically safe standard deviation."""
    if not values:
        return (0.0, 1.0)
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array))
    # A degenerate or single-valued column would otherwise divide by zero.
    return (mean, std if std > 1e-9 else 1.0)
