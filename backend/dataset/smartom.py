"""SMART-OM dataset indexer.

The SMART-OM release ships as a four-level directory tree::

    <root>/
      01. Normal/
        01. Unannotated/<site>/<image>
        02. Region annotation/<site or json dir>/...
        03. Full annotation/...
      02. Variation from normal/
        ... plus 04. Lesion annotation/
      03. OPMD/
      04. Oral Cancer/
      Descriptors/*.xlsx
      Metadata/Patient_s Metadata.xlsx

Nothing in this module assumes those names are stable: directory roles are
resolved by numeric prefix and matched case-insensitively, and every unmatched
path is reported rather than silently dropped.

The canonical sample set is the ``01. Unannotated`` tree. The other annotation
trees contain re-renderings of the *same* source images (boxes/masks burned in),
so treating them as additional samples would duplicate every observation two to
four times. See ``DECISIONS.md`` DEC-015.
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})

# Numeric prefix -> canonical diagnostic class.
CLASS_BY_PREFIX: Dict[str, str] = {
    "01": "normal",
    "02": "variation_from_normal",
    "03": "opmd",
    "04": "oral_cancer",
}

# Numeric prefix -> annotation tree role.
ANNOTATION_BY_PREFIX: Dict[str, str] = {
    "01": "unannotated",
    "02": "region",
    "03": "full",
    "04": "lesion",
}

# Filename site token -> canonical anatomical site.
SITE_TOKENS: Dict[str, str] = {
    "DT": "dorsal_tongue",
    "VT": "ventral_tongue",
    "LB": "left_buccal_mucosa",
    "RB": "right_buccal_mucosa",
    "UL": "upper_lip",
    "LL": "lower_lip",
    "UA": "upper_arch",
    "LA": "lower_arch",
}

# Directory site name fragment -> canonical anatomical site.
SITE_DIR_FRAGMENTS: Sequence[Tuple[str, str]] = (
    ("dorsal tongue", "dorsal_tongue"),
    ("ventral tongue", "ventral_tongue"),
    ("left buccal", "left_buccal_mucosa"),
    ("right buccal", "right_buccal_mucosa"),
    ("upper lip", "upper_lip"),
    ("lower lip", "lower_lip"),
    ("upper arch", "upper_arch"),
    ("lower arch", "lower_arch"),
)

_SMITA_RE = re.compile(r"(SMITA\s*0*(\d+))", re.IGNORECASE)
_CA_RE = re.compile(r"^CA[\s_-]*(\d+)", re.IGNORECASE)
_PURE_NUM_RE = re.compile(r"^(\d+)$")
_LEGACY_SITE_RE = re.compile(r"^(\d+)\s*-\s*_?([A-Z]{2})$", re.IGNORECASE)
_CAPTURE_RE = re.compile(r"_(R|W)_+", re.IGNORECASE)
# Suffixes appended by the annotation renderers; stripped before matching.
_ANNOTATION_SUFFIX_RE = re.compile(r"_(region|full|lesion)$", re.IGNORECASE)


class AnnotationKind(str, Enum):
    """Role of an annotation tree inside a diagnostic-class folder."""

    UNANNOTATED = "unannotated"
    REGION = "region"
    FULL = "full"
    LESION = "lesion"


@dataclass(frozen=True)
class Polygon:
    """A closed region-of-interest outline in pixel coordinates."""

    xs: Tuple[float, ...]
    ys: Tuple[float, ...]
    shape: str

    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Return ``(left, top, right, bottom)`` as integers."""
        return (
            int(min(self.xs)),
            int(min(self.ys)),
            int(max(self.xs)) + 1,
            int(max(self.ys)) + 1,
        )

    @property
    def area(self) -> float:
        """Shoelace area of the outline."""
        n = len(self.xs)
        if n < 3:
            return 0.0
        total = 0.0
        for i in range(n):
            j = (i + 1) % n
            total += self.xs[i] * self.ys[j] - self.xs[j] * self.ys[i]
        return abs(total) / 2.0


@dataclass
class ImageRecord:
    """One canonical dataset observation (a single unannotated capture)."""

    image_id: str
    """Stable identifier: ``<class>/<site>/<filename stem>``."""

    path: Path
    filename: str
    diagnostic_class: str
    site: str
    patient_id: str
    patient_id_source: str
    """How ``patient_id`` was derived: ``smita`` | ``ca_series`` | ``numeric`` |
    ``legacy_numeric`` | ``unresolved``."""

    capture_mode: Optional[str] = None
    """``R`` or ``W`` capture token from the filename, when present."""

    descriptor_label: Optional[str] = None
    """Primary lesion descriptor from the Descriptors workbooks."""

    descriptor_terms: Tuple[str, ...] = ()
    """All non-empty descriptor cells for the row (site, artefacts, ...)."""

    region_polygons: Tuple[Polygon, ...] = ()
    lesion_polygons: Tuple[Polygon, ...] = ()

    @property
    def has_lesion_roi(self) -> bool:
        return len(self.lesion_polygons) > 0

    @property
    def has_region_roi(self) -> bool:
        return len(self.region_polygons) > 0


@dataclass
class SmartOmIndex:
    """Fully resolved dataset index plus everything that failed to resolve."""

    root: Path
    records: List[ImageRecord] = field(default_factory=list)
    annotation_image_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    """``(diagnostic_class, annotation_kind) -> image count`` across all trees."""

    patient_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    clinical_findings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    descriptor_rows: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    unmatched_paths: List[str] = field(default_factory=list)
    unmatched_descriptor_keys: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def by_class(self) -> Dict[str, List[ImageRecord]]:
        out: Dict[str, List[ImageRecord]] = {}
        for record in self.records:
            out.setdefault(record.diagnostic_class, []).append(record)
        return out

    def by_patient(self) -> Dict[str, List[ImageRecord]]:
        out: Dict[str, List[ImageRecord]] = {}
        for record in self.records:
            out.setdefault(record.patient_id, []).append(record)
        return out


# --------------------------------------------------------------------------
# Root discovery
# --------------------------------------------------------------------------


def _looks_like_dataset_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    prefixes = {
        child.name.split(".")[0].strip()
        for child in path.iterdir()
        if child.is_dir()
    }
    return bool(prefixes & set(CLASS_BY_PREFIX))


def discover_dataset_root(start: Optional[Path] = None) -> Optional[Path]:
    """Locate the dataset root without assuming its name.

    Honours ``CARESCAN_DATASET_ROOT`` first, then searches ``start`` and its
    immediate subdirectories for a folder containing the numeric class prefixes.
    """
    env = os.environ.get("CARESCAN_DATASET_ROOT")
    if env:
        candidate = Path(env).expanduser()
        return candidate if _looks_like_dataset_root(candidate) else None

    base = Path(start) if start is not None else Path(__file__).resolve().parents[2]
    if _looks_like_dataset_root(base):
        return base
    for child in sorted(base.iterdir()):
        if child.is_dir() and not child.name.startswith(".") and _looks_like_dataset_root(child):
            return child
    return None


# --------------------------------------------------------------------------
# Path / filename parsing
# --------------------------------------------------------------------------


def _numeric_prefix(name: str) -> Optional[str]:
    stripped = name.strip()
    if len(stripped) >= 2 and stripped[:2].isdigit():
        return stripped[:2]
    return None


def resolve_site(site_dir: str, filename_stem: str) -> str:
    """Resolve anatomical site, preferring the filename token over the folder."""
    token_match = _CAPTURE_RE.search(filename_stem)
    if token_match:
        tail = filename_stem[token_match.end():]
        token = re.match(r"([A-Za-z]{2})", tail)
        if token and token.group(1).upper() in SITE_TOKENS:
            return SITE_TOKENS[token.group(1).upper()]

    legacy = _LEGACY_SITE_RE.match(filename_stem.strip())
    if legacy and legacy.group(2).upper() in SITE_TOKENS:
        return SITE_TOKENS[legacy.group(2).upper()]

    folded = site_dir.lower()
    for fragment, canonical in SITE_DIR_FRAGMENTS:
        if fragment in folded:
            return canonical
    return "unknown"


def resolve_patient_id(filename_stem: str, diagnostic_class: str) -> Tuple[str, str]:
    """Derive a patient identifier and record how it was derived.

    SMART-OM mixes several naming cohorts. ``SMITA#####`` filenames carry an
    explicit subject id. The remaining cohorts (``Ca N``, bare numerics, and
    ``N - SITE``) carry only a series number, which is namespaced per cohort
    because there is no evidence the numbering spaces are shared. When a stem
    matches nothing, the image becomes its own singleton patient so it can never
    silently join another subject's group.
    """
    stem = unicodedata.normalize("NFKC", filename_stem).strip()
    stem = _ANNOTATION_SUFFIX_RE.sub("", stem)

    smita = _SMITA_RE.search(stem)
    if smita:
        return f"SMITA{int(smita.group(2)):05d}", "smita"

    ca = _CA_RE.match(stem)
    if ca:
        return f"CA-{int(ca.group(1)):04d}", "ca_series"

    legacy = _LEGACY_SITE_RE.match(stem)
    if legacy:
        return f"SER-{diagnostic_class}-{int(legacy.group(1)):04d}", "legacy_numeric"

    pure = _PURE_NUM_RE.match(stem)
    if pure:
        return f"NUM-{int(pure.group(1)):05d}", "numeric"

    slug = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").upper() or "UNKNOWN"
    return f"UNRESOLVED-{diagnostic_class}-{slug}", "unresolved"


def resolve_capture_mode(filename_stem: str) -> Optional[str]:
    match = _CAPTURE_RE.search(filename_stem)
    return match.group(1).upper() if match else None


def descriptor_key(filename_stem: str) -> str:
    """Normalise a filename stem for descriptor/annotation cross-referencing."""
    stem = unicodedata.normalize("NFKC", str(filename_stem)).strip()
    stem = _ANNOTATION_SUFFIX_RE.sub("", stem)
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def scoped_key(diagnostic_class: str, filename_stem: str) -> str:
    """Class-scoped cross-reference key.

    Scoping is required, not cosmetic: 29 filename stems occur in more than one
    diagnostic-class folder (a subject's dorsal-tongue capture can be filed under
    Normal for one visit and under Variation from normal for another). An
    unscoped filename key makes a Normal image inherit the lesion descriptor and
    lesion polygon of its same-named Variation sibling.
    """
    return f"{diagnostic_class}::{descriptor_key(filename_stem)}"


# Descriptor/annotation sheet name fragment -> canonical diagnostic class.
SHEET_CLASS_FRAGMENTS: Sequence[Tuple[str, str]] = (
    ("variation", "variation_from_normal"),
    ("opmd", "opmd"),
    ("cancer", "oral_cancer"),
    ("normal", "normal"),
)


def resolve_sheet_class(sheet_name: str) -> Optional[str]:
    """Map a workbook sheet name onto a diagnostic class.

    ``variation`` is tested before ``normal`` because the sheet is literally
    named "Variation from normal".
    """
    folded = str(sheet_name).strip().lower()
    for fragment, canonical in SHEET_CLASS_FRAGMENTS:
        if fragment in folded:
            return canonical
    return None


# --------------------------------------------------------------------------
# VIA polygon annotations
# --------------------------------------------------------------------------


def _polygons_from_via_region(region: Any) -> Optional[Polygon]:
    if not isinstance(region, dict):
        return None
    shape = region.get("shape_attributes")
    if not isinstance(shape, dict):
        return None
    name = str(shape.get("name", "")).lower()

    if name in {"polygon", "polyline"}:
        xs = shape.get("all_points_x") or []
        ys = shape.get("all_points_y") or []
        if len(xs) < 3 or len(xs) != len(ys):
            return None
        return Polygon(tuple(float(v) for v in xs), tuple(float(v) for v in ys), name)

    if name == "rect":
        try:
            x, y = float(shape["x"]), float(shape["y"])
            w, h = float(shape["width"]), float(shape["height"])
        except (KeyError, TypeError, ValueError):
            return None
        if w <= 0 or h <= 0:
            return None
        return Polygon((x, x + w, x + w, x), (y, y, y + h, y + h), name)

    if name in {"ellipse", "circle"}:
        try:
            cx, cy = float(shape["cx"]), float(shape["cy"])
        except (KeyError, TypeError, ValueError):
            return None
        rx = float(shape.get("rx", shape.get("r", 0)) or 0)
        ry = float(shape.get("ry", shape.get("r", 0)) or 0)
        if rx <= 0 or ry <= 0:
            return None
        xs, ys = [], []
        for i in range(24):
            angle = 2 * math.pi * i / 24
            xs.append(cx + rx * math.cos(angle))
            ys.append(cy + ry * math.sin(angle))
        return Polygon(tuple(xs), tuple(ys), name)

    return None


def load_via_annotations(json_path: Path) -> Dict[str, List[Polygon]]:
    """Parse one VGG Image Annotator project file into ``filename key -> polygons``.

    Keys are unscoped :func:`descriptor_key` values; the caller scopes them with
    the diagnostic class of the tree the file was found in.
    """
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}

    metadata = payload.get("_via_img_metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        return {}

    out: Dict[str, List[Polygon]] = {}
    for entry in metadata.values():
        if not isinstance(entry, dict):
            continue
        filename = entry.get("filename")
        if not filename:
            continue
        key = descriptor_key(Path(str(filename)).stem)
        polygons = [
            poly
            for poly in (_polygons_from_via_region(r) for r in entry.get("regions") or [])
            if poly is not None and poly.area > 0
        ]
        if polygons:
            out.setdefault(key, []).extend(polygons)
    return out


# --------------------------------------------------------------------------
# Workbook parsing
# --------------------------------------------------------------------------


def _read_workbook(path: Path) -> Dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a declared dependency
        return {}
    try:
        book = pd.ExcelFile(path)
    except (OSError, ValueError):
        return {}
    return {name: book.parse(name) for name in book.sheet_names}


def _find_filename_column(columns: Iterable[Any]) -> Optional[Any]:
    """Locate the image-filename column in a descriptor sheet.

    Matching on ``"name"`` alone is not enough: the Normal sheets spell the
    header ``Fileame`` (a typo in the published dataset), which contains no
    ``name`` substring. Both spellings are covered by also accepting ``file``.
    """
    candidates = [c for c in columns if isinstance(c, str)]
    for token in ("file_name", "filename", "file name", "fileame", "file"):
        for col in candidates:
            if token in col.strip().lower():
                return col
    for col in candidates:
        if "name" in col.strip().lower():
            return col
    return None


def load_descriptors(root: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Load per-image lesion descriptors from the ``Descriptors`` workbooks.

    Returns descriptors keyed by :func:`scoped_key`, so a stem that appears under
    two diagnostic classes gets two independent descriptor entries.
    """
    warnings: List[str] = []
    descriptors: Dict[str, Dict[str, Any]] = {}

    descriptor_dir = next(
        (d for d in root.iterdir() if d.is_dir() and "descriptor" in d.name.lower()),
        None,
    )
    if descriptor_dir is None:
        warnings.append("No Descriptors directory found; per-image labels unavailable.")
        return descriptors, warnings

    for workbook in sorted(descriptor_dir.glob("*.xlsx")):
        kind = "region"
        lowered = workbook.name.lower()
        if "full" in lowered:
            kind = "full"
        elif "lesion" in lowered:
            kind = "lesion"

        for sheet_name, frame in _read_workbook(workbook).items():
            if frame is None or frame.empty:
                continue
            sheet_class = resolve_sheet_class(sheet_name)
            if sheet_class is None:
                warnings.append(
                    f"{workbook.name}:{sheet_name} maps to no diagnostic class; skipped."
                )
                continue
            name_col = _find_filename_column(frame.columns)
            if name_col is None:
                warnings.append(
                    f"{workbook.name}:{sheet_name} has no filename column; skipped."
                )
                continue
            term_cols = [
                c for c in frame.columns if c != name_col and str(c).strip().upper() != "S.NO"
            ]

            for _, row in frame.iterrows():
                raw_name = row[name_col]
                if raw_name is None or str(raw_name).strip() in {"", "nan", "NaN"}:
                    continue
                key = scoped_key(sheet_class, Path(str(raw_name)).stem)
                if key.endswith("::"):
                    continue
                terms = [
                    str(row[c]).strip()
                    for c in term_cols
                    if str(row.get(c, "")).strip() not in {"", "nan", "NaN", "-", "None"}
                ]
                entry = descriptors.setdefault(
                    key,
                    {
                        "filename": str(raw_name).strip(),
                        "sheet": str(sheet_name),
                        "diagnostic_class": sheet_class,
                        "terms": [],
                    },
                )
                entry["terms"].extend(terms)
                if kind == "lesion" and terms:
                    entry["lesion_label"] = terms[0]
                elif kind == "region" and terms and "region_label" not in entry:
                    entry["region_label"] = terms[0]

    return descriptors, warnings


def _flatten_metadata_header(frame: Any) -> Any:
    """Merge the SMART-OM two-row header into single column names.

    The workbook uses a merged header: row 0 of the data holds sub-headers for
    grouped columns (``Smoking`` -> ``Type``, ``Frequency``, ...) and pandas
    surfaces those groups as ``Unnamed: N``.
    """
    import pandas as pd

    if frame.empty:
        return frame

    sub_header = frame.iloc[0]
    columns: List[str] = []
    group = ""
    for col, sub in zip(frame.columns, sub_header):
        col_str = str(col)
        if not col_str.startswith("Unnamed"):
            group = col_str.strip()
        sub_str = "" if pd.isna(sub) else str(sub).strip()
        if col_str.startswith("Unnamed"):
            columns.append(f"{group} :: {sub_str}" if sub_str else f"{group} :: extra")
        elif sub_str and sub_str.lower() not in {group.lower(), "nan"}:
            columns.append(f"{group} :: {sub_str}")
        else:
            columns.append(group)

    body = frame.iloc[1:].copy()
    body.columns = columns
    return body.reset_index(drop=True)


def load_patient_metadata(
    root: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[str]]:
    """Load demographics/habits and clinical findings keyed by patient id."""
    warnings: List[str] = []
    demographics: Dict[str, Dict[str, Any]] = {}
    findings: Dict[str, Dict[str, Any]] = {}

    metadata_dir = next(
        (d for d in root.iterdir() if d.is_dir() and "metadata" in d.name.lower()),
        None,
    )
    if metadata_dir is None:
        warnings.append("No Metadata directory found; clinical features unavailable.")
        return demographics, findings, warnings

    workbook = next(iter(sorted(metadata_dir.glob("*.xlsx"))), None)
    if workbook is None:
        warnings.append("Metadata directory contains no .xlsx workbook.")
        return demographics, findings, warnings

    import pandas as pd

    for sheet_name, frame in _read_workbook(workbook).items():
        if frame is None or frame.empty:
            continue
        lowered = str(sheet_name).lower()
        target = findings if "finding" in lowered else demographics

        if "demograph" in lowered:
            frame = _flatten_metadata_header(frame)
        else:
            # Clinical findings sheet also carries a blank spacer row.
            id_col_guess = next(
                (c for c in frame.columns if isinstance(c, str) and "smita" in c.lower()),
                None,
            )
            if id_col_guess is not None and frame[id_col_guess].isna().iloc[0]:
                frame = frame.iloc[1:].reset_index(drop=True)

        id_col = next(
            (c for c in frame.columns if isinstance(c, str) and "smita" in c.lower()),
            None,
        )
        if id_col is None:
            warnings.append(f"Metadata sheet '{sheet_name}' has no SMITA_ID column.")
            continue

        for _, row in frame.iterrows():
            raw_id = row[id_col]
            if raw_id is None or pd.isna(raw_id):
                continue
            patient_id, source = resolve_patient_id(str(raw_id), "metadata")
            if source != "smita":
                continue
            record = {
                str(col): (None if pd.isna(val) else val)
                for col, val in row.items()
                if str(col) != str(id_col)
            }
            target[patient_id] = record

    return demographics, findings, warnings


# --------------------------------------------------------------------------
# Index construction
# --------------------------------------------------------------------------


def build_index(
    root: Optional[Path] = None,
    *,
    load_annotations: bool = True,
    load_metadata: bool = True,
) -> SmartOmIndex:
    """Walk the dataset and build a fully cross-referenced index.

    Raises:
        FileNotFoundError: if no dataset root can be resolved.
    """
    resolved = Path(root) if root is not None else discover_dataset_root()
    if resolved is None or not resolved.is_dir():
        raise FileNotFoundError(
            "SMART-OM dataset root not found. Set CARESCAN_DATASET_ROOT or pass root=."
        )

    index = SmartOmIndex(root=resolved)
    region_polys: Dict[str, List[Polygon]] = {}
    lesion_polys: Dict[str, List[Polygon]] = {}
    canonical: List[Tuple[Path, str, str, str]] = []  # path, class, annotation, site dir

    for class_dir in sorted(p for p in resolved.iterdir() if p.is_dir()):
        class_prefix = _numeric_prefix(class_dir.name)
        diagnostic_class = CLASS_BY_PREFIX.get(class_prefix or "")
        if diagnostic_class is None:
            continue

        for annotation_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            annotation = ANNOTATION_BY_PREFIX.get(_numeric_prefix(annotation_dir.name) or "")
            if annotation is None:
                index.unmatched_paths.append(str(annotation_dir.relative_to(resolved)))
                continue

            for sub_dir in sorted(p for p in annotation_dir.iterdir() if p.is_dir()):
                for entry in sorted(sub_dir.iterdir()):
                    if not entry.is_file():
                        continue
                    suffix = entry.suffix.lower()
                    if suffix in IMAGE_EXTENSIONS:
                        key = (diagnostic_class, annotation)
                        index.annotation_image_counts[key] = (
                            index.annotation_image_counts.get(key, 0) + 1
                        )
                        if annotation == AnnotationKind.UNANNOTATED.value:
                            canonical.append((entry, diagnostic_class, annotation, sub_dir.name))
                    elif suffix == ".json" and load_annotations:
                        parsed = load_via_annotations(entry)
                        sink = lesion_polys if annotation == "lesion" else region_polys
                        for key, polys in parsed.items():
                            sink.setdefault(f"{diagnostic_class}::{key}", []).extend(polys)
                    elif suffix not in {".xlsx", ".xls", ".csv", ".txt", ".md"}:
                        index.unmatched_paths.append(str(entry.relative_to(resolved)))

    descriptors, descriptor_warnings = load_descriptors(resolved)
    index.descriptor_rows = descriptors
    index.warnings.extend(descriptor_warnings)

    if load_metadata:
        demographics, findings, metadata_warnings = load_patient_metadata(resolved)
        index.patient_metadata = demographics
        index.clinical_findings = findings
        index.warnings.extend(metadata_warnings)

    used_descriptor_keys = set()
    # image_id is `<class>/<site>/<stem>`, which drops the file extension -- and the
    # dataset contains one genuine collision: `04. Oral Cancer/01. Unannotated/
    # 02. Ventral tongue/` holds both `Ca 2.jpg` (618x442) and `Ca 2.png` (789x402).
    # They are different captures, not two encodings of one image. Left alone, they
    # would share an identifier, which silently breaks every dict keyed by image_id
    # (descriptor lookup, quality exclusions, split manifests, feature caches) and
    # puts one image into a partition twice.
    #
    # Colliding IDs are therefore suffixed with the file extension. Only the *second*
    # and later occurrences change, so 2468 of 2469 identifiers keep the plain form
    # and existing manifests stay valid for them. `canonical` is built from sorted()
    # iteration, so which record keeps the plain ID is deterministic.
    seen_image_ids: Dict[str, int] = {}
    for path, diagnostic_class, _annotation, site_dir in canonical:
        stem = path.stem
        site = resolve_site(site_dir, stem)
        patient_id, source = resolve_patient_id(stem, diagnostic_class)
        # Descriptor rows are keyed by stem in the source workbooks, so colliding
        # files necessarily share one descriptor row. Recorded as-is rather than
        # guessed: the workbook does not say which file it describes.
        key = scoped_key(diagnostic_class, stem)
        descriptor = descriptors.get(key)
        if descriptor is not None:
            used_descriptor_keys.add(key)

        terms = tuple(dict.fromkeys(descriptor["terms"])) if descriptor else ()
        label = None
        if descriptor:
            label = descriptor.get("lesion_label") or descriptor.get("region_label")
            if label is None and terms:
                label = terms[0]

        image_id = f"{diagnostic_class}/{site}/{stem}"
        if image_id in seen_image_ids:
            collision = f"{image_id}{path.suffix.lower()}"
            if collision in seen_image_ids:
                # Same stem and same extension in two site directories mapping to one
                # site: fall back to an ordinal so the ID is still unique.
                collision = f"{image_id}{path.suffix.lower()}#{seen_image_ids[image_id]}"
            index.warnings.append(
                f"Duplicate image_id {image_id!r} from {path.name!r}; "
                f"disambiguated to {collision!r}."
            )
            seen_image_ids[image_id] += 1
            image_id = collision
        seen_image_ids.setdefault(image_id, 1)

        index.records.append(
            ImageRecord(
                image_id=image_id,
                path=path,
                filename=path.name,
                diagnostic_class=diagnostic_class,
                site=site,
                patient_id=patient_id,
                patient_id_source=source,
                capture_mode=resolve_capture_mode(stem),
                descriptor_label=label,
                descriptor_terms=terms,
                region_polygons=tuple(region_polys.get(key, ())),
                lesion_polygons=tuple(lesion_polys.get(key, ())),
            )
        )

    index.unmatched_descriptor_keys = sorted(set(descriptors) - used_descriptor_keys)
    return index


@lru_cache(maxsize=4)
def cached_index(root: Optional[str] = None) -> SmartOmIndex:
    """Process-cached index, for CLI tools that build it more than once."""
    return build_index(Path(root) if root else None)


def iter_image_paths(root: Path) -> Iterable[Path]:
    """Yield every image file under ``root``, regardless of tree role."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                yield Path(dirpath) / name
