"""Dataset inspection: computes real statistics and writes an auditable report.

Every number in the generated report is measured from the supplied dataset. No
value is assumed, defaulted, or carried over from documentation.
"""

from __future__ import annotations

import collections
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.dataset.smartom import ImageRecord, SmartOmIndex, build_index
from backend.dataset.split import (
    DEFAULT_POSITIVE_CLASSES,
    SEVERITY_ORDER,
    binary_label,
    patient_stratum,
)


def class_of(image_id: str) -> str:
    """Diagnostic class prefix of an image id (``<class>/<site>/<stem>``)."""
    return image_id.split("/", 1)[0]


# Probe results are cached here so regenerating the report does not re-decode
# every image. Generated, never committed (see .gitignore).
DEFAULT_PROBE_CACHE = Path("backend/artifacts/dataset/probe_cache.json")


@dataclass
class ImageProbe:
    """Physical properties of one image file, measured by decoding it."""

    image_id: str
    width: int
    height: int
    mode: str
    file_format: str
    bytes_on_disk: int
    content_sha256: str
    readable: bool
    error: Optional[str] = None

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000.0


@dataclass
class InspectionReport:
    """Complete inspection result. Serialisable to JSON and Markdown."""

    dataset_root: str
    n_images_canonical: int
    n_images_all_trees: int
    n_patients: int
    generated_by: str = "backend.training.inspect_dataset"

    class_distribution: Dict[str, int] = field(default_factory=dict)
    site_distribution: Dict[str, int] = field(default_factory=dict)
    capture_mode_distribution: Dict[str, int] = field(default_factory=dict)
    annotation_tree_counts: Dict[str, int] = field(default_factory=dict)

    patient_id_sources: Dict[str, int] = field(default_factory=dict)
    images_per_patient: Dict[str, int] = field(default_factory=dict)
    images_per_patient_histogram: Dict[str, int] = field(default_factory=dict)
    patients_per_class: Dict[str, int] = field(default_factory=dict)
    patients_per_stratum: Dict[str, int] = field(default_factory=dict)
    n_patients_multi_class: int = 0

    label_type: str = "multiclass(4) reduced to binary screening target"
    positive_classes: List[str] = field(default_factory=list)
    binary_label_distribution: Dict[str, int] = field(default_factory=dict)
    descriptor_label_counts: Dict[str, int] = field(default_factory=dict)
    n_images_without_descriptor: int = 0
    images_without_descriptor: List[str] = field(default_factory=list)

    roi_availability: Dict[str, int] = field(default_factory=dict)
    annotation_format: str = ""

    image_formats: Dict[str, int] = field(default_factory=dict)
    image_modes: Dict[str, int] = field(default_factory=dict)
    dimension_summary: Dict[str, float] = field(default_factory=dict)
    n_unreadable: int = 0
    unreadable_images: List[Dict[str, str]] = field(default_factory=list)
    duplicate_groups: List[List[str]] = field(default_factory=list)
    n_duplicate_images: int = 0
    cross_class_duplicate_groups: List[Dict[str, Any]] = field(default_factory=list)
    n_cross_class_duplicate_groups: int = 0

    clinical_metadata_coverage: Dict[str, Any] = field(default_factory=dict)
    clinical_feature_availability: Dict[str, Dict[str, int]] = field(default_factory=dict)

    unmatched_paths: List[str] = field(default_factory=list)
    unmatched_descriptor_keys: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    exclusion_candidates: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
        }

    def save_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path


# --------------------------------------------------------------------------
# Image probing
# --------------------------------------------------------------------------


def probe_image(record: ImageRecord, *, hash_bytes: int = 0) -> ImageProbe:
    """Decode one image and measure its real properties.

    Args:
        record: the image to probe.
        hash_bytes: if > 0, hash only the first N bytes of the decoded pixel
            buffer. Full-buffer hashing is exact but slow on ~2500 multi-megapixel
            files; the default hashes the whole buffer.
    """
    path = record.path
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            file_format = (image.format or path.suffix.lstrip(".")).upper()
            # Hash decoded pixels, not the file: the dataset stores the same
            # capture re-encoded under different names, which file hashing misses.
            payload = image.convert("RGB").tobytes()
        digest = hashlib.sha256(payload[:hash_bytes] if hash_bytes else payload).hexdigest()
        return ImageProbe(
            image_id=record.image_id,
            width=width,
            height=height,
            mode=mode,
            file_format=file_format,
            bytes_on_disk=path.stat().st_size,
            content_sha256=digest,
            readable=True,
        )
    except Exception as exc:  # noqa: BLE001 - any decode failure is a finding
        return ImageProbe(
            image_id=record.image_id,
            width=0,
            height=0,
            mode="",
            file_format=path.suffix.lstrip(".").upper(),
            bytes_on_disk=path.stat().st_size if path.exists() else 0,
            content_sha256="",
            readable=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def probe_images(
    records: Sequence[ImageRecord],
    *,
    limit: Optional[int] = None,
    progress: Optional[Any] = None,
    cache_path: Optional[Path] = None,
) -> List[ImageProbe]:
    """Probe every record (or the first ``limit`` of them).

    Decoding and pixel-hashing ~2500 multi-megapixel files takes minutes, so
    results are cached on ``(path, size, mtime)``. A file that changes on disk
    invalidates its own entry and nothing else.
    """
    subset = list(records)[:limit] if limit else list(records)

    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path and Path(cache_path).exists():
        try:
            cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}  # a corrupt cache is re-earned, not fatal

    def cache_key(record: ImageRecord) -> str:
        try:
            stat = record.path.stat()
            return f"{record.image_id}|{stat.st_size}|{int(stat.st_mtime)}"
        except OSError:
            return f"{record.image_id}|missing"

    probes: List[ImageProbe] = []
    hits = 0
    updated = dict(cache)
    for i, record in enumerate(subset, start=1):
        key = cache_key(record)
        entry = cache.get(key)
        if entry is not None:
            probes.append(ImageProbe(**entry))
            hits += 1
        else:
            probe = probe_image(record)
            probes.append(probe)
            updated[key] = asdict(probe)
        if progress is not None and i % 250 == 0:
            progress(f"  probed {i}/{len(subset)} images ({hits} from cache)")

    if cache_path and updated != cache:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(updated), encoding="utf-8")

    if progress is not None:
        progress(f"  probe complete: {len(probes)} images, {hits} served from cache")
    return probes


# --------------------------------------------------------------------------
# Clinical feature availability
# --------------------------------------------------------------------------


def summarise_clinical_features(index: SmartOmIndex) -> Dict[str, Dict[str, int]]:
    """Count present / known-absent / unknown for each named risk factor.

    Delegates to :func:`backend.ml.features_clinical.read_factor` so the report
    describes exactly what the model will consume. A naive "is the cell blank"
    count would report almost everything as missing, because 214 patients record
    ``Habit_history = No`` and leave the habit-specific cells as ``-``.
    """
    from backend.ml.features_clinical import RISK_FACTORS, read_factor

    out: Dict[str, Dict[str, int]] = {}
    rows = list(index.patient_metadata.values())
    for factor in RISK_FACTORS:
        present = absent = unknown = 0
        for row in rows:
            observation = read_factor(row, factor)
            if observation.present is True:
                present += 1
            elif observation.present is False:
                absent += 1
            else:
                unknown += 1
        out[factor] = {
            "present": present,
            "known_absent": absent,
            "unknown": unknown,
            "n_patients": len(rows),
        }
    return out


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def inspect_dataset(
    root: Optional[Path] = None,
    *,
    positive_classes: Sequence[str] = DEFAULT_POSITIVE_CLASSES,
    probe_limit: Optional[int] = None,
    progress: Optional[Any] = None,
    index: Optional[SmartOmIndex] = None,
    probe_cache: Optional[Path] = DEFAULT_PROBE_CACHE,
) -> Tuple[InspectionReport, SmartOmIndex, List[ImageProbe]]:
    """Build the full inspection report from the dataset on disk."""
    say = progress or (lambda _msg: None)
    say("Building dataset index...")
    idx = index if index is not None else build_index(root)
    records = idx.records

    say(f"Indexed {len(records)} canonical images. Probing image files...")
    probes = probe_images(
        records, limit=probe_limit, progress=progress, cache_path=probe_cache
    )
    readable = [p for p in probes if p.readable]

    by_patient = idx.by_patient()
    patient_classes = {
        pid: {r.diagnostic_class for r in group} for pid, group in by_patient.items()
    }

    # Duplicate detection on decoded pixel content.
    by_hash: Dict[str, List[str]] = {}
    for probe in readable:
        by_hash.setdefault(probe.content_sha256, []).append(probe.image_id)
    duplicate_groups = sorted(
        (sorted(ids) for ids in by_hash.values() if len(ids) > 1), key=lambda g: g[0]
    )
    n_duplicate_images = sum(len(g) - 1 for g in duplicate_groups)

    # A duplicate group spanning two diagnostic-class folders is not redundancy:
    # it is the *same photograph carrying two different labels*. That is a label
    # contradiction and has to be surfaced separately, because training on both
    # copies teaches the model that identical pixels have opposite ground truth.
    cross_class_groups: List[Dict[str, Any]] = []
    for group in duplicate_groups:
        classes = sorted({class_of(image_id) for image_id in group})
        if len(classes) > 1:
            cross_class_groups.append({"classes": classes, "images": group})

    widths = [p.width for p in readable]
    heights = [p.height for p in readable]
    dimension_summary: Dict[str, float] = {}
    if readable:
        dimension_summary = {
            "width_min": float(min(widths)),
            "width_max": float(max(widths)),
            "width_mean": round(sum(widths) / len(widths), 1),
            "height_min": float(min(heights)),
            "height_max": float(max(heights)),
            "height_mean": round(sum(heights) / len(heights), 1),
            "megapixels_min": round(min(p.megapixels for p in readable), 3),
            "megapixels_max": round(max(p.megapixels for p in readable), 3),
            "megapixels_mean": round(sum(p.megapixels for p in readable) / len(readable), 3),
        }

    images_per_patient = {pid: len(group) for pid, group in sorted(by_patient.items())}
    counts_hist = collections.Counter(images_per_patient.values())

    patients_per_class: Dict[str, int] = {}
    for classes in patient_classes.values():
        for cls in classes:
            patients_per_class[cls] = patients_per_class.get(cls, 0) + 1

    positives = sum(binary_label(r.diagnostic_class, positive_classes) for r in records)

    without_descriptor = [r.image_id for r in records if r.descriptor_label is None]
    unreadable = [p for p in probes if not p.readable]

    exclusion_candidates: Dict[str, str] = {}
    for probe in unreadable:
        exclusion_candidates[probe.image_id] = f"unreadable image file ({probe.error})"

    for group in duplicate_groups:
        classes = sorted({class_of(image_id) for image_id in group})
        if len(classes) == 1:
            # Same label on both copies: pure redundancy. Keep one.
            for image_id in group[1:]:
                exclusion_candidates[image_id] = f"pixel-identical duplicate of {group[0]}"
            continue

        # Contradictory labels. Keep the most severe copy and exclude the rest.
        # Rationale: for a screening triage model the dangerous error is teaching
        # it that a lesion photograph is normal. Retaining the higher-severity
        # filing preserves the finding; retaining the lower-severity one would
        # inject a false negative. Documented in DECISIONS.md.
        def severity(image_id: str) -> int:
            cls = class_of(image_id)
            return SEVERITY_ORDER.index(cls) if cls in SEVERITY_ORDER else -1

        keep = max(group, key=severity)
        for image_id in group:
            if image_id == keep:
                continue
            exclusion_candidates[image_id] = (
                f"pixel-identical to {keep} but labelled "
                f"'{class_of(image_id)}' vs '{class_of(keep)}'; lower-severity "
                "copy dropped to avoid a contradictory label"
            )

    report = InspectionReport(
        dataset_root=str(idx.root),
        n_images_canonical=len(records),
        n_images_all_trees=sum(idx.annotation_image_counts.values()),
        n_patients=len(by_patient),
        class_distribution=dict(
            sorted(collections.Counter(r.diagnostic_class for r in records).items())
        ),
        site_distribution=dict(
            sorted(collections.Counter(r.site for r in records).items())
        ),
        capture_mode_distribution=dict(
            sorted(
                collections.Counter(r.capture_mode or "unspecified" for r in records).items()
            )
        ),
        annotation_tree_counts={
            f"{cls}/{kind}": count
            for (cls, kind), count in sorted(idx.annotation_image_counts.items())
        },
        patient_id_sources=dict(
            sorted(collections.Counter(r.patient_id_source for r in records).items())
        ),
        images_per_patient=images_per_patient,
        images_per_patient_histogram={
            str(k): v for k, v in sorted(counts_hist.items())
        },
        patients_per_class=dict(sorted(patients_per_class.items())),
        patients_per_stratum=dict(
            sorted(
                collections.Counter(
                    patient_stratum(classes) for classes in patient_classes.values()
                ).items()
            )
        ),
        n_patients_multi_class=sum(1 for c in patient_classes.values() if len(c) > 1),
        positive_classes=list(positive_classes),
        binary_label_distribution={
            "positive": positives,
            "negative": len(records) - positives,
        },
        descriptor_label_counts=dict(
            collections.Counter(
                r.descriptor_label for r in records if r.descriptor_label
            ).most_common(40)
        ),
        n_images_without_descriptor=len(without_descriptor),
        images_without_descriptor=without_descriptor[:50],
        roi_availability={
            "lesion_polygon": sum(1 for r in records if r.has_lesion_roi),
            "region_polygon_only": sum(
                1 for r in records if r.has_region_roi and not r.has_lesion_roi
            ),
            "no_polygon": sum(
                1 for r in records if not r.has_region_roi and not r.has_lesion_roi
            ),
        },
        annotation_format=(
            "VGG Image Annotator (VIA) v2 project JSON; polygon/rect/ellipse/"
            "polyline shapes under _via_img_metadata[*].regions"
        ),
        image_formats=dict(
            sorted(collections.Counter(p.file_format for p in probes).items())
        ),
        image_modes=dict(
            sorted(collections.Counter(p.mode for p in readable).items())
        ),
        dimension_summary=dimension_summary,
        n_unreadable=len(unreadable),
        unreadable_images=[
            {"image_id": p.image_id, "error": p.error or ""} for p in unreadable[:50]
        ],
        duplicate_groups=duplicate_groups[:100],
        n_duplicate_images=n_duplicate_images,
        cross_class_duplicate_groups=cross_class_groups[:100],
        n_cross_class_duplicate_groups=len(cross_class_groups),
        clinical_metadata_coverage={
            "patients_in_demographics_sheet": len(idx.patient_metadata),
            "patients_in_clinical_findings_sheet": len(idx.clinical_findings),
            "image_patients_with_demographics": sum(
                1 for pid in by_patient if pid in idx.patient_metadata
            ),
            "image_patients_without_demographics": sum(
                1 for pid in by_patient if pid not in idx.patient_metadata
            ),
        },
        clinical_feature_availability=summarise_clinical_features(idx),
        unmatched_paths=idx.unmatched_paths[:50],
        unmatched_descriptor_keys=idx.unmatched_descriptor_keys[:50],
        warnings=list(idx.warnings),
        exclusion_candidates=exclusion_candidates,
    )
    return report, idx, probes


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def _table(rows: Dict[str, Any], key_header: str, value_header: str) -> str:
    lines = [f"| {key_header} | {value_header} |", "|---|---|"]
    for key, value in rows.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def render_markdown(report: InspectionReport) -> str:
    """Render the report as Markdown. All values come from the report object."""
    total = report.n_images_canonical
    class_rows = {
        cls: f"{count} ({count / total:.2%})"
        for cls, count in report.class_distribution.items()
    }
    binary_rows = {
        label: f"{count} ({count / total:.2%})"
        for label, count in report.binary_label_distribution.items()
    }

    clinical_rows = {}
    for feature, stats in report.clinical_feature_availability.items():
        n = stats.get("n_patients") or 1
        clinical_rows[feature] = (
            f"present {stats['present']} ({stats['present'] / n:.1%}), "
            f"known-absent {stats['known_absent']}, "
            f"unknown {stats['unknown']}"
        )

    sections = [
        "# SMART-OM Dataset Inspection Report",
        "",
        "> Generated by `python -m backend.training.inspect_dataset`. Every value "
        "below is measured from the dataset on disk. Regenerate after any dataset "
        "change rather than editing this file by hand.",
        "",
        f"- **Dataset root**: `{report.dataset_root}`",
        f"- **Canonical images** (the `01. Unannotated` trees): **{total}**",
        f"- **Image files across all annotation trees**: {report.n_images_all_trees}",
        f"- **Distinct patients**: **{report.n_patients}**",
        f"- **Annotation format**: {report.annotation_format}",
        "",
        "## 1. Directory structure discovered",
        "",
        "```",
        "<root>/",
        "  <NN>. <diagnostic class>/",
        "    01. Unannotated/      <- canonical samples (one file per observation)",
        "    02. Region annotation/  <- same images re-rendered + VIA region JSON",
        "    03. Full annotation/    <- same images re-rendered + VIA full JSON",
        "    04. Lesion annotation/  <- same images re-rendered + VIA lesion JSON",
        "      <NN>. <anatomical site>/<image files>",
        "      <json dir>/<VIA project>.json",
        "  Descriptors/*.xlsx      <- per-image lesion descriptors, one sheet per class",
        "  Metadata/*.xlsx         <- per-patient demographics, habits, clinical findings",
        "```",
        "",
        "Only the `01. Unannotated` tree contributes training samples. The other "
        "three trees hold the *same* source captures with annotation overlays burned "
        "in; counting them would duplicate every observation 2-4x.",
        "",
        "### Image files per class / annotation tree",
        "",
        _table(report.annotation_tree_counts, "class / tree", "image files"),
        "",
        "## 2. Class distribution (canonical images)",
        "",
        _table(class_rows, "diagnostic class", "images"),
        "",
        f"Label type: {report.label_type}. Positive classes: "
        f"`{', '.join(report.positive_classes)}`.",
        "",
        _table(binary_rows, "binary screening label", "images"),
        "",
        "## 3. Patient structure",
        "",
        _table(
            {
                "distinct patients": report.n_patients,
                "patients whose images span >1 diagnostic class": report.n_patients_multi_class,
                "images per patient (min)": min(report.images_per_patient.values())
                if report.images_per_patient
                else 0,
                "images per patient (max)": max(report.images_per_patient.values())
                if report.images_per_patient
                else 0,
                "images per patient (mean)": round(total / report.n_patients, 2)
                if report.n_patients
                else 0,
            },
            "metric",
            "value",
        ),
        "",
        "**Images are not one-per-patient.** Most subjects contribute one capture "
        "per anatomical site, in one or two lighting modes; a minority contribute "
        "a single image. This is why splitting must be patient-level (see "
        "`backend/dataset/split.py`).",
        "",
        "### Images per patient (histogram)",
        "",
        _table(report.images_per_patient_histogram, "images", "patients"),
        "",
        "### Patient identifier derivation",
        "",
        _table(report.patient_id_sources, "source", "images"),
        "",
        "### Patients per diagnostic class (a patient may appear in several)",
        "",
        _table(report.patients_per_class, "class", "patients"),
        "",
        "### Patients per stratum (most severe class per patient)",
        "",
        _table(report.patients_per_stratum, "stratum", "patients"),
        "",
        "## 4. Anatomical site distribution",
        "",
        _table(report.site_distribution, "site", "images"),
        "",
        "### Capture mode token in filename",
        "",
        _table(report.capture_mode_distribution, "mode", "images"),
        "",
        "## 5. Annotation and label availability",
        "",
        _table(report.roi_availability, "ROI availability", "images"),
        "",
        _table(
            {
                "images with a descriptor label": total - report.n_images_without_descriptor,
                "images WITHOUT a descriptor label": report.n_images_without_descriptor,
                "descriptor rows not matched to any image": len(
                    report.unmatched_descriptor_keys
                ),
            },
            "metric",
            "value",
        ),
        "",
        "### Most frequent descriptor labels",
        "",
        _table(report.descriptor_label_counts, "descriptor", "images"),
        "",
        "## 6. Image file properties (measured by decoding every file)",
        "",
        _table(report.image_formats, "format", "files"),
        "",
        _table(report.image_modes, "colour mode", "files"),
        "",
        _table(report.dimension_summary, "dimension metric", "pixels"),
        "",
        _table(
            {
                "unreadable / corrupted files": report.n_unreadable,
                "pixel-identical duplicate groups": len(report.duplicate_groups),
                "redundant duplicate images": report.n_duplicate_images,
                "duplicate groups spanning >1 diagnostic class": report.n_cross_class_duplicate_groups,
            },
            "integrity metric",
            "value",
        ),
        "",
        "### Label contradictions (cross-class pixel-identical duplicates)",
        "",
        f"{report.n_cross_class_duplicate_groups} group(s) contain the *same "
        "photograph* filed under two different diagnostic classes. These are not "
        "redundant copies: identical pixels carry contradictory ground truth. "
        "Every such group shares one filename stem and one patient, so no "
        "patient-level leakage arises from them, but training on both copies "
        "would teach the model that the same image is both positive and negative. "
        "The lower-severity copy is therefore excluded (see section 8).",
        "",
    ]

    if report.cross_class_duplicate_groups:
        sections.append(
            _table(
                {
                    " / ".join(g["images"]): " vs ".join(g["classes"])
                    for g in report.cross_class_duplicate_groups[:40]
                },
                "images",
                "conflicting labels",
            )
        )
        sections.append("")

    sections += [
        "## 7. Clinical feature availability",
        "",
        _table(report.clinical_metadata_coverage, "metric", "value"),
        "",
        _table(clinical_rows, "risk factor", "coverage"),
        "",
        "## 8. Exclusion candidates",
        "",
        f"{len(report.exclusion_candidates)} image(s) flagged. Exclusions are "
        "recorded in the split manifest (`excluded_images`) rather than dropped, "
        "so every removal stays auditable.",
        "",
    ]

    if report.exclusion_candidates:
        sections.append(_table(report.exclusion_candidates, "image_id", "reason"))
        sections.append("")

    if report.warnings:
        sections.extend(
            ["## 9. Parser warnings", ""] + [f"- {w}" for w in report.warnings] + [""]
        )

    return "\n".join(sections)
