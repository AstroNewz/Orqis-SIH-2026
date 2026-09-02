"""Model artifact persistence.

PART 24 requires every fitted object in the pipeline to be persisted and every
inference to be traceable to a model version. This module is the single place
that decides the on-disk layout, so nothing else in the codebase hard-codes a
path.

Layout under ``ARTIFACT_DIR`` (default ``backend/artifacts``)::

    dataset/
        split_manifest.json         patient-level split (backend.dataset.split)
        inspection_report.json      dataset inspection (backend.dataset.inspect)
        features_<extractor>.npz    cached image descriptors, training only
    models/<model_version>/
        metadata.json               versions, seeds, config, provenance
        pipeline.json               clinical encoder + fusion scalers + reducer meta
        reduction.npz               PCA mean / components / variances
        quantum.json                VQC parameters + training record
        calibration.json            fitted calibrator
    models/current                  text file naming the active model version

No pickle
---------
Every artifact is JSON or ``.npz``. Unpickling executes arbitrary code, and a
model artifact is exactly the kind of file that gets copied between machines and
checked into deployment images. JSON and ``npz`` are inert, diffable, and load
without matching library versions -- which matters when an artifact must still be
readable long after the environment that produced it is gone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from backend.core.config import Settings, settings as default_settings
from backend.ml.features_clinical import ClinicalFeatureEncoder
from backend.ml.features_image import resolve_extractor
from backend.ml.fusion import FeatureFusion
from backend.ml.pipeline import PIPELINE_VERSION, ClassicalPipeline
from backend.ml.reduction import DimensionalityReducer

CURRENT_POINTER = "current"

PIPELINE_FILE = "pipeline.json"
REDUCTION_ARRAYS_FILE = "reduction.npz"
METADATA_FILE = "metadata.json"
QUANTUM_FILE = "quantum.json"
CALIBRATION_FILE = "calibration.json"
BASELINES_FILE = "baselines.json"

SPLIT_MANIFEST_FILE = "split_manifest.json"
INSPECTION_REPORT_FILE = "inspection_report.json"


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing, unreadable, or internally inconsistent."""


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return path


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ArtifactError(f"Artifact not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"Artifact at {path} is not valid JSON: {exc}") from exc


@dataclass
class ArtifactStore:
    """Resolves artifact paths and reads/writes the fitted pipeline."""

    root: Path
    dataset_subdir: str = "dataset"
    models_subdir: str = "models"

    @classmethod
    def from_settings(cls, config: Optional[Settings] = None) -> "ArtifactStore":
        cfg = config or default_settings
        return cls(root=Path(cfg.artifact_dir))

    # ----------------------------------------------------------------- paths
    @property
    def dataset_dir(self) -> Path:
        return self.root / self.dataset_subdir

    @property
    def models_dir(self) -> Path:
        return self.root / self.models_subdir

    def model_dir(self, model_version: str) -> Path:
        # Version strings become directory names, so anything that could escape
        # the artifact root is rejected rather than sanitised silently.
        if not model_version or any(sep in model_version for sep in ("/", "\\", "..")):
            raise ArtifactError(f"Invalid model version for a directory name: {model_version!r}")
        return self.models_dir / model_version

    @property
    def split_manifest_path(self) -> Path:
        return self.dataset_dir / SPLIT_MANIFEST_FILE

    @property
    def inspection_report_path(self) -> Path:
        return self.dataset_dir / INSPECTION_REPORT_FILE

    def feature_cache_path(self, extractor: str) -> Path:
        return self.dataset_dir / f"features_{extractor}.npz"

    # -------------------------------------------------------- current model
    @property
    def current_pointer_path(self) -> Path:
        return self.models_dir / CURRENT_POINTER

    def set_current(self, model_version: str) -> Path:
        """Record which model version inference should load."""
        self.model_dir(model_version)  # validates the name
        self.current_pointer_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_pointer_path.write_text(model_version + "\n", encoding="utf-8")
        return self.current_pointer_path

    def current_version(self) -> Optional[str]:
        if not self.current_pointer_path.exists():
            return None
        value = self.current_pointer_path.read_text(encoding="utf-8").strip()
        return value or None

    def available_versions(self) -> list:
        if not self.models_dir.exists():
            return []
        return sorted(
            child.name
            for child in self.models_dir.iterdir()
            if child.is_dir() and (child / METADATA_FILE).exists()
        )

    def resolve_version(self, model_version: Optional[str] = None) -> str:
        """The requested version, else the recorded current one."""
        if model_version:
            return model_version
        current = self.current_version()
        if current:
            return current
        raise ArtifactError(
            f"No current model recorded at {self.current_pointer_path}. Train a model "
            "with `python -m backend.training.train_all` first."
        )

    # ----------------------------------------------------------- save/load
    def save_pipeline(
        self,
        pipeline: ClassicalPipeline,
        *,
        model_version: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        set_current: bool = True,
    ) -> Path:
        """Persist the fitted classical pipeline under ``model_version``."""
        if not pipeline.fitted:
            raise ArtifactError("Refusing to save an unfitted pipeline.")

        directory = self.model_dir(model_version)
        directory.mkdir(parents=True, exist_ok=True)

        # Arrays go to .npz; the JSON keeps only metadata, so pipeline.json stays
        # small enough to read by eye when auditing a deployed model.
        pipeline.reducer.save_arrays(directory / REDUCTION_ARRAYS_FILE)
        _write_json(
            directory / PIPELINE_FILE,
            {
                "pipeline_version": pipeline.version,
                "preprocessing_version": pipeline.preprocessing_version,
                "extractor_name": pipeline.extractor.name,
                "n_qubits": pipeline.n_qubits,
                "target_size": list(pipeline.target_size),
                "center_crop_fraction": pipeline.center_crop_fraction,
                "clinical_encoder": pipeline.clinical_encoder.to_dict(),
                "fusion": pipeline.fusion.to_dict(),
                "reduction": pipeline.reducer.to_dict(include_arrays=False),
                "reduction_arrays": REDUCTION_ARRAYS_FILE,
            },
        )

        metadata: Dict[str, Any] = {
            "model_version": model_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": pipeline.describe(),
        }
        metadata.update(extra_metadata or {})
        _write_json(directory / METADATA_FILE, metadata)

        if set_current:
            self.set_current(model_version)
        return directory

    def load_pipeline(self, model_version: Optional[str] = None) -> ClassicalPipeline:
        """Rebuild the fitted classical pipeline from disk."""
        version = self.resolve_version(model_version)
        directory = self.model_dir(version)
        payload = _read_json(directory / PIPELINE_FILE)

        reducer = DimensionalityReducer.from_dict(payload["reduction"])
        arrays_path = directory / payload.get("reduction_arrays", REDUCTION_ARRAYS_FILE)
        if reducer.method != "identity":
            if not arrays_path.exists():
                raise ArtifactError(
                    f"Reduction method is {reducer.method!r} but its arrays are missing "
                    f"at {arrays_path}. The artifact is incomplete."
                )
            reducer.load_arrays(arrays_path)

        pipeline = ClassicalPipeline(
            extractor=resolve_extractor(payload.get("extractor_name")),
            clinical_encoder=ClinicalFeatureEncoder.from_dict(payload["clinical_encoder"]),
            fusion=FeatureFusion.from_dict(payload["fusion"]),
            reducer=reducer,
            n_qubits=int(payload["n_qubits"]),
            version=str(payload.get("pipeline_version", PIPELINE_VERSION)),
            preprocessing_version=str(payload.get("preprocessing_version", "")),
            target_size=tuple(payload.get("target_size", (224, 224))),  # type: ignore[arg-type]
            center_crop_fraction=float(payload.get("center_crop_fraction", 0.80)),
        )
        metadata_path = directory / METADATA_FILE
        if metadata_path.exists():
            pipeline.training_metadata = _read_json(metadata_path)
        if not pipeline.fitted:
            raise ArtifactError(
                f"Pipeline artifact at {directory} loaded but is not fitted. "
                "It was probably written by an interrupted training run."
            )
        return pipeline

    # ------------------------------------------------- generic sub-artifacts
    def write_component(
        self, model_version: str, filename: str, payload: Dict[str, Any]
    ) -> Path:
        """Write a JSON sub-artifact (quantum model, calibrator) for a version."""
        return _write_json(self.model_dir(model_version) / filename, payload)

    def read_component(
        self, model_version: Optional[str], filename: str, *, required: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Read a JSON sub-artifact. ``required=False`` returns ``None`` if absent."""
        path = self.model_dir(self.resolve_version(model_version)) / filename
        if not path.exists() and not required:
            return None
        return _read_json(path)

    def metadata(self, model_version: Optional[str] = None) -> Dict[str, Any]:
        return _read_json(self.model_dir(self.resolve_version(model_version)) / METADATA_FILE)

    def update_metadata(self, model_version: str, updates: Dict[str, Any]) -> Path:
        """Merge keys into an existing metadata file, preserving what is there.

        Training runs in stages -- classical, then quantum, then calibration --
        and each stage records its own provenance without discarding the previous
        stage's.
        """
        path = self.model_dir(model_version) / METADATA_FILE
        current = _read_json(path) if path.exists() else {}
        current.update(updates)
        return _write_json(path, current)


@dataclass
class LoadedModel:
    """A fully assembled model: classical pipeline plus its recorded metadata.

    The quantum model and calibrator are loaded by their own modules from the
    same directory; this container carries the paths and metadata so nothing has
    to re-derive them.
    """

    model_version: str
    pipeline: ClassicalPipeline
    directory: Path
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def quantum_path(self) -> Path:
        return self.directory / QUANTUM_FILE

    @property
    def calibration_path(self) -> Path:
        return self.directory / CALIBRATION_FILE

    @property
    def has_quantum_model(self) -> bool:
        return self.quantum_path.exists()

    @property
    def has_calibrator(self) -> bool:
        return self.calibration_path.exists()


def load_model(
    model_version: Optional[str] = None, *, config: Optional[Settings] = None
) -> LoadedModel:
    """Load the classical pipeline and metadata for a model version."""
    store = ArtifactStore.from_settings(config)
    version = store.resolve_version(model_version)
    pipeline = store.load_pipeline(version)
    directory = store.model_dir(version)
    metadata_path = directory / METADATA_FILE
    return LoadedModel(
        model_version=version,
        pipeline=pipeline,
        directory=directory,
        metadata=_read_json(metadata_path) if metadata_path.exists() else {},
    )
