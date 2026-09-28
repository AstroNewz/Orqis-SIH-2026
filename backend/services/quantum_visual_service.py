"""E2 quantum-visual demonstrator: real per-request circuit execution (DEC-035).

This is the runtime companion to :mod:`backend.evaluation.e2_quantum_visual` (the
fit/evaluate driver) and :mod:`quantum_ml.visual_circuit` (the circuit). It loads the
*persisted* E2 bundle once, reconstructs the exact TRAIN-only objects the driver fitted,
and -- for one image's 576-d MobileNet ROI embedding -- executes the real 8-qubit
angle-encoded circuit on the exact Aer statevector, reads the 16 pre-registered local Z
observables, and applies the small logistic head. The result is a
:class:`~backend.schemas.inference.QuantumVisualSummary`.

Strictly a SECONDARY, EXPERIMENTAL signal (DEC-033/DEC-034: a demonstrator, never an
advantage claim). It is additive and non-clinical:

* it never headlines the verdict and never feeds ``probability`` / ``final_probability`` /
  any FHIR value -- the caller keeps the classical primary path untouched;
* its equal-dimension RFF control and raw PCA-8 control are computed on the *same* image
  and returned beside it, so the quantum number is never reported alone;
* nothing here is mocked or hardcoded: every value is either read back from the built
  circuit / persisted artifact (provenance) or produced by executing the circuit on this
  request's embedding.

Faithfulness to the evaluated pipeline is enforced, not assumed. On load the service
verifies the persisted preprocessor content hash, the module circuit/preprocessing
version strings, and the reconstructed circuit telemetry against the bundle; any
disagreement (or a missing artifact, or a missing quantum backend) leaves the service
"not ready", and the caller then simply omits the E2 card. The bundle is also asserted
to carry ``test_partition_used: false`` -- the held-out test partition never enters this
path.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.ml.artifacts import ArtifactError, ArtifactStore
from backend.ml.quantum_visual import (
    MOBILENET_DIM,
    QUANTUM_VISUAL_VERSION,
    LogisticHead,
    QuantumVisualPreprocessor,
    RandomFourierControl,
)
from backend.schemas.inference import QuantumVisualSummary
from quantum_ml.visual_circuit import VISUAL_CIRCUIT_VERSION, QuantumVisualFeatureMap

logger = logging.getLogger(__name__)

# These mirror ``backend.evaluation.e2_quantum_visual`` (the fit driver) but are declared
# here so the request path never imports the evaluation module (which pulls in sklearn and
# the whole training stack). The version guards below catch any real drift between the two.
E2_ARTIFACT_VERSION = "e2_quantum_visual"
E2_ARTIFACT_FILE = "e2_quantum_visual.json"

#: TRAIN+VALIDATION-only static report the driver writes at repo root. Read best-effort to
#: surface the KEEP/KILL verdict and matched-control PR-AUCs; the executed per-image
#: numbers never depend on it. Guarded so a test-partition report can never be surfaced.
E2_REPORT_FILE = "reports_e2_quantum_visual.json"

#: Structural telemetry keys that must match between the reconstructed circuit and the
#: persisted bundle. ``observable_labels`` is compared separately (list equality).
_TELEMETRY_GUARD_KEYS = (
    "circuit_version",
    "n_qubits",
    "n_reuploading_blocks",
    "n_encoding_parameters",
    "n_trainable_parameters",
    "two_qubit_gate_count",
    "circuit_depth",
    "state_dimension",
    "n_quantum_features",
    "backend_name",
    "seed",
)


class QuantumVisualUnavailable(RuntimeError):
    """The E2 demonstrator cannot run (no artifact, failed provenance, or bad input).

    Always caught by the caller, which then leaves ``quantum_visual=None`` and serves the
    classical primary result unchanged. It is never surfaced to a patient.
    """


def _clip01(value: float) -> float:
    """Clamp a head probability into ``[0, 1]`` (guards sigmoid float epsilon)."""
    return float(min(1.0, max(0.0, value)))


class QuantumVisualService:
    """Loads the persisted E2 bundle once and executes the real circuit per request."""

    def __init__(self, *, config: Optional[Settings] = None):
        self._cfg = config or default_settings
        self._lock = threading.Lock()
        self._loaded = False
        self._ready = False
        self._load_error: Optional[str] = None

        self._preprocessor: Optional[QuantumVisualPreprocessor] = None
        self._feature_map: Optional[QuantumVisualFeatureMap] = None
        self._rff: Optional[RandomFourierControl] = None
        self._heads: Dict[str, LogisticHead] = {}
        self._telemetry: Dict[str, Any] = {}
        self._observable_labels: Sequence[str] = ()
        self._aer_validation: Dict[str, Any] = {}
        self._fitted_on_condition: Optional[str] = None
        self._preprocessor_hash: Optional[str] = None

        # From the static TRAIN+VALIDATION report (best-effort, all optional).
        self._keep_kill_decision: Optional[str] = None
        self._quantum_pr_auc: Optional[float] = None
        self._rff_pr_auc: Optional[float] = None
        self._pca_pr_auc: Optional[float] = None

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        """Load and verify the E2 bundle once. Never raises; sets ``_ready``.

        A failure to load E2 is expected and non-fatal: the demonstrator is optional and
        the classical primary path must keep working. The reason is recorded in
        ``_load_error`` for ``/model/info`` and logged once.
        """
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                self._build()
                self._ready = True
                self._load_error = None
                logger.info(
                    "E2 quantum-visual demonstrator ready: %s | %d qubits, depth %d, "
                    "%d features | fitted_on=%s",
                    self._telemetry.get("circuit_version"),
                    int(self._telemetry.get("n_qubits", 0)),
                    int(self._telemetry.get("circuit_depth", 0)),
                    int(self._telemetry.get("n_quantum_features", 0)),
                    self._fitted_on_condition,
                )
            except QuantumVisualUnavailable as exc:
                self._ready = False
                self._load_error = str(exc)
                logger.info("E2 quantum-visual demonstrator unavailable: %s", exc)
            except Exception as exc:  # noqa: BLE001 - never let E2 loading break the app
                self._ready = False
                self._load_error = f"unexpected error loading E2 artifact: {exc}"
                logger.warning("E2 quantum-visual demonstrator failed to load: %s", exc)
            finally:
                self._loaded = True

    def _build(self) -> None:
        """Reconstruct and verify every E2 object from the persisted bundle."""
        store = ArtifactStore.from_settings(self._cfg)
        try:
            bundle = store.read_component(
                E2_ARTIFACT_VERSION, E2_ARTIFACT_FILE, required=False
            )
        except ArtifactError as exc:
            raise QuantumVisualUnavailable(f"E2 artifact unreadable: {exc}") from exc
        if bundle is None:
            raise QuantumVisualUnavailable(
                "no E2 artifact persisted (run backend.evaluation.e2_quantum_visual)."
            )

        # -- Firewall: the persisted bundle is fit on TRAIN only. Refuse anything else.
        if bundle.get("test_partition_used") is not False:
            raise QuantumVisualUnavailable(
                "E2 bundle does not certify test_partition_used=false; refusing to load."
            )

        # -- Version guards: a stored 8-vector must never be reinterpreted by changed code.
        if bundle.get("visual_circuit_version") != VISUAL_CIRCUIT_VERSION:
            raise QuantumVisualUnavailable(
                f"circuit version mismatch: bundle {bundle.get('visual_circuit_version')!r} "
                f"vs code {VISUAL_CIRCUIT_VERSION!r}."
            )
        if bundle.get("quantum_visual_version") != QUANTUM_VISUAL_VERSION:
            raise QuantumVisualUnavailable(
                f"preprocessing version mismatch: bundle "
                f"{bundle.get('quantum_visual_version')!r} vs code {QUANTUM_VISUAL_VERSION!r}."
            )

        # -- Preprocessor + content-hash integrity check.
        try:
            preprocessor = QuantumVisualPreprocessor.from_dict(bundle["preprocessor"])
        except (KeyError, TypeError, ValueError) as exc:
            raise QuantumVisualUnavailable(f"E2 preprocessor malformed: {exc}") from exc
        recorded_hash = bundle.get("preprocessor_artifact_hash")
        if recorded_hash and preprocessor.artifact_hash != recorded_hash:
            raise QuantumVisualUnavailable(
                "E2 preprocessor content hash does not match its provenance record; "
                "refusing to serve a possibly-reinterpreted 8-vector."
            )

        # -- Circuit rebuilt from the *bundle's* recorded topology, then verified to agree
        #    with the built circuit's own telemetry (a strong drift guard).
        recorded_telemetry = dict(bundle.get("circuit_telemetry") or {})
        feature_map = QuantumVisualFeatureMap(
            n_qubits=int(recorded_telemetry.get("n_qubits", 8)),
            n_blocks=int(recorded_telemetry.get("n_reuploading_blocks", 2)),
            seed=int(recorded_telemetry.get("seed", 7)),
        )
        built_telemetry = feature_map.telemetry()
        mismatches = {
            key: (built_telemetry.get(key), recorded_telemetry.get(key))
            for key in _TELEMETRY_GUARD_KEYS
            if built_telemetry.get(key) != recorded_telemetry.get(key)
        }
        if mismatches:
            raise QuantumVisualUnavailable(
                f"E2 circuit telemetry disagrees with the persisted bundle: {mismatches}."
            )
        observable_labels = list(built_telemetry.get("observable_labels", []))
        if observable_labels != list(recorded_telemetry.get("observable_labels", [])):
            raise QuantumVisualUnavailable(
                "E2 observable labels disagree with the persisted bundle."
            )

        # -- Matched control + the three heads (quantum / rff / pca).
        try:
            rff = RandomFourierControl.from_dict(bundle["rff_control"])
            heads = {
                name: LogisticHead.from_dict(bundle["heads"][name])
                for name in ("quantum", "rff", "pca")
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise QuantumVisualUnavailable(f"E2 control/head malformed: {exc}") from exc

        # -- Warm-up doubles as a final smoke test: if the exact Aer statevector cannot be
        #    produced for a trivial input, E2 is not ready (better here than mid-request).
        try:
            feature_map.transform(np.zeros((1, feature_map.n_qubits), dtype=np.float64))
        except Exception as exc:  # noqa: BLE001 - qiskit-aer absent or broken
            raise QuantumVisualUnavailable(
                f"E2 quantum backend could not execute the circuit: {exc}"
            ) from exc

        # -- Commit (all-or-nothing, mirroring InferenceService._load).
        self._preprocessor = preprocessor
        self._feature_map = feature_map
        self._rff = rff
        self._heads = heads
        self._telemetry = recorded_telemetry or built_telemetry
        self._observable_labels = observable_labels
        self._aer_validation = dict(bundle.get("aer_validation") or {})
        self._fitted_on_condition = bundle.get("fitted_on_condition")
        self._preprocessor_hash = recorded_hash or preprocessor.artifact_hash
        self._load_static_report(store, self._fitted_on_condition)

    def _load_static_report(self, store: ArtifactStore, condition: Optional[str]) -> None:
        """Best-effort read of the TRAIN+VALIDATION report for KEEP/KILL + PR-AUCs.

        Every field here is optional; the executed per-image numbers do not depend on it.
        A test-partition report (or any read error) is silently skipped -- the firewall is
        re-checked here, not merely trusted.
        """
        report = None
        for path in self._report_candidates(store):
            try:
                if path.exists():
                    report = json.loads(path.read_text(encoding="utf-8"))
                    break
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        if not isinstance(report, dict):
            return
        if report.get("test_partition_used") is not False:
            logger.info("E2 report skipped: it does not certify test_partition_used=false.")
            return

        key = condition or report.get("primary_condition")
        entry = ((report.get("conditions") or {}).get(key)) or {}
        decision = (entry.get("keep_kill") or {}).get("decision")
        matched = entry.get("matched_controls_vs_quantum") or {}
        controls = matched.get("controls") or {}
        self._keep_kill_decision = decision
        self._quantum_pr_auc = _opt_float(matched.get("quantum_validation_pr_auc"))
        self._rff_pr_auc = _opt_float((controls.get("rff") or {}).get("validation_pr_auc"))
        self._pca_pr_auc = _opt_float((controls.get("pca") or {}).get("validation_pr_auc"))

    @staticmethod
    def _report_candidates(store: ArtifactStore) -> Sequence[Path]:
        """Ordered, de-duplicated candidate locations for the static E2 report."""
        candidates = []
        try:
            candidates.append(store.root.parent.parent / E2_REPORT_FILE)
        except Exception:  # noqa: BLE001 - path arithmetic is best-effort
            pass
        candidates.append(Path.cwd() / E2_REPORT_FILE)
        candidates.append(Path(__file__).resolve().parents[2] / E2_REPORT_FILE)
        seen: Dict[str, Path] = {}
        for path in candidates:
            seen.setdefault(str(path), path)
        return list(seen.values())

    # ------------------------------------------------------------------ status
    @property
    def ready(self) -> bool:
        """Whether the E2 demonstrator can execute, without raising."""
        self._load()
        return self._ready

    def describe(self) -> Dict[str, Any]:
        """Provenance for ``/model/info``. Safe whether or not E2 loaded."""
        self._load()
        if not self._ready:
            return {"ready": False, "reason": self._load_error}
        return {
            "ready": True,
            "role": "experimental_secondary_quantum_visual_demonstrator",
            "advantage_claimed": False,
            "fitted_on_condition": self._fitted_on_condition,
            "preprocessor_artifact_hash": self._preprocessor_hash,
            "keep_kill_decision": self._keep_kill_decision,
            **{k: self._telemetry.get(k) for k in _TELEMETRY_GUARD_KEYS},
            "n_quantum_features": self._telemetry.get("n_quantum_features"),
            "aer_validation": self._aer_validation or None,
        }

    # --------------------------------------------------------------- execution
    def analyze_embedding(self, embedding: Sequence[float]) -> QuantumVisualSummary:
        """Execute the real E2 circuit on one 576-d MobileNet ROI embedding.

        Raises:
            QuantumVisualUnavailable: E2 is not ready, or the embedding is the wrong
                width / non-finite. The caller catches this and omits the E2 card.
        """
        self._load()
        if not self._ready:
            raise QuantumVisualUnavailable(
                self._load_error or "E2 quantum-visual demonstrator is not available."
            )
        assert self._preprocessor and self._feature_map and self._rff  # for type-checkers

        vector = np.asarray(embedding, dtype=np.float64).ravel()
        if vector.size != MOBILENET_DIM:
            raise QuantumVisualUnavailable(
                f"E2 expects a {MOBILENET_DIM}-d MobileNet embedding; got {vector.size}."
            )
        if not np.all(np.isfinite(vector)):
            raise QuantumVisualUnavailable("E2 embedding contains non-finite values.")
        row = vector.reshape(1, -1)

        started = time.perf_counter()
        # --- the real quantum stage: TRAIN-fitted angles -> exact Aer statevector -> the
        #     16 pre-registered local-Z observables -> the fitted logistic head.
        angles = self._preprocessor.to_angles(row)
        quantum_features = self._feature_map.transform(angles)  # (1, 16)
        secondary = float(self._heads["quantum"].scores(quantum_features)[0])
        # Entanglement-witness telemetry (0 on any separable state); NOT an advantage metric.
        correlations = self._feature_map.connected_correlations(quantum_features)[0]  # (8,)
        # --- matched controls on the SAME image, so the quantum number is never alone.
        pca_scores = self._preprocessor.pca_scores(row)
        rff_probability = float(self._heads["rff"].scores(self._rff.transform(pca_scores))[0])
        pca_probability = float(self._heads["pca"].scores(pca_scores)[0])
        execution_ms = (time.perf_counter() - started) * 1000.0

        feature_vector = quantum_features[0]
        abs_correlations = np.abs(correlations)
        telemetry = self._telemetry
        aer = self._aer_validation

        return QuantumVisualSummary(
            circuit_version=str(telemetry["circuit_version"]),
            n_qubits=int(telemetry["n_qubits"]),
            n_reuploading_blocks=int(telemetry["n_reuploading_blocks"]),
            n_encoding_parameters=int(telemetry["n_encoding_parameters"]),
            n_trainable_parameters=int(telemetry["n_trainable_parameters"]),
            two_qubit_gate_count=int(telemetry["two_qubit_gate_count"]),
            circuit_depth=int(telemetry["circuit_depth"]),
            state_dimension=int(telemetry["state_dimension"]),
            n_quantum_features=int(telemetry["n_quantum_features"]),
            backend_name=str(telemetry["backend_name"]),
            seed=int(telemetry["seed"]),
            observable_labels=list(self._observable_labels),
            secondary_probability=_clip01(secondary),
            quantum_feature_vector=[float(v) for v in feature_vector],
            # At n=1 the per-edge "mean" reduces to the edge's own |C_i|; kept per-edge so
            # the client can render all 8 ring edges of the entanglement witness.
            max_abs_connected_correlation=(
                float(np.max(abs_correlations)) if abs_correlations.size else 0.0
            ),
            mean_abs_connected_correlation_per_edge=[float(v) for v in abs_correlations],
            execution_time_ms=execution_ms,
            aer_max_abs_deviation=_opt_float(aer.get("max_abs_deviation")),
            aer_tolerance=_opt_float(aer.get("tolerance")),
            aer_validation_passed=(
                bool(aer["passed"]) if aer.get("passed") is not None else None
            ),
            rff_probability=_clip01(rff_probability),
            pca_probability=_clip01(pca_probability),
            keep_kill_decision=self._keep_kill_decision,
            quantum_validation_pr_auc=self._quantum_pr_auc,
            rff_validation_pr_auc=self._rff_pr_auc,
            pca_validation_pr_auc=self._pca_pr_auc,
            fitted_on_condition=self._fitted_on_condition,
            preprocessor_artifact_hash=self._preprocessor_hash,
        )


def _opt_float(value: Any) -> Optional[float]:
    """Coerce to ``float`` or ``None`` -- never raise for an absent/garbled report field."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_default_service: Optional[QuantumVisualService] = None
_default_lock = threading.Lock()


def get_quantum_visual_service(config: Optional[Settings] = None) -> QuantumVisualService:
    """Process-wide E2 service, so the artifact and warm circuit load once."""
    global _default_service
    if _default_service is None:
        with _default_lock:
            if _default_service is None:
                _default_service = QuantumVisualService(config=config)
    return _default_service


def reset_quantum_visual_service() -> None:
    """Drop the cached E2 service. For tests that swap artifacts or settings."""
    global _default_service
    with _default_lock:
        _default_service = None
