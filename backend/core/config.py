"""Application settings.

Every tunable lives here so that thresholds, seeds, and quantum parameters are
configuration rather than magic numbers scattered through the pipeline.
Secrets are read from the environment and never given real defaults.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: backend/core/config.py -> backend/core -> backend -> <root>
REPO_ROOT = Path(__file__).resolve().parents[2]

QuantumExecutionMode = Literal["ideal_simulation", "noisy_simulation", "ibm_hardware"]
CalibrationMethod = Literal["platt", "isotonic", "auto"]
FeatureMode = Literal["image_only", "clinical_only", "multimodal"]
OptimizerName = Literal["spsa", "qnspsa", "cobyla"]


class Settings(BaseSettings):
    """Application settings loaded from environment variables or defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ App
    APP_NAME: str = "CareScan Hybrid QML Backend"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    # ------------------------------------------------------------- Security
    SECRET_KEY: str = "insecure-default-key-for-dev-only-replace-in-prod"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ------------------------------------------------------------- Database
    # SQLite by default so `pytest` and local dev need no server or credentials.
    # PostgreSQL is the deployment target: set DATABASE_URL to a postgresql+psycopg2 URL.
    DATABASE_URL: str = "sqlite:///./carescan.db"

    # -------------------------------------------------------------- Quantum
    # Execution mode is what the pipeline honours. QUANTUM_BACKEND is retained
    # for backwards compatibility with the earlier two-value setting.
    QUANTUM_EXECUTION_MODE: QuantumExecutionMode = "ideal_simulation"
    QUANTUM_BACKEND: str = "aer_simulator"  # "aer_simulator" | "ibm_quantum"
    QUANTUM_QUBITS: int = 8  # Amplitude encoding: dimension = 2 ** QUANTUM_QUBITS
    QUANTUM_SHOTS: int = 1024
    QUANTUM_CIRCUIT_DEPTH: int = 2  # variational layers in the ansatz
    QUANTUM_OPTIMIZER: OptimizerName = "spsa"
    QUANTUM_NOISE_MODEL: str = "fake_sherbrooke"
    """IBM device whose published calibration data seeds the noise model when
    mode == noisy_simulation. A 127-qubit Eagle processor, so the error rates are
    representative of real hardware. Falls back to a documented synthetic
    depolarising model if qiskit-ibm-runtime is unavailable."""
    QUANTUM_ALLOW_HARDWARE_FALLBACK: bool = True
    """If IBM hardware is selected but unreachable, fall back to local simulation
    instead of failing the request. The Flutter client must never break because
    hardware is unavailable."""

    # -------------------------------------- IBM Quantum (optional, never required)
    IBMQ_API_KEY: Optional[str] = None
    IBMQ_BACKEND_NAME: str = "ibm_kingston"
    IBMQ_INSTANCE: Optional[str] = None
    IBMQ_CHANNEL: str = "ibm_quantum_platform"

    # ------------------------------------------------------- Reproducibility
    RANDOM_SEED: int = 42
    MODEL_VERSION: str = "carescan-hybrid-qml-0.1.0"

    # ------------------------------------------------------------- Artifacts
    ARTIFACT_DIR: Path = REPO_ROOT / "backend" / "artifacts"
    DATASET_ROOT: Optional[Path] = None  # None -> autodiscovery / CARESCAN_DATASET_ROOT

    # ----------------------------------------------------------- Calibration
    CALIBRATION_METHOD: CalibrationMethod = "auto"
    """``auto`` selects Platt vs isotonic by validation Brier score."""

    SCREENING_TARGET_SENSITIVITY: float = 0.85
    """Sensitivity the recommended operating threshold aims for.

    A calibrated model at this dataset's ~6% prevalence rarely emits a probability
    above 0.50, so a 0.50 cut-off on calibrated output misses almost every positive.
    ``backend.training.calibrate`` therefore derives a recommended threshold that
    reaches this sensitivity on out-of-fold probabilities and records it alongside
    the calibrator. 0.85 favours catching lesions over avoiding false alarms, which
    is the trade-off a screening tool should make -- but the value is an engineering
    choice, not a clinically validated one."""

    HIGH_RISK_TARGET_SPECIFICITY: float = 0.95
    """Specificity the recommended HIGH-risk band boundary aims for.

    The upper band answers a different question from the screening threshold: not
    "is this worth looking at" but "is this worth looking at first". It is therefore
    derived at high specificity, so the band stays small enough to be actionable.
    Also an engineering choice, not a validated cut-off."""

    # ------------------------------------------------------ Clinical decision
    # These two are the *fallback* band boundaries, used when a model version has
    # no calibration artifact to take them from. Prefer the per-model values in
    # calibration.json: band boundaries depend on the score distribution of the
    # specific trained model, so a global constant is wrong for any model whose
    # probabilities are distributed differently. Measured on v1-handcrafted, the
    # old 0.50/0.70 defaults were both unreachable -- the model's highest calibrated
    # probability on the test partition was 0.41, so at 0.50 the screen flagged
    # nothing whatsoever and the HIGH band was dead code. See DECISIONS.md.
    SCREENING_THRESHOLD: float = 0.50
    """Experimental screening threshold. NOT a clinically validated cut-off.

    Left at the neutral 0.50 deliberately: it is a *fallback* for an uncalibrated
    or artifact-less model, where 0.50 is the only defensible arbitrary choice. The
    inference path overrides it with the model-specific threshold recommended by
    ``backend.training.calibrate`` for ``SCREENING_TARGET_SENSITIVITY``."""

    HIGH_RISK_THRESHOLD: float = 0.70
    """Upper band boundary for the HIGH risk label. Also experimental, also a
    fallback, and also overridden by the per-model value derived for
    ``HIGH_RISK_TARGET_SPECIFICITY``."""

    # ------------------------------------------------------------- Inference
    FEATURE_MODE: FeatureMode = "multimodal"
    INFERENCE_TIMEOUT_SECONDS: float = 120.0
    RETAIN_UPLOADED_IMAGES: bool = False
    """Data minimisation: uploaded images are deleted once features are derived."""

    # ------------------------------------------------- Image quality control
    # Every limit below was set by measuring the metric's distribution over a
    # 400-image sample of the supplied dataset and placing the limit outside the
    # observed range of usable captures. See backend/ml/image_quality.py for the
    # measured percentiles and the reasoning. These are engineering
    # acceptability limits, NOT clinically validated cut-offs.
    QC_MIN_MEAN_LUMINANCE: float = 40.0  # observed min 65.5
    QC_MAX_MEAN_LUMINANCE: float = 235.0  # observed max 186.5
    QC_MIN_LAPLACIAN_VARIANCE: float = 12.0  # resolution-normalised; p1 = 12.6
    QC_MIN_SHORT_EDGE_PX: int = 224  # extractor input edge; observed min 280
    QC_MIN_ROI_FRACTION: float = 0.02  # observed min 0.06
    QC_MAX_CLIPPED_FRACTION: float = 0.25  # observed max 0.186
    QC_MIN_BYTES_PER_PIXEL: float = 0.05  # observed min 0.078

    @property
    def quantum_dimension(self) -> int:
        """Amplitude-encoding state dimension implied by the qubit count."""
        return 2 ** self.QUANTUM_QUBITS

    @property
    def artifact_dir(self) -> Path:
        return Path(self.ARTIFACT_DIR)

    @property
    def ibm_credentials_available(self) -> bool:
        return bool(self.IBMQ_API_KEY)


settings = Settings()
