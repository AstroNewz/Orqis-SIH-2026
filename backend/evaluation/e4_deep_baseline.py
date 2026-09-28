"""E4 deep rung -- a 1D residual CNN on the raw 12-lead waveform, and its fusion with the
tabular ceiling.

``e4_classical_baseline`` delivered the mission's first two classical rungs (simple logistic,
feature-engineered gradient boosting) plus a kernel arm and the matched 8/16/32-dimension
controls. The mission names **four** rungs -- "simple + feature-engineered + modern deep +
best fusion" -- and this module is the remaining two. Until it exists, the quantum gate stays
shut, because a circuit measured against a tabular-only ceiling may be beating a baseline
that a deep model already exceeds. That is exactly the straw man §7.4 condition 1 forbids.

Why a deep arm can legitimately beat the tabular one, and why it might not:

* The 97-dimensional ``ecg-v1`` vector is a *hand-built summary*. It keeps interval
  durations, per-lead ST/T amplitudes, axis and spectral content, and discards everything
  else -- beat-to-beat morphology change, P-wave detail, subtle notching, the actual shape
  of a depolarisation. A convolutional model reads the waveform those numbers were measured
  from, so anything the summary threw away is available to it.
* Against that, 17,418 training records is small for a model with this many parameters, and
  the published PTB-XL benchmarks put deep models only modestly ahead of strong feature
  baselines on the superclass tasks. A deep arm that does *not* win is an honest and
  perfectly publishable outcome; it is recorded either way.

**Protocol, identical in discipline to the tabular rungs.**

* Folds 1-7 fit, **fold 8 inner-validation**, folds 1-8 refit, fold 9 scored once,
  **fold 10 never touched** (its waveforms are not on this machine). The tabular arms chose
  hyperparameters by leave-one-fold-out CV across all of folds 1-8; eight refits of a CNN
  per candidate is not affordable, so the deep arm holds out a *single* TRAIN fold instead.
  That is a weaker selection estimate, not a weaker guarantee: fold 8 is still inside TRAIN,
  and fold 9 still never participates in any choice.
* The architecture, the learning rate and the **number of epochs** are all chosen by fold-8
  ROC-AUC. The operating threshold comes from the same fold-8 predictions, produced by a
  model that never saw fold 8.
* The fusion blender is fitted on fold 8 as well, on out-of-sample predictions from *both*
  members: the CNN trained on folds 1-7, and the gradient-boosted model's leave-one-fold-out
  prediction for fold 8. Neither member has seen fold 8, so the blender is not learning to
  trust whichever member memorised it.

**Nothing pooled across records enters the input.** The waveform cache stores decoded
physical millivolts and nothing else. This module applies one record-local, deterministic
0.5-40 Hz zero-phase band-pass -- the same conditioning ``backend/ml/features_ecg.py``
applies, so the two rungs see comparably conditioned signal -- and then feeds millivolts
directly. Amplitude is deliberately **not** normalised per record: absolute voltage is
diagnostic (hypertrophy is literally a voltage criterion), and dividing it out would destroy
the one signal most relevant to the failure mode the tabular ceiling is worst at. The input
``BatchNorm`` that follows is a model parameter fitted on TRAIN batches like every other
weight, and at evaluation it uses its TRAIN running statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt

from sklearn.linear_model import LogisticRegression

from backend.dataset.ptbxl import TRAIN_FOLDS, VALIDATION_FOLDS
from backend.evaluation.e4_classical_baseline import (
    BOOTSTRAP_DRAWS,
    FIXED_SPECIFICITY,
    SEED,
    Partition,
    bootstrap_roc_auc,
    build_estimator,
    build_partition,
    failure_map,
    guard_partitions,
    out_of_fold_scores,
    paired_bootstrap_delta,
    primary_labels,
    raw_scores,
    roc_auc,
    select_threshold,
)
from backend.evaluation.metrics import evaluate_predictions, sensitivity_at_specificity
from backend.ml.ecg_transform import fit_ecg_transform
from backend.training.prepare_ecg_features import (
    ECG_CACHE_VERSION,
    FEATURE_SET_VERSION,
    EcgCohort,
    load_cohort,
)
from backend.training.prepare_ecg_signals import SignalCache, load_signal_cache

logger = logging.getLogger(__name__)

E4_DEEP_VERSION = "v1-e4-deep-1"

INNER_VALIDATION_FOLD = 8
"""The single TRAIN fold held out for architecture, epoch and threshold selection."""

BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 40.0
BANDPASS_ORDER = 3
SAMPLING_FREQUENCY = 100.0

MAX_EPOCHS = 30
PATIENCE = 8
BATCH_SIZE = 128
WEIGHT_DECAY = 1e-4

STEM_POOL = 2
"""Max-pool after the stem, so the residual blocks run at 25 Hz-equivalent resolution.

This is a **compute** decision, and it is worth being explicit about why that is
legitimate. Measured on this machine, a stride-2-only stem at 32-192 channels costs 285 s
per epoch of forward/backward on 8 CPU threads; three architectures over 30 epochs is then
about ten hours, which is not a grid anyone will re-run. Pooling once after the stem and
halving the channel widths costs roughly an eighth of that.

The quantity that drove the change is wall-clock, which is independent of every label --
no fold-9 number influenced it, and none could have, because the deep arm had not been
scored when the timing was measured. The cost is genuine: 25 Hz resolves the QRS complex
(~80-100 ms) but blurs fine notching, so if the deep arm loses, "it was under-resolved" is
a live explanation and is recorded as one rather than being discovered afterwards.
"""

ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    # Three shapes that disagree about *where* the capacity should go, so the selection is
    # informative rather than a formality: depth with narrow kernels, the same depth wider,
    # and the same width with a receptive field long enough to span a whole QRS complex
    # (kernel 9 at the pooled 25 Hz covers ~360 ms, comfortably more than one complex).
    "resnet_small": {"widths": (16, 32, 64, 96), "kernel": 5, "dropout": 0.3},
    "resnet_wide": {"widths": (24, 48, 96, 128), "kernel": 5, "dropout": 0.3},
    "resnet_long_kernel": {"widths": (16, 32, 64, 96), "kernel": 9, "dropout": 0.3},
}

LEARNING_RATE = 1e-3

DEFAULT_FEATURE_CACHE = Path("backend/artifacts/dataset/ptbxl/features_ecg-v1.npz")
DEFAULT_SIGNAL_DIR = Path("backend/artifacts/dataset")
"""The two caches live in different directories because they have different lifetimes: the
feature cache sits beside the PTB-XL tree it was extracted from, while the 897 MiB waveform
cache is a rebuildable derivative. Both directories are gitignored."""

CLASSICAL_REFERENCE_ARM = "gbm@f97"
"""The tabular ceiling this rung is compared against and fused with.

``gbm@f97`` won the classical grid at ROC-AUC 0.940234 (DEC-044). Its hyperparameters are
**read back** from the saved classical report rather than re-selected here. Re-running
``select_configuration`` would re-open a choice that is already recorded, and a grid search
repeated with a different tie-breaking order could silently land on a different cell than
the one DEC-044 reports -- the reference arm would then no longer be the arm in the record.
Reading the recorded cell also costs one refit instead of a whole grid.
"""

DEFAULT_CLASSICAL_REPORT = Path("backend/artifacts/reports/e4_classical_baseline.json")
CLASSICAL_REFERENCE_TASK = "primary_norm_vs_abnormal"


class DeepBaselineError(RuntimeError):
    """Raised when the deep rung cannot be built, fitted or trusted."""


def load_reference_arm(
    report_path: Optional[Path] = None,
    *,
    arm: str = CLASSICAL_REFERENCE_ARM,
    task: str = CLASSICAL_REFERENCE_TASK,
) -> Dict[str, Any]:
    """Recover the recorded tabular winner: its model, its cell, and its published ROC-AUC."""
    path = Path(report_path) if report_path is not None else DEFAULT_CLASSICAL_REPORT
    if not path.exists():
        raise DeepBaselineError(
            f"{path} not found. Run `python -m backend.evaluation.e4_classical_baseline` "
            "first; the deep rung is compared against its recorded winner."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    try:
        arms = report["tasks"][task]["arms"]
    except (KeyError, TypeError) as error:
        raise DeepBaselineError(f"{path} has no task {task!r}: {error}") from error
    for entry in arms:
        if entry.get("arm") == arm:
            if not entry.get("selected_params"):
                raise DeepBaselineError(f"Arm {arm!r} in {path} records no selected_params.")
            return {
                "arm": arm,
                "model": entry.get("model", arm.split("@", 1)[0]),
                "selected_params": dict(entry["selected_params"]),
                "published_roc_auc": entry.get("validation_roc_auc"),
                "report_path": str(path),
                "baseline_version": report.get("baseline_version"),
            }
    raise DeepBaselineError(
        f"Arm {arm!r} is not in {path}; found {[e.get('arm') for e in arms]}."
    )


# --------------------------------------------------------------------------- conditioning
_BANDPASS_SOS = butter(
    BANDPASS_ORDER,
    [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
    btype="bandpass",
    fs=SAMPLING_FREQUENCY,
    output="sos",
)


def condition(batch: np.ndarray) -> np.ndarray:
    """Record-local 0.5-40 Hz zero-phase band-pass on ``(n, leads, samples)``.

    Record-local and deterministic, so it is fold-honest by construction: it is a function
    of one record's own samples and of nothing else in the corpus. This is the same property
    that makes ``features_ecg`` structurally safe, and the same reason no scaler is applied
    here -- a scaler would be a statistic pooled across records.
    """
    filtered = sosfiltfilt(_BANDPASS_SOS, np.asarray(batch, dtype=np.float64), axis=-1)
    return np.ascontiguousarray(filtered, dtype=np.float32)


# --------------------------------------------------------------------------------- model
class ResidualBlock1d(nn.Module):
    """Two convolutions and a projection shortcut."""

    def __init__(self, c_in: int, c_out: int, kernel: int, stride: int, dropout: float) -> None:
        super().__init__()
        padding = kernel // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.dropout = nn.Dropout(dropout)
        if stride == 1 and c_in == c_out:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.dropout(y)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.shortcut(x))


class EcgResNet1d(nn.Module):
    """A compact residual CNN over ``(batch, 12, 1000)`` millivolt waveforms."""

    def __init__(
        self,
        n_leads: int = 12,
        widths: Sequence[int] = (16, 32, 64, 96),
        kernel: int = 5,
        dropout: float = 0.3,
        stem_kernel: int = 7,
        stem_pool: int = STEM_POOL,
    ) -> None:
        super().__init__()
        # Fitted on TRAIN batches like every other parameter; at eval it uses the TRAIN
        # running statistics. It is here so the network does not have to learn the scale of
        # a millivolt, while the *relative* amplitude between records is preserved.
        self.input_norm = nn.BatchNorm1d(n_leads)
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, widths[0], stem_kernel, stride=2, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
            # 1000 -> 500 -> 250 samples. The residual stack is where essentially all the
            # arithmetic lives, so halving its input length here halves the whole model's
            # cost; see STEM_POOL for why that trade was taken and what it costs.
            nn.MaxPool1d(stem_pool) if stem_pool > 1 else nn.Identity(),
        )
        blocks: List[nn.Module] = []
        channels = widths[0]
        for width in widths[1:]:
            blocks.append(ResidualBlock1d(channels, width, kernel, stride=2, dropout=dropout))
            channels = width
        self.blocks = nn.Sequential(*blocks)
        self.head_dropout = nn.Dropout(dropout)
        # Average pooling reports how much of the record looks abnormal; max pooling reports
        # whether any part of it does. A 10-second strip can be diagnostic on one beat, so
        # both summaries are kept rather than choosing between them.
        self.classifier = nn.Linear(2 * channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(self.input_norm(x))
        x = self.blocks(x)
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)
        return self.classifier(self.head_dropout(pooled)).squeeze(-1)


def build_model(architecture: str, *, n_leads: int = 12, seed: int = SEED) -> EcgResNet1d:
    if architecture not in ARCHITECTURES:
        raise DeepBaselineError(
            f"Unknown architecture {architecture!r}; known: {sorted(ARCHITECTURES)}."
        )
    torch.manual_seed(seed)
    return EcgResNet1d(n_leads=n_leads, **ARCHITECTURES[architecture])


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ------------------------------------------------------------------------------ batching
def _iterate_batches(
    signals: np.ndarray,
    rows: np.ndarray,
    batch_size: int,
    *,
    order: Optional[np.ndarray] = None,
) -> Any:
    """Yield conditioned ``(batch, leads, samples)`` float32 blocks.

    ``rows`` indexes the memory-mapped cache, so only the batch is ever resident. The cache
    is ~900 MiB; holding it in RAM alongside the model is avoidable and therefore avoided.
    """
    sequence = np.arange(rows.size) if order is None else order
    for start in range(0, sequence.size, batch_size):
        picked = sequence[start : start + batch_size]
        # Memory-mapped fancy indexing requires sorted access to stay sequential; the
        # inverse permutation restores the caller's order so labels stay aligned.
        cache_rows = rows[picked]
        ordering = np.argsort(cache_rows, kind="stable")
        block = np.asarray(signals[cache_rows[ordering]], dtype=np.float32)
        restore = np.empty_like(ordering)
        restore[ordering] = np.arange(ordering.size)
        yield picked, condition(block[restore])


def predict_logits(
    model: EcgResNet1d,
    signals: np.ndarray,
    rows: np.ndarray,
    *,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Score every row once, in cache order, returning logits aligned with ``rows``."""
    model.eval()
    logits = np.full(rows.size, np.nan, dtype=np.float64)
    with torch.no_grad():
        for picked, block in _iterate_batches(signals, rows, batch_size):
            logits[picked] = model(torch.from_numpy(block)).double().numpy()
    if not np.isfinite(logits).all():
        raise DeepBaselineError("Some rows received no prediction.")
    return logits


@dataclass
class TrainingHistory:
    """Per-epoch record of a fit, kept so a reported epoch choice can be re-read."""

    architecture: str
    learning_rate: float
    epochs: List[Dict[str, Any]]
    best_epoch: int
    best_inner_roc_auc: Optional[float]
    seconds: float
    n_parameters: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "learning_rate": self.learning_rate,
            "n_parameters": self.n_parameters,
            "best_epoch": self.best_epoch,
            "best_inner_roc_auc": self.best_inner_roc_auc,
            "seconds": round(self.seconds, 1),
            "epochs": self.epochs,
        }


def train_model(
    architecture: str,
    signals: np.ndarray,
    train_rows: np.ndarray,
    train_y: np.ndarray,
    *,
    inner_rows: Optional[np.ndarray] = None,
    inner_y: Optional[np.ndarray] = None,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
    fixed_epochs: Optional[int] = None,
) -> Tuple[EcgResNet1d, TrainingHistory]:
    """Fit one configuration.

    With ``inner_rows``/``inner_y`` this is the *selection* fit: it scores the held-out
    inner fold after every epoch, keeps the best-scoring weights, and stops early. With
    ``fixed_epochs`` it is the *refit*: it runs a predetermined number of epochs on the
    larger partition and never looks at anything held out, because the choice was already
    made. Mixing the two -- refitting while peeking at a held-out score -- would turn the
    epoch count into a second, undeclared selection.
    """
    if (inner_rows is None) == (fixed_epochs is None):
        raise DeepBaselineError(
            "Pass exactly one of inner_rows (selection fit) or fixed_epochs (refit)."
        )
    torch.manual_seed(seed)
    model = build_model(architecture, n_leads=int(signals.shape[1]), seed=seed)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    # The cosine schedule always spans max_epochs, so a refit stopped at the selected epoch
    # follows exactly the learning-rate trajectory that produced that selection.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max_epochs)
    criterion = nn.BCEWithLogitsLoss()
    generator = np.random.default_rng(seed)

    total_epochs = int(fixed_epochs) if fixed_epochs is not None else int(max_epochs)
    labels = torch.from_numpy(np.asarray(train_y, dtype=np.float32))
    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_score = -np.inf
    best_epoch = total_epochs
    started = time.perf_counter()

    for epoch in range(1, total_epochs + 1):
        model.train()
        order = generator.permutation(train_rows.size)
        running = 0.0
        seen = 0
        for picked, block in _iterate_batches(signals, train_rows, batch_size, order=order):
            optimiser.zero_grad(set_to_none=True)
            logits = model(torch.from_numpy(block))
            loss = criterion(logits, labels[picked])
            loss.backward()
            optimiser.step()
            running += float(loss.item()) * picked.size
            seen += int(picked.size)
        scheduler.step()
        entry: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": round(running / max(seen, 1), 6),
            "learning_rate": round(float(scheduler.get_last_lr()[0]), 8),
        }
        if inner_rows is not None and inner_y is not None:
            inner_logits = predict_logits(model, signals, inner_rows, batch_size=batch_size)
            score = roc_auc(inner_y, inner_logits)
            entry["inner_roc_auc"] = None if score is None else round(float(score), 6)
            if score is not None and score > best_score:
                best_score = float(score)
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            history.append(entry)
            logger.info(
                "    epoch %2d/%d  loss %.4f  inner ROC-AUC %s",
                epoch,
                total_epochs,
                entry["train_loss"],
                "n/a" if score is None else f"{score:.6f}",
            )
            if epoch - best_epoch >= patience:
                logger.info("    early stop: %d epochs without improvement", patience)
                break
        else:
            history.append(entry)
            logger.info(
                "    epoch %2d/%d  loss %.4f (refit, nothing held out)",
                epoch,
                total_epochs,
                entry["train_loss"],
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, TrainingHistory(
        architecture=architecture,
        learning_rate=learning_rate,
        epochs=history,
        best_epoch=best_epoch,
        best_inner_roc_auc=None if best_score == -np.inf else round(best_score, 6),
        seconds=time.perf_counter() - started,
        n_parameters=count_parameters(model),
    )


# ------------------------------------------------------------------------------- helpers
def _logit(probability: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))


def split_inner(train: Partition) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean masks for the inner-train folds and the inner-validation fold."""
    folds = np.asarray(train.strat_folds, dtype=int)
    held = folds == INNER_VALIDATION_FOLD
    if not held.any():
        raise DeepBaselineError(
            f"TRAIN contains no fold {INNER_VALIDATION_FOLD}; inner validation is impossible."
        )
    if not (~held).any():
        raise DeepBaselineError("TRAIN collapses to the inner-validation fold alone.")
    inner_patients = np.unique(train.patients[held])
    outer_patients = np.unique(train.patients[~held])
    shared = np.intersect1d(inner_patients, outer_patients)
    if shared.size:
        raise DeepBaselineError(
            f"{shared.size} patient(s) span the inner split, first few: {shared[:5].tolist()}."
        )
    return ~held, held


def _arm_payload(
    name: str,
    probability: np.ndarray,
    validation: Partition,
    threshold: Dict[str, Any],
    *,
    bootstrap_draws: int,
    seed: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Every arm reports the same block, so arms are comparable line by line."""
    metrics = evaluate_predictions(
        validation.y, probability, threshold=threshold["threshold"]
    )
    payload: Dict[str, Any] = {
        "arm": name,
        "threshold": threshold,
        "validation": metrics.to_dict(),
        "validation_roc_auc": metrics.roc_auc,
        "validation_bootstrap_roc_auc": bootstrap_roc_auc(
            validation.y, probability, validation.patients, draws=bootstrap_draws, seed=seed
        ),
        "sensitivity_at_fixed_specificity": sensitivity_at_specificity(
            validation.y, probability, FIXED_SPECIFICITY
        ),
        "validation_scores": probability,
    }
    if extra:
        payload.update(extra)
    return payload


# ---------------------------------------------------------------------------------- run
def run(
    cache: Optional[Path] = None,
    signals_dir: Optional[Path] = None,
    *,
    classical_report: Optional[Path] = None,
    architectures: Optional[Sequence[str]] = None,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Select on fold 8, refit on folds 1-8, score fold 9 once. Fold 10 is never read."""
    torch.manual_seed(seed)
    started = time.perf_counter()

    cache = Path(cache) if cache is not None else DEFAULT_FEATURE_CACHE
    cohort = load_cohort(cache)
    signal_cache: SignalCache = load_signal_cache(
        Path(signals_dir) if signals_dir is not None else DEFAULT_SIGNAL_DIR
    )
    if set(signal_cache.metadata.get("folds", [])) & {10}:
        raise DeepBaselineError("The signal cache contains fold 10; refusing to run.")

    labels = primary_labels(cohort)
    transform = fit_ecg_transform(cohort, reduction="identity")
    if set(transform.fitted_on_folds) - set(TRAIN_FOLDS):
        raise DeepBaselineError(
            f"Representation fitted on non-TRAIN folds {transform.fitted_on_folds}."
        )
    train = build_partition(cohort, transform, labels, sorted(TRAIN_FOLDS), "TRAIN")
    validation = build_partition(
        cohort, transform, labels, sorted(VALIDATION_FOLDS), "VALIDATION"
    )
    guards = guard_partitions(train, validation)

    train_rows = signal_cache.rows_for(train.ecg_ids)
    validation_rows = signal_cache.rows_for(validation.ecg_ids)
    if np.intersect1d(train_rows, validation_rows).size:
        raise DeepBaselineError("A waveform row is shared between TRAIN and VALIDATION.")

    inner_train_mask, inner_validation_mask = split_inner(train)
    logger.info(
        "TRAIN %d rec / %d pt   inner-train %d   inner-val (fold %d) %d   VALIDATION %d rec / %d pt",
        train.n_records,
        train.n_patients,
        int(inner_train_mask.sum()),
        INNER_VALIDATION_FOLD,
        int(inner_validation_mask.sum()),
        validation.n_records,
        validation.n_patients,
    )

    # ---------------------------------------------------------------- architecture search
    wanted = list(architectures) if architectures else list(ARCHITECTURES)
    inner_rows = train_rows[inner_validation_mask]
    inner_y = train.y[inner_validation_mask]
    selection: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_inner_logits: Optional[np.ndarray] = None
    for architecture in wanted:
        logger.info("  selection fit: %s", architecture)
        candidate_model, history = train_model(
            architecture,
            signal_cache.signals,
            train_rows[inner_train_mask],
            train.y[inner_train_mask],
            inner_rows=inner_rows,
            inner_y=inner_y,
            max_epochs=max_epochs,
            batch_size=batch_size,
            seed=seed,
        )
        # ``train_model`` restored the best-epoch weights, so this *is* the folds-1-7 model
        # the selection chose. Scoring fold 8 with it now, rather than refitting the winner
        # afterwards, costs one forward pass instead of a second full training run -- and
        # removes the chance that a re-fit lands somewhere slightly different from the model
        # whose score justified the choice.
        candidate_logits = predict_logits(
            candidate_model, signal_cache.signals, inner_rows, batch_size=batch_size
        )
        candidate = history.to_dict()
        selection.append(candidate)
        if best is None or (candidate["best_inner_roc_auc"] or -1.0) > (
            best["best_inner_roc_auc"] or -1.0
        ):
            best = candidate
            best_inner_logits = candidate_logits
    if best is None or best_inner_logits is None:
        raise DeepBaselineError("No architecture was fitted.")
    logger.info(
        "  selected %s at epoch %d (inner ROC-AUC %s)",
        best["architecture"],
        best["best_epoch"],
        best["best_inner_roc_auc"],
    )

    # Fold-8 predictions from a model fitted on folds 1-7 only: out-of-sample, so they can
    # legitimately set the threshold and train the fusion blender.
    inner_logits = best_inner_logits
    recomputed = roc_auc(inner_y, inner_logits)
    if recomputed is None or abs(float(recomputed) - float(best["best_inner_roc_auc"])) > 1e-6:
        raise DeepBaselineError(
            f"Retained model scores {recomputed} on fold {INNER_VALIDATION_FOLD} but the "
            f"selection recorded {best['best_inner_roc_auc']}; the wrong weights are loaded."
        )
    deep_threshold = select_threshold(inner_y, _sigmoid(inner_logits))
    deep_threshold["selected_on"] = (
        f"TRAIN fold {INNER_VALIDATION_FOLD}, predicted by a model fitted on folds 1-7"
    )

    # ----------------------------------------------------------------------------- refit
    logger.info("  refit on folds 1-8 for %d epochs", best["best_epoch"])
    final_model, refit_history = train_model(
        best["architecture"],
        signal_cache.signals,
        train_rows,
        train.y,
        fixed_epochs=int(best["best_epoch"]),
        max_epochs=max_epochs,
        batch_size=batch_size,
        seed=seed,
    )
    deep_validation_logits = predict_logits(
        final_model, signal_cache.signals, validation_rows, batch_size=batch_size
    )
    deep_probability = _sigmoid(deep_validation_logits)

    arms: List[Dict[str, Any]] = [
        _arm_payload(
            f"cnn@{best['architecture']}",
            deep_probability,
            validation,
            deep_threshold,
            bootstrap_draws=bootstrap_draws,
            seed=seed,
            extra={
                "kind": "deep",
                "architecture": best["architecture"],
                "n_parameters": best["n_parameters"],
                "selected_epoch": best["best_epoch"],
                "input": "raw 12-lead waveform, 0.5-40 Hz record-local band-pass, millivolts",
                "input_dimension": int(signal_cache.n_leads * signal_cache.n_samples),
            },
        )
    ]

    # ------------------------------------------------------- the tabular arm, recomputed
    reference = load_reference_arm(classical_report)
    logger.info(
        "  refitting the recorded tabular winner %s %s",
        reference["arm"],
        json.dumps(reference["selected_params"], sort_keys=True),
    )
    tabular_out_of_fold, _ = out_of_fold_scores(
        reference["model"], reference["selected_params"], train, seed=seed
    )
    tabular_threshold = select_threshold(train.y, tabular_out_of_fold)
    tabular_estimator = build_estimator(
        reference["model"], reference["selected_params"], seed=seed
    )
    tabular_estimator.fit(train.features, train.y)
    tabular_probability = raw_scores(tabular_estimator, validation.features)
    tabular_roc_auc = roc_auc(validation.y, tabular_probability)
    published = reference["published_roc_auc"]
    # A silent drift here would mean the "ceiling" the deep arm is measured against is not
    # the ceiling on record, so the discrepancy is asserted rather than eyeballed.
    reproduces = (
        published is not None
        and tabular_roc_auc is not None
        and abs(float(tabular_roc_auc) - float(published)) < 1e-9
    )
    if not reproduces:
        raise DeepBaselineError(
            f"{reference['arm']} scored {tabular_roc_auc} here but {published} in "
            f"{reference['report_path']}. The reference arm must reproduce exactly, or the "
            "deep arm is being compared against a different baseline than the record names."
        )
    arms.append(
        _arm_payload(
            reference["arm"],
            tabular_probability,
            validation,
            tabular_threshold,
            bootstrap_draws=bootstrap_draws,
            seed=seed,
            extra={
                "kind": "tabular_reference",
                "selected_params": reference["selected_params"],
                "input": "97 ecg-v1 features",
                "input_dimension": int(train.features.shape[1]),
                "reproduces_recorded_roc_auc": reproduces,
                "recorded_roc_auc": published,
                "recorded_in": reference["report_path"],
                "note": (
                    "Refitted here on the cell recorded in the classical report, because "
                    "that report deliberately strips raw score vectors and the fusion needs "
                    "them. The hyperparameters are read back, not re-selected."
                ),
            },
        )
    )

    # ---------------------------------------------------------------------------- fusion
    # Both members are out-of-sample on fold 8: the CNN was fitted on folds 1-7, and the
    # tabular member's fold-8 values are its leave-one-fold-out predictions. A blender
    # fitted on in-sample predictions would learn to trust whichever member memorised more.
    tabular_inner = tabular_out_of_fold[inner_validation_mask]
    blend_train = np.column_stack([inner_logits, _logit(tabular_inner)])
    blender = LogisticRegression(max_iter=1000)
    blender.fit(blend_train, inner_y)
    blend_validation = np.column_stack(
        [deep_validation_logits, _logit(tabular_probability)]
    )
    fusion_probability = blender.predict_proba(blend_validation)[:, 1]
    blend_inner_probability = blender.predict_proba(blend_train)[:, 1]
    fusion_threshold = select_threshold(inner_y, blend_inner_probability)
    fusion_threshold["selected_on"] = (
        f"TRAIN fold {INNER_VALIDATION_FOLD}, both members out-of-sample"
    )
    arms.append(
        _arm_payload(
            "fusion@cnn+gbm",
            fusion_probability,
            validation,
            fusion_threshold,
            bootstrap_draws=bootstrap_draws,
            seed=seed,
            extra={
                "kind": "fusion",
                "members": [f"cnn@{best['architecture']}", reference["arm"]],
                "blender": "logistic regression on the two members' log-odds",
                "blender_coefficients": {
                    "deep": round(float(blender.coef_[0][0]), 6),
                    "tabular": round(float(blender.coef_[0][1]), 6),
                    "intercept": round(float(blender.intercept_[0]), 6),
                },
                "fitted_on": (
                    f"TRAIN fold {INNER_VALIDATION_FOLD} only, with both members "
                    "out-of-sample on it"
                ),
                "inner_roc_auc": roc_auc(inner_y, blend_inner_probability),
            },
        )
    )

    # ------------------------------------------------------------------------ comparison
    ranked = sorted(
        arms, key=lambda arm: (arm["validation_roc_auc"] is not None, arm["validation_roc_auc"]),
        reverse=True,
    )
    strongest = ranked[0]
    paired = [
        {
            "arm": arm["arm"],
            "versus": strongest["arm"],
            **paired_bootstrap_delta(
                validation.y,
                np.asarray(strongest["validation_scores"], dtype=np.float64),
                np.asarray(arm["validation_scores"], dtype=np.float64),
                validation.patients,
                draws=bootstrap_draws,
                seed=seed,
            ),
        }
        for arm in ranked[1:]
    ]

    deep_arm = arms[0]
    tabular_arm = arms[1]
    deep_vs_tabular = paired_bootstrap_delta(
        validation.y,
        np.asarray(deep_arm["validation_scores"], dtype=np.float64),
        np.asarray(tabular_arm["validation_scores"], dtype=np.float64),
        validation.patients,
        draws=bootstrap_draws,
        seed=seed,
    )

    report: Dict[str, Any] = {
        "deep_version": E4_DEEP_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "cache_version": ECG_CACHE_VERSION,
        "signal_cache_version": signal_cache.metadata.get("signal_cache_version"),
        "seed": seed,
        "bootstrap_draws": bootstrap_draws,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "protocol": {
            "fit_folds": sorted(TRAIN_FOLDS),
            "inner_validation_fold": INNER_VALIDATION_FOLD,
            "scored_folds": sorted(VALIDATION_FOLDS),
            "test_folds_read": [],
            "conditioning": (
                f"record-local {BANDPASS_LOW_HZ}-{BANDPASS_HIGH_HZ} Hz zero-phase "
                f"band-pass, order {BANDPASS_ORDER}; no amplitude normalisation"
            ),
            "selection_criterion": f"ROC-AUC on TRAIN fold {INNER_VALIDATION_FOLD}",
        },
        "cohort": {
            "train": train.summary(),
            "validation": validation.summary(),
            "guards": guards,
            "signal_cache": {
                key: signal_cache.metadata.get(key)
                for key in ("n_rows", "n_failures", "lead_names", "n_samples",
                            "sampling_frequency", "units", "fitted_statistics")
            },
        },
        "architecture_selection": selection,
        "selected": {
            "architecture": best["architecture"],
            "epoch": best["best_epoch"],
            "inner_roc_auc": best["best_inner_roc_auc"],
        },
        "threshold_source": (
            "the winning selection fit itself (folds 1-7), reused rather than refitted; "
            "its retained fold-8 ROC-AUC is asserted to match the selection record"
        ),
        "refit_history": refit_history.to_dict(),
        "arms": arms,
        "strongest_arm": strongest["arm"],
        "paired_vs_strongest": paired,
        "deep_vs_tabular": {
            "a": deep_arm["arm"],
            "b": tabular_arm["arm"],
            **deep_vs_tabular,
        },
        "failure_map": failure_map(strongest, validation, cohort, transform, labels),
        "total_seconds": round(time.perf_counter() - started, 1),
        "test_partition_used": False,
    }
    return report


def summary_lines(report: Dict[str, Any]) -> List[str]:
    lines: List[str] = [
        f"E4 deep rung {report['deep_version']}  (seed {report['seed']})",
        f"  TRAIN {report['cohort']['train']['n_records']} rec / "
        f"{report['cohort']['train']['n_patients']} pt   "
        f"VALIDATION {report['cohort']['validation']['n_records']} rec / "
        f"{report['cohort']['validation']['n_patients']} pt",
        f"  selected {report['selected']['architecture']} @ epoch "
        f"{report['selected']['epoch']} (inner ROC-AUC {report['selected']['inner_roc_auc']})",
        "",
        "  arm                        ROC-AUC     95% CI            bal.acc   sens@sp0.90",
    ]
    for arm in sorted(
        report["arms"],
        key=lambda a: (a["validation_roc_auc"] is not None, a["validation_roc_auc"]),
        reverse=True,
    ):
        interval = arm["validation_bootstrap_roc_auc"]
        auc = arm["validation_roc_auc"]
        sens = arm["sensitivity_at_fixed_specificity"]
        lines.append(
            "  {:<24} {:>9} {:>18}  {:>8}  {:>10}".format(
                arm["arm"],
                "n/a" if auc is None else f"{auc:.6f}",
                f"[{interval['p2_5']:.4f}, {interval['p97_5']:.4f}]"
                if interval.get("p2_5") is not None
                else "n/a",
                f"{arm['validation']['balanced_accuracy']:.4f}"
                if arm["validation"].get("balanced_accuracy") is not None
                else "n/a",
                f"{sens['sensitivity']:.4f}" if sens.get("sensitivity") is not None else "n/a",
            )
        )
    delta = report["deep_vs_tabular"]
    lines += [
        "",
        f"  {delta['a']} - {delta['b']} = {delta['observed_delta']:+.6f} "
        f"[{delta['p2_5']:+.5f}, {delta['p97_5']:+.5f}]  "
        f"excludes zero: {delta['excludes_zero']}",
        f"  total {report['total_seconds']} s   test_partition_used="
        f"{report['test_partition_used']}",
    ]
    return lines


def _strip_arrays(value: Any) -> Any:
    """Drop raw score vectors before serialising; keep the report auditable, not bulky."""
    if isinstance(value, dict):
        return {
            key: _strip_arrays(item)
            for key, item in value.items()
            if key != "validation_scores"
        }
    if isinstance(value, list):
        return [_strip_arrays(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="E4 deep rung: 1D CNN + fusion on PTB-XL.")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--signals", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--classical-report", type=Path, default=None)
    parser.add_argument("--architectures", default=None, help="Comma-separated subset.")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = run(
        cache=args.cache,
        signals_dir=args.signals,
        classical_report=args.classical_report,
        architectures=(
            [name.strip() for name in args.architectures.split(",") if name.strip()]
            if args.architectures
            else None
        ),
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    for line in summary_lines(report):
        logger.info("%s", line)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(_strip_arrays(report), indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info("")
        logger.info("Report written to %s", args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
