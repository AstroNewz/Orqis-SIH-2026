"""The deep rung on PTB-XL: record-locality, the inner split, and the recorded winner (E4).

:mod:`backend.evaluation.e4_deep_baseline` produced ``cnn@resnet_small`` 0.940476 and,
through the fusion it also produced, the bar ``fusion@cnn+gbm`` 0.946293 that every quantum
arm in this phase is measured against. It additionally exports ``split_inner``,
``condition``, ``predict_logits`` and ``_logit``, which :mod:`backend.evaluation.e4_quantum`
re-executes when it recovers those members -- so a defect here does not stay local, it
propagates into the quantum rung's baseline.

That module asserts its recovered members reproduce the recorded scores to 5e-7 and 1e-9.
Worth being precise about what that does and does not establish: it proves **determinism**,
not correctness. A systematic defect would reproduce exactly and pass. These tests cover the
part reproduction cannot, organised around the four ways this module could lie:

* it could **leak across the inner split**: put a patient on both sides of fold 8, or fit a
  conditioning statistic pooled across records rather than computed within one;
* it could **normalise away the signal**: apply an amplitude scaler, which would be a
  statistic over the corpus wearing the costume of preprocessing;
* it could **misalign scores and labels**: the batcher sorts rows for sequential memmap
  access and then restores the caller's order, and a wrong inverse permutation would
  scramble predictions against labels without raising anything;
* it could **drift from the record**: silently change an architecture's shape, so the
  ``resnet_small`` reproduced in E4-Q is not the ``resnet_small`` DEC-046 scored.

The fixtures are synthetic and tiny. Training is not exercised -- one real epoch costs 66
seconds and the suite must stay runnable -- so the trained-model path is covered by its
recorded reproduction in :mod:`backend.evaluation.e4_quantum`, and these tests cover the
deterministic scaffolding around it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.evaluation.e4_deep_baseline import (
    ARCHITECTURES,
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    CLASSICAL_REFERENCE_ARM,
    E4_DEEP_VERSION,
    INNER_VALIDATION_FOLD,
    SAMPLING_FREQUENCY,
    SEED,
    DeepBaselineError,
    _logit,
    _sigmoid,
    build_model,
    condition,
    count_parameters,
    load_reference_arm,
    predict_logits,
    split_inner,
)

N_LEADS = 12
N_SAMPLES = 1000

# DEC-046's recorded winner. Pinned here so a change to ARCHITECTURES that alters the
# model's size cannot pass silently -- the number is quoted in TASKS.md, DECISIONS.md and
# the phase doc, and e4_quantum refits this exact shape to recover its fusion member.
RECORDED_RESNET_SMALL_PARAMS = 126_649


def _partition(folds, *, patients=None, name: str = "TRAIN"):
    """A minimal stand-in carrying only what ``split_inner`` reads."""
    folds = np.asarray(folds, dtype=int)
    n = folds.size
    patients = np.arange(n) if patients is None else np.asarray(patients)
    return type(
        "P",
        (),
        {
            "name": name,
            "strat_folds": folds,
            "patients": patients,
            "y": np.zeros(n, dtype=int),
            "n_records": n,
        },
    )()


def _signals(n: int = 6, *, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) / SAMPLING_FREQUENCY
    base = np.sin(2.0 * np.pi * 8.0 * t)  # inside the 0.5-40 Hz band
    out = base[None, None, :] * rng.uniform(0.5, 2.0, size=(n, N_LEADS, 1))
    return (out + rng.normal(scale=0.05, size=(n, N_LEADS, N_SAMPLES))).astype(np.float32)


# ------------------------------------------------------------ it could leak across folds
class TestInnerSplit:
    """Fold 8 is the inner validation fold, and no patient may straddle it."""

    def test_the_split_is_on_fold_eight(self) -> None:
        assert INNER_VALIDATION_FOLD == 8
        inner_train, inner_validation = split_inner(_partition([1, 2, 8, 8, 3]))
        np.testing.assert_array_equal(inner_validation, [False, False, True, True, False])
        np.testing.assert_array_equal(inner_train, ~inner_validation)

    def test_the_two_masks_partition_every_row_exactly_once(self) -> None:
        inner_train, inner_validation = split_inner(_partition([1, 8, 2, 8, 3, 8]))
        assert not (inner_train & inner_validation).any()
        assert (inner_train | inner_validation).all()

    def test_a_patient_spanning_the_split_is_refused(self) -> None:
        """The defect this guard exists for: one patient with records on both sides."""
        with pytest.raises(DeepBaselineError, match="span the inner split"):
            split_inner(_partition([1, 8, 2], patients=[77, 77, 99]))

    def test_disjoint_patients_are_accepted(self) -> None:
        inner_train, inner_validation = split_inner(
            _partition([1, 1, 8, 8], patients=[10, 10, 20, 20])
        )
        assert inner_train.sum() == 2 and inner_validation.sum() == 2

    def test_train_without_fold_eight_is_refused_rather_than_silently_empty(self) -> None:
        with pytest.raises(DeepBaselineError, match="no fold 8"):
            split_inner(_partition([1, 2, 3]))

    def test_train_that_is_only_fold_eight_is_refused(self) -> None:
        with pytest.raises(DeepBaselineError, match="collapses"):
            split_inner(_partition([8, 8, 8]))


# --------------------------------------------------- it could normalise away the signal
class TestConditioning:
    """0.5-40 Hz, zero-phase, record-local, and deliberately not amplitude-normalised."""

    def test_conditioning_one_record_does_not_depend_on_its_neighbours(self) -> None:
        """The property that makes this stage fold-honest, tested as a property.

        A record conditioned inside one batch must be bit-identical to the same record
        conditioned inside a different batch. Any pooled statistic -- a corpus mean, a
        global scale -- would break this and nothing else in the pipeline would notice.
        """
        batch = _signals(6)
        alone = condition(batch[2:3])
        with_others = condition(batch)[2:3]
        np.testing.assert_array_equal(alone, with_others)

        shuffled = condition(batch[::-1])[::-1][2:3]
        np.testing.assert_array_equal(alone, shuffled)

    def test_amplitude_is_preserved_rather_than_normalised(self) -> None:
        """Doubling the input must double the output; a scaler would collapse both to one."""
        batch = _signals(3)
        once = condition(batch)
        twice = condition(batch * 2.0)
        np.testing.assert_allclose(twice, once * 2.0, rtol=1e-4, atol=1e-5)
        assert np.std(once) != pytest.approx(1.0, abs=0.05), "output looks standardised"

    def test_the_filter_is_zero_phase(self) -> None:
        """``sosfiltfilt`` must not shift the signal; a one-pass filter would delay it."""
        t = np.arange(N_SAMPLES) / SAMPLING_FREQUENCY
        pure = np.sin(2.0 * np.pi * 8.0 * t).astype(np.float32)
        out = condition(pure[None, None, :])[0, 0]
        middle = slice(200, 800)  # away from the edge transients filtfilt leaves
        correlation = np.corrcoef(out[middle], pure[middle])[0, 1]
        assert correlation > 0.99, f"in-band signal was phase-shifted (r={correlation:.3f})"

    def test_out_of_band_content_is_attenuated(self) -> None:
        t = np.arange(N_SAMPLES) / SAMPLING_FREQUENCY
        for frequency in (0.05, 48.0):
            assert frequency < BANDPASS_LOW_HZ or frequency > BANDPASS_HIGH_HZ
            tone = np.sin(2.0 * np.pi * frequency * t).astype(np.float32)
            out = condition(tone[None, None, :])[0, 0]
            assert np.std(out[200:800]) < 0.3 * np.std(tone[200:800])

    def test_conditioning_returns_contiguous_float32(self) -> None:
        """The model consumes this directly via ``torch.from_numpy``."""
        out = condition(_signals(4))
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]


# ------------------------------------------------- it could misalign scores and labels
class TestBatchOrdering:
    """The batcher sorts rows for sequential memmap reads and must restore caller order."""

    def test_predictions_follow_the_requested_row_order(self) -> None:
        """A wrong inverse permutation scrambles scores against labels and raises nothing.

        The model is a fixed random net, so the only thing under test is alignment: asking
        for rows in a shuffled order must permute the answers the same way.
        """
        signals = _signals(8, seed=11)
        model = build_model("resnet_small", seed=SEED)

        ascending = predict_logits(model, signals, np.arange(8), batch_size=3)
        shuffled_rows = np.array([5, 0, 7, 2, 6, 1, 4, 3])
        shuffled = predict_logits(model, signals, shuffled_rows, batch_size=3)
        np.testing.assert_allclose(shuffled, ascending[shuffled_rows], atol=1e-6)

    def test_batch_size_does_not_change_the_answer(self) -> None:
        signals = _signals(8, seed=11)
        model = build_model("resnet_small", seed=SEED)
        rows = np.arange(8)
        np.testing.assert_allclose(
            predict_logits(model, signals, rows, batch_size=8),
            predict_logits(model, signals, rows, batch_size=3),
            atol=1e-6,
        )

    def test_every_row_is_scored_exactly_once(self) -> None:
        signals = _signals(5, seed=2)
        model = build_model("resnet_small", seed=SEED)
        out = predict_logits(model, signals, np.arange(5), batch_size=2)
        assert out.shape == (5,)
        assert np.isfinite(out).all()


# ------------------------------------------------------- it could drift from the record
class TestRecordedArchitecture:
    """The winner E4-Q refits must be the winner DEC-046 scored."""

    def test_resnet_small_has_the_recorded_parameter_count(self) -> None:
        assert count_parameters(build_model("resnet_small")) == RECORDED_RESNET_SMALL_PARAMS

    def test_building_is_deterministic_for_a_seed(self) -> None:
        first = build_model("resnet_small", seed=SEED)
        second = build_model("resnet_small", seed=SEED)
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b)

    def test_a_different_seed_gives_different_weights(self) -> None:
        """Otherwise the determinism test above would pass vacuously."""
        first = build_model("resnet_small", seed=SEED)
        other = build_model("resnet_small", seed=SEED + 1)
        assert any(
            not torch.allclose(a, b) for a, b in zip(first.parameters(), other.parameters())
        )

    @pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
    def test_every_architecture_builds_and_emits_one_logit_per_record(
        self, architecture: str
    ) -> None:
        model = build_model(architecture)
        out = model(torch.from_numpy(condition(_signals(3))))
        assert out.shape == (3,)

    def test_the_three_architectures_disagree_about_capacity(self) -> None:
        """The selection is only informative if the candidates are genuinely different."""
        sizes = {name: count_parameters(build_model(name)) for name in ARCHITECTURES}
        assert len(set(sizes.values())) == len(sizes), sizes

    def test_an_unknown_architecture_is_refused(self) -> None:
        with pytest.raises(DeepBaselineError, match="Unknown architecture"):
            build_model("transformer_xl")


class TestReferenceArm:
    """The tabular member is read back from the record, never re-selected."""

    def test_the_reference_arm_is_the_recorded_classical_winner(self) -> None:
        assert CLASSICAL_REFERENCE_ARM == "gbm@f97"

    def test_a_missing_report_names_the_command_that_builds_it(self, tmp_path: Path) -> None:
        with pytest.raises(DeepBaselineError, match="e4_classical_baseline"):
            load_reference_arm(tmp_path / "absent.json")

    def test_an_absent_arm_is_refused_and_lists_what_was_found(self, tmp_path: Path) -> None:
        report = tmp_path / "classical.json"
        report.write_text(
            json.dumps(
                {"tasks": {"primary_norm_vs_abnormal": {"arms": [{"arm": "logistic@f97"}]}}}
            ),
            encoding="utf-8",
        )
        with pytest.raises(DeepBaselineError):
            load_reference_arm(report)


class TestLogitHelpers:
    """The blender consumes log-odds, so a saturated probability must not become infinite."""

    def test_logit_and_sigmoid_round_trip(self) -> None:
        probability = np.array([0.02, 0.25, 0.5, 0.75, 0.98])
        np.testing.assert_allclose(_sigmoid(_logit(probability)), probability, atol=1e-9)

    def test_saturated_probabilities_stay_finite(self) -> None:
        out = _logit(np.array([0.0, 1.0]))
        assert np.isfinite(out).all()
        assert out[0] < 0 < out[1]

    def test_the_clip_is_symmetric(self) -> None:
        out = _logit(np.array([0.0, 1.0]))
        assert out[0] == pytest.approx(-out[1])

    def test_logit_is_monotone(self) -> None:
        values = _logit(np.linspace(0.0, 1.0, 25))
        assert np.all(np.diff(values) >= 0)


def test_the_version_is_stamped() -> None:
    assert E4_DEEP_VERSION == "v1-e4-deep-1"
