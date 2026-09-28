# CareScan Platform Architecture (Backend)

**Status:** the spine and the HTTP surface are implemented and tested. One track is
serving, one is registered and waiting on a model. See §8 for exactly what is and is not
built.

`ARCHITECTURE.md` at the repository root describes the Flutter client. This document
describes the backend, and specifically the layer that turned it from *one model for one
condition* into a platform that can carry several.

---

## 1. Why this layer exists

CareScan was built as a single model for a single condition, and every layer said so. The
routes took an image. The service loaded one artifact version. `InferenceResult` described
a variational quantum classifier looking at an oral photograph, with field documentation
that is specific to that model and load-bearing for the client.

That was the correct shape for one model. It is the wrong shape for what the project is
now: a platform for early disease detection with an oral-lesion track and a 12-lead ECG
track, where the second shares almost everything with the first except its front end.

The naive fix — generalise `InferenceService`, widen `InferenceResult` with optional ECG
fields — trades a working single-condition system for a half-working two-condition one.
The documentation on `InferenceResult.classical_probability` and `primary_calibrated`
encodes a specific, hard-won decision about which model headlines the verdict (DEC-034);
diluting those fields to cover a second modality would make them mean less for both.

So the platform layer is **additive**. Nothing in `backend/ml/track.py`,
`backend/ml/tracks/` or `backend/routes/track_routes.py` is imported by the existing
screening path, and `/api/screening/analyze` returns exactly what it returned before.
`tests/test_tracks.py::test_the_existing_screening_surface_is_untouched` fails if that
stops being true.

## 2. The shared pipeline

Strip the modality away and both tracks run the same seven stages:

```
artifacts → input validation → features → representation
          → classical reference + optional quantum head → calibration → banded verdict
```

Only the first two stages are modality-specific. Everything from "representation" onward
is the same idea with different arrays in it. `backend/ml/track.py` names that shape as a
contract, so a third condition is a registration rather than a rewrite.

| Stage | Oral lesion | 12-lead ECG |
|---|---|---|
| Input | JPEG/PNG photograph | 12 × 1000 float array, mV, 100 Hz |
| Validation | quality report, ROI check | shape, lead order, finiteness, beat-template quality |
| Features | MobileNetV3-Small descriptor | 97 hand-built `ecg-v1` features |
| Representation | TRAIN-fitted reduction | TRAIN-fitted PCA / univariate selection |
| Classical reference | logistic regression | gradient-boosted trees |
| Quantum head | 8-qubit VQC (served, secondary) | **none — gated, see §6** |
| Calibration | Platt / isotonic, TRAIN-fitted | TRAIN-fitted threshold |
| Verdict | banded, classical headlines | banded |

## 3. The contract

Four Pydantic models and one Protocol, all in `backend/ml/track.py`.

**`DiseaseTrack`** is a `runtime_checkable` Protocol, deliberately not a base class. The
oral-lesion track is an *adapter* over the existing `InferenceService`
(`backend/ml/tracks/oral_lesion.py`); the service does not know the adapter exists and no
import runs in that direction. A protocol is what lets the oldest and most-tested path in
the repository stay exactly where it is.

**`InputSpec`** publishes what a caller must send in enough detail to build a client from
it alone — including `channel_names`. That field is not decoration. A caller that sends
the precordial leads first gets a confident and completely wrong answer, and nothing
downstream can detect it, so the lead order is published rather than inferred.

**`ValidationSummary`** is where the methodology lives (§4).

**`TrackAssessment`** is the shared answer vocabulary. Its core is small on purpose:
anything one track knows that the others do not — image quality, circuit depth, per-lead
findings — goes in `detail`, typed by that track, so no track grows a field for another
track's concerns.

**`TrackRegistry`** is thread-safe (FastAPI serves from a pool), refuses duplicate ids
without an explicit `replace=True` (last-write-wins would make the served model depend on
import order), and catches exceptions in `describe_all()` so one broken artifact directory
costs one card in the UI rather than the whole listing.

## 4. The part that is not plumbing

A track **cannot be registered without declaring how its headline number was obtained**:
which dataset, which partition, how many *patients*, and whether a frozen test partition
has ever been scored.

```python
frozen_test_evaluated: bool   # no default. Forgetting it is a ValidationError.

@property
def is_clinically_claimable(self) -> bool:
    return bool(self.frozen_test_evaluated)
```

`frozen_test_evaluated` has no default and is not defaulted to `True`. A model whose only
number comes from a partition that model selection has already seen is a legitimate
research result and an illegitimate clinical claim, and the platform should be able to say
which one it is holding without anyone having to remember to mention it.

Today **both tracks report `is_clinically_claimable == False`**, and that is the correct
answer for both. PTB-XL's `strat_fold` 10 has never been read on this machine. The
oral-lesion operating point was chosen on validation.

Three further honesty constraints are carried as data rather than as documentation:

- **`probability_is_calibrated`** — `False` when the number is a ranking score rather than
  a risk estimate. On the oral track the classical baseline headlines the verdict
  (DEC-034) and its score is *not* calibrated: the baselines are fitted with
  `class_weight="balanced"` against a 4.7% positive rate, which deliberately pushes their
  probabilities away from the base rate. The band is the verdict in that case; a client
  must not render the value as a percentage. The flag makes that checkable at render time.
- **`classical_probability` and `quantum_probability` are reported side by side, never
  blended.** No fusion weight has been validated, and a blended number would correspond to
  no model in any evaluation report.
- **`threshold_selected_on`** — a threshold chosen on the partition it is then reported
  against is self-selected, and naming its origin makes that visible.

`RiskBand.INDETERMINATE` exists for the same reason. Refusing to band is sometimes the
honest answer — a lead-off ECG, an unreadable photograph — and a platform that can only
say LOW or HIGH will say one of them anyway. The ECG track returns `INDETERMINATE` for a
record whose beat template could not be built, rather than banding it LOW.

## 5. HTTP surface

Mounted at `/api/tracks`, alongside (never replacing) the existing routes.

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/tracks` | Every track, **including unready ones**, with `unready_reason` |
| `GET` | `/api/tracks/{id}` | One descriptor |
| `POST` | `/api/tracks/{id}/analyze` | JSON payload |
| `POST` | `/api/tracks/{id}/analyze-image` | Multipart upload, 10 MB cap |

The listing includes broken tracks on purpose: a platform that hides them looks healthier
than it is, and a client that only learns about a condition when it works cannot tell the
user why it is missing.

JSON and multipart are separate endpoints rather than one polymorphic route. They have
different size limits and different failure modes, and an endpoint that guesses which it
received would guess wrong under exactly the conditions that matter.

Error mapping follows the existing screening router:

| Condition | Status |
|---|---|
| Unknown track | 404, naming what *is* registered |
| Payload violates the published `InputSpec` | 422 |
| Artifacts missing or unloadable | 503 |
| Upload too large | 413 |
| Anything else | 500, logged with traceback server-side, no detail to the client |

Validation runs **before** loading, so a malformed ECG sent to a track with no model
returns 422 describing the payload rather than 503 describing the server. The caller
should be told what *they* got wrong.

## 6. Where the quantum component actually is

`TrackDescriptor` carries `uses_quantum` **and** `quantum_role`. A platform that
advertises "quantum" without saying where should not be believed, including by its own
authors.

- **Oral lesion — `uses_quantum=True`.** An 8-qubit variational classifier over the
  reduced image descriptor, reported *alongside* the classical baseline, not in place of
  it. It ranks below the classical baseline on this dataset (ISS-008 / DEC-034), so the
  classical band headlines the verdict while the calibrated probability stays the quantum
  one — the only calibrated number available. Both travel in the response.
- **ECG — `uses_quantum=False`, `quantum_role=None`.** This is a statement about what is
  deployed, not about what is planned.

The ECG quantum head is gated behind a measured bar, not behind a schedule. The E4
classical work ran all four rungs the research mission named — simple, feature-engineered,
modern deep, best fusion — on `strat_fold` 9 (NORM-vs-abnormal, patient-level, DEC-044 then
DEC-046):

| Arm | ROC-AUC | 95% CI |
|---|---|---|
| **`fusion@cnn+gbm`** — the bar | **0.946293** | [0.9364, 0.9550] |
| `cnn@resnet_small` — 1D-CNN on raw 12-lead mV | 0.940476 | [0.9297, 0.9501] |
| `gbm@f97` — boosted trees on 97 `ecg-v1` features | 0.940234 | [0.9297, 0.9494] |

with matched-dimension controls at 0.915006 (d=8), 0.927408 (d=16) and 0.931961 (d=32), and
a permutation test on the tabular winner at p = 0.004975.

The two single arms **tie**: paired patient-clustered delta +0.000242, CI [−0.006101,
+0.006374], spanning zero. A 126k-parameter network reading 12,000 raw samples scores what
400 boosted trees read off 97 scalars. But their fusion beats both with intervals that
exclude zero, so they are not redundant — they disagree per record while agreeing in
aggregate.

That result raised the bar and sharpened the test. A quantum arm is not wired into this
track until it clears 0.946293 under the pre-registered five-condition standard —
patient-level validation, matched classical controls, permutation testing, bootstrap
confidence intervals, generalisation to the untouched test partition — **and** the control
DEC-046 added: a quantum fusion must be compared against a *classical* fusion of the same
shape. Beating a single classical model is no longer evidence of anything, because a second
classical representation already does that.

Until then the descriptor says there is no quantum component here, because there is not.

## 7. Adding a third condition

1. Implement `describe()` and `analyze()` — no inheritance required.
2. Declare a `ValidationSummary`, including `frozen_test_evaluated`. You cannot skip it;
   the model will not construct.
3. Add it to `install_default_tracks()` in `backend/ml/tracks/__init__.py`.

`__init__` must not touch the filesystem. Construction is cheap and loading is lazy, so a
deployment missing one model still starts, serves everything else, and reports the missing
track as unready with a reason.

## 8. Implementation status

| Component | File | Status |
|---|---|---|
| Contract + registry | `backend/ml/track.py` | Implemented, 20 tests |
| Oral-lesion adapter | `backend/ml/tracks/oral_lesion.py` | Implemented, serving |
| ECG track | `backend/ml/tracks/ecg.py` | Implemented, **awaiting a model** |
| HTTP surface | `backend/routes/track_routes.py` | Implemented, 36 tests |
| Wiring | `backend/main.py` | Implemented |

The ECG track is shipped **before** its model, on purpose. Everything a caller needs — the
input contract, the lead order, the provenance block, the failure vocabulary — is fixed
now, and the track reports `ready=false` naming the artifact it is missing. The
alternative, deciding what the platform is allowed to say about a number after seeing the
number, is precisely how a development ROC-AUC becomes a clinical claim.

What is outstanding is a persisted classifier bundle at
`backend/artifacts/models/ecg_classifier/current.joblib`. The evaluation harness does not
write one, and the reason changed when the deep rung landed. Classical arm selection is no
longer open — all four rungs are measured and the fusion won. What remains is that **every
number so far was selected on a development partition**: fold 9 chose the operating point,
the architecture was chosen on an inner TRAIN fold, and fold 10 has never been read.
Freezing a servable model on that basis would ship an operating point chosen against the
data it is reported on.

Dropping a bundle at that path makes the track live with no code change. The bundle carries
its estimator, threshold, feature-set version and transform in one file, so a model can
never be served with another model's operating point.
