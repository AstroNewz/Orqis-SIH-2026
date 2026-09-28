/// Client-side mirror of the backend's platform track contract
/// (`backend/ml/track.py`), as served by `GET /api/tracks`.
///
/// The backend puts a track's [TrackValidationSummary] inside every descriptor on
/// purpose: "putting it in the payload means a client cannot render a number as
/// validated by forgetting to ask". These models keep that property on this side of
/// the wire. A headline metric is never exposed as a bare number — it is only
/// reachable through accessors that carry its partition and its frozen-test status
/// with it, so the honest caption is the path of least resistance rather than an
/// extra step a screen has to remember.
///
/// Parsing is deliberately tolerant of *unknown* values and intolerant of *missing*
/// ones. A backend that adds a fourth modality must not crash an older client, but a
/// descriptor arriving without `frozen_test_evaluated` is not a descriptor whose
/// claims can be judged, and guessing a default there is how a development estimate
/// gets shown as validated.
library;

import 'package:carescan/core/errors/failures.dart';

/// What a track consumes. Determines the transport, not the science.
enum TrackModality {
  image('image'),
  signal('signal'),
  tabular('tabular'),

  /// A modality this build does not know about. Forward compatibility: a client
  /// that crashes on an unrecognised track cannot show the ones it does understand.
  unknown('unknown');

  const TrackModality(this.wireValue);

  final String wireValue;

  static TrackModality fromWire(String? value) {
    for (final modality in values) {
      if (modality.wireValue == value) return modality;
    }
    return TrackModality.unknown;
  }
}

/// What a caller must send, in enough detail that a client can be built from it alone.
class TrackInputSpec {
  const TrackInputSpec({
    required this.modality,
    required this.description,
    this.contentTypes = const [],
    this.shape,
    this.units,
    this.samplingFrequencyHz,
    this.channelNames,
    this.maxBytes,
  });

  final TrackModality modality;

  /// One sentence a UI can show, e.g. "a 10-second 12-lead ECG at 100 Hz, in millivolts".
  final String description;

  /// Acceptable MIME types, e.g. `['image/jpeg']`.
  final List<String> contentTypes;

  /// Expected array shape for signal or tabular tracks, e.g. `[12, 1000]`.
  final List<int>? shape;

  final String? units;
  final double? samplingFrequencyHz;

  /// Channel order for multi-channel signals. A caller that sends leads in a
  /// different order gets a different answer, so the order is part of the published
  /// contract rather than something to infer.
  final List<String>? channelNames;

  final int? maxBytes;

  factory TrackInputSpec.fromJson(Map<String, dynamic> json) {
    return TrackInputSpec(
      modality: TrackModality.fromWire(json['modality'] as String?),
      description: (json['description'] as String?) ?? '',
      contentTypes: _stringListOrEmpty(json['content_types']),
      shape: _intList(json['shape']),
      units: json['units'] as String?,
      samplingFrequencyHz: _toDouble(json['sampling_frequency_hz']),
      channelNames: _stringListOrNull(json['channel_names']),
      maxBytes: _toInt(json['max_bytes']),
    );
  }
}

/// How a track's headline number was obtained. Deliberately blunt.
///
/// Every field exists because omitting it is how screening models get oversold. A
/// number without its partition, its patient count and its test-set status is not a
/// result; it is a number.
class TrackValidationSummary {
  const TrackValidationSummary({
    required this.dataset,
    required this.task,
    required this.primaryMetric,
    required this.partitionScored,
    required this.frozenTestEvaluated,
    required this.partitionReusedForSelection,
    this.primaryMetricValue,
    this.confidenceInterval,
    this.nRecords,
    this.nPatients,
    this.notes,
  });

  /// Human-readable provenance, e.g. "PTB-XL v1.0.3 (PhysioNet, CC BY 4.0)".
  final String dataset;

  /// What the reported metric measures, e.g. "NORM vs abnormal".
  final String task;

  final String primaryMetric;
  final double? primaryMetricValue;

  /// `[low, high]`, bootstrapped over **patients** rather than records.
  final List<double>? confidenceInterval;

  /// The partition the headline number came from, named exactly, e.g.
  /// "strat_fold 9 (validation)".
  final String partitionScored;

  final int? nRecords;
  final int? nPatients;

  /// `true` only if a partition untouched by every selection decision has been
  /// scored. When `false` the headline number is a *development* estimate: it comes
  /// from data model selection has already seen, so it is optimistically biased by
  /// an unknown amount, and the track must not be described as validated.
  final bool frozenTestEvaluated;

  /// `true` when [partitionScored] also informed model selection. Paired with
  /// [frozenTestEvaluated] this distinguishes "not yet tested" from "tested".
  final bool partitionReusedForSelection;

  final String? notes;

  /// Whether this track may be described as validated. Never true without a
  /// frozen-test evaluation, regardless of how good the development number looks.
  ///
  /// Mirrors `ValidationSummary.is_clinically_claimable` on the backend.
  bool get isClinicallyClaimable => frozenTestEvaluated;

  /// The headline metric formatted with the qualification it requires, or `null`
  /// when the track has not published a number.
  ///
  /// This is the only formatter offered for the metric, and it is why
  /// [primaryMetricValue] should not be interpolated into a UI string directly: an
  /// unvalidated number always reads as a development estimate here, so a screen
  /// cannot present one as validated by forgetting the distinction.
  String? get headline {
    final value = primaryMetricValue;
    if (value == null) return null;
    final metric = '$primaryMetric ${value.toStringAsFixed(4)}';
    final interval = confidenceInterval;
    final withInterval = (interval != null && interval.length == 2)
        ? '$metric [${interval[0].toStringAsFixed(4)}, ${interval[1].toStringAsFixed(4)}]'
        : metric;
    return frozenTestEvaluated
        ? '$withInterval on $partitionScored'
        : '$withInterval on $partitionScored — development estimate, '
              'not a validated result';
  }

  /// One sentence naming what the number is and is not, for a caption or tooltip.
  String get provenance {
    final cohort = [
      if (nRecords != null) '$nRecords records',
      if (nPatients != null) '$nPatients patients',
    ].join(' / ');
    final scope = cohort.isEmpty
        ? partitionScored
        : '$partitionScored, $cohort';
    if (frozenTestEvaluated) {
      return '$task on $dataset. Scored on $scope, a partition untouched by '
          'model selection.';
    }
    final reuse = partitionReusedForSelection
        ? ' This partition also informed model selection, so the number is '
              'optimistically biased by an unknown amount.'
        : '';
    return '$task on $dataset. Scored on $scope. No frozen test has been '
        'evaluated, so this is a development estimate, not a validated result.$reuse';
  }

  factory TrackValidationSummary.fromJson(Map<String, dynamic> json) {
    // Required, and not defaulted. A descriptor without this flag is one whose
    // claims cannot be judged, and defaulting it either way is a lie in one
    // direction or a hidden track in the other.
    final frozen = json['frozen_test_evaluated'];
    if (frozen is! bool) {
      throw const ServerFailure(
        'A track described itself without frozen_test_evaluated. Refusing to '
        'display a validation claim that cannot be checked.',
      );
    }
    return TrackValidationSummary(
      dataset: (json['dataset'] as String?) ?? 'unspecified',
      task: (json['task'] as String?) ?? 'unspecified',
      primaryMetric: (json['primary_metric'] as String?) ?? 'unspecified',
      primaryMetricValue: _toDouble(json['primary_metric_value']),
      confidenceInterval: _doubleListOrNull(json['confidence_interval']),
      partitionScored: (json['partition_scored'] as String?) ?? 'unspecified',
      nRecords: _toInt(json['n_records']),
      nPatients: _toInt(json['n_patients']),
      frozenTestEvaluated: frozen,
      // Defaults to the backend's own default (true), which is the cautious
      // reading: assume the partition was reused unless told otherwise.
      partitionReusedForSelection:
          json['partition_reused_for_selection'] as bool? ?? true,
      notes: json['notes'] as String?,
    );
  }
}

/// Everything a client needs to decide whether to offer a track, and how.
///
/// Mirrors the backend's `TrackDescriptor`.
class ScreeningTrack {
  const ScreeningTrack({
    required this.trackId,
    required this.displayName,
    required this.condition,
    required this.modality,
    required this.inputSpec,
    required this.validation,
    required this.modelVersion,
    required this.ready,
    required this.primaryModel,
    required this.usesQuantum,
    required this.disclaimer,
    this.unreadyReason,
    this.quantumRole,
  });

  final String trackId;
  final String displayName;

  /// The condition screened for, in patient-facing words.
  final String condition;

  final TrackModality modality;
  final TrackInputSpec inputSpec;
  final TrackValidationSummary validation;

  final String modelVersion;

  /// Whether the track can currently run. `GET /api/tracks` lists unready tracks
  /// too, so this is expected to be `false` sometimes rather than exceptional.
  final bool ready;

  /// Why [ready] is `false` — missing artifacts, a failed load, an untrained model.
  final String? unreadyReason;

  /// Which member headlines the verdict. Not assumed to be the quantum one.
  final String primaryModel;

  final bool usesQuantum;

  /// What the quantum component actually does here, in one phrase, or `null` if
  /// there is none. A platform that advertises "quantum" without saying where
  /// should not be believed, including by its own authors.
  final String? quantumRole;

  final String disclaimer;

  /// Whether a UI should let the user start a scan on this track.
  bool get isSelectable => ready;

  /// Why the track cannot be selected, in words a UI can show, or `null` when it can.
  String? get blockedReason {
    if (ready) return null;
    final reason = unreadyReason;
    if (reason != null && reason.isNotEmpty) return reason;
    return 'This screening track is not available in this build.';
  }

  /// A short badge for the quantum story, or `null` when there is nothing to claim.
  ///
  /// Returns the declared [quantumRole] rather than the word "quantum" alone, so a
  /// track cannot advertise the label without saying where it applies.
  String? get quantumBadge {
    if (!usesQuantum) return null;
    final role = quantumRole;
    if (role == null || role.isEmpty) return null;
    return role;
  }

  factory ScreeningTrack.fromJson(Map<String, dynamic> json) {
    final inputSpecJson = json['input_spec'];
    final validationJson = json['validation'];
    if (inputSpecJson is! Map<String, dynamic>) {
      throw const ServerFailure(
        'A track described itself without an input_spec.',
      );
    }
    if (validationJson is! Map<String, dynamic>) {
      throw const ServerFailure(
        'A track described itself without a validation summary. Refusing to '
        'offer a screening track whose provenance is unknown.',
      );
    }
    final trackId = json['track_id'] as String?;
    if (trackId == null || trackId.isEmpty) {
      throw const ServerFailure('A track described itself without a track_id.');
    }
    return ScreeningTrack(
      trackId: trackId,
      displayName: (json['display_name'] as String?) ?? trackId,
      condition: (json['condition'] as String?) ?? '',
      modality: TrackModality.fromWire(json['modality'] as String?),
      inputSpec: TrackInputSpec.fromJson(inputSpecJson),
      validation: TrackValidationSummary.fromJson(validationJson),
      modelVersion: (json['model_version'] as String?) ?? 'unknown',
      // Absent `ready` reads as not ready: a track that cannot state it is
      // loadable should not be offered as though it were.
      ready: json['ready'] as bool? ?? false,
      unreadyReason: json['unready_reason'] as String?,
      primaryModel: (json['primary_model'] as String?) ?? 'unknown',
      usesQuantum: json['uses_quantum'] as bool? ?? false,
      quantumRole: json['quantum_role'] as String?,
      disclaimer: (json['disclaimer'] as String?) ?? '',
    );
  }
}

/// The parsed result of `GET /api/tracks`, including what was thrown away.
///
/// A malformed descriptor does not fail the whole catalogue — one track the backend
/// describes badly should not hide the others. But it is not dropped silently
/// either: every refusal is recorded in [rejected] with its reason, because a
/// descriptor quietly disappearing from a screening platform is indistinguishable
/// from a schema regression, and that is precisely the failure that should be loud.
class TrackCatalogue {
  const TrackCatalogue({required this.tracks, required this.rejected});

  final List<ScreeningTrack> tracks;

  /// One human-readable reason per descriptor this client refused to parse. Empty
  /// on a healthy response.
  final List<String> rejected;

  bool get hasRejections => rejected.isNotEmpty;

  /// Tracks a UI may offer a scan on.
  List<ScreeningTrack> get selectable =>
      tracks.where((track) => track.isSelectable).toList(growable: false);

  /// Tracks listed but not runnable, so a UI can grey them out with a reason
  /// rather than pretending the platform is smaller than it is.
  List<ScreeningTrack> get unavailable =>
      tracks.where((track) => !track.isSelectable).toList(growable: false);

  /// Tracks whose headline number rests on a frozen-test evaluation. The
  /// complement is not "worse tracks" — it is tracks whose numbers are
  /// development estimates and must be captioned as such.
  List<ScreeningTrack> get clinicallyClaimable => tracks
      .where((track) => track.validation.isClinicallyClaimable)
      .toList(growable: false);

  /// Parses a `GET /api/tracks` array, skipping and recording what it cannot read.
  ///
  /// Throws a [ServerFailure] when a non-empty response yields no usable track at
  /// all: that is a changed contract, and returning an empty catalogue would render
  /// as "no screening tracks configured", which is a different and false statement.
  factory TrackCatalogue.fromJsonList(List<dynamic> items) {
    final tracks = <ScreeningTrack>[];
    final rejected = <String>[];

    for (var index = 0; index < items.length; index++) {
      final item = items[index];
      if (item is! Map<String, dynamic>) {
        rejected.add('Entry $index was not a JSON object.');
        continue;
      }
      try {
        tracks.add(ScreeningTrack.fromJson(item));
      } on Failure catch (failure) {
        final id = item['track_id'] ?? 'entry $index';
        rejected.add('$id: ${failure.message}');
      } catch (error) {
        final id = item['track_id'] ?? 'entry $index';
        rejected.add('$id: unreadable descriptor ($error).');
      }
    }

    if (tracks.isEmpty && items.isNotEmpty) {
      throw ServerFailure(
        'The backend listed ${items.length} screening track(s) but none could be '
        'read. This build may be older than the API. First reason: '
        '${rejected.isEmpty ? 'unknown' : rejected.first}',
      );
    }

    return TrackCatalogue(
      tracks: List.unmodifiable(tracks),
      rejected: List.unmodifiable(rejected),
    );
  }
}

// ---------------------------------------------------------------- JSON coercion
// FastAPI serialises an int-valued float as `1` rather than `1.0`, so a plain
// `as double` cast fails on values that are mathematically fine. These helpers
// widen numbers and drop malformed entries rather than throwing, since a mangled
// *optional* field should not hide an otherwise usable track. The required fields
// are checked explicitly in the factories above instead.

double? _toDouble(Object? value) {
  if (value is num) return value.toDouble();
  if (value is String) return double.tryParse(value);
  return null;
}

int? _toInt(Object? value) {
  if (value is int) return value;
  if (value is num) return value.toInt();
  if (value is String) return int.tryParse(value);
  return null;
}

List<String> _stringListOrEmpty(Object? value) {
  if (value is! List) return const [];
  return value.whereType<String>().toList(growable: false);
}

/// Absent, malformed and empty all collapse to `null`. For an optional list the
/// three are the same thing to a caller — "nothing to show" — and distinguishing
/// them would only push the check into every widget.
List<String>? _stringListOrNull(Object? value) {
  final parsed = _stringListOrEmpty(value);
  return parsed.isEmpty ? null : parsed;
}

List<int>? _intList(Object? value) {
  if (value is! List) return null;
  final parsed = value.map(_toInt).whereType<int>().toList(growable: false);
  return parsed.isEmpty ? null : parsed;
}

List<double>? _doubleListOrNull(Object? value) {
  if (value is! List) return null;
  final parsed = value
      .map(_toDouble)
      .whereType<double>()
      .toList(growable: false);
  return parsed.isEmpty ? null : parsed;
}
