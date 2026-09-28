import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/data/datasources/track_remote_data_source.dart';
import 'package:carescan/data/repositories/api_track_repository.dart';
import 'package:carescan/features/tracks/models/screening_track.dart';
import 'package:carescan/features/tracks/repositories/track_repository.dart';
import 'package:flutter_test/flutter_test.dart';

/// A thrown type the repository has no mapping for, so the fallback branch has
/// something to catch that is not already a [Failure].
class _Unmapped implements Exception {
  const _Unmapped();
  @override
  String toString() => 'the data source threw something unplanned';
}

class FakeTrackRemoteDataSource implements TrackRemoteDataSource {
  TrackCatalogue? catalogueResult;
  Object? catalogueError;

  ScreeningTrack? trackResult;
  Object? trackError;

  /// Every id `getTrack` was called with, so a test can prove the repository
  /// forwards rather than transforms.
  final List<String> requestedIds = [];

  @override
  Future<TrackCatalogue> getTracks() async {
    if (catalogueError != null) throw catalogueError!;
    return catalogueResult ??
        const TrackCatalogue(tracks: [], rejected: []);
  }

  @override
  Future<ScreeningTrack> getTrack(String trackId) async {
    requestedIds.add(trackId);
    if (trackError != null) throw trackError!;
    return trackResult ?? _track();
  }
}

ScreeningTrack _track({
  String trackId = 'ecg-ptbxl',
  bool ready = true,
  bool frozenTestEvaluated = false,
}) {
  return ScreeningTrack(
    trackId: trackId,
    displayName: 'ECG screening',
    condition: 'Abnormal resting ECG',
    modality: TrackModality.signal,
    inputSpec: const TrackInputSpec(
      modality: TrackModality.signal,
      description: 'a 10-second 12-lead ECG at 100 Hz, in millivolts',
    ),
    validation: TrackValidationSummary(
      dataset: 'PTB-XL v1.0.3 (PhysioNet, CC BY 4.0)',
      task: 'NORM vs abnormal',
      primaryMetric: 'ROC-AUC',
      primaryMetricValue: 0.946293,
      partitionScored: 'strat_fold 9 (validation)',
      frozenTestEvaluated: frozenTestEvaluated,
      partitionReusedForSelection: true,
    ),
    modelVersion: 'ecg-v1',
    ready: ready,
    primaryModel: 'fusion@cnn+gbm',
    usesQuantum: false,
    disclaimer: 'Screening aid only.',
  );
}

void main() {
  group('ApiTrackRepository', () {
    late FakeTrackRemoteDataSource fakeDataSource;
    late ApiTrackRepository repository;

    setUp(() {
      fakeDataSource = FakeTrackRemoteDataSource();
      repository = ApiTrackRepository(remoteDataSource: fakeDataSource);
    });

    group('getTracks', () {
      test('wraps a catalogue in Success', () async {
        fakeDataSource.catalogueResult = TrackCatalogue(
          tracks: [_track()],
          rejected: const [],
        );

        final result = await repository.getTracks();

        expect(result, isA<Success<TrackCatalogue>>());
        final catalogue = (result as Success<TrackCatalogue>).value;
        expect(catalogue.tracks, hasLength(1));
        expect(catalogue.tracks.single.trackId, 'ecg-ptbxl');
      });

      test('an empty catalogue is a Success, not an Error', () async {
        // "The backend has no tracks configured" is a real answer, and the
        // screen renders it as an empty state. Mapping it to Error would show a
        // connection problem the user does not have.
        fakeDataSource.catalogueResult = const TrackCatalogue(
          tracks: [],
          rejected: [],
        );

        final result = await repository.getTracks();

        expect(result, isA<Success<TrackCatalogue>>());
        expect((result as Success<TrackCatalogue>).value.tracks, isEmpty);
      });

      test('a partial catalogue stays a Success and keeps its rejections',
          () async {
        // Some descriptors were unreadable, others were fine. Collapsing this to
        // an Error would hide the readable tracks behind one malformed sibling.
        fakeDataSource.catalogueResult = TrackCatalogue(
          tracks: [_track()],
          rejected: const ['oral-lesion: unreadable descriptor.'],
        );

        final result = await repository.getTracks();

        expect(result, isA<Success<TrackCatalogue>>());
        final catalogue = (result as Success<TrackCatalogue>).value;
        expect(catalogue.tracks, hasLength(1));
        expect(catalogue.hasRejections, isTrue);
        expect(catalogue.rejected.single, contains('oral-lesion'));
      });

      test('passes a Failure through unchanged, preserving its type',
          () async {
        // The screen distinguishes "the backend is unreachable" from "the
        // backend sent something unreadable". Re-wrapping every failure as one
        // type would erase that and make both read the same to the user.
        fakeDataSource.catalogueError =
            const NetworkFailure('Unable to connect to backend server');

        final result = await repository.getTracks();

        expect(result, isA<Error<TrackCatalogue>>());
        final failure = (result as Error<TrackCatalogue>).failure;
        expect(failure, isA<NetworkFailure>());
        expect(failure.message, contains('Unable to connect'));
      });

      test('a ServerFailure stays a ServerFailure', () async {
        fakeDataSource.catalogueError = const ServerFailure(
          'A track described itself without frozen_test_evaluated.',
        );

        final result = await repository.getTracks();

        final failure = (result as Error<TrackCatalogue>).failure;
        expect(failure, isA<ServerFailure>());
        expect(failure.message, contains('frozen_test_evaluated'));
      });

      test('maps a non-Failure throw to UnknownFailure instead of escaping',
          () async {
        fakeDataSource.catalogueError = const _Unmapped();

        final result = await repository.getTracks();

        expect(result, isA<Error<TrackCatalogue>>());
        final failure = (result as Error<TrackCatalogue>).failure;
        expect(failure, isA<UnknownFailure>());
        expect(failure.message, contains('unplanned'));
      });

      test('an Error, not just an Exception, is still contained', () async {
        // The bare catch must be untyped. `on Exception` would let an Error
        // (a type cast, a null assertion) propagate out of the repository and
        // crash the widget that awaited it.
        fakeDataSource.catalogueError = ArgumentError('bad state');

        final result = await repository.getTracks();

        expect(result, isA<Error<TrackCatalogue>>());
        expect((result as Error<TrackCatalogue>).failure, isA<UnknownFailure>());
      });
    });

    group('getTrack', () {
      test('wraps a track in Success', () async {
        fakeDataSource.trackResult = _track(trackId: 'oral-lesion');

        final result = await repository.getTrack('oral-lesion');

        expect(result, isA<Success<ScreeningTrack>>());
        expect(
          (result as Success<ScreeningTrack>).value.trackId,
          'oral-lesion',
        );
      });

      test('forwards the id verbatim rather than normalising it', () async {
        await repository.getTrack('ECG-PTBXL');

        expect(fakeDataSource.requestedIds, ['ECG-PTBXL']);
      });

      test('an unknown track is a ValidationFailure, not a ServerFailure',
          () async {
        // Asking for a track that does not exist is this build being older than
        // the API, not the backend being down, and the two want different UI.
        fakeDataSource.trackError =
            const ValidationFailure('No screening track named "nope" exists.');

        final result = await repository.getTrack('nope');

        final failure = (result as Error<ScreeningTrack>).failure;
        expect(failure, isA<ValidationFailure>());
        expect(failure, isNot(isA<ServerFailure>()));
      });

      test('an empty id is refused without reaching the data source', () async {
        // The data source refuses locally; the repository must surface that as
        // an Error rather than letting the throw escape.
        repository = ApiTrackRepository(
          remoteDataSource: TrackRemoteDataSourceImpl(),
        );

        final result = await repository.getTrack('');

        expect(result, isA<Error<ScreeningTrack>>());
        expect(
          (result as Error<ScreeningTrack>).failure,
          isA<ValidationFailure>(),
        );
      });

      test('maps a non-Failure throw to UnknownFailure', () async {
        fakeDataSource.trackError = const _Unmapped();

        final result = await repository.getTrack('ecg-ptbxl');

        expect((result as Error<ScreeningTrack>).failure, isA<UnknownFailure>());
      });

      test('the validation summary survives the repository unaltered',
          () async {
        // The repository must not helpfully "complete" a development estimate.
        // A track whose frozen test has not been run arrives at the screen
        // still saying so.
        fakeDataSource.trackResult = _track(frozenTestEvaluated: false);

        final result = await repository.getTrack('ecg-ptbxl');

        final track = (result as Success<ScreeningTrack>).value;
        expect(track.validation.isClinicallyClaimable, isFalse);
        expect(track.validation.headline, contains('development estimate'));
      });
    });

    test('constructs its own data source when none is injected', () {
      // The default constructor is what production uses; a compile-time-only
      // dependency would go unexercised until runtime.
      expect(ApiTrackRepository(), isA<TrackRepository>());
    });
  });
}
