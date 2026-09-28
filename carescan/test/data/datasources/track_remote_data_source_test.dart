import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:carescan/core/constants/api_constants.dart';
import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/data/datasources/track_remote_data_source.dart';
import 'package:carescan/features/tracks/models/screening_track.dart';

/// Cover for the platform track catalogue client.
///
/// The fixture below mirrors `backend/ml/track.py`'s `TrackDescriptor` field for
/// field. Those models carry no Pydantic aliases (only
/// `ConfigDict(protected_namespaces=())`, which exists because `model_version`
/// collides with Pydantic's reserved prefix), so the wire format is plain
/// snake_case and these keys are the real ones.
///
/// The tests are organised around the failure modes that would otherwise be silent:
/// a development number rendered as validated, a track vanishing from the catalogue
/// without anyone noticing, and a 404 for a mistyped id reading as a server outage.

class MockHttpHeaders implements HttpHeaders {
  @override
  ContentType? contentType;

  @override
  int contentLength = 0;

  @override
  void set(String name, Object value, {bool preserveHeaderCase = false}) {}

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class MockHttpClientResponse extends Stream<List<int>>
    implements HttpClientResponse {
  @override
  final int statusCode;
  final String _body;

  MockHttpClientResponse(this.statusCode, this._body);

  @override
  StreamSubscription<List<int>> listen(
    void Function(List<int> event)? onData, {
    Function? onError,
    void Function()? onDone,
    bool? cancelOnError,
  }) {
    return Stream.value(utf8.encode(_body)).listen(
      onData,
      onError: onError,
      onDone: onDone,
      cancelOnError: cancelOnError,
    );
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class MockHttpClientRequest implements HttpClientRequest {
  final int _statusCode;
  final String _responseBody;

  @override
  final HttpHeaders headers = MockHttpHeaders();

  MockHttpClientRequest(this._statusCode, this._responseBody);

  @override
  void write(Object? obj) {}

  @override
  void add(List<int> data) {}

  @override
  Future<HttpClientResponse> close() async =>
      MockHttpClientResponse(_statusCode, _responseBody);

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class MockHttpClient implements HttpClient {
  int responseStatusCode = 200;
  String responseBody = '[]';
  Exception? throwException;

  /// Every URI the client asked for, so a test can assert the path actually built
  /// from `ApiConstants` -- and assert that no request was made at all when the
  /// data source should have refused locally.
  final List<Uri> requestedUris = [];

  @override
  Duration? connectionTimeout;

  @override
  Future<HttpClientRequest> getUrl(Uri url) async {
    requestedUris.add(url);
    if (throwException != null) throw throwException!;
    return MockHttpClientRequest(responseStatusCode, responseBody);
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

/// A fully-populated descriptor in the backend's own shape.
Map<String, dynamic> trackFixture({
  Map<String, dynamic> overrides = const {},
  Map<String, dynamic> validationOverrides = const {},
}) {
  final validation = <String, dynamic>{
    'dataset': 'PTB-XL v1.0.3 (PhysioNet, CC BY 4.0)',
    'task': 'NORM vs abnormal',
    'primary_metric': 'ROC-AUC',
    'primary_metric_value': 0.946293,
    'confidence_interval': [0.9364, 0.9550],
    'partition_scored': 'strat_fold 9 (validation)',
    'n_records': 2146,
    'n_patients': 1917,
    'frozen_test_evaluated': false,
    'partition_reused_for_selection': true,
    'notes': 'Development estimate.',
    ...validationOverrides,
  };

  return <String, dynamic>{
    'track_id': 'ecg-ptbxl',
    'display_name': '12-lead ECG screening',
    'condition': 'Abnormal ECG',
    'modality': 'signal',
    'input_spec': {
      'modality': 'signal',
      'content_types': ['application/json'],
      'description': 'a 10-second 12-lead ECG at 100 Hz, in millivolts',
      'shape': [12, 1000],
      'units': 'mV',
      'sampling_frequency_hz': 100.0,
      'channel_names': ['I', 'II', 'III'],
      'max_bytes': 1048576,
    },
    'validation': validation,
    'model_version': 'v1-e4-classical-1',
    'ready': true,
    'unready_reason': null,
    'primary_model': 'fusion@cnn+gbm',
    'uses_quantum': false,
    'quantum_role': null,
    'disclaimer': 'Research use only. Not a diagnosis.',
    ...overrides,
  };
}

void main() {
  group('ScreeningTrack.fromJson', () {
    test('parses every field of a full descriptor', () {
      final track = ScreeningTrack.fromJson(trackFixture());

      expect(track.trackId, 'ecg-ptbxl');
      expect(track.displayName, '12-lead ECG screening');
      expect(track.condition, 'Abnormal ECG');
      expect(track.modality, TrackModality.signal);
      expect(track.modelVersion, 'v1-e4-classical-1');
      expect(track.ready, isTrue);
      expect(track.unreadyReason, isNull);
      expect(track.primaryModel, 'fusion@cnn+gbm');
      expect(track.usesQuantum, isFalse);
      expect(track.disclaimer, 'Research use only. Not a diagnosis.');

      expect(track.inputSpec.modality, TrackModality.signal);
      expect(track.inputSpec.contentTypes, ['application/json']);
      expect(track.inputSpec.shape, [12, 1000]);
      expect(track.inputSpec.units, 'mV');
      expect(track.inputSpec.samplingFrequencyHz, 100.0);
      expect(track.inputSpec.channelNames, ['I', 'II', 'III']);
      expect(track.inputSpec.maxBytes, 1048576);

      expect(track.validation.dataset, contains('PTB-XL'));
      expect(track.validation.primaryMetricValue, closeTo(0.946293, 1e-9));
      expect(track.validation.confidenceInterval, [0.9364, 0.9550]);
      expect(track.validation.nRecords, 2146);
      expect(track.validation.nPatients, 1917);
      expect(track.validation.frozenTestEvaluated, isFalse);
    });

    test('an unknown modality does not crash the parse', () {
      // A backend that adds a modality must not take out an older client: it
      // could otherwise hide every track the client *does* understand.
      final track = ScreeningTrack.fromJson(
        trackFixture(overrides: {'modality': 'genomic'}),
      );
      expect(track.modality, TrackModality.unknown);
    });

    test('a missing track_id is refused', () {
      expect(
        () => ScreeningTrack.fromJson(trackFixture(overrides: {'track_id': null})),
        throwsA(isA<ServerFailure>()),
      );
    });

    test('a missing validation summary is refused', () {
      expect(
        () => ScreeningTrack.fromJson(trackFixture(overrides: {'validation': null})),
        throwsA(isA<ServerFailure>()),
      );
    });

    test('a missing input_spec is refused', () {
      expect(
        () => ScreeningTrack.fromJson(trackFixture(overrides: {'input_spec': null})),
        throwsA(isA<ServerFailure>()),
      );
    });

    test('an absent ready flag reads as NOT ready', () {
      // The conservative direction. A track that cannot state it is loadable must
      // not be offered as though it were.
      final json = trackFixture()..remove('ready');
      expect(ScreeningTrack.fromJson(json).ready, isFalse);
    });

    test('display_name falls back to the track id rather than being blank', () {
      final json = trackFixture()..remove('display_name');
      expect(ScreeningTrack.fromJson(json).displayName, 'ecg-ptbxl');
    });
  });

  group('the frozen-test flag is required, never defaulted', () {
    // This is the load-bearing guard in the whole file. Defaulting this flag
    // either way is a lie in one direction (a development estimate shown as
    // validated) or a hidden track in the other.
    test('an absent frozen_test_evaluated is refused', () {
      final json = trackFixture();
      (json['validation'] as Map<String, dynamic>).remove('frozen_test_evaluated');
      expect(
        () => ScreeningTrack.fromJson(json),
        throwsA(
          isA<ServerFailure>().having(
            (f) => f.message,
            'message',
            contains('frozen_test_evaluated'),
          ),
        ),
      );
    });

    test('a non-boolean frozen_test_evaluated is refused', () {
      // A truthy string must not sneak past as `true`.
      for (final bad in <Object?>['true', 1, 'false', 0, null]) {
        expect(
          () => ScreeningTrack.fromJson(
            trackFixture(validationOverrides: {'frozen_test_evaluated': bad}),
          ),
          throwsA(isA<ServerFailure>()),
          reason: 'frozen_test_evaluated: $bad should be refused',
        );
      }
    });

    test('partition_reused_for_selection defaults to the cautious reading', () {
      final json = trackFixture();
      (json['validation'] as Map<String, dynamic>)
          .remove('partition_reused_for_selection');
      // Assume the partition WAS reused unless told otherwise; matches the
      // backend's own default of True.
      expect(
        ScreeningTrack.fromJson(json).validation.partitionReusedForSelection,
        isTrue,
      );
    });
  });

  group('TrackValidationSummary honesty accessors', () {
    TrackValidationSummary summaryWith(Map<String, dynamic> overrides) =>
        ScreeningTrack.fromJson(trackFixture(validationOverrides: overrides))
            .validation;

    test('isClinicallyClaimable tracks the frozen-test flag only', () {
      expect(
        summaryWith({'frozen_test_evaluated': false}).isClinicallyClaimable,
        isFalse,
      );
      expect(
        summaryWith({'frozen_test_evaluated': true}).isClinicallyClaimable,
        isTrue,
      );
    });

    test('a high number without a frozen test is still not claimable', () {
      // The number here is the highest in the phase. It changes nothing.
      final summary = summaryWith({
        'primary_metric_value': 0.946329,
        'frozen_test_evaluated': false,
      });
      expect(summary.isClinicallyClaimable, isFalse);
      expect(summary.headline, contains('development estimate'));
    });

    test('an unvalidated headline always says so', () {
      final headline = summaryWith({'frozen_test_evaluated': false}).headline!;
      expect(headline, contains('ROC-AUC 0.9463'));
      expect(headline, contains('[0.9364, 0.9550]'));
      expect(headline, contains('strat_fold 9 (validation)'));
      expect(headline, contains('development estimate'));
      expect(headline, contains('not a validated result'));
    });

    test('a frozen-test headline does not carry the caveat', () {
      final headline = summaryWith({
        'frozen_test_evaluated': true,
        'partition_scored': 'strat_fold 10 (frozen test)',
      }).headline!;
      expect(headline, contains('strat_fold 10 (frozen test)'));
      expect(headline, isNot(contains('development estimate')));
    });

    test('headline is null when no number was published', () {
      expect(summaryWith({'primary_metric_value': null}).headline, isNull);
    });

    test('headline omits the interval when none was published', () {
      final headline = summaryWith({'confidence_interval': null}).headline!;
      expect(headline, contains('ROC-AUC 0.9463'));
      expect(headline, isNot(contains('[')));
    });

    test('provenance names the optimistic bias when the partition was reused', () {
      final provenance = summaryWith({
        'frozen_test_evaluated': false,
        'partition_reused_for_selection': true,
      }).provenance;
      expect(provenance, contains('2146 records / 1917 patients'));
      expect(provenance, contains('optimistically biased'));
      expect(provenance, contains('not a validated result'));
    });

    test('provenance of a frozen-test number says the partition was untouched', () {
      final provenance = summaryWith({
        'frozen_test_evaluated': true,
        'partition_reused_for_selection': false,
      }).provenance;
      expect(provenance, contains('untouched by model selection'));
      expect(provenance, isNot(contains('development estimate')));
    });

    test('an int-valued metric from JSON widens to double', () {
      // FastAPI serialises 1.0 as `1`; a plain `as double` cast would throw on a
      // value that is mathematically fine.
      final summary = summaryWith({'primary_metric_value': 1, 'confidence_interval': [0, 1]});
      expect(summary.primaryMetricValue, 1.0);
      expect(summary.confidenceInterval, [0.0, 1.0]);
      expect(summary.headline, contains('1.0000'));
    });
  });

  group('ScreeningTrack availability and quantum labelling', () {
    test('a ready track is selectable with no blocked reason', () {
      final track = ScreeningTrack.fromJson(trackFixture());
      expect(track.isSelectable, isTrue);
      expect(track.blockedReason, isNull);
    });

    test('an unready track reports the backend reason verbatim', () {
      final track = ScreeningTrack.fromJson(trackFixture(overrides: {
        'ready': false,
        'unready_reason': 'Model artifacts are missing.',
      }));
      expect(track.isSelectable, isFalse);
      expect(track.blockedReason, 'Model artifacts are missing.');
    });

    test('an unready track with no stated reason still explains itself', () {
      final track = ScreeningTrack.fromJson(trackFixture(overrides: {
        'ready': false,
        'unready_reason': null,
      }));
      expect(track.blockedReason, isNotNull);
      expect(track.blockedReason, contains('not available'));
    });

    test('quantum cannot be advertised without saying where it applies', () {
      // uses_quantum true but no declared role: no badge. A platform that claims
      // "quantum" without naming the component should not be believed.
      for (final role in <Object?>[null, '']) {
        final track = ScreeningTrack.fromJson(trackFixture(overrides: {
          'uses_quantum': true,
          'quantum_role': role,
        }));
        expect(track.usesQuantum, isTrue);
        expect(track.quantumBadge, isNull, reason: 'role: $role');
      }
    });

    test('a declared quantum role is surfaced as the badge', () {
      final track = ScreeningTrack.fromJson(trackFixture(overrides: {
        'uses_quantum': true,
        'quantum_role': 'ZZ feature map over 8 PCA components',
      }));
      expect(track.quantumBadge, 'ZZ feature map over 8 PCA components');
    });

    test('a role on a non-quantum track produces no badge', () {
      final track = ScreeningTrack.fromJson(trackFixture(overrides: {
        'uses_quantum': false,
        'quantum_role': 'leftover text',
      }));
      expect(track.quantumBadge, isNull);
    });
  });

  group('TrackCatalogue.fromJsonList', () {
    test('an empty catalogue parses without throwing', () {
      final catalogue = TrackCatalogue.fromJsonList([]);
      expect(catalogue.tracks, isEmpty);
      expect(catalogue.hasRejections, isFalse);
    });

    test('one bad descriptor does not lose the good ones', () {
      final bad = trackFixture(overrides: {'track_id': 'broken-track'});
      (bad['validation'] as Map<String, dynamic>).remove('frozen_test_evaluated');

      final catalogue = TrackCatalogue.fromJsonList([trackFixture(), bad]);

      expect(catalogue.tracks, hasLength(1));
      expect(catalogue.tracks.single.trackId, 'ecg-ptbxl');
      // ...but the loss is recorded, naming the track. A descriptor silently
      // vanishing is indistinguishable from a schema regression.
      expect(catalogue.hasRejections, isTrue);
      expect(catalogue.rejected.single, contains('broken-track'));
      expect(catalogue.rejected.single, contains('frozen_test_evaluated'));
    });

    test('a non-object entry is rejected by index', () {
      final catalogue = TrackCatalogue.fromJsonList([trackFixture(), 'nonsense']);
      expect(catalogue.tracks, hasLength(1));
      expect(catalogue.rejected.single, contains('Entry 1'));
    });

    test('a non-empty list that yields nothing usable is a contract break', () {
      // Returning an empty catalogue here would render as "no screening tracks
      // configured", which is a different and false statement.
      final bad = trackFixture();
      (bad['validation'] as Map<String, dynamic>).remove('frozen_test_evaluated');
      expect(
        () => TrackCatalogue.fromJsonList([bad]),
        throwsA(
          isA<ServerFailure>()
              .having((f) => f.message, 'message', contains('none could be read')),
        ),
      );
    });

    test('partitions tracks by availability and by claimability', () {
      final catalogue = TrackCatalogue.fromJsonList([
        trackFixture(overrides: {'track_id': 'ready-unvalidated', 'ready': true}),
        trackFixture(
          overrides: {'track_id': 'unready', 'ready': false},
        ),
        trackFixture(
          overrides: {'track_id': 'validated', 'ready': true},
          validationOverrides: {'frozen_test_evaluated': true},
        ),
      ]);

      expect(
        catalogue.selectable.map((t) => t.trackId),
        ['ready-unvalidated', 'validated'],
      );
      expect(catalogue.unavailable.map((t) => t.trackId), ['unready']);
      expect(catalogue.clinicallyClaimable.map((t) => t.trackId), ['validated']);
    });

    test('the parsed lists are unmodifiable', () {
      final catalogue = TrackCatalogue.fromJsonList([trackFixture()]);
      expect(
        () => catalogue.tracks.add(catalogue.tracks.first),
        throwsUnsupportedError,
      );
    });
  });

  group('TrackRemoteDataSourceImpl', () {
    late MockHttpClient mockHttpClient;
    late TrackRemoteDataSourceImpl dataSource;

    setUp(() {
      mockHttpClient = MockHttpClient();
      dataSource = TrackRemoteDataSourceImpl(httpClient: mockHttpClient);
      ApiConstants.setBaseUrl('http://localhost:8000');
    });

    tearDown(ApiConstants.resetBaseUrl);

    group('getTracks', () {
      test('requests /api/tracks and parses the catalogue', () async {
        mockHttpClient.responseBody = jsonEncode([
          trackFixture(),
          trackFixture(overrides: {'track_id': 'oral-lesion', 'modality': 'image'}),
        ]);

        final catalogue = await dataSource.getTracks();

        expect(mockHttpClient.requestedUris.single.path, '/api/tracks');
        expect(catalogue.tracks, hasLength(2));
        expect(catalogue.tracks.last.modality, TrackModality.image);
        expect(catalogue.hasRejections, isFalse);
      });

      test('a JSON object instead of a list is a ServerFailure', () async {
        mockHttpClient.responseBody = jsonEncode({'tracks': []});
        expect(
          () => dataSource.getTracks(),
          throwsA(isA<ServerFailure>()),
        );
      });

      test('a non-2xx response surfaces the backend detail', () async {
        mockHttpClient.responseStatusCode = 500;
        mockHttpClient.responseBody = jsonEncode({'detail': 'registry exploded'});
        expect(
          () => dataSource.getTracks(),
          throwsA(
            isA<ServerFailure>()
                .having((f) => f.message, 'message', contains('registry exploded')),
          ),
        );
      });

      test('unparsable JSON is a ServerFailure, not a crash', () async {
        mockHttpClient.responseBody = 'not json at all';
        expect(
          () => dataSource.getTracks(),
          throwsA(
            isA<ServerFailure>()
                .having((f) => f.message, 'message', contains('Malformed')),
          ),
        );
      });

      test('a socket error becomes a NetworkFailure', () async {
        mockHttpClient.throwException = const SocketException('no route to host');
        expect(
          () => dataSource.getTracks(),
          throwsA(isA<NetworkFailure>()),
        );
      });

      test('a timeout becomes a NetworkFailure', () async {
        mockHttpClient.throwException = TimeoutException('slow');
        expect(
          () => dataSource.getTracks(),
          throwsA(isA<NetworkFailure>()),
        );
      });
    });

    group('getTrack', () {
      test('requests /api/tracks/{id} and parses the descriptor', () async {
        mockHttpClient.responseBody = jsonEncode(trackFixture());

        final track = await dataSource.getTrack('ecg-ptbxl');

        expect(mockHttpClient.requestedUris.single.path, '/api/tracks/ecg-ptbxl');
        expect(track.trackId, 'ecg-ptbxl');
      });

      test('an id needing escaping is encoded into the path', () async {
        mockHttpClient.responseBody = jsonEncode(trackFixture());
        await dataSource.getTrack('odd id/../etc');
        // Uri.encodeComponent escapes the separators, so the segment cannot climb
        // out of /api/tracks/.
        expect(mockHttpClient.requestedUris.single.path, startsWith('/api/tracks/'));
        expect(mockHttpClient.requestedUris.single.pathSegments, hasLength(3));
      });

      test('404 is a ValidationFailure, not a ServerFailure', () async {
        // The caller named a track that does not exist. Reporting that as a
        // ServerFailure would make a mistyped id read as a backend outage.
        mockHttpClient.responseStatusCode = 404;
        mockHttpClient.responseBody = jsonEncode({'detail': 'Unknown track'});
        expect(
          () => dataSource.getTrack('no-such-track'),
          throwsA(
            isA<ValidationFailure>()
                .having((f) => f.message, 'message', contains('no-such-track')),
          ),
        );
      });

      test('503 from an unloadable track stays a ServerFailure', () async {
        mockHttpClient.responseStatusCode = 503;
        mockHttpClient.responseBody = jsonEncode({'detail': 'artifacts missing'});
        expect(
          () => dataSource.getTrack('ecg-ptbxl'),
          throwsA(
            isA<ServerFailure>()
                .having((f) => f.message, 'message', contains('artifacts missing')),
          ),
        );
      });

      test('an empty id is refused locally without a request', () async {
        await expectLater(
          dataSource.getTrack(''),
          throwsA(isA<ValidationFailure>()),
        );
        expect(mockHttpClient.requestedUris, isEmpty);
      });

      test('a JSON list instead of an object is a ServerFailure', () async {
        mockHttpClient.responseBody = jsonEncode([trackFixture()]);
        expect(
          () => dataSource.getTrack('ecg-ptbxl'),
          throwsA(isA<ServerFailure>()),
        );
      });

      test('a descriptor missing the frozen flag is refused end to end', () async {
        // The guard holds through the transport layer, not just in the model test.
        final bad = trackFixture();
        (bad['validation'] as Map<String, dynamic>).remove('frozen_test_evaluated');
        mockHttpClient.responseBody = jsonEncode(bad);
        expect(
          () => dataSource.getTrack('ecg-ptbxl'),
          throwsA(
            isA<ServerFailure>().having(
              (f) => f.message,
              'message',
              contains('frozen_test_evaluated'),
            ),
          ),
        );
      });
    });
  });
}
