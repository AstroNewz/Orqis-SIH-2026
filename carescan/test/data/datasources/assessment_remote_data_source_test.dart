import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:carescan/core/constants/api_constants.dart';
import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/data/datasources/assessment_remote_data_source.dart';
import '../../support/image_fixtures.dart';

class MockHttpHeaders implements HttpHeaders {
  @override
  ContentType? contentType;

  @override
  int contentLength = 0;

  // The multipart upload leg sets these; a mock header bag just accepts them.
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

  // The upload leg streams the multipart body through add(); the mock ignores it.
  @override
  void add(List<int> data) {}

  @override
  Future<HttpClientResponse> close() async {
    return MockHttpClientResponse(_statusCode, _responseBody);
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class MockHttpClient implements HttpClient {
  int responseStatusCode = 200;
  String responseBody = '{}';
  Exception? throwException;

  // The production flow uploads the capture first, then analyzes the server-side
  // copy. The upload leg is answered here so tests can focus on the analyze
  // response via [responseStatusCode] / [responseBody].
  int uploadStatusCode = 201;
  String uploadResponseBody = '{"image_path": "/server/uploads/scan.jpg"}';

  @override
  Duration? connectionTimeout;

  @override
  Future<HttpClientRequest> postUrl(Uri url) async {
    if (throwException != null) throw throwException!;
    if (url.path.contains('/upload')) {
      return MockHttpClientRequest(uploadStatusCode, uploadResponseBody);
    }
    return MockHttpClientRequest(responseStatusCode, responseBody);
  }

  @override
  Future<HttpClientRequest> getUrl(Uri url) async {
    if (throwException != null) throw throwException!;
    return MockHttpClientRequest(responseStatusCode, responseBody);
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

void main() {
  group('AssessmentRemoteDataSourceImpl', () {
    late MockHttpClient mockHttpClient;
    late AssessmentRemoteDataSourceImpl dataSource;
    late Directory tempDir;
    late String tempImagePath;

    setUp(() {
      mockHttpClient = MockHttpClient();
      dataSource = AssessmentRemoteDataSourceImpl(httpClient: mockHttpClient);
      ApiConstants.setBaseUrl('http://localhost:8000');
      // A real file on disk: submitAssessment uploads the capture before it
      // analyzes, and the upload reads these bytes.
      tempDir = Directory.systemTemp.createTempSync('carescan_ds_test');
      tempImagePath = '${tempDir.path}/scan.jpg';
      File(tempImagePath).writeAsBytesSync(qualityFixture());
    });

    tearDown(() {
      ApiConstants.resetBaseUrl();
      if (tempDir.existsSync()) tempDir.deleteSync(recursive: true);
    });

    group('submitAssessment', () {
      test('successfully parses POST /api/screening/analyze response', () async {
        final mockJsonResponse = jsonEncode({
          'id': 'res-101',
          'assessmentId': 'assess-101',
          'riskLevel': 'LOW RISK',
          'details': 'No abnormalities detected.',
          'finalProbability': 0.12,
          'quantumProbability': 0.10,
          'classicalProbability': 0.14,
        });

        mockHttpClient.responseStatusCode = 200;
        mockHttpClient.responseBody = mockJsonResponse;

        final result = await dataSource.submitAssessment(tempImagePath);

        expect(result.id, 'res-101');
        expect(result.assessmentId, 'assess-101');
        expect(result.riskLevel, 'LOW RISK');
        expect(result.details, 'No abnormalities detected.');
      });

      test('throws ServerFailure on non-2xx status code', () async {
        final errorResponse = jsonEncode({
          'detail': 'Invalid image format',
        });

        mockHttpClient.responseStatusCode = 400;
        mockHttpClient.responseBody = errorResponse;

        expect(
          () => dataSource.submitAssessment(tempImagePath),
          throwsA(
            isA<ServerFailure>().having(
              (f) => f.message,
              'message',
              contains('Invalid image format'),
            ),
          ),
        );
      });

      test('throws NetworkFailure on SocketException', () async {
        mockHttpClient.throwException = const SocketException('Connection refused');

        expect(
          () => dataSource.submitAssessment(tempImagePath),
          throwsA(
            isA<NetworkFailure>().having(
              (f) => f.message,
              'message',
              contains('Unable to connect'),
            ),
          ),
        );
      });

      test('throws NetworkFailure on TimeoutException', () async {
        mockHttpClient.throwException = TimeoutException('Timeout');

        expect(
          () => dataSource.submitAssessment(tempImagePath),
          throwsA(
            isA<NetworkFailure>().having(
              (f) => f.message,
              'message',
              contains('timed out'),
            ),
          ),
        );
      });
    });

    group('getAssessmentHistory', () {
      test('successfully parses GET /api/patients/{id}/history response', () async {
        final mockHistoryResponse = jsonEncode([
          {
            'assessment': {
              'id': 'assess-201',
              'imagePath': '/images/201.jpg',
              'timestamp': '2026-08-30T10:00:00Z',
              'type': 'Intra-oral Scan',
            },
            'result': {
              'id': 'res-201',
              'assessmentId': 'assess-201',
              'riskLevel': 'HIGH RISK',
              'details': 'Follow-up clinical biopsy recommended.',
            },
          },
        ]);

        mockHttpClient.responseStatusCode = 200;
        mockHttpClient.responseBody = mockHistoryResponse;

        final history = await dataSource.getAssessmentHistory('patient-123');

        expect(history.length, 1);
        expect(history.first.assessment.id, 'assess-201');
        expect(history.first.assessment.type, 'Intra-oral Scan');
        expect(history.first.result.riskLevel, 'HIGH RISK');
        expect(history.first.result.details, contains('biopsy'));
      });

      test('throws ServerFailure on 500 internal server error', () async {
        mockHttpClient.responseStatusCode = 500;
        mockHttpClient.responseBody = jsonEncode({'detail': 'Database error'});

        expect(
          () => dataSource.getAssessmentHistory('patient-123'),
          throwsA(
            isA<ServerFailure>().having(
              (f) => f.message,
              'message',
              contains('Database error'),
            ),
          ),
        );
      });

      test('throws NetworkFailure on SocketException during history fetch', () async {
        mockHttpClient.throwException = const SocketException('Host unreachable');

        expect(
          () => dataSource.getAssessmentHistory('patient-123'),
          throwsA(isA<NetworkFailure>()),
        );
      });
    });
  });
}
