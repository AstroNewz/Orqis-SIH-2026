import 'package:flutter_test/flutter_test.dart';
import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/data/datasources/assessment_remote_data_source.dart';
import 'package:carescan/data/repositories/api_assessment_repository.dart';
import 'package:carescan/features/assessment/models/assessment.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';

class FakeAssessmentRemoteDataSource implements AssessmentRemoteDataSource {
  AssessmentResult? submitResult;
  Exception? submitException;
  List<HistoryEntry>? historyResult;
  Exception? historyException;

  @override
  Future<AssessmentResult> submitAssessment(
    String imagePath, {
    String? patientId,
    bool isMock = false,
  }) async {
    if (submitException != null) {
      throw submitException!;
    }
    return submitResult ??
        const AssessmentResult(
          id: 'res-1',
          assessmentId: 'assess-1',
          riskLevel: 'LOW RISK',
          details: 'Normal',
        );
  }

  @override
  Future<List<HistoryEntry>> getAssessmentHistory(String patientId) async {
    if (historyException != null) {
      throw historyException!;
    }
    return historyResult ?? [];
  }
}

void main() {
  group('ApiAssessmentRepository', () {
    late FakeAssessmentRemoteDataSource fakeDataSource;
    late ApiAssessmentRepository repository;

    setUp(() {
      fakeDataSource = FakeAssessmentRemoteDataSource();
      repository = ApiAssessmentRepository(
        remoteDataSource: fakeDataSource,
        defaultPatientId: 'test-patient-id',
      );
    });

    test('submitAssessment returns Success when remote datasource succeeds', () async {
      const mockResult = AssessmentResult(
        id: 'api-res-1',
        assessmentId: 'assess-1',
        riskLevel: 'LOW RISK',
        details: 'Normal oral mucosa',
      );
      fakeDataSource.submitResult = mockResult;

      final result = await repository.submitAssessment('/mock/path.jpg');

      expect(result, isA<Success<AssessmentResult>>());
      result.fold(
        (failure) => fail('Expected Success, got Error'),
        (data) {
          expect(data.id, 'api-res-1');
          expect(data.riskLevel, 'LOW RISK');
        },
      );
    });

    test('submitAssessment returns Error with ServerFailure when server errors', () async {
      fakeDataSource.submitException = const ServerFailure('Server Error (500): Internal error');

      final result = await repository.submitAssessment('/mock/path.jpg');

      expect(result, isA<Error<AssessmentResult>>());
      result.fold(
        (failure) {
          expect(failure, isA<ServerFailure>());
          expect(failure.message, contains('Server Error'));
        },
        (data) => fail('Expected Error, got Success'),
      );
    });

    test('submitAssessment returns Error with NetworkFailure on timeout/socket error', () async {
      fakeDataSource.submitException = const NetworkFailure('Unable to connect to backend');

      final result = await repository.submitAssessment('/mock/path.jpg');

      expect(result, isA<Error<AssessmentResult>>());
      result.fold(
        (failure) {
          expect(failure, isA<NetworkFailure>());
          expect(failure.message, contains('Unable to connect'));
        },
        (data) => fail('Expected Error, got Success'),
      );
    });

    test('getAssessmentHistory returns Success with list of entries', () async {
      final entries = [
        HistoryEntry(
          assessment: Assessment(
            id: 'assess-10',
            imagePath: '/images/10.jpg',
            timestamp: DateTime(2026, 8, 30),
            type: 'Intra-oral Scan',
          ),
          result: const AssessmentResult(
            id: 'res-10',
            assessmentId: 'assess-10',
            riskLevel: 'LOW RISK',
            details: 'Normal',
          ),
        ),
      ];
      fakeDataSource.historyResult = entries;

      final result = await repository.getAssessmentHistory();

      expect(result, isA<Success<List<HistoryEntry>>>());
      result.fold(
        (failure) => fail('Expected Success, got Error'),
        (data) {
          expect(data.length, 1);
          expect(data.first.assessment.id, 'assess-10');
        },
      );
    });

    test('getAssessmentHistory returns Error on NetworkFailure', () async {
      fakeDataSource.historyException = const NetworkFailure('Connection timed out');

      final result = await repository.getAssessmentHistory();

      expect(result, isA<Error<List<HistoryEntry>>>());
      result.fold(
        (failure) => expect(failure, isA<NetworkFailure>()),
        (data) => fail('Expected Error, got Success'),
      );
    });
  });
}
