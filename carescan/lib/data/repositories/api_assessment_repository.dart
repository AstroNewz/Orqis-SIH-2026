import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/data/datasources/assessment_remote_data_source.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';
import 'package:carescan/features/assessment/repositories/assessment_repository.dart';

/// Production implementation of [AssessmentRepository] connecting to the
/// FastAPI backend through [AssessmentRemoteDataSource].
class ApiAssessmentRepository implements AssessmentRepository {
  final AssessmentRemoteDataSource _remoteDataSource;
  final String? defaultPatientId;

  ApiAssessmentRepository({
    AssessmentRemoteDataSource? remoteDataSource,
    this.defaultPatientId,
  }) : _remoteDataSource =
            remoteDataSource ?? AssessmentRemoteDataSourceImpl();

  @override
  Future<Result<AssessmentResult>> submitAssessment(String imagePath) async {
    try {
      final result = await _remoteDataSource.submitAssessment(
        imagePath,
        patientId: defaultPatientId,
      );
      return Success(result);
    } on Failure catch (failure) {
      return Error(failure);
    } catch (e) {
      return Error(UnknownFailure('Unexpected error: $e'));
    }
  }

  @override
  Future<Result<List<HistoryEntry>>> getAssessmentHistory() async {
    try {
      final patientId = defaultPatientId ?? 'default_patient';
      final history = await _remoteDataSource.getAssessmentHistory(patientId);
      return Success(history);
    } on Failure catch (failure) {
      return Error(failure);
    } catch (e) {
      return Error(UnknownFailure('Unexpected error: $e'));
    }
  }
}
