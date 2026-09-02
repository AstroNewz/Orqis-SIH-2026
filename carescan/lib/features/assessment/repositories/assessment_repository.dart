import 'package:carescan/core/errors/result.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';

abstract class AssessmentRepository {
  /// Submits an image for assessment and returns the result.
  Future<Result<AssessmentResult>> submitAssessment(String imagePath);

  /// Retrieves the history of past assessments.
  Future<Result<List<HistoryEntry>>> getAssessmentHistory();
}
