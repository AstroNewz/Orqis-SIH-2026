import 'package:carescan/core/errors/result.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';
import 'package:carescan/features/assessment/models/localization_result.dart';

abstract class AssessmentRepository {
  /// Submits an image for assessment and returns the result.
  Future<Result<AssessmentResult>> submitAssessment(String imagePath);

  /// Retrieves the history of past assessments.
  Future<Result<List<HistoryEntry>>> getAssessmentHistory();

  /// Locates the lesion ROI for a non-gating overlay. Never affects the verdict.
  Future<Result<LocalizationResult>> localize(String imagePath);
}
