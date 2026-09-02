import 'package:carescan/core/errors/result.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';
import 'package:carescan/features/assessment/models/assessment.dart';
import 'package:carescan/features/assessment/repositories/assessment_repository.dart';

class MockAssessmentRepository implements AssessmentRepository {
  @override
  Future<Result<AssessmentResult>> submitAssessment(String imagePath) async {
    // Simulate network delay
    await Future.delayed(const Duration(seconds: 2));

    return const Success(
      AssessmentResult(
        id: 'test-result-id',
        assessmentId: 'test-assessment-id',
        riskLevel: 'MOCK LOW RISK',
        details:
            'TEST DATA: Analysis complete. No significant anomalies detected.',
      ),
    );
  }

  @override
  Future<Result<List<HistoryEntry>>> getAssessmentHistory() async {
    await Future.delayed(const Duration(seconds: 1));

    return Success([
      HistoryEntry(
        assessment: Assessment(
          id: 'test-1',
          imagePath: '/mock/path/1',
          timestamp: DateTime.now().subtract(const Duration(days: 2)),
          type: 'TEST DATA: Skin Scan',
        ),
        result: const AssessmentResult(
          id: 'test-res-1',
          assessmentId: 'test-1',
          riskLevel: 'MOCK LOW RISK',
          details: 'TEST DATA: Normal appearance.',
        ),
      ),
      HistoryEntry(
        assessment: Assessment(
          id: 'test-2',
          imagePath: '/mock/path/2',
          timestamp: DateTime.now().subtract(const Duration(days: 6)),
          type: 'TEST DATA: Mole Scan',
        ),
        result: const AssessmentResult(
          id: 'test-res-2',
          assessmentId: 'test-2',
          riskLevel: 'MOCK LOW RISK',
          details: 'TEST DATA: Normal appearance.',
        ),
      ),
    ]);
  }
}
