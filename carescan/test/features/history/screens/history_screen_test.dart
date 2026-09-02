import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/features/assessment/models/assessment.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';
import 'package:carescan/features/assessment/repositories/assessment_repository.dart';
import 'package:carescan/features/history/history_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

class _FakeAssessmentRepository implements AssessmentRepository {
  final Result<List<HistoryEntry>> historyResult;

  _FakeAssessmentRepository(this.historyResult);

  @override
  Future<Result<List<HistoryEntry>>> getAssessmentHistory() async =>
      historyResult;

  @override
  Future<Result<AssessmentResult>> submitAssessment(String imagePath) async =>
      const Success(
        AssessmentResult(
          id: '1',
          assessmentId: '1',
          riskLevel: 'Low Risk',
          details: 'Normal',
        ),
      );
}

void main() {
  Widget buildTestWidget({required AssessmentRepository repository}) {
    return MaterialApp(
      theme: AppTheme.theme,
      home: HistoryScreen(repository: repository),
    );
  }

  group('HistoryScreen', () {
    testWidgets('renders history items on success', (
      WidgetTester tester,
    ) async {
      final entries = [
        HistoryEntry(
          assessment: Assessment(
            id: 'asm-1',
            timestamp: DateTime(2023, 10, 24),
            imagePath: 'path1',
            type: 'Skin Scan • Full Body',
          ),
          result: const AssessmentResult(
            id: 'res-1',
            assessmentId: 'asm-1',
            riskLevel: 'Low Risk',
            details: 'All clear',
          ),
        ),
      ];

      final repo = _FakeAssessmentRepository(Success(entries));
      await tester.pumpWidget(buildTestWidget(repository: repo));
      await tester.pumpAndSettle();

      expect(find.text('Assessment History'), findsOneWidget);
      expect(find.text('Skin Scan • Full Body'), findsOneWidget);
      expect(find.text('LOW RISK'), findsOneWidget);
    });

    testWidgets('renders empty view when history is empty', (
      WidgetTester tester,
    ) async {
      final repo = _FakeAssessmentRepository(const Success([]));
      await tester.pumpWidget(buildTestWidget(repository: repo));
      await tester.pumpAndSettle();

      expect(find.text('No Assessments Yet'), findsOneWidget);
      expect(
        find.text(
          'Your past scan results will appear here once you complete your first assessment.',
        ),
        findsOneWidget,
      );
    });

    testWidgets('renders error view on failure and allows retry', (
      WidgetTester tester,
    ) async {
      final repo = _FakeAssessmentRepository(
        const Error(ServerFailure('Connection failed')),
      );
      await tester.pumpWidget(buildTestWidget(repository: repo));
      await tester.pumpAndSettle();

      expect(find.text('Could Not Load History'), findsOneWidget);
      expect(find.text('Retry'), findsOneWidget);
    });
  });
}
