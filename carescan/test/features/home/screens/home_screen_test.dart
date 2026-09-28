import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/features/assessment/models/assessment.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';
import 'package:carescan/features/assessment/models/localization_result.dart';
import 'package:carescan/features/assessment/repositories/assessment_repository.dart';
import 'package:carescan/features/home/home_screen.dart';
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

  @override
  Future<Result<LocalizationResult>> localize(String imagePath) async =>
      const Success(LocalizationResult.none());
}

void main() {
  Widget buildTestWidget({required AssessmentRepository repository}) {
    return MaterialApp(
      theme: AppTheme.theme,
      home: HomeScreen(repository: repository),
    );
  }

  group('HomeScreen', () {
    testWidgets('renders hero card and recent assessments on success', (
      WidgetTester tester,
    ) async {
      final entries = [
        HistoryEntry(
          assessment: Assessment(
            id: 'asm-1',
            timestamp: DateTime(2023, 10, 12),
            imagePath: 'path1',
            type: 'Oral health screening',
          ),
          result: const AssessmentResult(
            id: 'res-1',
            assessmentId: 'asm-1',
            riskLevel: 'Low Risk',
            primaryRiskLevel: 'High Risk',
            details: 'All clear',
          ),
        ),
      ];

      final repo = _FakeAssessmentRepository(Success(entries));
      await tester.pumpWidget(buildTestWidget(repository: repo));
      await tester.pumpAndSettle();

      expect(find.text('Screen early.\nUnderstand sooner.'), findsOneWidget);
      expect(find.text('Start Screening'), findsOneWidget);
      expect(find.text('Oral health screening'), findsOneWidget);
      expect(find.text('HIGH RISK'), findsOneWidget);
    });

    testWidgets('renders empty state when no recent assessments', (
      WidgetTester tester,
    ) async {
      final repo = _FakeAssessmentRepository(const Success([]));
      await tester.pumpWidget(buildTestWidget(repository: repo));
      await tester.pumpAndSettle();

      expect(
        find.text('No screenings yet'),
        findsOneWidget,
      );
    });

    testWidgets('renders error state and retry on failure', (
      WidgetTester tester,
    ) async {
      final repo = _FakeAssessmentRepository(
        const Error(ServerFailure('Failed to load')),
      );
      await tester.pumpWidget(buildTestWidget(repository: repo));
      await tester.pumpAndSettle();

      expect(find.text('Screening history is unavailable'), findsOneWidget);
      expect(find.text('Try Again'), findsOneWidget);
    });
  });
}
