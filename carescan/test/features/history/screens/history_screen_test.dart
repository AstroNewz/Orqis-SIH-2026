import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/features/assessment/models/assessment.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';
import 'package:carescan/features/assessment/models/localization_result.dart';
import 'package:carescan/features/assessment/repositories/assessment_repository.dart';
import 'package:carescan/features/history/history_screen.dart';
import 'package:carescan/l10n/l10n.dart';
import 'package:carescan/shared/widgets/screening_record.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

class _FakeAssessmentRepository implements AssessmentRepository {
  final Result<List<HistoryEntry>> historyResult;

  /// How many times the screen asked for the list. Retry has to actually
  /// re-query; a button that only clears the error would look identical.
  int historyCalls = 0;

  _FakeAssessmentRepository(this.historyResult);

  @override
  Future<Result<List<HistoryEntry>>> getAssessmentHistory() async {
    historyCalls++;
    return historyResult;
  }

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

HistoryEntry _entry({
  required String id,
  required DateTime at,
  String type = 'Oral Scan',
  String riskLevel = 'Low Risk',
}) => HistoryEntry(
  assessment: Assessment(
    id: id,
    timestamp: at,
    imagePath: 'path-$id',
    type: type,
  ),
  result: AssessmentResult(
    id: 'res-$id',
    assessmentId: id,
    riskLevel: riskLevel,
    details: 'All clear',
  ),
);

void main() {
  Widget buildTestWidget({
    required AssessmentRepository repository,
    Locale locale = const Locale('en'),
  }) {
    return MaterialApp(
      locale: locale,
      theme: AppTheme.theme,
      localizationsDelegates: AppLocalizations.localizationsDelegates,
      supportedLocales: AppLocalizations.supportedLocales,
      home: HistoryScreen(repository: repository),
    );
  }

  group('HistoryScreen', () {
    testWidgets('renders history items on success', (
      WidgetTester tester,
    ) async {
      final entries = [
        _entry(
          id: 'asm-1',
          at: DateTime(2023, 10, 24),
          type: 'Skin Scan • Full Body',
        ),
      ];

      final repo = _FakeAssessmentRepository(Success(entries));
      await tester.pumpWidget(buildTestWidget(repository: repo));
      await tester.pumpAndSettle();

      expect(find.text('History'), findsOneWidget);
      expect(find.text('Skin Scan • Full Body'), findsOneWidget);
      expect(find.text('LOW RISK'), findsOneWidget);
    });

    testWidgets('lists the newest screening first', (WidgetTester tester) async {
      // Server order is not guaranteed, and Home sorts descending. The two
      // views disagreeing about which screening is most recent is the bug.
      final repo = _FakeAssessmentRepository(
        Success([
          _entry(id: 'older', at: DateTime(2023, 1, 1), type: 'Older Scan'),
          _entry(id: 'newer', at: DateTime(2024, 6, 9), type: 'Newer Scan'),
        ]),
      );
      await tester.pumpWidget(buildTestWidget(repository: repo));
      await tester.pumpAndSettle();

      final newer = tester.getTopLeft(find.text('Newer Scan')).dy;
      final older = tester.getTopLeft(find.text('Older Scan')).dy;
      expect(newer, lessThan(older));
    });

    testWidgets('a row opens its result', (WidgetTester tester) async {
      // The row carries a chevron. Before this it was decoration: the card had
      // no gesture at all, so the affordance promised navigation that the
      // screen could not perform.
      final repo = _FakeAssessmentRepository(
        Success([_entry(id: 'asm-1', at: DateTime(2024, 3, 3))]),
      );
      await tester.pumpWidget(buildTestWidget(repository: repo));
      await tester.pumpAndSettle();

      expect(
        find.descendant(
          of: find.byType(ScreeningRecord),
          matching: find.byType(InkWell),
        ),
        findsOneWidget,
      );
      final inkWell = tester.widget<InkWell>(
        find.descendant(
          of: find.byType(ScreeningRecord),
          matching: find.byType(InkWell),
        ),
      );
      expect(inkWell.onTap, isNotNull);
    });

    testWidgets('a high-risk band is not drawn like a low-risk one', (
      WidgetTester tester,
    ) async {
      // Every band used to render in the same primary teal, so the strongest
      // result the app can produce was indistinguishable from the weakest.
      final repo = _FakeAssessmentRepository(
        Success([
          _entry(id: 'hi', at: DateTime(2024, 5, 5), riskLevel: 'HIGH_RISK'),
        ]),
      );
      await tester.pumpWidget(buildTestWidget(repository: repo));
      await tester.pumpAndSettle();

      expect(find.text('HIGH RISK'), findsOneWidget);
      // Not colour alone: the band also changes icon.
      expect(find.byIcon(Icons.warning_amber_rounded), findsOneWidget);
    });

    testWidgets('renders empty view when history is empty', (
      WidgetTester tester,
    ) async {
      final repo = _FakeAssessmentRepository(const Success([]));
      await tester.pumpWidget(buildTestWidget(repository: repo));
      await tester.pumpAndSettle();

      expect(find.text('No screenings yet'), findsOneWidget);
      expect(
        find.text(
          'Your completed screenings will appear here. Start with a guided oral photograph.',
        ),
        findsOneWidget,
      );
      expect(find.text('Start your first screening'), findsOneWidget);
    });

    testWidgets('renders error view on failure and allows retry', (
      WidgetTester tester,
    ) async {
      final repo = _FakeAssessmentRepository(
        const Error(ServerFailure('Connection failed')),
      );
      await tester.pumpWidget(buildTestWidget(repository: repo));
      await tester.pumpAndSettle();

      expect(find.text('Screening history is unavailable'), findsOneWidget);
      expect(find.text('Try Again'), findsOneWidget);
      // The transport detail stays out of the patient's way.
      expect(find.text('Connection failed'), findsNothing);

      expect(repo.historyCalls, 1);
      await tester.tap(find.text('Try Again'));
      await tester.pumpAndSettle();
      expect(repo.historyCalls, 2);
    });

    testWidgets('renders in Hindi without falling back to English', (
      WidgetTester tester,
    ) async {
      final repo = _FakeAssessmentRepository(const Success([]));
      await tester.pumpWidget(
        buildTestWidget(repository: repo, locale: const Locale('hi')),
      );
      await tester.pumpAndSettle();

      expect(find.text('इतिहास'), findsOneWidget);
      expect(find.text('अभी कोई स्क्रीनिंग नहीं हुई'), findsOneWidget);
      expect(find.text('No screenings yet'), findsNothing);
    });
  });
}
