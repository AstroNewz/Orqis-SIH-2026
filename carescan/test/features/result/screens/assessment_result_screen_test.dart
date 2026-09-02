import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/result/screens/assessment_result_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  Widget buildTestWidget({AssessmentResult? result, String? imagePath}) {
    return MaterialApp(
      theme: AppTheme.theme,
      home: AssessmentResultScreen(result: result, imagePath: imagePath),
    );
  }

  group('AssessmentResultScreen', () {
    testWidgets('renders default low risk result and details correctly', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(buildTestWidget());

      expect(find.text('Assessment'), findsOneWidget);
      expect(find.text('LOW RISK'), findsOneWidget);
      expect(find.text('Why this result?'), findsOneWidget);
      expect(find.text('Next Steps'), findsOneWidget);
      expect(find.text('Save to History'), findsOneWidget);
      expect(find.text('Discard'), findsOneWidget);
    });

    testWidgets('renders custom assessment result risk level', (
      WidgetTester tester,
    ) async {
      const customResult = AssessmentResult(
        id: 'test-12345678',
        assessmentId: 'asm-1',
        riskLevel: 'Moderate Risk',
        details: 'Areas of concern detected requiring follow up.',
      );

      await tester.pumpWidget(buildTestWidget(result: customResult));

      expect(find.text('MODERATE RISK'), findsOneWidget);
      expect(
        find.text('Areas of concern detected requiring follow up.'),
        findsOneWidget,
      );
      expect(find.text('ID: test-123'), findsOneWidget);
    });
  });
}
