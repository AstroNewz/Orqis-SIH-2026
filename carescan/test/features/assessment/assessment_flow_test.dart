import 'package:carescan/app.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/preview/screens/image_preview_screen.dart';
import 'package:carescan/features/result/screens/assessment_result_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';

void main() {
  group('Assessment Flow Integration', () {
    testWidgets('Full navigation from Home to Preview, Analyzing and Result', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pumpAndSettle();

      // Verify we start at Home Dashboard
      expect(find.text('Good morning, Alex'), findsOneWidget);
      expect(find.text('Start a New Assessment'), findsOneWidget);

      // Verify bottom nav items exist
      expect(find.text('Home'), findsOneWidget);
      expect(find.text('History'), findsOneWidget);
      expect(find.text('Profile'), findsOneWidget);

      // Navigate to History tab
      await tester.tap(find.text('History'));
      await tester.pumpAndSettle();
      expect(find.text('Assessment History'), findsOneWidget);

      // Navigate to Profile/Settings tab
      await tester.tap(find.text('Profile'));
      await tester.pumpAndSettle();
      expect(find.text('Personal Information'), findsOneWidget);
      expect(find.text('Notifications'), findsOneWidget);

      // Switch back to Home
      await tester.tap(find.text('Home'));
      await tester.pumpAndSettle();
      expect(find.text('Start a New Assessment'), findsOneWidget);
    });

    testWidgets('Preview screen transitions to Analyzing and Result', (
      WidgetTester tester,
    ) async {
      final testRouter = GoRouter(
        initialLocation: '/preview',
        routes: [
          GoRoute(
            path: '/preview',
            builder: (context, state) =>
                const ImagePreviewScreen(imagePath: 'dummy/path.jpg'),
          ),
          GoRoute(
            path: '/analyzing',
            builder: (context, state) =>
                const Scaffold(body: Text('Analyzing Mock')),
          ),
          GoRoute(
            path: '/result',
            builder: (context, state) => const AssessmentResultScreen(
              result: AssessmentResult(
                id: 'res-999',
                assessmentId: 'asm-999',
                riskLevel: 'Low Risk',
                details: 'Analysis completed successfully.',
              ),
            ),
          ),
        ],
      );

      await tester.pumpWidget(MaterialApp.router(routerConfig: testRouter));
      await tester.pumpAndSettle();

      expect(find.text('Preview'), findsOneWidget);
      expect(find.text('Looks good'), findsOneWidget);

      // Tap Use This Image
      await tester.tap(find.text('Use This Image'));
      await tester.pumpAndSettle();

      expect(find.text('Analyzing Mock'), findsOneWidget);
    });
  });
}
