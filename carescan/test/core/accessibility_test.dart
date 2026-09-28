import 'package:carescan/app.dart';
import 'package:carescan/features/auth/prototype_session.dart';
import 'package:carescan/features/preview/screens/image_preview_screen.dart';
import 'package:carescan/features/result/screens/assessment_result_screen.dart';
import 'package:carescan/shared/widgets/app_button.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('Accessibility & Semantics Verification', () {
    testWidgets('HomeScreen accessibility checks', (WidgetTester tester) async {
      // `AppRouter`'s redirect sends every route to /welcome until a prototype
      // session exists, so pumping MyApp alone lands on onboarding rather than
      // home. Enter as a guest to reach the screen under test.
      appSession.continueAsGuest();
      addTearDown(appSession.logout);

      await tester.pumpWidget(const MyApp());
      await tester.pumpAndSettle();

      final SemanticsHandle handle = tester.ensureSemantics();

      // Check key actionable elements have labels. The app bar's profile action
      // is icon-only, so its tooltip is what a screen reader announces.
      expect(find.byTooltip('Profile'), findsOneWidget);
      expect(find.text('Start Screening'), findsOneWidget);

      handle.dispose();
    });

    testWidgets('ImagePreviewScreen touch targets meet minimum 48dp size', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        const MaterialApp(home: ImagePreviewScreen(imagePath: 'dummy.jpg')),
      );
      await tester.pumpAndSettle();

      final useImageButton = tester.getSize(
        find.ancestor(
          of: find.text('Use This Image'),
          matching: find.byType(AppButton),
        ),
      );
      expect(useImageButton.height, greaterThanOrEqualTo(48.0));

      final retakeButton = tester.getSize(
        find.ancestor(
          of: find.text('Retake'),
          matching: find.byType(OutlinedButton),
        ),
      );
      expect(retakeButton.height, greaterThanOrEqualTo(48.0));
    });

    testWidgets('AssessmentResultScreen touch targets meet minimum 48dp size', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        const MaterialApp(home: AssessmentResultScreen()),
      );
      await tester.pumpAndSettle();

      final saveButton = tester.getSize(
        find.ancestor(
          of: find.text('View History'),
          matching: find.byType(AppButton),
        ),
      );
      expect(saveButton.height, greaterThanOrEqualTo(48.0));

      final discardButton = tester.getSize(
        find.ancestor(
          of: find.text('New Scan'),
          matching: find.byType(AppButton),
        ),
      );
      expect(discardButton.height, greaterThanOrEqualTo(48.0));
    });
  });
}
