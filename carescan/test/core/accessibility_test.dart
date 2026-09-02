import 'package:carescan/app.dart';
import 'package:carescan/features/preview/screens/image_preview_screen.dart';
import 'package:carescan/features/result/screens/assessment_result_screen.dart';
import 'package:carescan/shared/widgets/app_button.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('Accessibility & Semantics Verification', () {
    testWidgets('HomeScreen accessibility checks', (WidgetTester tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.pumpAndSettle();

      final SemanticsHandle handle = tester.ensureSemantics();

      // Check key actionable elements have labels
      expect(find.byIcon(Icons.settings), findsOneWidget);
      expect(find.text('Take a Photo'), findsOneWidget);

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
          of: find.text('Save to History'),
          matching: find.byType(AppButton),
        ),
      );
      expect(saveButton.height, greaterThanOrEqualTo(48.0));

      final discardButton = tester.getSize(
        find.ancestor(
          of: find.text('Discard'),
          matching: find.byType(AppButton),
        ),
      );
      expect(discardButton.height, greaterThanOrEqualTo(48.0));
    });
  });
}
