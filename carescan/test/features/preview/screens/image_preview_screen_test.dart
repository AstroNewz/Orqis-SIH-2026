import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/features/preview/screens/image_preview_screen.dart';
import 'package:carescan/shared/widgets/empty_state_widget.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  Widget buildTestWidget({String? imagePath}) {
    return MaterialApp(
      theme: AppTheme.theme,
      home: ImagePreviewScreen(imagePath: imagePath),
    );
  }

  group('ImagePreviewScreen', () {
    testWidgets('shows empty state when no image path is provided', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(buildTestWidget(imagePath: null));

      expect(find.byType(EmptyStateWidget), findsOneWidget);
      expect(find.textContaining('No Image'), findsOneWidget);

      // Retake button should be present
      expect(find.text('Retake'), findsOneWidget);
    });
  });
}
