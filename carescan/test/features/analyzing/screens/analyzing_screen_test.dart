import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/features/analyzing/screens/analyzing_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  Widget buildTestWidget() {
    return MaterialApp(theme: AppTheme.theme, home: const AnalyzingScreen());
  }

  group('AnalyzingScreen', () {
    testWidgets('renders analyzing screen correctly', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(buildTestWidget());

      expect(find.text('Analyzing your image...'), findsOneWidget);
      expect(
        find.byType(CircularProgressIndicator),
        findsNothing,
      ); // We use custom animation
      expect(find.byIcon(Icons.image_search_rounded), findsOneWidget);
    });
  });
}
