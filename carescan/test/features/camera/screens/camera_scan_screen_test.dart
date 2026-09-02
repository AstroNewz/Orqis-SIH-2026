import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/features/camera/screens/camera_scan_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  Widget buildTestWidget() {
    return MaterialApp(theme: AppTheme.theme, home: const CameraScanScreen());
  }

  group('CameraScanScreen', () {
    testWidgets('shows loading state initially', (WidgetTester tester) async {
      await tester.pumpWidget(buildTestWidget());

      expect(find.byType(CircularProgressIndicator), findsOneWidget);
      expect(find.bySemanticsLabel('Initializing camera'), findsOneWidget);
    });
  });
}
