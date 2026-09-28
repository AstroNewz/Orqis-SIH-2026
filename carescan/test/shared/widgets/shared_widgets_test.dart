import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:carescan/shared/widgets/app_button.dart';
import 'package:carescan/shared/widgets/app_card.dart';
import 'package:carescan/shared/widgets/loading_indicator.dart';
import 'package:carescan/shared/widgets/error_state_widget.dart';
import 'package:carescan/shared/widgets/empty_state_widget.dart';
import 'package:carescan/core/theme/app_theme.dart';

void main() {
  Widget buildTestWidget(Widget child) {
    return MaterialApp(
      theme: AppTheme.theme,
      home: Scaffold(body: child),
    );
  }

  group('Shared Widgets', () {
    testWidgets('AppButton renders label and taps', (
      WidgetTester tester,
    ) async {
      bool tapped = false;
      await tester.pumpWidget(
        buildTestWidget(
          AppButton(label: 'Test Button', onPressed: () => tapped = true),
        ),
      );

      expect(find.text('Test Button'), findsOneWidget);
      await tester.tap(find.byType(ElevatedButton));
      expect(tapped, isTrue);
    });

    testWidgets('AppCard renders child', (WidgetTester tester) async {
      await tester.pumpWidget(
        buildTestWidget(const AppCard(child: Text('Card Content'))),
      );

      expect(find.byType(Card), findsOneWidget);
      expect(find.text('Card Content'), findsOneWidget);
    });

    testWidgets('LoadingIndicator renders spinner and optional message', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        buildTestWidget(const LoadingIndicator(message: 'Loading Data...')),
      );

      expect(find.byType(CircularProgressIndicator), findsOneWidget);
      expect(find.text('Loading Data...'), findsOneWidget);
    });

    testWidgets('ErrorStateWidget renders details and retry', (
      WidgetTester tester,
    ) async {
      bool retried = false;
      await tester.pumpWidget(
        buildTestWidget(
          ErrorStateWidget(
            title: 'Error',
            message: 'Something went wrong',
            onRetry: () => retried = true,
          ),
        ),
      );

      expect(find.text('Error'), findsOneWidget);
      expect(find.text('Something went wrong'), findsOneWidget);
      // The retry label is localised now, so it reads the same as every other
      // retry affordance in the app rather than being the one English word
      // under a translated title and message.
      expect(find.text('Try Again'), findsOneWidget);

      await tester.tap(find.text('Try Again'));
      expect(retried, isTrue);
    });

    testWidgets('EmptyStateWidget renders message and action', (
      WidgetTester tester,
    ) async {
      bool actionTapped = false;
      await tester.pumpWidget(
        buildTestWidget(
          EmptyStateWidget(
            message: 'No Data',
            actionLabel: 'Refresh',
            onAction: () => actionTapped = true,
          ),
        ),
      );

      expect(find.text('No Data'), findsOneWidget);
      expect(find.text('Refresh'), findsOneWidget);

      await tester.tap(find.text('Refresh'));
      expect(actionTapped, isTrue);
    });
  });
}
