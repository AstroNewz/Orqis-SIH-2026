import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/result/screens/assessment_result_screen.dart';
import 'package:carescan/l10n/l10n.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

/// Everything the screen actually put on screen, including `RichText` spans.
///
/// `find.text` only matches a `Text` whose `data` is set, so it cannot see a
/// label/value pair built from spans. The DEC-034 assertions below are negative
/// — "no percentage is rendered for the primary model" — and a negative claim
/// checked with a finder that cannot see half the widgets would pass whether or
/// not the rule held.
String _renderedText(WidgetTester tester) {
  final parts = <String>[];
  for (final widget in tester.allWidgets) {
    if (widget is Text) {
      parts.add(widget.data ?? widget.textSpan?.toPlainText() ?? '');
    } else if (widget is RichText) {
      parts.add(widget.text.toPlainText());
    }
  }
  return parts.join('\n');
}

void main() {
  Widget buildTestWidget({
    AssessmentResult? result,
    String? imagePath,
    Locale locale = const Locale('en'),
  }) {
    return MaterialApp(
      locale: locale,
      theme: AppTheme.theme,
      localizationsDelegates: AppLocalizations.localizationsDelegates,
      supportedLocales: AppLocalizations.supportedLocales,
      home: AssessmentResultScreen(result: result, imagePath: imagePath),
    );
  }

  group('AssessmentResultScreen', () {
    testWidgets('an absent result does not read as low risk', (
      WidgetTester tester,
    ) async {
      // The screen is reachable from a history row whose payload failed to
      // decode. It used to default the band to LOW RISK and print "below the
      // screening threshold" — telling the patient they were fine on the
      // strength of no result at all.
      await tester.pumpWidget(buildTestWidget());

      expect(find.text('Screening Result'), findsOneWidget);
      expect(find.text('Assessment unavailable'), findsOneWidget);
      expect(find.text('LOW RISK'), findsNothing);
      expect(
        find.text(
          'A usable screening result is not available. '
          'Please check your history or try again.',
        ),
        findsOneWidget,
      );
      expect(find.text('View History'), findsOneWidget);
      expect(find.text('New Scan'), findsOneWidget);
    });

    testWidgets('an unrecognised band does not read as low risk', (
      WidgetTester tester,
    ) async {
      // Same failure mode from the other direction: the old substring chain
      // fell through to low risk for every value it did not recognise.
      await tester.pumpWidget(
        buildTestWidget(
          result: const AssessmentResult(
            id: 'res-1',
            assessmentId: 'asm-1',
            riskLevel: 'INDETERMINATE',
            details: 'Server could not classify this capture.',
          ),
        ),
      );

      expect(find.text('Assessment unavailable'), findsOneWidget);
      expect(find.text('LOW RISK'), findsNothing);
    });

    testWidgets('a raw backend band is normalised, not printed', (
      WidgetTester tester,
    ) async {
      // The hero used to uppercase the backend string and render it directly,
      // so the API's own enum spelling reached the patient.
      await tester.pumpWidget(
        buildTestWidget(
          result: const AssessmentResult(
            id: 'res-2',
            assessmentId: 'asm-2',
            riskLevel: 'HIGH_RISK',
            details: 'Findings warrant review.',
          ),
        ),
      );

      expect(find.text('HIGH RISK'), findsOneWidget);
      expect(find.text('HIGH_RISK'), findsNothing);
      // Not colour alone.
      expect(find.byIcon(Icons.warning_amber_rounded), findsOneWidget);
    });

    testWidgets('renders the band, the server notice and the short id', (
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
      // The server's own notice is rendered verbatim rather than paraphrased.
      expect(
        find.text('Areas of concern detected requiring follow up.'),
        findsOneWidget,
      );
      expect(find.text('ID: test-123'), findsOneWidget);
    });

    testWidgets('an uncalibrated primary model gets no percentage (DEC-034)', (
      WidgetTester tester,
    ) async {
      const result = AssessmentResult(
        id: 'res-3',
        assessmentId: 'asm-3',
        riskLevel: 'Low Risk',
        details: 'Screening complete.',
        primaryModel: 'logistic_regression',
        primaryRiskLevel: 'HIGH_RISK',
        // A ranking score, not a probability. It must not be shown as one.
        primaryProbability: 0.87,
        primaryCalibrated: false,
      );

      await tester.pumpWidget(buildTestWidget(result: result));

      // The headline is the primary model's band, not the quantum riskLevel.
      expect(find.text('HIGH RISK'), findsOneWidget);
      expect(find.text('LOW RISK'), findsNothing);

      final text = _renderedText(tester);
      expect(text, contains('logistic regression'));
      expect(
        text,
        contains(
          'The primary model provides a ranking score, '
          'not a calibrated disease probability.',
        ),
      );
      // No percentage anywhere: the primary score is the only number on this
      // result, and it is not a calibrated probability.
      expect(text, isNot(contains('%')));
      expect(text, isNot(contains('87')));
    });

    testWidgets('the calibrated quantum probability is a labelled secondary', (
      WidgetTester tester,
    ) async {
      const result = AssessmentResult(
        id: 'res-4',
        assessmentId: 'asm-4',
        riskLevel: 'Low Risk',
        details: 'Screening complete.',
        primaryModel: 'logistic_regression',
        primaryRiskLevel: 'LOW_RISK',
        primaryCalibrated: false,
        finalProbability: 0.123,
      );

      await tester.pumpWidget(buildTestWidget(result: result));

      final text = _renderedText(tester);
      // The one percentage the screen may render, under its own heading.
      expect(find.text('Experimental Research Analysis'), findsOneWidget);
      expect(text, contains('Secondary calibrated probability: 12.3%'));
      expect(
        text,
        contains('Quantum advantage and clinical validation '
            'have not been established.'),
      );
    });

    testWidgets('no quantum probability means no research card', (
      WidgetTester tester,
    ) async {
      // A card whose only content is "not available" is noise on a clinical
      // result, so the section is withheld rather than emptied.
      await tester.pumpWidget(
        buildTestWidget(
          result: const AssessmentResult(
            id: 'res-5',
            assessmentId: 'asm-5',
            riskLevel: 'Low Risk',
            details: 'Screening complete.',
            primaryModel: 'logistic_regression',
            primaryRiskLevel: 'LOW_RISK',
            primaryCalibrated: false,
          ),
        ),
      );

      expect(find.text('Experimental Research Analysis'), findsNothing);
      expect(_renderedText(tester), isNot(contains('%')));
    });

    testWidgets('a stub result is announced and describes no model', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        buildTestWidget(
          result: const AssessmentResult(
            id: 'res-6',
            assessmentId: 'asm-6',
            riskLevel: 'Low Risk',
            details: 'Stubbed.',
            primaryModel: 'logistic_regression',
            primaryCalibrated: false,
            finalProbability: 0.4,
            isMock: true,
          ),
        ),
      );

      expect(find.text('DEMO DATA — not a clinical result'), findsOneWidget);
      // Model provenance is withheld for a stub: there is no model behind it,
      // and a readout would present the stub as a real measurement.
      expect(find.text('Understanding this result'), findsNothing);
      expect(find.text('Experimental Research Analysis'), findsNothing);
    });

    testWidgets('renders in Hindi without falling back to English', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        buildTestWidget(
          locale: const Locale('hi'),
          result: const AssessmentResult(
            id: 'res-7',
            assessmentId: 'asm-7',
            riskLevel: 'HIGH_RISK',
            details: 'Findings warrant review.',
          ),
        ),
      );

      expect(find.text('स्क्रीनिंग परिणाम'), findsOneWidget);
      expect(find.text('अधिक जोखिम'), findsOneWidget);
      expect(find.text('अब क्या करें'), findsOneWidget);
      expect(find.text('इतिहास देखें'), findsOneWidget);
      expect(find.text('HIGH RISK'), findsNothing);
      expect(find.text('What to do next'), findsNothing);
    });
  });
}
