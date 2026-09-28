import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/features/camera/widgets/oral_capture_guide.dart';
import 'package:carescan/l10n/l10n.dart';

void main() {
  for (final language in ['en','hi']) {
    testWidgets('capture reference fits a small phone in $language', (tester) async {
      tester.view.physicalSize = const Size(320, 640);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      await tester.pumpWidget(MaterialApp(
        locale: Locale(language), theme: AppTheme.dark,
        localizationsDelegates: AppLocalizations.localizationsDelegates,
        supportedLocales: AppLocalizations.supportedLocales,
        home: const Scaffold(body: CaptureGuideContent()),
      ));
      await tester.pumpAndSettle();
      expect(find.byType(OralReferenceGraphic), findsOneWidget);
      expect(tester.takeException(), isNull);
      await tester.drag(find.byType(SingleChildScrollView), const Offset(0, -600));
      await tester.pumpAndSettle();
      expect(tester.takeException(), isNull);
    });
  }
}
