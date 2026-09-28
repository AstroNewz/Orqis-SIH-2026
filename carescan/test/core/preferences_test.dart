import 'package:carescan/core/preferences/app_preferences.dart';
import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/l10n/app_localizations.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  test('language and theme survive a new preferences instance', () async {
    SharedPreferences.setMockInitialValues({});
    final first = AppPreferences();
    await first.load();
    expect(await first.setLocale(const Locale('hi')), isTrue);
    expect(await first.setTheme(ThemeMode.dark), isTrue);
    final restored = AppPreferences();
    await restored.load();
    expect(restored.locale.languageCode, 'hi');
    expect(restored.themeMode, ThemeMode.dark);
    expect(await restored.setLocale(const Locale('fr')), isFalse);
  });
  testWidgets('Hindi delegates and dark text render together', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        locale: const Locale('hi'),
        theme: AppTheme.dark,
        localizationsDelegates: AppLocalizations.localizationsDelegates,
        supportedLocales: AppLocalizations.supportedLocales,
        home: Builder(
          builder: (context) => Scaffold(
            body: Text(AppLocalizations.of(context)!.startScreening),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();
    expect(find.text('स्क्रीनिंग शुरू करें'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}
