import 'package:flutter/material.dart';
import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/core/preferences/app_preferences.dart';
import 'package:carescan/l10n/app_localizations.dart';
import 'package:carescan/navigation/app_router.dart';

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) => ListenableBuilder(
    listenable: appPreferences,
    builder: (context, _) => MaterialApp.router(
      title: 'CareScan',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light,
      darkTheme: AppTheme.dark,
      themeMode: appPreferences.themeMode,
      locale: appPreferences.locale,
      localizationsDelegates: AppLocalizations.localizationsDelegates,
      supportedLocales: AppLocalizations.supportedLocales,
      routerConfig: AppRouter.router,
    ),
  );
}
