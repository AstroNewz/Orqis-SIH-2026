import 'package:flutter/material.dart';

import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/navigation/app_router.dart';

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'CareScan',
      theme: AppTheme.theme,
      routerConfig: AppRouter.router,
    );
  }
}
