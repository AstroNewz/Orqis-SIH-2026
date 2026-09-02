import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import 'package:carescan/navigation/scaffold_with_nav.dart';
import 'package:carescan/features/home/home_screen.dart';
import 'package:carescan/features/history/history_screen.dart';
import 'package:carescan/features/settings/settings_screen.dart';
import 'package:carescan/features/camera/screens/camera_scan_screen.dart';
import 'package:carescan/features/preview/screens/image_preview_screen.dart';
import 'package:carescan/features/analyzing/screens/analyzing_screen.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/result/screens/assessment_result_screen.dart';

// Placeholder screens for T-NAV-01 & T-NAV-02

class PlaceholderScreen extends StatelessWidget {
  final String title;
  const PlaceholderScreen({super.key, required this.title});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(title)),
      body: Center(child: Text(title)),
    );
  }
}

final _rootNavigatorKey = GlobalKey<NavigatorState>();

class AppRouter {
  static final GoRouter router = GoRouter(
    initialLocation: '/',
    navigatorKey: _rootNavigatorKey,
    routes: [
      StatefulShellRoute.indexedStack(
        builder: (context, state, navigationShell) {
          return ScaffoldWithNav(navigationShell: navigationShell);
        },
        branches: [
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: '/',
                builder: (context, state) => const HomeScreen(),
              ),
            ],
          ),
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: '/blogs',
                builder: (context, state) =>
                    const PlaceholderScreen(title: 'Blogs'),
              ),
            ],
          ),
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: '/history',
                builder: (context, state) => const HistoryScreen(),
              ),
            ],
          ),
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: '/profile',
                builder: (context, state) => const SettingsScreen(),
              ),
            ],
          ),
        ],
      ),
      // Push routes
      GoRoute(
        path: '/camera',
        parentNavigatorKey: _rootNavigatorKey,
        builder: (context, state) => const CameraScanScreen(),
      ),
      GoRoute(
        path: '/preview',
        parentNavigatorKey: _rootNavigatorKey,
        builder: (context, state) {
          final imagePath = state.extra as String?;
          return ImagePreviewScreen(imagePath: imagePath);
        },
      ),
      GoRoute(
        path: '/analyzing',
        parentNavigatorKey: _rootNavigatorKey,
        builder: (context, state) {
          final imagePath = state.extra as String?;
          return AnalyzingScreen(imagePath: imagePath);
        },
      ),
      GoRoute(
        path: '/result',
        parentNavigatorKey: _rootNavigatorKey,
        builder: (context, state) {
          final result = state.extra as AssessmentResult?;
          return AssessmentResultScreen(result: result);
        },
      ),
    ],
  );
}
