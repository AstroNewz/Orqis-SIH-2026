import 'package:flutter/material.dart';
import 'package:carescan/features/auth/prototype_session.dart';
import 'package:carescan/features/auth/welcome_screen.dart';
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
import 'package:carescan/features/tracks/screens/tracks_screen.dart';
import 'package:carescan/features/education/screens/education_screen.dart';
import 'package:carescan/features/education/screens/article_screen.dart';

final _rootNavigatorKey = GlobalKey<NavigatorState>();

class AppRouter {
  static final GoRouter router = GoRouter(
    initialLocation: '/',
    refreshListenable: appSession,
    redirect: (context, state) {
      if (appSession.identity == null && state.uri.path != '/welcome') {
        return '/welcome';
      }
      if (appSession.identity != null && state.uri.path == '/welcome') {
        return '/';
      }
      return null;
    },
    navigatorKey: _rootNavigatorKey,
    routes: [
      GoRoute(
        path: '/welcome',
        builder: (context, state) => const WelcomeScreen(),
      ),
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
                builder: (context, state) => const EducationScreen(),
                // Nested rather than pushed on the root navigator, so the
                // bottom bar stays visible while reading. Browsing an editorial
                // section is moving around inside a tab, not leaving for a
                // one-way flow the way /camera and /result are.
                routes: [
                  GoRoute(
                    path: ':articleId',
                    builder: (context, state) => ArticleScreen(
                      articleId: state.pathParameters['articleId'],
                    ),
                  ),
                ],
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
      // Pushed rather than made a fifth shell branch: the bottom bar's four
      // indices are load-bearing (see `scaffold_with_nav.dart` and the
      // navigation test that asserts them), and the track catalogue is
      // something you consult, not a place you live.
      GoRoute(
        path: '/tracks',
        parentNavigatorKey: _rootNavigatorKey,
        builder: (context, state) => const TracksScreen(),
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
          final extra = state.extra;
          // Preferred: the analyzing screen hands over the verdict plus the local
          // capture path so the analyzed image can be shown.
          if (extra is ({AssessmentResult result, String? imagePath})) {
            return AssessmentResultScreen(
              result: extra.result,
              imagePath: extra.imagePath,
            );
          }
          // Back-compat: a bare result (or nothing) still renders.
          return AssessmentResultScreen(result: extra as AssessmentResult?);
        },
      ),
    ],
  );
}
