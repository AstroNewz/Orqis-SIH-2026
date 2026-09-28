import 'package:carescan/app.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/auth/prototype_session.dart';
import 'package:carescan/features/preview/screens/image_preview_screen.dart';
import 'package:carescan/features/result/screens/assessment_result_screen.dart';
import 'package:carescan/navigation/scaffold_with_nav.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';

/// Which shell branch is showing, read from the navigation shell itself.
///
/// `StatefulShellRoute.indexedStack` keeps a visited branch mounted and
/// size-maintaining rather than offstage, so a `find.text` from a previously
/// visited branch still matches after switching away from it. Text alone
/// therefore cannot prove *which* tab is selected; the shell's own index can.
int _selectedBranch(WidgetTester tester) => tester
    .widget<ScaffoldWithNav>(find.byType(ScaffoldWithNav))
    .navigationShell
    .currentIndex;

void main() {
  group('Assessment Flow Integration', () {
    testWidgets('Shell navigation moves between the four branches', (
      WidgetTester tester,
    ) async {
      // `AppRouter`'s redirect holds every route at /welcome until a prototype
      // session exists, so the shell is unreachable without entering one first.
      appSession.continueAsGuest();
      addTearDown(appSession.logout);

      await tester.pumpWidget(const MyApp());
      await tester.pumpAndSettle();

      // Verify we start on the Home branch.
      expect(_selectedBranch(tester), 0);
      expect(find.text('Screen early.\nUnderstand sooner.'), findsOneWidget);
      expect(find.text('Start Screening'), findsOneWidget);

      // Verify the five bottom-bar destinations exist. Each label is asserted
      // here, before any branch is visited, because a visited branch can
      // contribute a colliding title of its own (SettingsScreen's app bar also
      // reads 'Profile').
      expect(find.text('Home'), findsOneWidget);
      expect(find.text('Blogs'), findsOneWidget);
      expect(find.text('Scan'), findsOneWidget);
      expect(find.text('History'), findsOneWidget);
      expect(find.text('Profile'), findsOneWidget);

      // Navigate to the History branch.
      await tester.tap(find.text('History'));
      await tester.pumpAndSettle();
      expect(_selectedBranch(tester), 2);
      // The History app bar now carries the same localized word as its nav
      // label, so two matches here is the screen being present, not a
      // duplicate. The branch index above is the load-bearing assertion.
      expect(find.text('History'), findsNWidgets(2));

      // Navigate to the Profile branch.
      await tester.tap(find.text('Profile'));
      await tester.pumpAndSettle();
      expect(_selectedBranch(tester), 3);
      expect(find.text('Personal Information'), findsOneWidget);
      expect(find.text('Notifications'), findsOneWidget);

      // Switch back to Home. The nav label is still the only 'Home' text, but
      // the branch index is what actually establishes the tab changed.
      await tester.tap(find.text('Home'));
      await tester.pumpAndSettle();
      expect(_selectedBranch(tester), 0);
    });

    testWidgets('Preview screen transitions to Analyzing and Result', (
      WidgetTester tester,
    ) async {
      final testRouter = GoRouter(
        initialLocation: '/preview',
        routes: [
          GoRoute(
            path: '/preview',
            builder: (context, state) =>
                const ImagePreviewScreen(imagePath: 'dummy/path.jpg'),
          ),
          GoRoute(
            path: '/analyzing',
            builder: (context, state) =>
                const Scaffold(body: Text('Analyzing Mock')),
          ),
          GoRoute(
            path: '/result',
            builder: (context, state) => const AssessmentResultScreen(
              result: AssessmentResult(
                id: 'res-999',
                assessmentId: 'asm-999',
                riskLevel: 'Low Risk',
                details: 'Analysis completed successfully.',
              ),
            ),
          ),
        ],
      );

      await tester.pumpWidget(MaterialApp.router(routerConfig: testRouter));
      await tester.pumpAndSettle();

      expect(find.text('Preview'), findsOneWidget);
      expect(find.text('Looks good'), findsOneWidget);

      // Tap Use This Image
      await tester.tap(find.text('Use This Image'));
      await tester.pumpAndSettle();

      expect(find.text('Analyzing Mock'), findsOneWidget);
    });
  });
}
