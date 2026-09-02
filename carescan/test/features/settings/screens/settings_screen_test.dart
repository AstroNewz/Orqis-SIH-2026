import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/features/settings/models/user_settings.dart';
import 'package:carescan/features/settings/repositories/settings_repository.dart';
import 'package:carescan/features/settings/settings_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

class _FakeSettingsRepository implements SettingsRepository {
  Result<UserSettings> getResult;
  UserSettings? lastUpdated;

  _FakeSettingsRepository(this.getResult);

  @override
  Future<Result<UserSettings>> getUserSettings() async => getResult;

  @override
  Future<Result<void>> updateUserSettings(UserSettings settings) async {
    lastUpdated = settings;
    return const Success(null);
  }
}

void main() {
  Widget buildTestWidget({required SettingsRepository repository}) {
    return MaterialApp(
      theme: AppTheme.theme,
      home: SettingsScreen(repository: repository),
    );
  }

  group('SettingsScreen', () {
    testWidgets('renders profile and settings tiles on success', (
      WidgetTester tester,
    ) async {
      final repo = _FakeSettingsRepository(
        const Success(UserSettings(notificationsEnabled: true)),
      );

      await tester.pumpWidget(buildTestWidget(repository: repo));
      await tester.pumpAndSettle();

      expect(find.text('Profile'), findsOneWidget);
      expect(find.text('Alex Johnson'), findsOneWidget);
      expect(find.text('Personal Information'), findsOneWidget);
      expect(find.text('Notifications'), findsOneWidget);
      expect(find.text('Version 1.0.0'), findsOneWidget);
      expect(find.text('Log Out'), findsOneWidget);
    });

    testWidgets('toggling notifications updates repository and UI', (
      WidgetTester tester,
    ) async {
      final repo = _FakeSettingsRepository(
        const Success(UserSettings(notificationsEnabled: true)),
      );

      await tester.pumpWidget(buildTestWidget(repository: repo));
      await tester.pumpAndSettle();

      final switchFinder = find.byType(Switch);
      expect(switchFinder, findsOneWidget);
      expect(tester.widget<Switch>(switchFinder).value, true);

      await tester.tap(switchFinder);
      await tester.pumpAndSettle();

      expect(repo.lastUpdated?.notificationsEnabled, false);
    });

    testWidgets('renders error view on failure and allows retry', (
      WidgetTester tester,
    ) async {
      final repo = _FakeSettingsRepository(
        const Error(ServerFailure('Failed')),
      );

      await tester.pumpWidget(buildTestWidget(repository: repo));
      await tester.pumpAndSettle();

      expect(find.text('Could Not Load Settings'), findsOneWidget);
      expect(find.text('Retry'), findsOneWidget);
    });
  });
}
