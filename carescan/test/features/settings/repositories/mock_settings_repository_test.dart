import 'package:flutter_test/flutter_test.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/features/settings/models/user_settings.dart';
import 'package:carescan/features/settings/repositories/mock_settings_repository.dart';

void main() {
  group('MockSettingsRepository', () {
    late MockSettingsRepository repository;

    setUp(() {
      repository = MockSettingsRepository();
    });

    test('getUserSettings returns default Success', () async {
      final result = await repository.getUserSettings();

      expect(result, isA<Success>());
      result.fold((failure) => fail('Expected Success, got Error'), (data) {
        expect(data.notificationsEnabled, isTrue);
        expect(data.themeMode, 'system');
      });
    });

    test('updateUserSettings updates and saves settings', () async {
      const newSettings = UserSettings(
        notificationsEnabled: false,
        themeMode: 'dark',
      );

      final updateResult = await repository.updateUserSettings(newSettings);
      expect(updateResult, isA<Success>());

      final getResult = await repository.getUserSettings();
      getResult.fold((failure) => fail('Expected Success, got Error'), (data) {
        expect(data.notificationsEnabled, isFalse);
        expect(data.themeMode, 'dark');
      });
    });
  });
}
