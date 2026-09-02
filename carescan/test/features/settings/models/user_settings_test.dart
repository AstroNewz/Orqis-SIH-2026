import 'package:flutter_test/flutter_test.dart';
import 'package:carescan/features/settings/models/user_settings.dart';

void main() {
  group('UserSettings Model', () {
    const tSettings = UserSettings(
      notificationsEnabled: true,
      dataSharingEnabled: false,
      themeMode: 'system',
    );

    final tJson = {
      'notificationsEnabled': true,
      'dataSharingEnabled': false,
      'themeMode': 'system',
    };

    test('fromJson should return a valid model', () {
      final result = UserSettings.fromJson(tJson);
      expect(result, equals(tSettings));
    });

    test('toJson should return a JSON map containing the proper data', () {
      final result = tSettings.toJson();
      expect(result, equals(tJson));
    });

    test('fromJson should handle missing fields with defaults', () {
      final result = UserSettings.fromJson(const {});
      expect(result.notificationsEnabled, true);
      expect(result.dataSharingEnabled, false);
      expect(result.themeMode, 'system');
    });
  });
}
