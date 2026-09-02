import 'package:carescan/core/errors/result.dart';
import 'package:carescan/features/settings/models/user_settings.dart';
import 'package:carescan/features/settings/repositories/settings_repository.dart';

class MockSettingsRepository implements SettingsRepository {
  UserSettings _settings = const UserSettings(
    notificationsEnabled: true,
    dataSharingEnabled: false,
    themeMode: 'system',
  );

  @override
  Future<Result<UserSettings>> getUserSettings() async {
    await Future.delayed(const Duration(milliseconds: 500));
    return Success(_settings);
  }

  @override
  Future<Result<void>> updateUserSettings(UserSettings settings) async {
    await Future.delayed(const Duration(milliseconds: 500));
    _settings = settings;
    return const Success(null);
  }
}
