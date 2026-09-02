import 'package:carescan/core/errors/result.dart';
import 'package:carescan/features/settings/models/user_settings.dart';

abstract class SettingsRepository {
  /// Retrieves the user's settings.
  Future<Result<UserSettings>> getUserSettings();

  /// Updates the user's settings.
  Future<Result<void>> updateUserSettings(UserSettings settings);
}
