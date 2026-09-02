class UserSettings {
  final bool notificationsEnabled;
  final bool dataSharingEnabled;
  final String themeMode; // 'system', 'light', 'dark'

  const UserSettings({
    this.notificationsEnabled = true,
    this.dataSharingEnabled = false,
    this.themeMode = 'system',
  });

  factory UserSettings.fromJson(Map<String, dynamic> json) {
    return UserSettings(
      notificationsEnabled: json['notificationsEnabled'] as bool? ?? true,
      dataSharingEnabled: json['dataSharingEnabled'] as bool? ?? false,
      themeMode: json['themeMode'] as String? ?? 'system',
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'notificationsEnabled': notificationsEnabled,
      'dataSharingEnabled': dataSharingEnabled,
      'themeMode': themeMode,
    };
  }

  UserSettings copyWith({
    bool? notificationsEnabled,
    bool? dataSharingEnabled,
    String? themeMode,
  }) {
    return UserSettings(
      notificationsEnabled: notificationsEnabled ?? this.notificationsEnabled,
      dataSharingEnabled: dataSharingEnabled ?? this.dataSharingEnabled,
      themeMode: themeMode ?? this.themeMode,
    );
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is UserSettings &&
        other.notificationsEnabled == notificationsEnabled &&
        other.dataSharingEnabled == dataSharingEnabled &&
        other.themeMode == themeMode;
  }

  @override
  int get hashCode {
    return Object.hash(notificationsEnabled, dataSharingEnabled, themeMode);
  }
}
