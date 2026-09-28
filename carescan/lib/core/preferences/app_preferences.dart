import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Only non-sensitive appearance preferences are kept in SharedPreferences.
class AppPreferences extends ChangeNotifier {
  AppPreferences();
  SharedPreferences? _storage;
  Locale _locale = const Locale('en');
  ThemeMode _themeMode = ThemeMode.system;
  Locale get locale => _locale;
  ThemeMode get themeMode => _themeMode;

  Future<void> load() async {
    try {
      _storage ??= await SharedPreferences.getInstance();
      _locale = Locale(_storage!.getString('language') == 'hi' ? 'hi' : 'en');
      _themeMode = switch (_storage!.getString('theme')) {
        'dark' => ThemeMode.dark,
        'light' => ThemeMode.light,
        _ => ThemeMode.system,
      };
      notifyListeners();
    } catch (_) {
      /* Defaults remain usable if platform storage is unavailable. */
    }
  }

  Future<bool> setLocale(Locale value) async {
    if (!['en', 'hi'].contains(value.languageCode)) return false;
    try {
      _storage ??= await SharedPreferences.getInstance();
      if (!await _storage!.setString('language', value.languageCode)) {
        return false;
      }
      _locale = value;
      notifyListeners();
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<bool> setTheme(ThemeMode value) async {
    try {
      _storage ??= await SharedPreferences.getInstance();
      if (!await _storage!.setString('theme', value.name)) return false;
      _themeMode = value;
      notifyListeners();
      return true;
    } catch (_) {
      return false;
    }
  }
}

final appPreferences = AppPreferences();
