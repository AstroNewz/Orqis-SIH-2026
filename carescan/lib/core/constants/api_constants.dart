import 'dart:io';

/// API configuration and endpoint definitions for CareScan backend communication.
/// Supports Android emulator, iOS simulator, and physical mobile devices via configurable LAN IP.
class ApiConstants {
  ApiConstants._();

  /// Compile-time configurable base URL (e.g. `flutter run --dart-define=BACKEND_BASE_URL=http://192.168.1.100:8000`)
  static const String _envBaseUrl = String.fromEnvironment('BACKEND_BASE_URL');

  /// Runtime override for base URL (useful for testing or dynamic discovery)
  static String? _customBaseUrl;

  /// Configure custom base URL at runtime (e.g., setting Mac LAN IP for physical iPhone)
  static void setBaseUrl(String url) {
    _customBaseUrl = url;
  }

  /// Reset to platform default base URL
  static void resetBaseUrl() {
    _customBaseUrl = null;
  }

  /// Resolved base URL supporting:
  /// 1. Explicit runtime override (`setBaseUrl`)
  /// 2. Dart compile-time environment define (`--dart-define=BACKEND_BASE_URL=...`)
  /// 3. Android Emulator (`http://10.0.2.2:8000`)
  /// 4. iOS Simulator / Desktop (`http://localhost:8000`)
  static String get baseUrl {
    if (_customBaseUrl != null && _customBaseUrl!.isNotEmpty) {
      return _customBaseUrl!;
    }

    if (_envBaseUrl.isNotEmpty) {
      return _envBaseUrl;
    }

    if (Platform.isAndroid) {
      // Android Emulator loopback alias to host machine
      return 'http://10.0.2.2:8000';
    }

    // iOS Simulator, macOS desktop, or default local host
    return 'http://localhost:8000';
  }

  // API Endpoints
  static const String analyzePath = '/api/screening/analyze';
  static const String uploadPath = '/api/screening/upload';
  static const String localizePath = '/api/localize';
  static const String healthPath = '/health';

  /// The platform track catalogue. Lists every screening track the backend knows
  /// about, including the ones that are not ready, each with the validation
  /// provenance behind its headline number.
  static const String tracksPath = '/api/tracks';

  static String trackDetailPath(String trackId) => '/api/tracks/$trackId';

  static String patientHistoryPath(String patientId) =>
      '/api/patients/$patientId/history';

  static String screeningDetailPath(String screeningId) =>
      '/api/screening/$screeningId';

  static String resultDetailPath(String screeningId) =>
      '/api/results/$screeningId';

  // Request Timeouts
  static const Duration connectTimeout = Duration(seconds: 10);
  static const Duration receiveTimeout = Duration(seconds: 30);
}
