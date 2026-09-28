import 'dart:convert';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

abstract interface class ProfileStore {
  Future<String?> read();
  Future<void> write(String value);
}

class SecureProfileStore implements ProfileStore {
  const SecureProfileStore();
  static const _storage = FlutterSecureStorage();
  @override
  Future<String?> read() => _storage.read(key: 'carescan.prototype.profile');
  @override
  Future<void> write(String value) =>
      _storage.write(key: 'carescan.prototype.profile', value: value);
}

@immutable
class PrototypeIdentity {
  const PrototypeIdentity({
    required this.id,
    required this.name,
    required this.isGuest,
  });
  final String id;
  final String name;
  final bool isGuest;
}

/// Device-only demo access. This is deliberately NOT an authentication provider.
class PrototypeSession extends ChangeNotifier {
  PrototypeSession({this.store = const SecureProfileStore()});
  final ProfileStore store;
  PrototypeIdentity? savedProfile;
  PrototypeIdentity? identity;
  bool storageAvailable = true;

  Future<void> load() async {
    try {
      final raw = await store.read();
      if (raw != null) {
        final data = jsonDecode(raw);
        if (data is Map<String, dynamic> &&
            data['id'] is String &&
            data['name'] is String &&
            RegExp(r'^prototype-[a-f0-9]{32}$')
                .hasMatch(data['id'] as String)) {
          savedProfile = PrototypeIdentity(
            id: data['id'] as String,
            name: data['name'] as String,
            isGuest: false,
          );
        }
      }
      storageAvailable = true;
    } catch (_) {
      storageAvailable = false;
    }
    notifyListeners();
  }

  Future<bool> createProfile(String name) async {
    final clean = name.trim();
    if (clean.length < 2 || clean.length > 60 || savedProfile != null) {
      return false;
    }
    final profile = PrototypeIdentity(
      id: _newId(),
      name: clean,
      isGuest: false,
    );
    try {
      await store.write(jsonEncode({'id': profile.id, 'name': profile.name}));
      savedProfile = profile;
      identity = profile;
      storageAvailable = true;
      notifyListeners();
      return true;
    } catch (_) {
      storageAvailable = false;
      notifyListeners();
      return false;
    }
  }

  bool login() {
    if (savedProfile == null) return false;
    identity = savedProfile;
    notifyListeners();
    return true;
  }

  void continueAsGuest() {
    identity = PrototypeIdentity(id: _newId(), name: '', isGuest: true);
    notifyListeners();
  }

  void logout() {
    identity = null;
    notifyListeners();
  }

  String _newId() {
    final random = Random.secure();
    return 'prototype-${List.generate(16, (_) => random.nextInt(256).toRadixString(16).padLeft(2, '0')).join()}';
  }
}

final appSession = PrototypeSession();
