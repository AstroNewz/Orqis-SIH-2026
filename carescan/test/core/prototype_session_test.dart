import 'package:carescan/features/auth/prototype_session.dart';
import 'package:flutter_test/flutter_test.dart';

class MemoryProfileStore implements ProfileStore {
  String? value;
  @override
  Future<String?> read() async => value;
  @override
  Future<void> write(String text) async {
    value = text;
  }
}

void main() {
  test(
    'saved local identity survives restart; guest IDs are isolated',
    () async {
      final storage = MemoryProfileStore();
      final session = PrototypeSession(store: storage);
      expect(await session.createProfile('Demo user'), isTrue);
      final id = session.identity!.id;
      session.logout();
      expect(session.identity, isNull);
      final reopened = PrototypeSession(store: storage);
      await reopened.load();
      expect(reopened.identity, isNull);
      expect(reopened.login(), isTrue);
      expect(reopened.identity!.id, id);
      reopened.logout();
      reopened.continueAsGuest();
      final guestId = reopened.identity!.id;
      expect(guestId, isNot(id));
      reopened.logout();
      reopened.continueAsGuest();
      expect(reopened.identity!.id, isNot(guestId));
      expect(await reopened.createProfile('Replacement'), isFalse);
    },
  );
}
