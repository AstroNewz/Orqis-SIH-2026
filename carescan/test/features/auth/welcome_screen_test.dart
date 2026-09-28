import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/features/auth/prototype_session.dart';
import 'package:carescan/features/auth/welcome_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import '../../core/prototype_session_test.dart' show MemoryProfileStore;

void main() {
  testWidgets('guest entry creates a separate explicit guest session', (tester) async {
    final session = PrototypeSession(store: MemoryProfileStore());
    await tester.pumpWidget(MaterialApp(theme: AppTheme.light, home: WelcomeScreen(session: session)));
    await tester.ensureVisible(find.text('Continue as guest'));
    await tester.tap(find.text('Continue as guest'));
    expect(session.identity?.isGuest, isTrue);
    expect(session.savedProfile, isNull);
  });
  testWidgets('create profile validates name and creates no password field', (tester) async {
    final session = PrototypeSession(store: MemoryProfileStore());
    await tester.pumpWidget(MaterialApp(theme: AppTheme.light, home: WelcomeScreen(session: session)));
    await tester.ensureVisible(find.text('Create account'));
    await tester.tap(find.text('Create account'));
    await tester.pumpAndSettle();
    expect(find.byType(TextFormField), findsOneWidget);
    await tester.enterText(find.byType(TextFormField), 'Demo visitor');
    await tester.ensureVisible(find.text('Create demo profile'));
    await tester.tap(find.text('Create demo profile'));
    await tester.pumpAndSettle();
    expect(session.identity?.name, 'Demo visitor');
    expect(tester.takeException(), isNull);
  });
  testWidgets('a rejected name is marked on the field, not only under it', (tester) async {
    final session = PrototypeSession(store: MemoryProfileStore());
    await tester.pumpWidget(MaterialApp(theme: AppTheme.light, home: WelcomeScreen(session: session)));
    await tester.ensureVisible(find.text('Create account'));
    await tester.tap(find.text('Create account'));
    await tester.pumpAndSettle();
    await tester.enterText(find.byType(TextFormField), 'a');
    await tester.ensureVisible(find.text('Create demo profile'));
    await tester.tap(find.text('Create demo profile'));
    await tester.pumpAndSettle();

    // The validator fired and nothing was created.
    expect(session.identity, isNull);
    final decorator = tester.widget<InputDecorator>(find.byType(InputDecorator));
    expect(decorator.decoration.errorText, isNotNull);
    // And the field itself is marked. This is the only validated input in the
    // app, and until inputDecorationTheme.errorBorder was set it resolved to
    // the idle outline, so the rejection was carried by the helper text alone.
    expect(decorator.decoration.errorBorder?.borderSide.color, AppTheme.light.colorScheme.error);
  });
}
