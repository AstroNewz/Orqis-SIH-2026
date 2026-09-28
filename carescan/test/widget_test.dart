import 'package:flutter_test/flutter_test.dart';
import 'package:carescan/app.dart';
import 'package:carescan/features/auth/prototype_session.dart';
void main() {
  testWidgets('App opens honest prototype onboarding before session entry', (tester) async {
    appSession.logout();
    await tester.pumpWidget(const MyApp());
    await tester.pumpAndSettle();
    expect(find.text('Login'), findsOneWidget);
    expect(find.text('Create account'), findsOneWidget);
    expect(find.text('Continue as guest'), findsOneWidget);
    expect(find.text('Alex Johnson'), findsNothing);
  });
}
