import 'package:flutter_test/flutter_test.dart';

import 'package:carescan/app.dart';

void main() {
  testWidgets('App renders Home Dashboard initially', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const MyApp());
    await tester.pumpAndSettle();

    expect(find.text('Good morning, Alex'), findsOneWidget);
    expect(find.text('Take a Photo'), findsOneWidget);
  });
}
