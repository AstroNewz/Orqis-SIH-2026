import 'package:carescan/core/theme/app_shapes.dart';
import 'package:carescan/core/theme/app_theme.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

/// The radius the theme hands to cards and buttons.
///
/// Read back out of the built theme rather than from a literal here, so that
/// changing the theme without changing the token — the exact drift [AppShapes]
/// existed to prevent and did not — fails this file.
BorderRadius _shapeRadius(ShapeBorder? shape) {
  expect(shape, isA<RoundedRectangleBorder>());
  final radius = (shape! as RoundedRectangleBorder).borderRadius;
  expect(radius, isA<BorderRadius>());
  return radius as BorderRadius;
}

OutlineInputBorder _outline(InputBorder? border, String name) {
  expect(
    border,
    isNotNull,
    reason: 'inputDecorationTheme.$name is not set, so Flutter falls back to '
        '`border` and the state is invisible',
  );
  expect(border, isA<OutlineInputBorder>());
  return border! as OutlineInputBorder;
}

void main() {
  final themes = <String, ThemeData>{
    'light': AppTheme.light,
    'dark': AppTheme.dark,
  };

  themes.forEach((name, theme) {
    final inputs = theme.inputDecorationTheme;

    group('$name theme', () {
      test('a focused input looks different from an idle one', () {
        final enabled = _outline(inputs.enabledBorder, 'enabledBorder');
        final focused = _outline(inputs.focusedBorder, 'focusedBorder');

        // WCAG 2.4.7 Focus Visible. With `focusedBorder` unset, Flutter
        // resolves the focused state to `border`, which carried the same
        // outlineVariant side as `enabledBorder` — so focusing the only text
        // field in the app changed nothing on screen. A keyboard or switch
        // user had no way to see where they were.
        expect(
          focused.borderSide,
          isNot(enabled.borderSide),
          reason: 'focus must change something visible',
        );
        expect(focused.borderSide.color, theme.colorScheme.primary);
      });

      test('focus is not signalled by colour alone', () {
        final enabled = _outline(inputs.enabledBorder, 'enabledBorder');
        final focused = _outline(inputs.focusedBorder, 'focusedBorder');

        // DESIGN.md: information is never carried by colour alone. Teal
        // against grey is the wrong side of that rule on its own, so the
        // focused ring is also thicker.
        expect(focused.borderSide.width, greaterThan(enabled.borderSide.width));
      });

      test('an invalid input is distinguishable from a valid one', () {
        final enabled = _outline(inputs.enabledBorder, 'enabledBorder');
        final error = _outline(inputs.errorBorder, 'errorBorder');
        final focusedError = _outline(
          inputs.focusedErrorBorder,
          'focusedErrorBorder',
        );

        expect(error.borderSide.color, theme.colorScheme.error);
        expect(focusedError.borderSide.color, theme.colorScheme.error);
        expect(error.borderSide, isNot(enabled.borderSide));
        // An invalid field that is also focused must still read as focused.
        expect(
          focusedError.borderSide.width,
          greaterThan(error.borderSide.width),
        );
      });

      test('every input border uses the shared radius token', () {
        for (final entry in <String, InputBorder?>{
          'border': inputs.border,
          'enabledBorder': inputs.enabledBorder,
          'focusedBorder': inputs.focusedBorder,
          'errorBorder': inputs.errorBorder,
          'focusedErrorBorder': inputs.focusedErrorBorder,
        }.entries) {
          expect(
            _outline(entry.value, entry.key).borderRadius,
            AppShapes.radiusMd,
            reason: '${entry.key} does not use AppShapes.radiusMd',
          );
        }
      });

      test('cards, buttons and inputs all share one radius token', () {
        // The token file claimed 8 while every surface in the app rendered 12,
        // and nothing imported the token, so nothing caught it. These three
        // assertions are what make the token load-bearing instead of
        // decorative.
        expect(_shapeRadius(theme.cardTheme.shape), AppShapes.radiusMd);
        expect(
          _shapeRadius(
            theme.filledButtonTheme.style?.shape?.resolve(<WidgetState>{}),
          ),
          AppShapes.radiusMd,
        );
        expect(
          _outline(inputs.border, 'border').borderRadius,
          AppShapes.radiusMd,
        );
      });
    });
  });

  testWidgets('a text field in the app theme shows focus when focused', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: AppTheme.light,
        home: const Scaffold(body: TextField()),
      ),
    );

    InputDecorator decorator() =>
        tester.widget<InputDecorator>(find.byType(InputDecorator));

    expect(decorator().isFocused, isFalse);

    await tester.tap(find.byType(TextField));
    await tester.pumpAndSettle();

    // The resolved decoration is the theme's after `applyDefaults`, so this
    // asserts the running widget — not just the ThemeData — has a focused
    // border to draw now that it is in the focused state.
    expect(decorator().isFocused, isTrue);
    final focused = decorator().decoration.focusedBorder;
    expect(focused, isNotNull);
    expect(
      focused!.borderSide.color,
      AppTheme.light.colorScheme.primary,
      reason: 'the focused field has no focus ring to paint',
    );
  });
}
