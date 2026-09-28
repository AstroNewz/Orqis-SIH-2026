import 'package:flutter/material.dart';

/// The corner radii the app actually renders.
///
/// This file used to declare a 4 / 8 / 16 scale that nothing imported, while
/// every surface in the app hand-wrote `BorderRadius.circular(12)`. The token
/// was not merely unused — it disagreed with the running app, and because it
/// was unused nothing could notice. Both tokens below are now read by
/// `AppTheme` and by the widgets, and `test/core/theme/app_theme_test.dart`
/// fails if the theme and the token drift apart again.
///
/// Only values the app genuinely uses are declared here. A radius that exists
/// as a constant but appears on no surface is how the previous version went
/// wrong, and it invites a future widget to reach for a plausible-looking token
/// that matches nothing beside it.
///
/// VERIFY WITH STITCH: 12 and 999 are what this build renders, which is why
/// adopting them changes no pixels. Whether 12 is the *designed* radius is a
/// separate question that needs the Stitch file (ISS-014).
class AppShapes {
  /// Cards, buttons, inputs, list tiles — every rectangular surface.
  static const BorderRadius radiusMd = BorderRadius.all(Radius.circular(12.0));

  /// Pills: category chips and status badges.
  static const BorderRadius radiusFull = BorderRadius.all(
    Radius.circular(999.0),
  );
}
