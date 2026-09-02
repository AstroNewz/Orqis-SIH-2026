import 'package:flutter/material.dart';

/// Design tokens for colors.
///
/// All visual values come from the design token files. Hardcoded styling in feature code is avoided.
class AppColors {
  // TODO: Verify with Stitch - Exact color values require Stitch inspection.

  /// Primary brand color
  static const Color primary = Color(0xFF6200EE);

  /// Darker/lighter primary
  static const Color primaryVariant = Color(0xFF3700B3);

  /// Secondary accent
  static const Color secondary = Color(0xFF03DAC6);

  /// Screen backgrounds
  static const Color background = Color(0xFFF6F6F6);

  /// Card/container surfaces
  static const Color surface = Color(0xFFFFFFFF);

  /// Error indicators
  static const Color error = Color(0xFFB00020);

  /// Text/icons on primary
  static const Color onPrimary = Color(0xFFFFFFFF);

  /// Text/icons on background
  static const Color onBackground = Color(0xFF000000);

  /// Text/icons on surface
  static const Color onSurface = Color(0xFF000000);

  /// Text/icons on error
  static const Color onError = Color(0xFFFFFFFF);

  /// Divider lines
  static const Color divider = Color(0xFFE0E0E0);

  /// Disabled elements
  static const Color disabled = Color(0xFF9E9E9E);
}
