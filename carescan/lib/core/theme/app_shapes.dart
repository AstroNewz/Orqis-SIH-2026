import 'package:flutter/material.dart';

/// Design tokens for shapes (border radii, etc.).
///
/// Exact radii require Stitch inspection. Hardcoded magic numbers are avoided.
class AppShapes {
  // TODO: Verify with Stitch - Exact radii require Stitch inspection.

  /// Small elements (chips, badges)
  static const BorderRadius radiusSm = BorderRadius.all(Radius.circular(4.0));

  /// Cards, inputs
  static const BorderRadius radiusMd = BorderRadius.all(Radius.circular(8.0));

  /// Bottom sheets, dialogs
  static const BorderRadius radiusLg = BorderRadius.all(Radius.circular(16.0));

  /// Circular elements
  static const BorderRadius radiusFull = BorderRadius.all(
    Radius.circular(999.0),
  );
}
