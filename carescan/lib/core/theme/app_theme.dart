import 'package:flutter/material.dart';

import 'app_colors.dart';
import 'app_typography.dart';
import 'app_shapes.dart';

/// Centralized application theme.
class AppTheme {
  /// The light theme for the application.
  static ThemeData get theme {
    final colorScheme = const ColorScheme(
      brightness: Brightness.light,
      primary: AppColors.primary,
      onPrimary: AppColors.onPrimary,
      secondary: AppColors.secondary,
      onSecondary: AppColors.onBackground,
      error: AppColors.error,
      onError: AppColors.onError,
      surface: AppColors.surface,
      onSurface: AppColors.onSurface,
    );

    return ThemeData(
      useMaterial3: true,
      colorScheme: colorScheme,
      scaffoldBackgroundColor: AppColors.background,
      dividerColor: AppColors.divider,
      disabledColor: AppColors.disabled,
      textTheme: AppTypography.textTheme,
      cardTheme: const CardThemeData(
        color: AppColors.surface,
        shape: RoundedRectangleBorder(borderRadius: AppShapes.radiusMd),
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: AppColors.primary,
        foregroundColor: AppColors.onPrimary,
        elevation: 0,
        centerTitle: true,
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: AppColors.primary,
          foregroundColor: AppColors.onPrimary,
          shape: const RoundedRectangleBorder(borderRadius: AppShapes.radiusSm),
        ),
      ),
    );
  }
}
