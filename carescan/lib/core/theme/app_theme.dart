import 'package:flutter/material.dart';

import 'app_colors.dart';
import 'app_shapes.dart';
import 'app_typography.dart';

abstract final class AppTheme {
  static ThemeData get theme => light;
  static ThemeData get light => _build(false);
  static ThemeData get dark => _build(true);

  static ThemeData _build(bool dark) {
    final scheme =
        ColorScheme.fromSeed(
          seedColor: AppColors.primary,
          brightness: dark ? Brightness.dark : Brightness.light,
        ).copyWith(
          primary: dark ? AppColors.darkPrimary : AppColors.primary,
          onPrimary: dark ? AppColors.darkBackground : Colors.white,
          primaryContainer: dark
              ? AppColors.darkElevated
              : AppColors.softSurface,
          onPrimaryContainer: dark
              ? AppColors.darkPrimary
              : AppColors.primaryVariant,
          surface: dark ? AppColors.darkSurface : AppColors.surface,
          surfaceContainerLowest: dark
              ? AppColors.darkBackground
              : AppColors.surface,
          surfaceContainerLow: dark
              ? AppColors.darkSurface
              : AppColors.background,
          surfaceContainer: dark
              ? AppColors.darkElevated
              : AppColors.softSurface,
          surfaceContainerHigh: dark
              ? AppColors.darkElevated
              : AppColors.softSurface,
          surfaceContainerHighest: dark
              ? AppColors.darkBorder
              : AppColors.divider,
          onSurface: dark ? AppColors.darkText : AppColors.onSurface,
          onSurfaceVariant: dark
              ? AppColors.darkSecondaryText
              : AppColors.secondaryText,
          outline: dark ? AppColors.darkSecondaryText : AppColors.disabled,
          outlineVariant: dark ? AppColors.darkBorder : AppColors.divider,
          error: dark ? const Color(0xFFFFB4AB) : AppColors.error,
          tertiary: dark ? const Color(0xFFE5BA73) : AppColors.warning,
        );
    const shape = RoundedRectangleBorder(borderRadius: AppShapes.radiusMd);
    final baseButton = FilledButton.styleFrom(
      minimumSize: const Size(48, 52),
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
      shape: shape,
      textStyle: AppTypography.textTheme.labelLarge,
    );
    return ThemeData(
      useMaterial3: true,
      colorScheme: scheme,
      scaffoldBackgroundColor: dark
          ? AppColors.darkBackground
          : AppColors.background,
      textTheme: AppTypography.textTheme.apply(
        bodyColor: scheme.onSurface,
        displayColor: scheme.onSurface,
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: dark ? AppColors.darkBackground : AppColors.background,
        foregroundColor: scheme.onSurface,
        surfaceTintColor: Colors.transparent,
        centerTitle: false,
        elevation: 0,
        titleTextStyle: AppTypography.textTheme.titleLarge?.copyWith(
          color: scheme.onSurface,
        ),
      ),
      cardTheme: CardThemeData(
        color: scheme.surface,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: shape,
      ),
      filledButtonTheme: FilledButtonThemeData(style: baseButton),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: baseButton.copyWith(
          backgroundColor: WidgetStatePropertyAll(scheme.primary),
          foregroundColor: WidgetStatePropertyAll(scheme.onPrimary),
          elevation: const WidgetStatePropertyAll(0),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: baseButton.copyWith(
          side: WidgetStatePropertyAll(
            BorderSide(color: scheme.outlineVariant),
          ),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          minimumSize: const Size(48, 48),
          textStyle: AppTypography.textTheme.labelLarge,
        ),
      ),
      iconButtonTheme: IconButtonThemeData(
        style: IconButton.styleFrom(minimumSize: const Size(48, 48)),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: scheme.surface,
        contentPadding: const EdgeInsets.all(16),
        border: OutlineInputBorder(
          borderRadius: AppShapes.radiusMd,
          borderSide: BorderSide(color: scheme.outlineVariant),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: AppShapes.radiusMd,
          borderSide: BorderSide(color: scheme.outlineVariant),
        ),
        // Every state below has to be spelled out. Flutter resolves an unset
        // state border to `border`, so leaving these off did not fall back to
        // a Material default — it painted the *idle* outline in every state,
        // and focusing the field changed nothing at all (WCAG 2.4.7).
        //
        // Thicker as well as coloured: DESIGN.md forbids carrying information
        // by colour alone, and teal-on-grey at 1 px is colour alone.
        focusedBorder: OutlineInputBorder(
          borderRadius: AppShapes.radiusMd,
          borderSide: BorderSide(color: scheme.primary, width: 2),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: AppShapes.radiusMd,
          borderSide: BorderSide(color: scheme.error),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderRadius: AppShapes.radiusMd,
          borderSide: BorderSide(color: scheme.error, width: 2),
        ),
      ),
      dividerTheme: DividerThemeData(
        color: scheme.outlineVariant,
        thickness: 1,
      ),
      snackBarTheme: const SnackBarThemeData(
        behavior: SnackBarBehavior.floating,
      ),
      bottomSheetTheme: BottomSheetThemeData(
        backgroundColor: scheme.surface,
        showDragHandle: true,
      ),
      visualDensity: VisualDensity.standard,
    );
  }
}
