import 'package:flutter/material.dart';

enum AppButtonVariant { primary, secondary, text }

class AppButton extends StatelessWidget {
  final String label;
  final VoidCallback? onPressed;
  final AppButtonVariant variant;

  const AppButton({
    super.key,
    required this.label,
    required this.onPressed,
    this.variant = AppButtonVariant.primary,
  });

  @override
  Widget build(BuildContext context) {
    // Touch target >= 48dp required by accessibility guidelines
    const minimumSize = Size(64, 48);

    switch (variant) {
      case AppButtonVariant.primary:
        return ElevatedButton(
          onPressed: onPressed,
          style: ElevatedButton.styleFrom(minimumSize: minimumSize),
          child: Text(label),
        );
      case AppButtonVariant.secondary:
        return OutlinedButton(
          onPressed: onPressed,
          style: OutlinedButton.styleFrom(minimumSize: minimumSize),
          child: Text(label),
        );
      case AppButtonVariant.text:
        return TextButton(
          onPressed: onPressed,
          style: TextButton.styleFrom(minimumSize: minimumSize),
          child: Text(label),
        );
    }
  }
}
