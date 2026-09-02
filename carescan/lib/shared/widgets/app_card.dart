import 'package:flutter/material.dart';
import 'package:carescan/core/theme/app_spacing.dart';

class AppCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;

  const AppCard({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(AppSpacing.md),
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(padding: padding, child: child),
    );
  }
}
