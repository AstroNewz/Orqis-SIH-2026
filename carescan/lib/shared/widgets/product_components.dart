import 'package:flutter/material.dart';
import 'package:carescan/l10n/l10n.dart';
import 'package:carescan/core/theme/app_shapes.dart';

class CareScanBrand extends StatelessWidget {
  const CareScanBrand({super.key, this.large = false});
  final bool large;
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: large ? 52 : 36,
          height: large ? 52 : 36,
          decoration: BoxDecoration(
            color: theme.colorScheme.primary,
            borderRadius: BorderRadius.circular(10),
          ),
          child: Icon(
            Icons.add_rounded,
            color: theme.colorScheme.onPrimary,
            size: large ? 38 : 28,
          ),
        ),
        const SizedBox(width: 10),
        Text(
          context.l10n.appName,
          style:
              (large
                      ? theme.textTheme.headlineLarge
                      : theme.textTheme.titleLarge)
                  ?.copyWith(letterSpacing: -0.5),
        ),
      ],
    );
  }
}

/// Keeps editorial and clinical content readable on phones and tablets.
class ContentPane extends StatelessWidget {
  const ContentPane({
    super.key,
    required this.child,
    this.maxWidth = 720,
    this.padding = const EdgeInsets.all(24),
  });
  final Widget child;
  final double maxWidth;
  final EdgeInsetsGeometry padding;
  @override
  Widget build(BuildContext context) => Align(
    alignment: Alignment.topCenter,
    child: ConstrainedBox(
      constraints: BoxConstraints(maxWidth: maxWidth),
      child: Padding(padding: padding, child: child),
    ),
  );
}

class InfoNote extends StatelessWidget {
  const InfoNote({
    super.key,
    required this.text,
    this.icon = Icons.info_outline,
    this.warning = false,
  });
  final String text;
  final IconData icon;
  final bool warning;
  @override
  Widget build(BuildContext context) {
    final s = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: s.surfaceContainer,
        borderRadius: AppShapes.radiusMd,
        border: Border(
          left: BorderSide(color: warning ? s.tertiary : s.primary, width: 3),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 20, color: warning ? s.tertiary : s.primary),
          const SizedBox(width: 12),
          Expanded(
            child: Text(text, style: Theme.of(context).textTheme.bodyMedium),
          ),
        ],
      ),
    );
  }
}

class SectionHeading extends StatelessWidget {
  const SectionHeading({
    super.key,
    required this.title,
    this.action,
    this.onAction,
  });
  final String title;
  final String? action;
  final VoidCallback? onAction;
  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.only(top: 24, bottom: 12),
    child: Row(
      children: [
        Expanded(
          child: Text(title, style: Theme.of(context).textTheme.titleLarge),
        ),
        if (action != null)
          TextButton(onPressed: onAction, child: Text(action!)),
      ],
    ),
  );
}
