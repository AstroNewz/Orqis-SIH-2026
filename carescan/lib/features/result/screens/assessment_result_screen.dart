import 'dart:io';

import 'package:carescan/core/theme/app_spacing.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/shared/widgets/app_button.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

class AssessmentResultScreen extends StatelessWidget {
  final AssessmentResult? result;
  final String? imagePath;

  const AssessmentResultScreen({super.key, this.result, this.imagePath});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final riskLevel = (result?.riskLevel ?? 'LOW RISK').toUpperCase();
    final isLowRisk = riskLevel.contains('LOW');
    final details =
        result?.details ??
        'Based on the visual analysis, this area appears normal with no immediate concerns.';

    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => context.go('/home'),
        ),
        title: const Text('Assessment'),
        centerTitle: true,
        actions: [
          IconButton(icon: const Icon(Icons.more_horiz), onPressed: () {}),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(
            horizontal: AppSpacing.md,
            vertical: AppSpacing.sm,
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Hero Card
              Container(
                padding: const EdgeInsets.symmetric(
                  vertical: AppSpacing.xl,
                  horizontal: AppSpacing.lg,
                ),
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceContainerLowest,
                  borderRadius: BorderRadius.circular(24),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withValues(alpha: 0.04),
                      blurRadius: 16,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Column(
                  children: [
                    Container(
                      width: 96,
                      height: 96,
                      decoration: BoxDecoration(
                        color: isLowRisk
                            ? theme.colorScheme.primaryContainer
                            : theme.colorScheme.errorContainer,
                        shape: BoxShape.circle,
                      ),
                      child: Icon(
                        isLowRisk
                            ? Icons.check_circle_rounded
                            : Icons.warning_rounded,
                        size: 48,
                        color: isLowRisk
                            ? theme.colorScheme.onPrimaryContainer
                            : theme.colorScheme.onErrorContainer,
                      ),
                    ),
                    const SizedBox(height: AppSpacing.md),
                    Text(
                      riskLevel,
                      style: theme.textTheme.headlineMedium?.copyWith(
                        color: isLowRisk
                            ? theme.colorScheme.primary
                            : theme.colorScheme.error,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: AppSpacing.xs),
                    Text(
                      details,
                      style: theme.textTheme.bodyLarge?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
              const SizedBox(height: AppSpacing.md),

              // Image Reference (if available)
              if (imagePath != null && imagePath!.isNotEmpty)
                Container(
                  height: 192,
                  margin: const EdgeInsets.only(bottom: AppSpacing.md),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(20),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.04),
                        blurRadius: 16,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(20),
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        Image.file(
                          File(imagePath!),
                          fit: BoxFit.cover,
                          errorBuilder: (context, error, stackTrace) =>
                              Container(
                                color: theme.colorScheme.surfaceContainer,
                                child: const Center(
                                  child: Icon(Icons.image_outlined, size: 48),
                                ),
                              ),
                        ),
                        Positioned(
                          bottom: 8,
                          right: 8,
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 4,
                            ),
                            decoration: BoxDecoration(
                              color: theme.colorScheme.surfaceContainerLowest
                                  .withValues(alpha: 0.8),
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(
                                  Icons.visibility_outlined,
                                  size: 14,
                                  color: theme.colorScheme.onSurfaceVariant,
                                ),
                                const SizedBox(width: 4),
                                Text(
                                  'Analyzed Area',
                                  style: theme.textTheme.labelMedium?.copyWith(
                                    color: theme.colorScheme.onSurfaceVariant,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

              // Details Grid / List
              // Card 1: Why this result?
              Container(
                padding: const EdgeInsets.all(AppSpacing.lg),
                margin: const EdgeInsets.only(bottom: AppSpacing.md),
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceContainerLowest,
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withValues(alpha: 0.04),
                      blurRadius: 16,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(
                          Icons.analytics_outlined,
                          size: 20,
                          color: theme.colorScheme.secondary,
                        ),
                        const SizedBox(width: AppSpacing.xs),
                        Text(
                          'Why this result?',
                          style: theme.textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: AppSpacing.sm),
                    Text(
                      'The scan detected typical pigmentation, uniform borders, and normal texture consistency. No asymmetry or significant color variations indicative of anomalies were found by the analysis engine.',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                        height: 1.4,
                      ),
                    ),
                  ],
                ),
              ),

              // Card 2: Next Steps
              Container(
                padding: const EdgeInsets.all(AppSpacing.lg),
                margin: const EdgeInsets.only(bottom: AppSpacing.md),
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceContainerLowest,
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withValues(alpha: 0.04),
                      blurRadius: 16,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(
                          Icons.arrow_forward_rounded,
                          size: 20,
                          color: theme.colorScheme.secondary,
                        ),
                        const SizedBox(width: AppSpacing.xs),
                        Text(
                          'Next Steps',
                          style: theme.textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: AppSpacing.sm),
                    _buildBulletItem(
                      context,
                      icon: Icons.calendar_month_outlined,
                      text: 'Monitor the area and consider re-scanning in 2 weeks for tracking.',
                    ),
                    const SizedBox(height: AppSpacing.sm),
                    _buildBulletItem(
                      context,
                      icon: Icons.shield_outlined,
                      text: 'Continue regular sun protection routines.',
                    ),
                  ],
                ),
              ),

              // Metadata Section
              Container(
                padding: const EdgeInsets.all(AppSpacing.md),
                margin: const EdgeInsets.only(bottom: AppSpacing.xl),
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceContainer,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Row(
                      children: [
                        Icon(
                          Icons.schedule_outlined,
                          size: 16,
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                        const SizedBox(width: AppSpacing.xs),
                        Text(
                          'Today • Recent Scan',
                          style: theme.textTheme.labelMedium?.copyWith(
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                      ],
                    ),
                    Row(
                      children: [
                        Icon(
                          Icons.fingerprint_outlined,
                          size: 16,
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                        const SizedBox(width: AppSpacing.xs),
                        Text(
                          result != null
                              ? 'ID: ${result!.id.substring(0, 8)}'
                              : 'ID: ASM-9284-B',
                          style: theme.textTheme.labelMedium?.copyWith(
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),

              // Action Buttons
              SizedBox(
                width: double.infinity,
                child: AppButton(
                  label: 'Save to History',
                  onPressed: () => context.go('/history'),
                  variant: AppButtonVariant.primary,
                ),
              ),
              const SizedBox(height: AppSpacing.sm),
              SizedBox(
                width: double.infinity,
                child: AppButton(
                  label: 'Discard',
                  onPressed: () => context.go('/home'),
                  variant: AppButtonVariant.secondary,
                ),
              ),
              const SizedBox(height: AppSpacing.lg),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildBulletItem(
    BuildContext context, {
    required IconData icon,
    required String text,
  }) {
    final theme = Theme.of(context);
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 18, color: theme.colorScheme.primary),
        const SizedBox(width: AppSpacing.sm),
        Expanded(
          child: Text(
            text,
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ),
      ],
    );
  }
}
