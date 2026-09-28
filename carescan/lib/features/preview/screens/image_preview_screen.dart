import 'dart:io';

import 'package:carescan/core/theme/app_shapes.dart';
import 'package:carescan/core/theme/app_spacing.dart';
import 'package:carescan/shared/widgets/app_button.dart';
import 'package:carescan/shared/widgets/empty_state_widget.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

class ImagePreviewScreen extends StatelessWidget {
  final String? imagePath;

  const ImagePreviewScreen({super.key, this.imagePath});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final hasImage = imagePath != null && imagePath!.isNotEmpty;

    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => context.pop(),
        ),
        title: const Text('Preview'),
        centerTitle: true,
        actions: const [SizedBox(width: 48)], // For balance
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(
            horizontal: AppSpacing.md,
            vertical: AppSpacing.md,
          ),
          child: Column(
            children: [
              // Image Preview Card
              Expanded(
                child: Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: theme.colorScheme.surfaceContainer,
                    borderRadius: AppShapes.radiusMd,
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.04),
                        blurRadius: 20,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: ClipRRect(
                    borderRadius: AppShapes.radiusMd,
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        if (hasImage)
                          Image.file(
                            File(imagePath!),
                            fit: BoxFit.cover,
                            errorBuilder: (context, error, stackTrace) =>
                                const Center(
                                  child: Text('Failed to load image'),
                                ),
                          )
                        else
                          const Center(
                            child: EmptyStateWidget(
                              icon: Icons.image_not_supported_rounded,
                              message: 'No Image. Please return to the camera and capture an image.',
                            ),
                          ),
                        // Quality Check Badge
                        if (hasImage)
                          Positioned(
                            top: 16,
                            right: 16,
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 6,
                              ),
                              decoration: BoxDecoration(
                                color: const Color(0xFFE6F4EA)
                                    .withValues(alpha: 0.9),
                                borderRadius: BorderRadius.circular(24),
                              ),
                              child: const Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Icon(
                                    Icons.check_circle_rounded,
                                    size: 16,
                                    color: Color(0xFF137333),
                                  ),
                                  SizedBox(width: 4),
                                  Text(
                                    'Looks good',
                                    style: TextStyle(
                                      color: Color(0xFF137333),
                                      fontWeight: FontWeight.w600,
                                      fontSize: 12,
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
              ),
              const SizedBox(height: AppSpacing.xl),
              // Instructions / Feedback
              Text(
                'Image is clear',
                style: theme.textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: AppSpacing.xs),
              Text(
                'Make sure all details are visible and in focus before proceeding.',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: AppSpacing.xl),
              // Action Buttons
              SizedBox(
                width: double.infinity,
                child: AppButton(
                  label: 'Use This Image',
                  onPressed: hasImage
                      ? () => context.push('/analyzing', extra: imagePath)
                      : null,
                  variant: AppButtonVariant.primary,
                ),
              ),
              const SizedBox(height: AppSpacing.md),
              SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  onPressed: () => context.pop(), // Returns to camera
                  icon: const Icon(Icons.refresh_rounded),
                  label: const Text('Retake'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
