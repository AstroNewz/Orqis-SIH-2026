import 'dart:io';

import 'package:carescan/core/di/service_locator.dart';
import 'package:carescan/features/assessment/models/localization_result.dart';
import 'package:carescan/features/assessment/repositories/assessment_repository.dart';
import 'package:flutter/material.dart';

/// Shows the analyzed capture with an optional, non-gating ROI overlay (DEC-034).
///
/// The box is a *visual aid* only: it is fetched from `POST /api/localize`, which
/// never touches the screening verdict. Everything here fails silently -- a
/// localiser that is unavailable, unsure, or errors just leaves the plain capture
/// on screen -- so the overlay can never break or delay the result.
///
/// The image is drawn with [BoxFit.contain] and the box is mapped from normalised
/// coordinates against the source dimensions the backend reports, so it lands on the
/// right pixels whatever the display size and aspect ratio.
class LocalizedCaptureView extends StatefulWidget {
  final String imagePath;
  final AssessmentRepository? repository;
  final double height;

  const LocalizedCaptureView({
    super.key,
    required this.imagePath,
    this.repository,
    this.height = 220,
  });

  @override
  State<LocalizedCaptureView> createState() => _LocalizedCaptureViewState();
}

class _LocalizedCaptureViewState extends State<LocalizedCaptureView> {
  LocalizationResult? _box;

  @override
  void initState() {
    super.initState();
    _localize();
  }

  Future<void> _localize() async {
    final repo = widget.repository ?? appAssessmentRepository;
    try {
      final result = await repo.localize(widget.imagePath);
      if (!mounted) return;
      result.fold(
        (_) {}, // fail-silent: no overlay on failure
        (loc) {
          if (loc.hasDrawableBox) {
            setState(() => _box = loc);
          }
        },
      );
    } catch (_) {
      // Overlay is advisory: swallow anything so the result is never blocked.
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final box = _box;

    return Container(
      height: widget.height,
      margin: const EdgeInsets.only(bottom: 16),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHigh,
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
              File(widget.imagePath),
              fit: BoxFit.contain,
              errorBuilder: (context, error, stackTrace) => Container(
                color: theme.colorScheme.surfaceContainer,
                child: const Center(
                  child: Icon(Icons.image_outlined, size: 48),
                ),
              ),
            ),
            if (box != null)
              Positioned.fill(
                child: CustomPaint(
                  painter: _RoiBoxPainter(
                    box: box.boxNormalised!,
                    sourceWidth: box.sourceWidth.toDouble(),
                    sourceHeight: box.sourceHeight.toDouble(),
                    stroke: theme.colorScheme.primary,
                    fill: theme.colorScheme.primary.withValues(alpha: 0.12),
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
                  color: theme.colorScheme.surfaceContainerLowest.withValues(
                    alpha: 0.85,
                  ),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      box != null
                          ? Icons.center_focus_strong_outlined
                          : Icons.visibility_outlined,
                      size: 14,
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      box != null ? 'Region of interest' : 'Analyzed capture',
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
    );
  }
}

/// Paints one normalised ROI box over a [BoxFit.contain] image.
///
/// The painter is told the source pixel dimensions so it can reproduce the exact
/// letterboxing `BoxFit.contain` applied, then place the box within the fitted image
/// rect. This keeps the overlay aligned regardless of the widget's own aspect ratio.
class _RoiBoxPainter extends CustomPainter {
  final List<double> box; // [x0, y0, x1, y1] in [0, 1]
  final double sourceWidth;
  final double sourceHeight;
  final Color stroke;
  final Color fill;

  _RoiBoxPainter({
    required this.box,
    required this.sourceWidth,
    required this.sourceHeight,
    required this.stroke,
    required this.fill,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (sourceWidth <= 0 || sourceHeight <= 0 || box.length != 4) return;

    // Reproduce BoxFit.contain: uniform scale, centred with letterbox padding.
    final scale = (size.width / sourceWidth) < (size.height / sourceHeight)
        ? size.width / sourceWidth
        : size.height / sourceHeight;
    final displayWidth = sourceWidth * scale;
    final displayHeight = sourceHeight * scale;
    final offsetX = (size.width - displayWidth) / 2;
    final offsetY = (size.height - displayHeight) / 2;

    final left = offsetX + box[0].clamp(0.0, 1.0) * displayWidth;
    final top = offsetY + box[1].clamp(0.0, 1.0) * displayHeight;
    final right = offsetX + box[2].clamp(0.0, 1.0) * displayWidth;
    final bottom = offsetY + box[3].clamp(0.0, 1.0) * displayHeight;

    if (right <= left || bottom <= top) return;

    final rect = RRect.fromLTRBR(
      left,
      top,
      right,
      bottom,
      const Radius.circular(8),
    );

    canvas.drawRRect(rect, Paint()..color = fill);
    canvas.drawRRect(
      rect,
      Paint()
        ..color = stroke
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5,
    );
  }

  @override
  bool shouldRepaint(covariant _RoiBoxPainter oldDelegate) {
    return box != oldDelegate.box ||
        sourceWidth != oldDelegate.sourceWidth ||
        sourceHeight != oldDelegate.sourceHeight ||
        stroke != oldDelegate.stroke ||
        fill != oldDelegate.fill;
  }
}
