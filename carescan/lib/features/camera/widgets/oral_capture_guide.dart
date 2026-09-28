import 'package:flutter/material.dart';
import 'package:carescan/l10n/l10n.dart';
import 'package:carescan/shared/widgets/product_components.dart';
import 'package:carescan/core/theme/app_shapes.dart';

Future<void> showCaptureGuide(BuildContext context) =>
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      builder: (context) => FractionallySizedBox(
        heightFactor: .92,
        child: Scaffold(
          appBar: AppBar(
            title: Text(context.l10n.captureGuide),
            automaticallyImplyLeading: false,
            actions: [
              IconButton(
                tooltip: context.l10n.close,
                onPressed: () => Navigator.pop(context),
                icon: const Icon(Icons.close),
              ),
            ],
          ),
          body: const CaptureGuideContent(),
        ),
      ),
    );

class CaptureGuideContent extends StatelessWidget {
  const CaptureGuideContent({super.key});
  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    return SingleChildScrollView(
      child: ContentPane(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const OralReferenceGraphic(),
            const SizedBox(height: 12),
            Text(l.guideIntro, style: Theme.of(context).textTheme.bodySmall),
            const SizedBox(height: 24),
            _item(
              context,
              Icons.wb_sunny_outlined,
              l.guideLighting,
              l.guideLightingBody,
            ),
            _item(
              context,
              Icons.center_focus_strong,
              l.guidePosition,
              l.guidePositionBody,
            ),
            _item(
              context,
              Icons.straighten,
              l.guideDistance,
              l.guideDistanceBody,
            ),
            _item(
              context,
              Icons.pan_tool_outlined,
              l.guideSteady,
              l.guideSteadyBody,
            ),
            InfoNote(text: l.disclaimer),
          ],
        ),
      ),
    );
  }

  Widget _item(
    BuildContext context,
    IconData icon,
    String title,
    String body,
  ) => Padding(
    padding: const EdgeInsets.only(bottom: 24),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, color: Theme.of(context).colorScheme.primary),
        const SizedBox(width: 16),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 6),
              Text(body, style: Theme.of(context).textTheme.bodyLarge),
            ],
          ),
        ),
      ],
    ),
  );
}

/// Original, code-native reference illustration. Not a patient image or model output.
class OralReferenceGraphic extends StatelessWidget {
  const OralReferenceGraphic({super.key, this.height = 220});
  final double height;
  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Semantics(
      image: true,
      label: context.l10n.guidePositionBody,
      child: Container(
        height: height,
        decoration: BoxDecoration(
          color: scheme.surfaceContainer,
          borderRadius: AppShapes.radiusMd,
        ),
        child: CustomPaint(
          painter: _ReferencePainter(
            scheme.primary,
            scheme.onSurfaceVariant,
            scheme.surface,
          ),
        ),
      ),
    );
  }
}

class _ReferencePainter extends CustomPainter {
  const _ReferencePainter(this.accent, this.line, this.surface);
  final Color accent, line, surface;
  @override
  void paint(Canvas canvas, Size size) {
    canvas.save();
    final scale = (size.width / 360).clamp(0.0, size.height / 230);
    canvas.translate(
      (size.width - 360 * scale) / 2,
      (size.height - 230 * scale) / 2,
    );
    canvas.scale(scale);
    final stroke = Paint()
      ..color = line.withValues(alpha: .65)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..strokeCap = StrokeCap.round;
    final face = Path()
      ..moveTo(98, 15)
      ..cubicTo(83, 68, 90, 162, 128, 199)
      ..quadraticBezierTo(180, 242, 232, 199)
      ..cubicTo(270, 162, 277, 68, 262, 15);
    canvas.drawPath(face, stroke);
    canvas.drawPath(
      Path()
        ..moveTo(166, 39)
        ..lineTo(156, 69)
        ..quadraticBezierTo(180, 80, 204, 69)
        ..lineTo(194, 39),
      stroke,
    );
    final mouth = Rect.fromCenter(
      center: const Offset(180, 131),
      width: 124,
      height: 82,
    );
    canvas.drawOval(mouth, Paint()..color = line.withValues(alpha: .18));
    canvas.drawOval(
      mouth,
      stroke
        ..color = accent
        ..strokeWidth = 3,
    );
    final teeth = Path()
      ..moveTo(129, 112)
      ..quadraticBezierTo(180, 129, 231, 112)
      ..lineTo(222, 101)
      ..quadraticBezierTo(180, 93, 138, 101)
      ..close();
    canvas.drawPath(teeth, Paint()..color = surface);
    canvas.drawPath(
      teeth,
      stroke
        ..color = line.withValues(alpha: .6)
        ..strokeWidth = 1,
    );
    for (final x in [153.0, 171.0, 189.0, 207.0]) {
      canvas.drawLine(Offset(x, 100), Offset(x, 121), stroke);
    }
    final tongue = Path()
      ..moveTo(143, 155)
      ..cubicTo(143, 129, 217, 129, 217, 155)
      ..quadraticBezierTo(180, 182, 143, 155);
    canvas.drawPath(tongue, Paint()..color = accent.withValues(alpha: .25));
    canvas.drawPath(tongue, stroke..color = accent.withValues(alpha: .7));
    canvas.drawLine(const Offset(180, 144), const Offset(180, 164), stroke);
    final target = RRect.fromRectAndRadius(
      const Rect.fromLTWH(101, 82, 158, 98),
      const Radius.circular(18),
    );
    canvas.drawRRect(
      target,
      stroke
        ..color = accent.withValues(alpha: .4)
        ..strokeWidth = 1,
    );
    for (final x in [101.0, 259.0]) {
      final dx = x == 101 ? 15.0 : -15.0;
      for (final y in [82.0, 180.0]) {
        final dy = y == 82 ? 15.0 : -15.0;
        canvas.drawPath(
          Path()
            ..moveTo(x + dx, y)
            ..lineTo(x, y)
            ..lineTo(x, y + dy),
          stroke
            ..color = accent
            ..strokeWidth = 3,
        );
      }
    }
    // Phone silhouette demonstrates upright orientation, separate from anatomy.
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        const Rect.fromLTWH(303, 75, 35, 65),
        const Radius.circular(6),
      ),
      stroke
        ..color = line
        ..strokeWidth = 1.5,
    );
    canvas.drawCircle(const Offset(320, 84), 2, Paint()..color = accent);
    canvas.drawLine(const Offset(313, 132), const Offset(328, 132), stroke);
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant _ReferencePainter old) =>
      accent != old.accent || line != old.line || surface != old.surface;
}

class OralFramingOverlay extends StatelessWidget {
  const OralFramingOverlay({super.key, required this.color});
  final Color color;
  @override
  Widget build(BuildContext context) => IgnorePointer(
    child: CustomPaint(
      painter: _FramingPainter(color),
      child: const SizedBox.expand(),
    ),
  );
}

class _FramingPainter extends CustomPainter {
  const _FramingPainter(this.color);
  final Color color;
  @override
  void paint(Canvas canvas, Size size) {
    final target = Rect.fromCenter(
      center: Offset(size.width / 2, size.height / 2),
      width: size.width * .76,
      height: size.height * .40,
    );
    final mask = Path()
      ..fillType = PathFillType.evenOdd
      ..addRect(Offset.zero & size)
      ..addOval(target);
    canvas.drawPath(mask, Paint()..color = Colors.black.withValues(alpha: .48));
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    canvas.drawOval(target, paint);
    final face = Rect.fromCenter(
      center: Offset(size.width / 2, size.height * .43),
      width: size.width * .88,
      height: size.height * .86,
    );
    canvas.drawArc(
      face,
      .12,
      2.9,
      false,
      paint
        ..color = Colors.white.withValues(alpha: .35)
        ..strokeWidth = 1,
    );
    canvas.drawLine(
      Offset(target.center.dx - 12, target.center.dy),
      Offset(target.center.dx + 12, target.center.dy),
      paint..color = color.withValues(alpha: .7),
    );
    canvas.drawLine(
      Offset(target.center.dx, target.center.dy - 12),
      Offset(target.center.dx, target.center.dy + 12),
      paint,
    );
  }

  @override
  bool shouldRepaint(covariant _FramingPainter old) => color != old.color;
}
