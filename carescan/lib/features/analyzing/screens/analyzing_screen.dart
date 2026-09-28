import 'dart:async';
import 'dart:math';

import 'package:carescan/core/di/service_locator.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/core/theme/app_spacing.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/assessment/repositories/assessment_repository.dart';
import 'package:carescan/shared/widgets/error_state_widget.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

class AnalyzingScreen extends StatefulWidget {
  final String? imagePath;
  final AssessmentRepository? repository;

  const AnalyzingScreen({super.key, this.imagePath, this.repository});

  @override
  State<AnalyzingScreen> createState() => _AnalyzingScreenState();
}

class _AnalyzingScreenState extends State<AnalyzingScreen>
    with TickerProviderStateMixin {
  late final AssessmentRepository _repository;
  late AnimationController _pulseController;
  late AnimationController _spinController;

  Timer? _analysisTimer;
  bool _hasError = false;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _repository = widget.repository ?? appAssessmentRepository;
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2000),
    )..repeat();

    _spinController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 8),
    )..repeat();

    _startAnalysis();
  }

  void _startAnalysis() {
    setState(() {
      _hasError = false;
      _errorMessage = null;
    });
    _analysisTimer?.cancel();

    // Simulate analyzing time and submit to repository
    _analysisTimer = Timer(const Duration(seconds: 2), () async {
      final Result<AssessmentResult> result = await _repository
          .submitAssessment(widget.imagePath ?? '');

      if (!mounted) return;

      result.fold(
        (failure) {
          setState(() {
            _hasError = true;
            _errorMessage = failure.message;
          });
        },
        (assessmentResult) {
          // Carry the on-device capture forward so the result screen can show the
          // exact image that was analyzed, alongside the persisted verdict.
          context.push(
            '/result',
            extra: (result: assessmentResult, imagePath: widget.imagePath),
          );
        },
      );
    });
  }

  @override
  void dispose() {
    _analysisTimer?.cancel();
    _pulseController.dispose();
    _spinController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_hasError) {
      return Scaffold(
        body: Center(
          child: ErrorStateWidget(
            title: 'Analysis Failed',
            message:
                _errorMessage ?? 'There was an issue processing your image.',
            onRetry: _startAnalysis,
          ),
        ),
      );
    }

    final theme = Theme.of(context);

    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
      body: SafeArea(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppSpacing.xl),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // Animated Progress Indicator
                SizedBox(
                  width: 192,
                  height: 192,
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      // Pulsing Rings
                      _buildPulseRing(
                        0.0,
                        theme.colorScheme.primary.withValues(alpha: 0.2),
                      ),
                      _buildPulseRing(
                        0.4,
                        theme.colorScheme.primary.withValues(alpha: 0.3),
                      ),
                      _buildPulseRing(
                        0.8,
                        theme.colorScheme.primary.withValues(alpha: 0.4),
                      ),

                      // Rotating Dash Ring
                      RotationTransition(
                        turns: _spinController,
                        child: SizedBox(
                          width: 192,
                          height: 192,
                          child: CustomPaint(
                            painter: _DashedCirclePainter(
                              color: theme.colorScheme.primaryContainer,
                            ),
                          ),
                        ),
                      ),

                      // Central Icon Container
                      Container(
                        width: 96,
                        height: 96,
                        decoration: BoxDecoration(
                          color: theme.colorScheme.surface,
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: theme.colorScheme.surfaceContainerHighest,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withValues(alpha: 0.04),
                              blurRadius: 20,
                              offset: const Offset(0, 4),
                            ),
                          ],
                        ),
                        child: Icon(
                          Icons.image_search_rounded,
                          size: 40,
                          color: theme.colorScheme.primary,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: AppSpacing.xl),
                // Text Content
                Text(
                  'Analyzing your image...',
                  style: theme.textTheme.headlineMedium?.copyWith(
                    color: theme.colorScheme.primary,
                    fontWeight: FontWeight.w600,
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: AppSpacing.sm),
                Text(
                  'This may take a few moments. We are ensuring a thorough assessment.',
                  style: theme.textTheme.bodyLarge?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                    height: 1.5,
                  ),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildPulseRing(double delay, Color color) {
    return AnimatedBuilder(
      animation: _pulseController,
      builder: (context, child) {
        // Calculate progress accounting for delay
        double rawProgress = _pulseController.value - delay;
        if (rawProgress < 0) rawProgress += 1.0;

        // Opacity fades out as it expands
        final opacity = 1.0 - rawProgress;
        // Size expands from center
        final size = 96 + (96 * rawProgress); // From inner circle to full 192

        return Opacity(
          opacity: opacity,
          child: Container(
            width: size,
            height: size,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(color: color, width: 4),
            ),
          ),
        );
      },
    );
  }
}

class _DashedCirclePainter extends CustomPainter {
  final Color color;

  _DashedCirclePainter({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final radius = size.width / 2;
    final center = Offset(radius, radius);
    final circumference = 2 * pi * radius;

    // Dash and gap lengths
    const dashLength = 10.0;
    const gapLength = 15.0;
    final totalLength = dashLength + gapLength;

    final dashCount = (circumference / totalLength).floor();
    final sweepAngle = (dashLength / circumference) * 2 * pi;
    final gapAngle = (gapLength / circumference) * 2 * pi;

    for (var i = 0; i < dashCount; i++) {
      final startAngle = i * (sweepAngle + gapAngle);
      canvas.drawArc(
        Rect.fromCircle(center: center, radius: radius - 4),
        startAngle,
        sweepAngle,
        false,
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _DashedCirclePainter oldDelegate) {
    return color != oldDelegate.color;
  }
}
