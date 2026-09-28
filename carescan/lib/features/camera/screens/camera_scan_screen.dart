import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:image_picker/image_picker.dart';
import 'package:carescan/core/theme/app_shapes.dart';
import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/l10n/l10n.dart';
import 'package:carescan/features/camera/widgets/oral_capture_guide.dart';

class CameraScanScreen extends StatefulWidget {
  const CameraScanScreen({super.key});
  @override
  State<CameraScanScreen> createState() => _CameraScanScreenState();
}

class _CameraScanScreenState extends State<CameraScanScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  List<CameraDescription> _cameras = [];
  int _cameraIndex = 0;
  int _generation = 0;
  bool _loading = true,
      _denied = false,
      _failed = false,
      _busy = false,
      _torch = false;
  bool _helper = true, _away = false;
  double? _luminance;
  DateTime _lastSample = DateTime.fromMillisecondsSinceEpoch(0);

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize();
  }

  @override
  void dispose() {
    ++_generation;
    WidgetsBinding.instance.removeObserver(this);
    unawaited(_release());
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      ++_generation;
      unawaited(_release());
    } else if (state == AppLifecycleState.resumed && !_away && !_busy) {
      _initialize();
    }
  }

  Future<void> _release() async {
    final controller = _controller;
    _controller = null;
    if (controller != null) {
      try {
        await controller.dispose();
      } catch (_) {
        /* Already released by the platform. */
      }
    }
  }

  Future<void> _initialize() async {
    final generation = ++_generation;
    if (!mounted) return;
    setState(() {
      _loading = true;
      _failed = false;
      _denied = false;
      _luminance = null;
      _torch = false;
    });
    await _release();
    try {
      if (_cameras.isEmpty) _cameras = await availableCameras();
      if (!mounted || generation != _generation) return;
      if (_cameras.isEmpty) throw CameraException('NoCamera', '');
      final controller = CameraController(
        _cameras[_cameraIndex % _cameras.length],
        ResolutionPreset.high,
        enableAudio: false,
      );
      _controller = controller;
      await controller.initialize();
      if (!mounted || generation != _generation) return;
      try {
        await controller.setFlashMode(FlashMode.off);
      } catch (_) {}
      try {
        await controller.startImageStream((frame) {
          if (!mounted || generation != _generation || _busy) return;
          final now = DateTime.now();
          if (now.difference(_lastSample).inMilliseconds < 500) return;
          _lastSample = now;
          final measured = _measureLight(frame);
          if (measured != null) setState(() => _luminance = measured);
        });
      } catch (_) {
        /* Some platforms do not support streams; guide remains manual. */
      }
      if (mounted && generation == _generation) {
        setState(() => _loading = false);
      }
    } on CameraException catch (error) {
      if (mounted && generation == _generation) {
        setState(() {
          _loading = false;
          _failed = true;
          _denied =
              error.code.contains('AccessDenied') ||
              error.code.contains('Restricted');
        });
      }
    } catch (_) {
      if (mounted && generation == _generation) {
        setState(() {
          _loading = false;
          _failed = true;
        });
      }
    }
  }

  /// Samples actual camera luminance in the central region. No anatomy or distance inference.
  double? _measureLight(CameraImage frame) {
    if (frame.planes.isEmpty) return null;
    final plane = frame.planes.first;
    final bgra = frame.format.group == ImageFormatGroup.bgra8888;
    if (!bgra && frame.format.group != ImageFormatGroup.yuv420) return null;
    double sum = 0;
    var count = 0;
    final stride = plane.bytesPerPixel ?? (bgra ? 4 : 1);
    for (var y = frame.height ~/ 3; y < frame.height * 2 ~/ 3; y += 12) {
      for (var x = frame.width ~/ 4; x < frame.width * 3 ~/ 4; x += 12) {
        final offset = y * plane.bytesPerRow + x * stride;
        if (offset + (bgra ? 2 : 0) >= plane.bytes.length) continue;
        sum += bgra
            ? .114 * plane.bytes[offset] +
                  .587 * plane.bytes[offset + 1] +
                  .299 * plane.bytes[offset + 2]
            : plane.bytes[offset];
        count++;
      }
    }
    return count == 0 ? null : sum / count;
  }

  Future<void> _showPreview(String path) async {
    _away = true;
    ++_generation;
    await _release();
    if (!mounted) return;
    await context.push('/preview', extra: path);
    _away = false;
    if (mounted) await _initialize();
  }

  Future<void> _capture() async {
    final controller = _controller;
    if (_busy || controller == null || !controller.value.isInitialized) return;
    setState(() => _busy = true);
    try {
      if (controller.value.isStreamingImages) {
        await controller.stopImageStream();
      }
      final file = await controller.takePicture();
      if (mounted) await _showPreview(file.path);
    } catch (_) {
      if (mounted) {
        _message(context.l10n.captureFailed);
        await _initialize();
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _gallery() async {
    if (_busy) return;
    setState(() => _busy = true);
    _away = true;
    ++_generation;
    await _release();
    try {
      final file = await ImagePicker().pickImage(
        source: ImageSource.gallery,
        requestFullMetadata: false,
      );
      if (file != null && mounted) {
        await _showPreview(file.path);
      } else {
        _away = false;
        if (mounted) await _initialize();
      }
    } catch (_) {
      _away = false;
      if (mounted) {
        _message(context.l10n.galleryFailed);
        await _initialize();
      }
    } finally {
      _away = false;
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _flash() async {
    if (_busy || _controller == null) return;
    try {
      await _controller!.setFlashMode(_torch ? FlashMode.off : FlashMode.torch);
      if (mounted) setState(() => _torch = !_torch);
    } catch (_) {
      if (mounted) _message(context.l10n.flashUnavailable);
    }
  }

  void _message(String message) =>
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text(message)));

  @override
  Widget build(BuildContext context) => Theme(
    data: AppTheme.dark,
    child: Builder(builder: _buildCamera),
  );
  Widget _buildCamera(BuildContext context) {
    final l = context.l10n;
    final theme = Theme.of(context);
    final light = _luminance;
    final tooDark = light != null && light < 40;
    final tooBright = light != null && light > 235;
    final lightGood = light != null && !tooDark && !tooBright;
    final color = tooDark || tooBright
        ? theme.colorScheme.tertiary
        : lightGood
        ? theme.colorScheme.primary
        : theme.colorScheme.onSurfaceVariant;
    final readyText = tooDark
        ? l.moreLight
        : tooBright
        ? l.lessLight
        : lightGood
        ? l.readinessGood
        : l.manualReady;
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          tooltip: l.back,
          icon: const Icon(Icons.close),
          onPressed: () => context.canPop() ? context.pop() : context.go('/'),
        ),
        title: Text(l.captureTitle),
        actions: [
          IconButton(
            tooltip: l.howToCapture,
            onPressed: () => showCaptureGuide(context),
            icon: const Icon(Icons.help_outline),
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: Stack(
                fit: StackFit.expand,
                children: [
                  if (_loading)
                    Center(
                      child: CircularProgressIndicator(
                        semanticsLabel: l.cameraLoading,
                      ),
                    )
                  else if (_failed)
                    SingleChildScrollView(
                      child: Padding(
                        padding: const EdgeInsets.all(24),
                        child: Column(
                          children: [
                            const SizedBox(height: 32),
                            const Icon(Icons.no_photography_outlined, size: 48),
                            const SizedBox(height: 16),
                            Text(
                              l.cameraError,
                              style: theme.textTheme.titleLarge,
                            ),
                            const SizedBox(height: 12),
                            Text(
                              _denied ? l.cameraPermission : l.cameraFailure,
                              textAlign: TextAlign.center,
                            ),
                            const SizedBox(height: 20),
                            FilledButton(
                              onPressed: _initialize,
                              child: Text(l.retry),
                            ),
                          ],
                        ),
                      ),
                    )
                  else ...[
                    if (_controller case final controller?)
                      Center(
                        child: AspectRatio(
                          aspectRatio: controller.value.isInitialized
                              ? 1 / controller.value.aspectRatio
                              : .75,
                          child: CameraPreview(controller),
                        ),
                      ),
                    OralFramingOverlay(color: color),
                    Align(
                      alignment: Alignment.topCenter,
                      child: Padding(
                        padding: const EdgeInsets.all(16),
                        child: Text(
                          l.centerMouth,
                          textAlign: TextAlign.center,
                          style: theme.textTheme.titleMedium,
                        ),
                      ),
                    ),
                  ],
                  if (_helper && !_failed && !_loading)
                    Align(
                      alignment: Alignment.bottomCenter,
                      child: Container(
                        margin: const EdgeInsets.all(12),
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: theme.colorScheme.surface,
                          borderRadius: AppShapes.radiusMd,
                        ),
                        child: Row(
                          children: [
                            const SizedBox(
                              width: 84,
                              child: OralReferenceGraphic(height: 74),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: TextButton(
                                onPressed: () => showCaptureGuide(context),
                                child: Text(l.howToCapture),
                              ),
                            ),
                            IconButton(
                              tooltip: l.close,
                              onPressed: () => setState(() => _helper = false),
                              icon: const Icon(Icons.close),
                            ),
                          ],
                        ),
                      ),
                    ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
              child: Semantics(
                liveRegion: true,
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      lightGood
                          ? Icons.check_circle_outline
                          : Icons.info_outline,
                      size: 18,
                      color: color,
                    ),
                    const SizedBox(width: 8),
                    Flexible(
                      child: Text(
                        readyText,
                        textAlign: TextAlign.center,
                        style: theme.textTheme.titleSmall?.copyWith(
                          color: color,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Text(
                l.manualFraming,
                textAlign: TextAlign.center,
                style: theme.textTheme.bodySmall,
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  IconButton(
                    tooltip: l.gallery,
                    onPressed: _busy ? null : _gallery,
                    icon: const Icon(Icons.photo_library_outlined),
                  ),
                  IconButton(
                    tooltip: l.flash,
                    onPressed: _busy || _loading || _failed ? null : _flash,
                    icon: Icon(_torch ? Icons.flash_on : Icons.flash_off),
                  ),
                  Semantics(
                    label: l.capture,
                    button: true,
                    child: SizedBox.square(
                      dimension: 72,
                      child: FilledButton(
                        style: FilledButton.styleFrom(
                          shape: const CircleBorder(),
                          padding: EdgeInsets.zero,
                        ),
                        onPressed:
                            _busy || _loading || _failed || tooDark || tooBright
                            ? null
                            : _capture,
                        child: _busy
                            ? const SizedBox.square(
                                dimension: 24,
                                child: CircularProgressIndicator(),
                              )
                            : const Icon(Icons.camera_alt_outlined, size: 30),
                      ),
                    ),
                  ),
                  IconButton(
                    tooltip: l.flipCamera,
                    onPressed: _busy || _cameras.length < 2
                        ? null
                        : () {
                            _cameraIndex++;
                            _initialize();
                          },
                    icon: const Icon(Icons.flip_camera_android_outlined),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
