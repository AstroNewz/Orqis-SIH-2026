import 'package:camera/camera.dart';
import 'package:carescan/core/errors/async_state.dart';
import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/theme/app_spacing.dart';
import 'package:carescan/shared/widgets/error_state_widget.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

class CameraScanScreen extends StatefulWidget {
  const CameraScanScreen({super.key});

  @override
  State<CameraScanScreen> createState() => _CameraScanScreenState();
}

class _CameraScanScreenState extends State<CameraScanScreen>
    with SingleTickerProviderStateMixin {
  CameraController? _controller;
  List<CameraDescription> _cameras = [];
  bool _isRearCameraSelected = true;
  bool _isFlashOn = false;

  AsyncState<void> _state = const AsyncLoading();
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 1),
    )..repeat(reverse: true);
    _pulseAnimation = Tween<double>(begin: 0.5, end: 1.0).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
    _initCamera();
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _initCamera() async {
    setState(() => _state = const AsyncLoading());
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        setState(
          () => _state = const AsyncError(CameraFailure('No cameras found')),
        );
        return;
      }
      await _setupCameraController();
    } on CameraException catch (e) {
      if (e.code == 'CameraAccessDenied' ||
          e.code == 'CameraAccessDeniedWithoutPrompt') {
        setState(
          () => _state = const AsyncError(
            CameraFailure('Camera permission denied'),
          ),
        );
      } else {
        setState(() => _state = const AsyncError(CameraFailure()));
      }
    } catch (e) {
      setState(() => _state = const AsyncError(CameraFailure()));
    }
  }

  Future<void> _setupCameraController() async {
    final camera = _cameras.firstWhere(
      (c) => _isRearCameraSelected
          ? c.lensDirection == CameraLensDirection.back
          : c.lensDirection == CameraLensDirection.front,
      orElse: () => _cameras.first,
    );

    _controller = CameraController(
      camera,
      ResolutionPreset.high,
      enableAudio: false,
    );

    await _controller!.initialize();
    await _controller!.setFlashMode(
      _isFlashOn ? FlashMode.torch : FlashMode.off,
    );

    if (!mounted) return;
    setState(() => _state = const AsyncSuccess(null));
  }

  Future<void> _toggleCamera() async {
    if (_cameras.length < 2) return;
    _isRearCameraSelected = !_isRearCameraSelected;
    await _setupCameraController();
  }

  Future<void> _toggleFlash() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    _isFlashOn = !_isFlashOn;
    await _controller!.setFlashMode(
      _isFlashOn ? FlashMode.torch : FlashMode.off,
    );
    setState(() {});
  }

  Future<void> _captureImage() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (_controller!.value.isTakingPicture) return;

    try {
      final XFile file = await _controller!.takePicture();
      if (!mounted) return;
      // Navigate to Image Preview (T-SCREEN-03) and pass the file path
      context.push('/preview', extra: file.path);
    } catch (e) {
      // Show snackbar or handle error
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Failed to capture image')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: _state.when(
        initial: () => const SizedBox.shrink(),
        loading: () => const Center(
          child: CircularProgressIndicator(
            semanticsLabel: 'Initializing camera',
          ),
        ),
        success: (_) => _buildCameraUI(),
        empty: () => const SizedBox.shrink(),
        error: (failure) => Center(
          child: ErrorStateWidget(
            title: 'Camera Error',
            message: failure.message,
            onRetry: _initCamera,
          ),
        ),
      ),
    );
  }

  Widget _buildCameraUI() {
    final theme = Theme.of(context);

    return Stack(
      fit: StackFit.expand,
      children: [
        // Camera Preview
        if (_controller != null && _controller!.value.isInitialized)
          CameraPreview(_controller!),

        // Dark Overlay
        Container(color: Colors.black.withValues(alpha: 0.2)),

        // Top Navigation / Controls
        SafeArea(
          child: Align(
            alignment: Alignment.topCenter,
            child: Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: AppSpacing.md,
                vertical: AppSpacing.md,
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  _buildIconButton(
                    icon: Icons.close_rounded,
                    label: 'Go back',
                    onTap: () => context.pop(),
                  ),
                  Row(
                    children: [
                      _buildIconButton(
                        icon: _isFlashOn
                            ? Icons.flash_on_rounded
                            : Icons.flash_off_rounded,
                        label: 'Toggle Flash',
                        onTap: _toggleFlash,
                      ),
                      const SizedBox(width: AppSpacing.md),
                      _buildIconButton(
                        icon: Icons.help_outline_rounded,
                        label: 'Help',
                        onTap: () {},
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),

        // Main Scanning Area
        SafeArea(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Instruction Text
              Container(
                margin: const EdgeInsets.only(bottom: AppSpacing.xl),
                padding: const EdgeInsets.symmetric(
                  horizontal: AppSpacing.lg,
                  vertical: AppSpacing.sm,
                ),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.5),
                  borderRadius: BorderRadius.circular(32),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text(
                      'Position the area inside the frame',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Ensure good lighting for analysis',
                      style: TextStyle(
                        color: theme.colorScheme.primaryContainer,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),

              // Scanner Frame with Cutout
              Container(
                width: 300,
                height: 300,
                decoration: BoxDecoration(
                  border: Border.all(
                    color: Colors.white.withValues(alpha: 0.2),
                    width: 1,
                  ),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Stack(
                  children: [
                    _buildCorner(Alignment.topLeft),
                    _buildCorner(Alignment.topRight),
                    _buildCorner(Alignment.bottomLeft),
                    _buildCorner(Alignment.bottomRight),

                    // Scanning Line
                    Center(
                      child: AnimatedBuilder(
                        animation: _pulseAnimation,
                        builder: (context, child) {
                          return Opacity(
                            opacity: _pulseAnimation.value,
                            child: Container(
                              height: 2,
                              decoration: BoxDecoration(
                                color: theme.colorScheme.primaryContainer,
                                boxShadow: [
                                  BoxShadow(
                                    color: theme.colorScheme.primaryContainer
                                        .withValues(alpha: 0.8),
                                    blurRadius: 8,
                                  ),
                                ],
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                  ],
                ),
              ),

              // Document Type Indicator
              Padding(
                padding: const EdgeInsets.only(top: AppSpacing.xl),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 8,
                      ),
                      decoration: BoxDecoration(
                        color: theme.colorScheme.primaryContainer,
                        borderRadius: BorderRadius.circular(24),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.description,
                            size: 16,
                            color: theme.colorScheme.onPrimaryContainer,
                          ),
                          const SizedBox(width: 8),
                          Text(
                            'Document',
                            style: TextStyle(
                              color: theme.colorScheme.onPrimaryContainer,
                              fontWeight: FontWeight.w500,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 8,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.4),
                        borderRadius: BorderRadius.circular(24),
                        border: Border.all(
                          color: Colors.white.withValues(alpha: 0.3),
                        ),
                      ),
                      child: const Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.qr_code_scanner,
                            size: 16,
                            color: Colors.white,
                          ),
                          SizedBox(width: 8),
                          Text(
                            'QR Code',
                            style: TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w500,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),

        // Bottom Controls (Shutter)
        SafeArea(
          child: Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.only(bottom: AppSpacing.xl),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // Gallery Button
                  Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.4),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: Colors.white.withValues(alpha: 0.2),
                      ),
                    ),
                    child: const Icon(
                      Icons.photo_library_rounded,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(width: 32),
                  // Shutter Button
                  GestureDetector(
                    onTap: _captureImage,
                    child: Container(
                      width: 80,
                      height: 80,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.white, width: 4),
                      ),
                      child: Center(
                        child: Container(
                          width: 64,
                          height: 64,
                          decoration: const BoxDecoration(
                            color: Colors.white,
                            shape: BoxShape.circle,
                          ),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 32),
                  // Switch Camera Button
                  _buildIconButton(
                    icon: Icons.flip_camera_ios_rounded,
                    label: 'Switch Camera',
                    onTap: _toggleCamera,
                  ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildIconButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Semantics(
        label: label,
        button: true,
        child: Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(
            color: Colors.black.withValues(alpha: 0.4),
            shape: BoxShape.circle,
          ),
          child: Icon(icon, color: Colors.white),
        ),
      ),
    );
  }

  Widget _buildCorner(Alignment alignment) {
    final theme = Theme.of(context);
    final color = theme.colorScheme.primaryContainer;

    // Determine borders based on alignment
    final top =
        alignment == Alignment.topLeft || alignment == Alignment.topRight;
    final left =
        alignment == Alignment.topLeft || alignment == Alignment.bottomLeft;

    return Align(
      alignment: alignment,
      child: Transform.translate(
        offset: Offset(left ? -2 : 2, top ? -2 : 2),
        child: Container(
          width: 32,
          height: 32,
          decoration: BoxDecoration(
            border: Border(
              top: top ? BorderSide(color: color, width: 4) : BorderSide.none,
              bottom: !top
                  ? BorderSide(color: color, width: 4)
                  : BorderSide.none,
              left: left ? BorderSide(color: color, width: 4) : BorderSide.none,
              right: !left
                  ? BorderSide(color: color, width: 4)
                  : BorderSide.none,
            ),
            borderRadius: BorderRadius.only(
              topLeft: top && left ? const Radius.circular(8) : Radius.zero,
              topRight: top && !left ? const Radius.circular(8) : Radius.zero,
              bottomLeft: !top && left ? const Radius.circular(8) : Radius.zero,
              bottomRight: !top && !left
                  ? const Radius.circular(8)
                  : Radius.zero,
            ),
          ),
        ),
      ),
    );
  }
}
