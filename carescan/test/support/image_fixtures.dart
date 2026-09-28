import 'dart:math';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

/// Synthetic color/texture fixture for engineering thresholds; never patient data.
Uint8List qualityFixture({int width = 256, int height = 256, int r = 180, int g = 75, int b = 90, bool texture = true}) {
  final image = img.Image(width: width, height: height);
  final random = Random(7);
  for (final pixel in image) {
    final noise = texture ? random.nextInt(31) - 15 : 0;
    pixel.setRgb((r + noise).clamp(0,255), (g + noise).clamp(0,255), (b + noise).clamp(0,255));
  }
  return Uint8List.fromList(img.encodePng(image));
}
