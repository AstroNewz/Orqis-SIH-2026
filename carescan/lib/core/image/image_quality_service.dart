import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:carescan/core/errors/failures.dart';

enum ImageIssue {
  dark,
  bright,
  blurry,
  resolution,
  format,
  tooLarge,
  oralFraming,
  unreadable,
}

@immutable
class ImageQualityReport {
  const ImageQualityReport({
    this.issue,
    this.width = 0,
    this.height = 0,
    this.meanLuminance,
    this.focusVariance,
    this.clippedFraction,
    this.redTissueFraction,
  });
  final ImageIssue? issue;
  final int width, height;
  final double? meanLuminance,
      focusVariance,
      clippedFraction,
      redTissueFraction;
  bool get accepted => issue == null;
}

class ImageInputFailure extends Failure {
  const ImageInputFailure(this.issue) : super('Image input validation failed');
  final ImageIssue issue;
}

abstract interface class ImageInputValidator {
  Future<ImageQualityReport> inspect(String path);
}

class LocalImageInputValidator implements ImageInputValidator {
  const LocalImageInputValidator();
  static const maxBytes = 25 * 1024 * 1024;
  @override
  Future<ImageQualityReport> inspect(String path) async {
    try {
      if (!RegExp(
        r'\.(jpe?g|png|webp|bmp)$',
        caseSensitive: false,
      ).hasMatch(path)) {
        return const ImageQualityReport(issue: ImageIssue.format);
      }
      final file = File(path);
      final length = await file.length();
      if (length > maxBytes) {
        return const ImageQualityReport(issue: ImageIssue.tooLarge);
      }
      if (length == 0) {
        return const ImageQualityReport(issue: ImageIssue.unreadable);
      }
      final bytes = await file.readAsBytes();
      return await compute(inspectImageBytes, bytes);
    } catch (_) {
      return const ImageQualityReport(issue: ImageIssue.unreadable);
    }
  }
}

/// Engineering checks, NOT a clinical test or trained anatomy classifier.
/// Exposure/focus limits mirror backend/core/config.py (40..235, focus >=12).
/// Red chromatic fraction is a conservative *plausibility heuristic*. It can
/// reject usable images and accept red objects. User confirmation and the
/// backend's stronger acquisition checks remain required.
ImageQualityReport inspectImageBytes(Uint8List bytes) {
  if (bytes.length > LocalImageInputValidator.maxBytes) {
    return const ImageQualityReport(issue: ImageIssue.tooLarge);
  }
  try {
    final decoder = img.findDecoderForData(bytes);
    final info = decoder?.startDecode(bytes);
    if (info == null) return const ImageQualityReport(issue: ImageIssue.format);
    if (info.width * info.height > 24000000 ||
        info.width > 12000 ||
        info.height > 12000) {
      return const ImageQualityReport(issue: ImageIssue.tooLarge);
    }
    final decoded = decoder!.decodeFrame(0);
    if (decoded == null) {
      return const ImageQualityReport(issue: ImageIssue.unreadable);
    }
    final original = img.bakeOrientation(decoded);
    final width = original.width, height = original.height;
    if (width < 224 || height < 224) {
      return ImageQualityReport(
        issue: ImageIssue.resolution,
        width: width,
        height: height,
      );
    }
    final shortEdge = width < height ? width : height;
    final image = shortEdge > 512
        ? img.copyResize(
            original,
            width: (width * 512 / shortEdge).round(),
            height: (height * 512 / shortEdge).round(),
            interpolation: img.Interpolation.nearest,
          )
        : original;
    final w = image.width, h = image.height;
    final gray = Float64List(w * h);
    var clipped = 0, red = 0, centerCount = 0;
    double total = 0;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final p = image.getPixel(x, y);
        final r = p.r.toDouble(), g = p.g.toDouble(), b = p.b.toDouble();
        final luma = .299 * r + .587 * g + .114 * b;
        gray[y * w + x] = luma;
        total += luma;
        if (r <= .5 ||
            g <= .5 ||
            b <= .5 ||
            r >= 254.5 ||
            g >= 254.5 ||
            b >= 254.5) {
          clipped++;
        }
        if (x >= w ~/ 5 && x < w * 4 ~/ 5 && y >= h ~/ 5 && y < h * 4 ~/ 5) {
          centerCount++;
          if (r > 60 &&
              r > g * 1.2 &&
              r > b * 1.1 &&
              r / (r + g + b + 1) > .45) {
            red++;
          }
        }
      }
    }
    double sum = 0, squares = 0;
    var n = 0;
    for (var y = 1; y < h - 1; y++) {
      for (var x = 1; x < w - 1; x++) {
        final i = y * w + x;
        final v =
            gray[i - w] + gray[i + w] + gray[i - 1] + gray[i + 1] - 4 * gray[i];
        sum += v;
        squares += v * v;
        n++;
      }
    }
    final brightness = total / gray.length;
    final focus = squares / n - (sum / n) * (sum / n);
    final clipping = clipped / gray.length;
    final tissue = red / centerCount;
    final issue = brightness < 40
        ? ImageIssue.dark
        : brightness > 235
        ? ImageIssue.bright
        : clipping > .25
        ? (brightness >= 128 ? ImageIssue.bright : ImageIssue.dark)
        : focus < 12
        ? ImageIssue.blurry
        : tissue < .18
        ? ImageIssue.oralFraming
        : null;
    return ImageQualityReport(
      issue: issue,
      width: width,
      height: height,
      meanLuminance: brightness,
      focusVariance: focus,
      clippedFraction: clipping,
      redTissueFraction: tissue,
    );
  } catch (_) {
    return const ImageQualityReport(issue: ImageIssue.unreadable);
  }
}
