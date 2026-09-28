/// Client-side view of `POST /api/localize` -- the non-gating ROI overlay (DEC-034).
///
/// Localisation is a *visual aid*, never part of the screening verdict: the UI draws
/// a box when [localized] is true and shows nothing otherwise. So this model carries
/// only what the overlay needs, and a missing or malformed field degrades to "no box"
/// rather than an error. Coordinates are resolution-independent: [boxNormalised] is
/// `[x0, y0, x1, y1]` in `[0, 1]`, mapped against [sourceWidth] / [sourceHeight] at
/// draw time so the box lands correctly whatever the display size.
class LocalizationResult {
  final String status;
  final bool localized;
  final double confidence;
  final List<double>? boxNormalised;
  final List<int>? roiBoxPixels;
  final int sourceWidth;
  final int sourceHeight;
  final String? roiSource;
  final List<String> reasons;
  final String localizerVersion;

  const LocalizationResult({
    required this.status,
    required this.localized,
    required this.confidence,
    required this.sourceWidth,
    required this.sourceHeight,
    this.boxNormalised,
    this.roiBoxPixels,
    this.roiSource,
    this.reasons = const [],
    this.localizerVersion = '',
  });

  /// A result that draws nothing -- used by the mock repository and as a safe
  /// fallback so the overlay is always optional.
  const LocalizationResult.none()
    : status = 'rejected',
      localized = false,
      confidence = 0.0,
      boxNormalised = null,
      roiBoxPixels = null,
      sourceWidth = 0,
      sourceHeight = 0,
      roiSource = null,
      reasons = const [],
      localizerVersion = '';

  /// True only when there is a well-formed box the overlay can actually draw.
  bool get hasDrawableBox =>
      localized &&
      boxNormalised != null &&
      boxNormalised!.length == 4 &&
      sourceWidth > 0 &&
      sourceHeight > 0;

  factory LocalizationResult.fromJson(Map<String, dynamic> json) {
    return LocalizationResult(
      status: (json['status'] as String?) ?? 'rejected',
      localized: (json['localized'] as bool?) ?? false,
      confidence: _toDouble(json['confidence']) ?? 0.0,
      boxNormalised: _toDoubleList(json['boxNormalised']),
      roiBoxPixels: _toIntList(json['roiBoxPixels']),
      sourceWidth: (json['sourceWidth'] as num?)?.toInt() ?? 0,
      sourceHeight: (json['sourceHeight'] as num?)?.toInt() ?? 0,
      roiSource: json['roiSource'] as String?,
      reasons:
          (json['reasons'] as List?)?.map((e) => e.toString()).toList() ??
          const [],
      localizerVersion: (json['localizerVersion'] as String?) ?? '',
    );
  }
}

double? _toDouble(dynamic value) {
  if (value == null) return null;
  if (value is num) return value.toDouble();
  if (value is String) return double.tryParse(value);
  return null;
}

List<double>? _toDoubleList(dynamic value) {
  if (value is! List) return null;
  final out = <double>[];
  for (final e in value) {
    final d = _toDouble(e);
    if (d == null) return null; // a malformed box is no box, not a partial one
    out.add(d);
  }
  return out;
}

List<int>? _toIntList(dynamic value) {
  if (value is! List) return null;
  final out = <int>[];
  for (final e in value) {
    if (e is num) {
      out.add(e.toInt());
    } else {
      return null;
    }
  }
  return out;
}
