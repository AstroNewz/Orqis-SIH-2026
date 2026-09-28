import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:carescan/core/image/image_quality_service.dart';
import '../support/image_fixtures.dart';

void main() {
  test('measures real exposure, focus and resolution for a usable synthetic fixture', () {
    final report = inspectImageBytes(qualityFixture());
    expect(report.accepted, isTrue);
    expect(report.width, 256);
    expect(report.focusVariance, greaterThan(12));
    expect(report.meanLuminance, inInclusiveRange(40, 235));
  });
  test('rejects dark, glare, blur, undersized and corrupt input', () {
    expect(inspectImageBytes(qualityFixture(r: 10,g: 10,b: 10)).issue, ImageIssue.dark);
    expect(inspectImageBytes(qualityFixture(r: 250,g: 250,b: 250)).issue, ImageIssue.bright);
    expect(inspectImageBytes(qualityFixture(texture: false)).issue, ImageIssue.blurry);
    expect(inspectImageBytes(qualityFixture(width: 100)).issue, ImageIssue.resolution);
    expect(inspectImageBytes(Uint8List.fromList([1,2,3])).accepted, isFalse);
  });
  test('rejects textured neutral wall, blue sky and low-red hand-like color fixtures', () {
    for (final color in [(140,140,140),(80,140,200),(210,170,130)]) {
      final report = inspectImageBytes(qualityFixture(r: color.$1, g: color.$2, b: color.$3));
      expect(report.issue, ImageIssue.oralFraming);
    }
  });
  test('documents that a red object may pass; this is not object recognition', () {
    expect(inspectImageBytes(qualityFixture(r: 200,g: 80,b: 70)).accepted, isTrue);
  });
}
