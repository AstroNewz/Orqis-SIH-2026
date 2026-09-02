import 'package:flutter_test/flutter_test.dart';
import 'package:carescan/features/assessment/models/assessment.dart';

void main() {
  group('Assessment Model', () {
    final tTimestamp = DateTime.parse('2023-10-12T10:00:00.000Z');
    final tAssessment = Assessment(
      id: 'test-id',
      imagePath: '/path/to/image.jpg',
      timestamp: tTimestamp,
      type: 'Skin Scan • Full Body',
    );

    final tJson = {
      'id': 'test-id',
      'imagePath': '/path/to/image.jpg',
      'timestamp': '2023-10-12T10:00:00.000Z',
      'type': 'Skin Scan • Full Body',
    };

    test('fromJson should return a valid model', () {
      final result = Assessment.fromJson(tJson);
      expect(result, equals(tAssessment));
    });

    test('toJson should return a JSON map containing the proper data', () {
      final result = tAssessment.toJson();
      expect(result, equals(tJson));
    });

    test('copyWith should return a new object with updated values', () {
      final updated = tAssessment.copyWith(id: 'new-id');
      expect(updated.id, 'new-id');
      expect(updated.imagePath, tAssessment.imagePath);
      expect(updated.timestamp, tAssessment.timestamp);
      expect(updated.type, tAssessment.type);
    });
  });
}
