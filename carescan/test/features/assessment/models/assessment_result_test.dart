import 'package:flutter_test/flutter_test.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';

void main() {
  group('AssessmentResult Model', () {
    const tResult = AssessmentResult(
      id: 'result-id',
      assessmentId: 'test-id',
      riskLevel: 'Low Risk',
      details: 'Analysis complete.',
    );

    final tJson = {
      'id': 'result-id',
      'assessmentId': 'test-id',
      'riskLevel': 'Low Risk',
      'details': 'Analysis complete.',
    };

    test('fromJson should return a valid model', () {
      final result = AssessmentResult.fromJson(tJson);
      expect(result, equals(tResult));
    });

    test('toJson should return a JSON map containing the proper data', () {
      final result = tResult.toJson();
      expect(result, equals(tJson));
    });

    test('copyWith should return a new object with updated values', () {
      final updated = tResult.copyWith(riskLevel: 'High Risk');
      expect(updated.riskLevel, 'High Risk');
      expect(updated.id, tResult.id);
      expect(updated.assessmentId, tResult.assessmentId);
      expect(updated.details, tResult.details);
    });
  });
}
