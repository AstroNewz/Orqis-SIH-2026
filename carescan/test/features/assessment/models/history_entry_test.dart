import 'package:flutter_test/flutter_test.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';
import 'package:carescan/features/assessment/models/assessment.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';

void main() {
  group('HistoryEntry Model', () {
    final tTimestamp = DateTime.parse('2023-10-12T10:00:00.000Z');

    final tAssessment = Assessment(
      id: 'test-id',
      imagePath: '/path/to/image.jpg',
      timestamp: tTimestamp,
      type: 'Skin Scan • Full Body',
    );

    const tResult = AssessmentResult(
      id: 'result-id',
      assessmentId: 'test-id',
      riskLevel: 'Low Risk',
      details: 'Analysis complete.',
    );

    final tHistoryEntry = HistoryEntry(
      assessment: tAssessment,
      result: tResult,
    );

    final tJson = {
      'assessment': {
        'id': 'test-id',
        'imagePath': '/path/to/image.jpg',
        'timestamp': '2023-10-12T10:00:00.000Z',
        'type': 'Skin Scan • Full Body',
      },
      'result': {
        'id': 'result-id',
        'assessmentId': 'test-id',
        'riskLevel': 'Low Risk',
        'details': 'Analysis complete.',
        // Design-A verdict fields (DEC-034), null when absent so toJson round-trips.
        'primaryModel': null,
        'primaryRiskLevel': null,
        'primaryProbability': null,
        'primaryThreshold': null,
        'primaryCalibrated': null,
        'finalProbability': null,
        'quantumProbability': null,
        'classicalProbability': null,
        'classification': null,
        'modelVersion': null,
        'isMock': false,
      },
    };

    test('fromJson should return a valid model', () {
      final result = HistoryEntry.fromJson(tJson);
      expect(result, equals(tHistoryEntry));
    });

    test('toJson should return a JSON map containing the proper data', () {
      final result = tHistoryEntry.toJson();
      expect(result, equals(tJson));
    });
  });
}
