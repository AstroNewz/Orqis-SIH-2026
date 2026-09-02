import 'package:flutter_test/flutter_test.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/features/assessment/repositories/mock_assessment_repository.dart';

void main() {
  group('MockAssessmentRepository', () {
    late MockAssessmentRepository repository;

    setUp(() {
      repository = MockAssessmentRepository();
    });

    test('submitAssessment returns Success with test data', () async {
      final result = await repository.submitAssessment('test/path');

      expect(result, isA<Success>());
      result.fold((failure) => fail('Expected Success, got Error'), (data) {
        expect(data.riskLevel, contains('MOCK'));
        expect(data.details, contains('TEST DATA'));
      });
    });

    test(
      'getAssessmentHistory returns Success with list of test entries',
      () async {
        final result = await repository.getAssessmentHistory();

        expect(result, isA<Success>());
        result.fold((failure) => fail('Expected Success, got Error'), (data) {
          expect(data, isNotEmpty);
          expect(data.first.assessment.type, contains('TEST DATA'));
          expect(data.first.result.riskLevel, contains('MOCK'));
        });
      },
    );
  });
}
