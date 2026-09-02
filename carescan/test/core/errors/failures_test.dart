import 'package:carescan/core/errors/failures.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('Failures', () {
    test('ServerFailure has correct default message and equality', () {
      const failure1 = ServerFailure();
      const failure2 = ServerFailure();
      const failure3 = ServerFailure('Custom server error');

      expect(failure1.message, 'Server Error');
      expect(failure1, equals(failure2));
      expect(failure1, isNot(equals(failure3)));
    });

    test('CacheFailure has correct default message', () {
      const failure = CacheFailure();
      expect(failure.message, 'Cache Error');
    });

    test('ValidationFailure has correct default message', () {
      const failure = ValidationFailure();
      expect(failure.message, 'Validation Error');
    });

    test('CameraFailure has correct default message', () {
      const failure = CameraFailure();
      expect(failure.message, 'Camera Error');
    });

    test('UnknownFailure has correct default message', () {
      const failure = UnknownFailure();
      expect(failure.message, 'Unknown Error');
    });
  });
}
