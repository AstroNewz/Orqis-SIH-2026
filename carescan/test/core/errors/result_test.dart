import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('Result', () {
    test('Success should store value and support equality', () {
      const result1 = Success<int>(42);
      const result2 = Success<int>(42);
      const result3 = Success<int>(10);

      expect(result1.value, 42);
      expect(result1, equals(result2));
      expect(result1, isNot(equals(result3)));
    });

    test('Error should store failure and support equality', () {
      const result1 = Error<int>(ServerFailure());
      const result2 = Error<int>(ServerFailure());
      const result3 = Error<int>(CacheFailure());

      expect(result1.failure, isA<ServerFailure>());
      expect(result1, equals(result2));
      expect(result1, isNot(equals(result3)));
    });

    test('fold should correctly handle Success', () {
      const Result<int> result = Success(42);

      final output = result.fold(
        (failure) => 'Error',
        (data) => 'Success $data',
      );

      expect(output, 'Success 42');
    });

    test('fold should correctly handle Error', () {
      const Result<int> result = Error(ServerFailure());

      final output = result.fold(
        (failure) => 'Error ${failure.message}',
        (data) => 'Success $data',
      );

      expect(output, 'Error Server Error');
    });
  });
}
