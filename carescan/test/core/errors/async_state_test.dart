import 'package:carescan/core/errors/async_state.dart';
import 'package:carescan/core/errors/failures.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('AsyncState', () {
    test('AsyncSuccess should store data and support equality', () {
      const state1 = AsyncSuccess<String>('Data');
      const state2 = AsyncSuccess<String>('Data');
      const state3 = AsyncSuccess<String>('Different');

      expect(state1.data, 'Data');
      expect(state1, equals(state2));
      expect(state1, isNot(equals(state3)));
    });

    test('AsyncError should store failure and support equality', () {
      const state1 = AsyncError<String>(ServerFailure());
      const state2 = AsyncError<String>(ServerFailure());
      const state3 = AsyncError<String>(CacheFailure());

      expect(state1.failure, isA<ServerFailure>());
      expect(state1, equals(state2));
      expect(state1, isNot(equals(state3)));
    });

    test('when should route to correct callback', () {
      final List<AsyncState<int>> states = [
        const AsyncInitial(),
        const AsyncLoading(),
        const AsyncSuccess(42),
        const AsyncEmpty(),
        const AsyncError(ServerFailure()),
      ];

      final results = states.map((state) {
        return state.when(
          initial: () => 'Initial',
          loading: () => 'Loading',
          success: (data) => 'Success $data',
          empty: () => 'Empty',
          error: (failure) => 'Error ${failure.message}',
        );
      }).toList();

      expect(results, [
        'Initial',
        'Loading',
        'Success 42',
        'Empty',
        'Error Server Error',
      ]);
    });
  });
}
