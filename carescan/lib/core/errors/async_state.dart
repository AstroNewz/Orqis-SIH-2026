import 'failures.dart';

sealed class AsyncState<T> {
  const AsyncState();

  /// Helper to safely pattern match against the states.
  R when<R>({
    required R Function() initial,
    required R Function() loading,
    required R Function(T data) success,
    required R Function() empty,
    required R Function(Failure failure) error,
  }) {
    return switch (this) {
      AsyncInitial() => initial(),
      AsyncLoading() => loading(),
      AsyncSuccess(data: final data) => success(data),
      AsyncEmpty() => empty(),
      AsyncError(failure: final failure) => error(failure),
    };
  }
}

class AsyncInitial<T> extends AsyncState<T> {
  const AsyncInitial();
}

class AsyncLoading<T> extends AsyncState<T> {
  const AsyncLoading();
}

class AsyncSuccess<T> extends AsyncState<T> {
  final T data;

  const AsyncSuccess(this.data);

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is AsyncSuccess<T> && other.data == data;
  }

  @override
  int get hashCode => data.hashCode;
}

class AsyncEmpty<T> extends AsyncState<T> {
  const AsyncEmpty();
}

class AsyncError<T> extends AsyncState<T> {
  final Failure failure;

  const AsyncError(this.failure);

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is AsyncError<T> && other.failure == failure;
  }

  @override
  int get hashCode => failure.hashCode;
}
