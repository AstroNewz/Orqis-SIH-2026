import 'failures.dart';

sealed class Result<T> {
  const Result();

  /// Functional fold to handle both success and error cases easily.
  R fold<R>(R Function(Failure failure) onError, R Function(T data) onSuccess) {
    return switch (this) {
      Success(value: final data) => onSuccess(data),
      Error(failure: final failure) => onError(failure),
    };
  }
}

class Success<T> extends Result<T> {
  final T value;

  const Success(this.value);

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is Success<T> && other.value == value;
  }

  @override
  int get hashCode => value.hashCode;
}

class Error<T> extends Result<T> {
  final Failure failure;

  const Error(this.failure);

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is Error<T> && other.failure == failure;
  }

  @override
  int get hashCode => failure.hashCode;
}
