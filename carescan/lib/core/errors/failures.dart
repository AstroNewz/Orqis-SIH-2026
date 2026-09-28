/// Base type for every recoverable error surfaced across a layer boundary.
///
/// Implements [Exception] because data sources *throw* failures (see
/// `AssessmentRemoteDataSource`), which repositories catch via `on Failure`.
/// Declaring the relationship lets `Exception`-typed test doubles and error
/// channels hold a failure without a cast.
abstract class Failure implements Exception {
  final String message;

  const Failure(this.message);

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is Failure && other.message == message;
  }

  @override
  int get hashCode => message.hashCode;

  @override
  String toString() => '$runtimeType: $message';
}

class ServerFailure extends Failure {
  const ServerFailure([super.message = 'Server Error']);
}

class CacheFailure extends Failure {
  const CacheFailure([super.message = 'Cache Error']);
}

class ValidationFailure extends Failure {
  const ValidationFailure([super.message = 'Validation Error']);
}

class CameraFailure extends Failure {
  const CameraFailure([super.message = 'Camera Error']);
}

class NetworkFailure extends Failure {
  const NetworkFailure([super.message = 'Network connection failure']);
}

class UnknownFailure extends Failure {
  const UnknownFailure([super.message = 'Unknown Error']);
}
