import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/data/datasources/track_remote_data_source.dart';
import 'package:carescan/features/tracks/models/screening_track.dart';
import 'package:carescan/features/tracks/repositories/track_repository.dart';

/// Production implementation of [TrackRepository] over `GET /api/tracks`.
///
/// Thin by design: the descriptor parsing, the refusal to read a descriptor
/// without `frozen_test_evaluated`, and the record of what was rejected all live
/// in the models, where they are enforced once for every caller rather than
/// re-implemented per screen. This layer exists to turn thrown [Failure]s into
/// [Result]s so a widget never has to wrap a call in try/catch.
class ApiTrackRepository implements TrackRepository {
  final TrackRemoteDataSource _remoteDataSource;

  ApiTrackRepository({TrackRemoteDataSource? remoteDataSource})
    : _remoteDataSource = remoteDataSource ?? TrackRemoteDataSourceImpl();

  @override
  Future<Result<TrackCatalogue>> getTracks() async {
    try {
      final catalogue = await _remoteDataSource.getTracks();
      return Success(catalogue);
    } on Failure catch (failure) {
      return Error(failure);
    } catch (e) {
      return Error(UnknownFailure('Unexpected error: $e'));
    }
  }

  @override
  Future<Result<ScreeningTrack>> getTrack(String trackId) async {
    try {
      final track = await _remoteDataSource.getTrack(trackId);
      return Success(track);
    } on Failure catch (failure) {
      return Error(failure);
    } catch (e) {
      return Error(UnknownFailure('Unexpected error: $e'));
    }
  }
}
