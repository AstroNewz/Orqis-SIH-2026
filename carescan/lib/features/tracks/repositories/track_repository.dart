import 'package:carescan/core/errors/result.dart';
import 'package:carescan/features/tracks/models/screening_track.dart';

/// Reads the platform's screening track catalogue.
///
/// Read-only on purpose. Scans still run through the assessment repository and
/// `/api/screening/analyze`; this exists so the client can *see* the platform —
/// which conditions exist, which are ready, and what validation stands behind
/// each published number — without re-routing the path the product depends on.
abstract class TrackRepository {
  /// Every track the backend knows about, ready or not.
  ///
  /// A returned [TrackCatalogue] may carry a non-empty `rejected` list: some
  /// descriptors were unreadable while others were fine. That is a partial
  /// success, not a failure, and it is deliberately not collapsed into an
  /// [Error] — hiding the readable tracks because one was malformed would make
  /// the platform look smaller than it is.
  Future<Result<TrackCatalogue>> getTracks();

  /// One track by id.
  Future<Result<ScreeningTrack>> getTrack(String trackId);
}
