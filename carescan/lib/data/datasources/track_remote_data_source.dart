import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:carescan/core/constants/api_constants.dart';
import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/features/tracks/models/screening_track.dart';

/// Reads the backend's platform track catalogue (`GET /api/tracks`).
///
/// Read surface only, and deliberately so. The analyze endpoints on
/// `/api/tracks/{id}/analyze` are not wired here: the shipping capture flow already
/// runs through `/api/screening/analyze`, and re-routing the one path the product
/// depends on would be a destructive replacement of working code rather than the
/// additive platform surface DEC-045 describes. What was missing was not another way
/// to score an image — it was any way for the client to *see* the platform: which
/// conditions exist, which are ready, and what validation each number actually has.
abstract class TrackRemoteDataSource {
  /// Lists every screening track the backend knows about, ready or not.
  Future<TrackCatalogue> getTracks();

  /// Fetches one track by id.
  ///
  /// Throws a [ValidationFailure] when no such track exists, distinct from the
  /// [ServerFailure] raised when the track exists but the server cannot describe it.
  Future<ScreeningTrack> getTrack(String trackId);
}

class TrackRemoteDataSourceImpl implements TrackRemoteDataSource {
  final HttpClient _httpClient;

  TrackRemoteDataSourceImpl({HttpClient? httpClient})
    : _httpClient = httpClient ?? HttpClient() {
    _httpClient.connectionTimeout = ApiConstants.connectTimeout;
  }

  @override
  Future<TrackCatalogue> getTracks() async {
    final uri = Uri.parse('${ApiConstants.baseUrl}${ApiConstants.tracksPath}');

    try {
      final request = await _httpClient
          .getUrl(uri)
          .timeout(ApiConstants.connectTimeout);
      request.headers.contentType = ContentType.json;

      final response = await request.close().timeout(
        ApiConstants.receiveTimeout,
      );
      final responseBody = await utf8.decoder.bind(response).join();

      if (response.statusCode >= 200 && response.statusCode < 300) {
        final decoded = jsonDecode(responseBody);
        if (decoded is! List) {
          throw const ServerFailure(
            'The track catalogue was not a list of tracks.',
          );
        }
        return TrackCatalogue.fromJsonList(decoded);
      }
      throw ServerFailure(
        _extractErrorMessage(responseBody, response.statusCode),
      );
    } on SocketException catch (e) {
      throw NetworkFailure('Unable to connect to backend server: ${e.message}');
    } on TimeoutException {
      throw const NetworkFailure(
        'Request timed out while fetching the track catalogue',
      );
    } on HttpException catch (e) {
      throw NetworkFailure('HTTP error: ${e.message}');
    } on FormatException catch (e) {
      throw ServerFailure('Malformed track catalogue response: ${e.message}');
    } on Failure {
      rethrow;
    } catch (e) {
      throw NetworkFailure('Unexpected network error: $e');
    }
  }

  @override
  Future<ScreeningTrack> getTrack(String trackId) async {
    if (trackId.isEmpty) {
      throw const ValidationFailure('No screening track was requested.');
    }

    final uri = Uri.parse(
      '${ApiConstants.baseUrl}'
      '${ApiConstants.trackDetailPath(Uri.encodeComponent(trackId))}',
    );

    try {
      final request = await _httpClient
          .getUrl(uri)
          .timeout(ApiConstants.connectTimeout);
      request.headers.contentType = ContentType.json;

      final response = await request.close().timeout(
        ApiConstants.receiveTimeout,
      );
      final responseBody = await utf8.decoder.bind(response).join();

      if (response.statusCode >= 200 && response.statusCode < 300) {
        final decoded = jsonDecode(responseBody);
        if (decoded is! Map<String, dynamic>) {
          throw const ServerFailure(
            'The track description was not a JSON object.',
          );
        }
        return ScreeningTrack.fromJson(decoded);
      }

      // 404 is the caller naming a track that does not exist, not a server fault.
      // Mapping it to ServerFailure alongside a genuine 500 would make "this build
      // asked for a track the backend does not have" read as "the backend is down".
      if (response.statusCode == HttpStatus.notFound) {
        throw ValidationFailure('No screening track named "$trackId" exists.');
      }
      throw ServerFailure(
        _extractErrorMessage(responseBody, response.statusCode),
      );
    } on SocketException catch (e) {
      throw NetworkFailure('Unable to connect to backend server: ${e.message}');
    } on TimeoutException {
      throw const NetworkFailure(
        'Request timed out while fetching the screening track',
      );
    } on HttpException catch (e) {
      throw NetworkFailure('HTTP error: ${e.message}');
    } on FormatException catch (e) {
      throw ServerFailure('Malformed track response: ${e.message}');
    } on Failure {
      rethrow;
    } catch (e) {
      throw NetworkFailure('Unexpected network error: $e');
    }
  }

  String _extractErrorMessage(String responseBody, int statusCode) {
    try {
      final jsonMap = jsonDecode(responseBody);
      if (jsonMap is Map<String, dynamic> && jsonMap.containsKey('detail')) {
        return 'Server Error ($statusCode): ${jsonMap['detail']}';
      }
    } catch (_) {}
    return 'Server returned error status code: $statusCode';
  }
}
