import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:carescan/core/constants/api_constants.dart';
import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/image/image_quality_service.dart';
import 'package:carescan/features/assessment/models/assessment.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';
import 'package:carescan/features/assessment/models/localization_result.dart';

abstract class AssessmentRemoteDataSource {
  /// Submits an image path and clinical metadata to FastAPI backend for QML analysis.
  Future<AssessmentResult> submitAssessment(
    String imagePath, {
    String? patientId,
    bool isMock = false,
  });

  /// Retrieves assessment history for a specific patient ID.
  Future<List<HistoryEntry>> getAssessmentHistory(String patientId);

  /// Locates the lesion ROI in a local capture for a non-gating visual overlay.
  Future<LocalizationResult> localizeImage(String imagePath);
}

class AssessmentRemoteDataSourceImpl implements AssessmentRemoteDataSource {
  final HttpClient _httpClient;

  AssessmentRemoteDataSourceImpl({HttpClient? httpClient})
    : _httpClient = httpClient ?? HttpClient() {
    _httpClient.connectionTimeout = ApiConstants.connectTimeout;
  }

  @override
  Future<AssessmentResult> submitAssessment(
    String imagePath, {
    String? patientId,
    bool isMock = false,
  }) async {
    // A camera capture lives only on the device, so the backend cannot read that
    // path. Upload the bytes first and analyze the server-side copy it returns. The
    // mock path scores nothing and needs no image, so it skips the upload.
    String backendImagePath = imagePath;
    if (!isMock) {
      if (imagePath.isEmpty) {
        throw const ValidationFailure(
          'No image to analyze. Capture a photo and try again.',
        );
      }
      backendImagePath = await _uploadImage(imagePath, patientId: patientId);
    }

    final uri = Uri.parse('${ApiConstants.baseUrl}${ApiConstants.analyzePath}');

    final payload = {
      'patient_id': patientId,
      'image_path': backendImagePath,
      'scan_type': 'Intra-oral Scan',
      'is_mock': isMock,
    };

    try {
      final request = await _httpClient
          .postUrl(uri)
          .timeout(ApiConstants.connectTimeout);
      request.headers.contentType = ContentType.json;
      request.write(jsonEncode(payload));

      final response = await request.close().timeout(
        ApiConstants.receiveTimeout,
      );
      final responseBody = await utf8.decoder.bind(response).join();

      if (response.statusCode >= 200 && response.statusCode < 300) {
        final Map<String, dynamic> jsonMap = jsonDecode(responseBody);
        return AssessmentResult.fromJson(jsonMap);
      } else {
        final errorMessage = _extractErrorMessage(
          responseBody,
          response.statusCode,
        );
        throw ServerFailure(errorMessage);
      }
    } on SocketException catch (e) {
      throw NetworkFailure('Unable to connect to backend server: ${e.message}');
    } on TimeoutException {
      throw const NetworkFailure(
        'Request timed out while waiting for analysis',
      );
    } on HttpException catch (e) {
      throw NetworkFailure('HTTP error: ${e.message}');
    } on FormatException catch (e) {
      throw ServerFailure('Malformed server response: ${e.message}');
    } on Failure {
      rethrow;
    } catch (e) {
      throw NetworkFailure('Unexpected network error: $e');
    }
  }

  @override
  Future<List<HistoryEntry>> getAssessmentHistory(String patientId) async {
    final uri = Uri.parse(
      '${ApiConstants.baseUrl}${ApiConstants.patientHistoryPath(patientId)}',
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
        final List<dynamic> jsonList = jsonDecode(responseBody);
        return jsonList.map((item) {
          final map = item as Map<String, dynamic>;
          final assessmentMap = map['assessment'] as Map<String, dynamic>;
          final resultMap = map['result'] as Map<String, dynamic>?;

          return HistoryEntry(
            assessment: Assessment.fromJson(assessmentMap),
            result: resultMap != null
                ? AssessmentResult.fromJson(resultMap)
                : AssessmentResult(
                    id: 'pending-${assessmentMap['id']}',
                    assessmentId: assessmentMap['id'] as String,
                    riskLevel: 'PENDING',
                    details: 'Analysis in progress',
                  ),
          );
        }).toList();
      } else {
        final errorMessage = _extractErrorMessage(
          responseBody,
          response.statusCode,
        );
        throw ServerFailure(errorMessage);
      }
    } on SocketException catch (e) {
      throw NetworkFailure('Unable to connect to backend server: ${e.message}');
    } on TimeoutException {
      throw const NetworkFailure('Request timed out while fetching history');
    } on HttpException catch (e) {
      throw NetworkFailure('HTTP error: ${e.message}');
    } on FormatException catch (e) {
      throw ServerFailure('Malformed history response: ${e.message}');
    } on Failure {
      rethrow;
    } catch (e) {
      throw NetworkFailure('Unexpected network error: $e');
    }
  }

  /// Posts the capture's bytes to `POST /api/localize` and returns the ROI
  /// overlay result.
  ///
  /// Independent of analysis by design (DEC-034): this only asks where the lesion
  /// is for a visual box, never re-runs the screening pipeline. Built as a raw
  /// `multipart/form-data` body with a single `file` part, since this client depends
  /// only on `dart:io`. Failures propagate as [Failure]s for the caller to swallow --
  /// the overlay is advisory, so a localiser that is unavailable simply draws nothing.
  @override
  Future<LocalizationResult> localizeImage(String imagePath) async {
    if (imagePath.isEmpty) {
      throw const ValidationFailure('No image to localise.');
    }
    final file = File(imagePath);
    if (!await file.exists()) {
      throw const ValidationFailure(
        'The captured image could not be found on the device.',
      );
    }
    final bytes = await file.readAsBytes();
    final filename = imagePath.split(RegExp(r'[\\/]')).last;
    final boundary =
        '----carescan${DateTime.now().microsecondsSinceEpoch.toRadixString(16)}';

    final prologue = StringBuffer()
      ..write('--$boundary\r\n')
      ..write(
        'Content-Disposition: form-data; name="file"; filename="$filename"\r\n',
      )
      ..write('Content-Type: ${_imageContentType(filename)}\r\n\r\n');
    final epilogue = '\r\n--$boundary--\r\n';

    final bodyBytes =
        (BytesBuilder()
              ..add(utf8.encode(prologue.toString()))
              ..add(bytes)
              ..add(utf8.encode(epilogue)))
            .takeBytes();

    final uri = Uri.parse(
      '${ApiConstants.baseUrl}${ApiConstants.localizePath}',
    );

    try {
      final request = await _httpClient
          .postUrl(uri)
          .timeout(ApiConstants.connectTimeout);
      request.headers.set(
        HttpHeaders.contentTypeHeader,
        'multipart/form-data; boundary=$boundary',
      );
      request.headers.contentLength = bodyBytes.length;
      request.add(bodyBytes);

      final response = await request.close().timeout(
        ApiConstants.receiveTimeout,
      );
      final responseBody = await utf8.decoder.bind(response).join();

      if (response.statusCode >= 200 && response.statusCode < 300) {
        final Map<String, dynamic> jsonMap = jsonDecode(responseBody);
        return LocalizationResult.fromJson(jsonMap);
      }
      throw ServerFailure(
        _extractErrorMessage(responseBody, response.statusCode),
      );
    } on SocketException catch (e) {
      throw NetworkFailure('Unable to connect to backend server: ${e.message}');
    } on TimeoutException {
      throw const NetworkFailure('Localisation timed out');
    } on HttpException catch (e) {
      throw NetworkFailure('HTTP error during localisation: ${e.message}');
    } on FormatException catch (e) {
      throw ServerFailure('Malformed localisation response: ${e.message}');
    } on Failure {
      rethrow;
    } catch (e) {
      throw NetworkFailure('Unexpected error during localisation: $e');
    }
  }

  /// Streams a local capture to `POST /api/screening/upload` and returns the
  /// server-side path the analyze call can read.
  ///
  /// Built as a raw `multipart/form-data` body because this client depends only on
  /// `dart:io` (no `http`/`dio` package): a boundary, one `file` part carrying the
  /// image bytes, and an optional `patient_id` text part. The backend chooses the
  /// stored filename itself and only trusts the extension, so the part's declared
  /// content type is advisory.
  Future<String> _uploadImage(String imagePath, {String? patientId}) async {
    final quality = await const LocalImageInputValidator().inspect(imagePath);
    if (!quality.accepted) throw ImageInputFailure(quality.issue!);
    final file = File(imagePath);
    if (!await file.exists()) {
      throw const ValidationFailure(
        'The captured image could not be found on the device.',
      );
    }
    final bytes = await file.readAsBytes();
    final filename = imagePath.split(RegExp(r'[\\/]')).last;
    final boundary =
        '----carescan${DateTime.now().microsecondsSinceEpoch.toRadixString(16)}';

    final prologue = StringBuffer()
      ..write('--$boundary\r\n')
      ..write(
        'Content-Disposition: form-data; name="file"; filename="$filename"\r\n',
      )
      ..write('Content-Type: ${_imageContentType(filename)}\r\n\r\n');

    final epilogue = StringBuffer()..write('\r\n');
    if (patientId != null && patientId.isNotEmpty) {
      epilogue
        ..write('--$boundary\r\n')
        ..write('Content-Disposition: form-data; name="patient_id"\r\n\r\n')
        ..write('$patientId\r\n');
    }
    epilogue.write('--$boundary--\r\n');

    final bodyBytes =
        (BytesBuilder()
              ..add(utf8.encode(prologue.toString()))
              ..add(bytes)
              ..add(utf8.encode(epilogue.toString())))
            .takeBytes();

    final uri = Uri.parse('${ApiConstants.baseUrl}${ApiConstants.uploadPath}');

    try {
      final request = await _httpClient
          .postUrl(uri)
          .timeout(ApiConstants.connectTimeout);
      request.headers.set(
        HttpHeaders.contentTypeHeader,
        'multipart/form-data; boundary=$boundary',
      );
      request.headers.contentLength = bodyBytes.length;
      request.add(bodyBytes);

      final response = await request.close().timeout(
        ApiConstants.receiveTimeout,
      );
      final responseBody = await utf8.decoder.bind(response).join();

      if (response.statusCode >= 200 && response.statusCode < 300) {
        final Map<String, dynamic> jsonMap = jsonDecode(responseBody);
        final serverPath = jsonMap['image_path'] as String?;
        if (serverPath == null || serverPath.isEmpty) {
          throw const ServerFailure(
            'Upload succeeded but the server returned no image path.',
          );
        }
        return serverPath;
      }
      throw ServerFailure(
        _extractErrorMessage(responseBody, response.statusCode),
      );
    } on SocketException catch (e) {
      throw NetworkFailure('Unable to connect to backend server: ${e.message}');
    } on TimeoutException {
      throw const NetworkFailure('Upload timed out while sending the image');
    } on HttpException catch (e) {
      throw NetworkFailure('HTTP error during upload: ${e.message}');
    } on FormatException catch (e) {
      throw ServerFailure('Malformed upload response: ${e.message}');
    } on Failure {
      rethrow;
    } catch (e) {
      throw NetworkFailure('Unexpected error during upload: $e');
    }
  }

  String _imageContentType(String filename) {
    final lower = filename.toLowerCase();
    if (lower.endsWith('.png')) return 'image/png';
    if (lower.endsWith('.webp')) return 'image/webp';
    if (lower.endsWith('.bmp')) return 'image/bmp';
    return 'image/jpeg';
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
