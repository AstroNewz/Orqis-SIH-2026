import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:carescan/core/constants/api_constants.dart';
import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/features/assessment/models/assessment.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';

abstract class AssessmentRemoteDataSource {
  /// Submits an image path and clinical metadata to FastAPI backend for QML analysis.
  Future<AssessmentResult> submitAssessment(
    String imagePath, {
    String? patientId,
    bool isMock = false,
  });

  /// Retrieves assessment history for a specific patient ID.
  Future<List<HistoryEntry>> getAssessmentHistory(String patientId);
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
    final uri = Uri.parse('${ApiConstants.baseUrl}${ApiConstants.analyzePath}');

    final payload = {
      'patient_id': patientId,
      'image_path': imagePath,
      'scan_type': 'Intra-oral Scan',
      'is_mock': isMock,
    };

    try {
      final request = await _httpClient
          .postUrl(uri)
          .timeout(ApiConstants.connectTimeout);
      request.headers.contentType = ContentType.json;
      request.write(jsonEncode(payload));

      final response = await request.close().timeout(ApiConstants.receiveTimeout);
      final responseBody = await utf8.decoder.bind(response).join();

      if (response.statusCode >= 200 && response.statusCode < 300) {
        final Map<String, dynamic> jsonMap = jsonDecode(responseBody);
        return AssessmentResult.fromJson(jsonMap);
      } else {
        final errorMessage = _extractErrorMessage(responseBody, response.statusCode);
        throw ServerFailure(errorMessage);
      }
    } on SocketException catch (e) {
      throw NetworkFailure('Unable to connect to backend server: ${e.message}');
    } on TimeoutException {
      throw const NetworkFailure('Request timed out while waiting for analysis');
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

      final response = await request.close().timeout(ApiConstants.receiveTimeout);
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
        final errorMessage = _extractErrorMessage(responseBody, response.statusCode);
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
