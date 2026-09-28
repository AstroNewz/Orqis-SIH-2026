import 'package:flutter/foundation.dart';
import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/data/repositories/api_assessment_repository.dart';
import 'package:carescan/features/auth/prototype_session.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';
import 'package:carescan/features/assessment/models/localization_result.dart';
import 'package:carescan/features/assessment/repositories/assessment_repository.dart';

/// Changes after a saved screening so retained shell tabs refresh their records.
final screeningRevision = ValueNotifier<int>(0);
String? _boundPatient;
AssessmentRepository? _boundRepository;
AssessmentRepository get appAssessmentRepository {
  final id = appSession.identity?.id;
  if (id == null) return const _NoSessionRepository();
  if (_boundPatient != id) {
    _boundPatient = id;
    _boundRepository = ApiAssessmentRepository(defaultPatientId: id);
  }
  return _boundRepository!;
}

class _NoSessionRepository implements AssessmentRepository {
  const _NoSessionRepository();
  @override
  Future<Result<List<HistoryEntry>>> getAssessmentHistory() async =>
      const Success([]);
  @override
  Future<Result<AssessmentResult>> submitAssessment(String imagePath) async =>
      const Error(ValidationFailure('No active prototype session'));
  @override
  Future<Result<LocalizationResult>> localize(String imagePath) async =>
      const Success(LocalizationResult.none());
}
