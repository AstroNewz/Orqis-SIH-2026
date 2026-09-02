import 'package:carescan/features/assessment/models/assessment.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';

class HistoryEntry {
  final Assessment assessment;
  final AssessmentResult result;

  const HistoryEntry({required this.assessment, required this.result});

  factory HistoryEntry.fromJson(Map<String, dynamic> json) {
    return HistoryEntry(
      assessment: Assessment.fromJson(
        json['assessment'] as Map<String, dynamic>,
      ),
      result: AssessmentResult.fromJson(json['result'] as Map<String, dynamic>),
    );
  }

  Map<String, dynamic> toJson() {
    return {'assessment': assessment.toJson(), 'result': result.toJson()};
  }

  HistoryEntry copyWith({Assessment? assessment, AssessmentResult? result}) {
    return HistoryEntry(
      assessment: assessment ?? this.assessment,
      result: result ?? this.result,
    );
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is HistoryEntry &&
        other.assessment == assessment &&
        other.result == result;
  }

  @override
  int get hashCode {
    return Object.hash(assessment, result);
  }
}
