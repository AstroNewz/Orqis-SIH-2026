class AssessmentResult {
  final String id;
  final String assessmentId;
  final String riskLevel;
  final String details;

  const AssessmentResult({
    required this.id,
    required this.assessmentId,
    required this.riskLevel,
    required this.details,
  });

  factory AssessmentResult.fromJson(Map<String, dynamic> json) {
    return AssessmentResult(
      id: json['id'] as String,
      assessmentId: json['assessmentId'] as String,
      riskLevel: json['riskLevel'] as String,
      details: json['details'] as String,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'assessmentId': assessmentId,
      'riskLevel': riskLevel,
      'details': details,
    };
  }

  AssessmentResult copyWith({
    String? id,
    String? assessmentId,
    String? riskLevel,
    String? details,
  }) {
    return AssessmentResult(
      id: id ?? this.id,
      assessmentId: assessmentId ?? this.assessmentId,
      riskLevel: riskLevel ?? this.riskLevel,
      details: details ?? this.details,
    );
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is AssessmentResult &&
        other.id == id &&
        other.assessmentId == assessmentId &&
        other.riskLevel == riskLevel &&
        other.details == details;
  }

  @override
  int get hashCode {
    return Object.hash(id, assessmentId, riskLevel, details);
  }
}
