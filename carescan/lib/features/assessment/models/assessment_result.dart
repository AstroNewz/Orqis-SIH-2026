class AssessmentResult {
  final String id;
  final String assessmentId;
  final String riskLevel;
  final String details;

  // Displayed (headline) verdict: the strongest validated model's band (DEC-034).
  // When primaryModel is the classical baseline, primaryCalibrated is false and
  // primaryProbability is an uncalibrated ranking score -- never shown as a percentage.
  final String? primaryModel;
  final String? primaryRiskLevel;
  final double? primaryProbability;
  final double? primaryThreshold;
  final bool? primaryCalibrated;

  // Calibrated quantum probability (the honest secondary readout) and provenance.
  final double? finalProbability;
  final double? quantumProbability;
  final double? classicalProbability;
  final String? classification;
  final String? modelVersion;
  final bool isMock;

  const AssessmentResult({
    required this.id,
    required this.assessmentId,
    required this.riskLevel,
    required this.details,
    this.primaryModel,
    this.primaryRiskLevel,
    this.primaryProbability,
    this.primaryThreshold,
    this.primaryCalibrated,
    this.finalProbability,
    this.quantumProbability,
    this.classicalProbability,
    this.classification,
    this.modelVersion,
    this.isMock = false,
  });

  factory AssessmentResult.fromJson(Map<String, dynamic> json) {
    return AssessmentResult(
      id: json['id'] as String,
      assessmentId: json['assessmentId'] as String,
      riskLevel: json['riskLevel'] as String,
      details: json['details'] as String,
      primaryModel: json['primaryModel'] as String?,
      primaryRiskLevel: json['primaryRiskLevel'] as String?,
      primaryProbability: _toDouble(json['primaryProbability']),
      primaryThreshold: _toDouble(json['primaryThreshold']),
      primaryCalibrated: json['primaryCalibrated'] as bool?,
      finalProbability: _toDouble(json['finalProbability']),
      quantumProbability: _toDouble(json['quantumProbability']),
      classicalProbability: _toDouble(json['classicalProbability']),
      classification: json['classification'] as String?,
      modelVersion: json['modelVersion'] as String?,
      isMock: (json['isMock'] as bool?) ?? false,
    );
  }

  /// The band to display as the verdict: the validated primary model's band when the
  /// backend supplied one, otherwise the (quantum) risk level. Kept here so every
  /// screen headlines the same value.
  String get displayRiskLevel =>
      (primaryRiskLevel != null && primaryRiskLevel!.isNotEmpty)
      ? primaryRiskLevel!
      : riskLevel;

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'assessmentId': assessmentId,
      'riskLevel': riskLevel,
      'details': details,
      'primaryModel': primaryModel,
      'primaryRiskLevel': primaryRiskLevel,
      'primaryProbability': primaryProbability,
      'primaryThreshold': primaryThreshold,
      'primaryCalibrated': primaryCalibrated,
      'finalProbability': finalProbability,
      'quantumProbability': quantumProbability,
      'classicalProbability': classicalProbability,
      'classification': classification,
      'modelVersion': modelVersion,
      'isMock': isMock,
    };
  }

  AssessmentResult copyWith({
    String? id,
    String? assessmentId,
    String? riskLevel,
    String? details,
    String? primaryModel,
    String? primaryRiskLevel,
    double? primaryProbability,
    double? primaryThreshold,
    bool? primaryCalibrated,
    double? finalProbability,
    double? quantumProbability,
    double? classicalProbability,
    String? classification,
    String? modelVersion,
    bool? isMock,
  }) {
    return AssessmentResult(
      id: id ?? this.id,
      assessmentId: assessmentId ?? this.assessmentId,
      riskLevel: riskLevel ?? this.riskLevel,
      details: details ?? this.details,
      primaryModel: primaryModel ?? this.primaryModel,
      primaryRiskLevel: primaryRiskLevel ?? this.primaryRiskLevel,
      primaryProbability: primaryProbability ?? this.primaryProbability,
      primaryThreshold: primaryThreshold ?? this.primaryThreshold,
      primaryCalibrated: primaryCalibrated ?? this.primaryCalibrated,
      finalProbability: finalProbability ?? this.finalProbability,
      quantumProbability: quantumProbability ?? this.quantumProbability,
      classicalProbability: classicalProbability ?? this.classicalProbability,
      classification: classification ?? this.classification,
      modelVersion: modelVersion ?? this.modelVersion,
      isMock: isMock ?? this.isMock,
    );
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is AssessmentResult &&
        other.id == id &&
        other.assessmentId == assessmentId &&
        other.riskLevel == riskLevel &&
        other.details == details &&
        other.primaryModel == primaryModel &&
        other.primaryRiskLevel == primaryRiskLevel &&
        other.primaryProbability == primaryProbability &&
        other.primaryThreshold == primaryThreshold &&
        other.primaryCalibrated == primaryCalibrated &&
        other.finalProbability == finalProbability &&
        other.quantumProbability == quantumProbability &&
        other.classicalProbability == classicalProbability &&
        other.classification == classification &&
        other.modelVersion == modelVersion &&
        other.isMock == isMock;
  }

  @override
  int get hashCode {
    return Object.hash(
      id,
      assessmentId,
      riskLevel,
      details,
      primaryModel,
      primaryRiskLevel,
      primaryProbability,
      primaryThreshold,
      primaryCalibrated,
      finalProbability,
      quantumProbability,
      classicalProbability,
      classification,
      modelVersion,
      isMock,
    );
  }
}

/// Parses a JSON number that may arrive as an int, a double, or a numeric string.
double? _toDouble(dynamic value) {
  if (value == null) return null;
  if (value is num) return value.toDouble();
  if (value is String) return double.tryParse(value);
  return null;
}
