class Assessment {
  final String id;
  final String imagePath;
  final DateTime timestamp;
  final String type;

  const Assessment({
    required this.id,
    required this.imagePath,
    required this.timestamp,
    required this.type,
  });

  factory Assessment.fromJson(Map<String, dynamic> json) {
    return Assessment(
      id: json['id'] as String,
      imagePath: json['imagePath'] as String,
      timestamp: DateTime.parse(json['timestamp'] as String),
      type: json['type'] as String,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'imagePath': imagePath,
      'timestamp': timestamp.toIso8601String(),
      'type': type,
    };
  }

  Assessment copyWith({
    String? id,
    String? imagePath,
    DateTime? timestamp,
    String? type,
  }) {
    return Assessment(
      id: id ?? this.id,
      imagePath: imagePath ?? this.imagePath,
      timestamp: timestamp ?? this.timestamp,
      type: type ?? this.type,
    );
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is Assessment &&
        other.id == id &&
        other.imagePath == imagePath &&
        other.timestamp == timestamp &&
        other.type == type;
  }

  @override
  int get hashCode {
    return Object.hash(id, imagePath, timestamp, type);
  }
}
