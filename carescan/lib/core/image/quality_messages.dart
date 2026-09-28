import 'package:carescan/l10n/l10n.dart';

import 'image_quality_service.dart';

String qualityTitle(AppLocalizations l, ImageIssue issue) => switch (issue) {
  ImageIssue.dark => l.qualityDark,
  ImageIssue.bright => l.qualityBright,
  ImageIssue.blurry => l.qualityBlur,
  ImageIssue.resolution => l.qualitySmall,
  ImageIssue.format => l.qualityFormat,
  ImageIssue.tooLarge => l.qualitySize,
  ImageIssue.oralFraming => l.qualityContext,
  ImageIssue.unreadable => l.qualityUnavailable,
};
