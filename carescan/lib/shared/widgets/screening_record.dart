import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';
import 'package:carescan/l10n/l10n.dart';
import 'package:carescan/core/theme/app_shapes.dart';

enum ScreeningBand { low, moderate, high, pending, unavailable }

ScreeningBand screeningBand(String value) =>
    switch (value.toUpperCase().replaceAll('_', ' ').trim()) {
      'LOW' || 'LOW RISK' => ScreeningBand.low,
      'MODERATE' || 'MODERATE RISK' => ScreeningBand.moderate,
      'HIGH' || 'HIGH RISK' => ScreeningBand.high,
      'PENDING' || 'PROCESSING' => ScreeningBand.pending,
      _ => ScreeningBand.unavailable,
    };
String bandLabel(BuildContext context, String value) =>
    switch (screeningBand(value)) {
      ScreeningBand.low => context.l10n.lowRisk,
      ScreeningBand.moderate => context.l10n.moderateRisk,
      ScreeningBand.high => context.l10n.highRisk,
      ScreeningBand.pending => context.l10n.pendingRisk,
      ScreeningBand.unavailable => context.l10n.unknownRisk,
    };
IconData bandIcon(ScreeningBand band) => switch (band) {
  ScreeningBand.high => Icons.warning_amber_rounded,
  ScreeningBand.moderate => Icons.info_outline,
  ScreeningBand.low => Icons.check_circle_outline,
  _ => Icons.hourglass_empty,
};

class RiskBadge extends StatelessWidget {
  const RiskBadge({super.key, required this.riskLevel});
  final String riskLevel;
  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final band = screeningBand(riskLevel);
    final color = switch (band) {
      ScreeningBand.high => scheme.error,
      ScreeningBand.moderate => scheme.tertiary,
      _ => scheme.onSurfaceVariant,
    };
    return Wrap(
      crossAxisAlignment: WrapCrossAlignment.center,
      spacing: 6,
      children: [
        Icon(bandIcon(band), color: color, size: 18),
        Text(
          bandLabel(context, riskLevel),
          style: Theme.of(context).textTheme.labelMedium
              ?.copyWith(color: color, fontWeight: FontWeight.w600),
        ),
      ],
    );
  }
}

class ScreeningRecord extends StatelessWidget {
  const ScreeningRecord({super.key, required this.entry});
  final HistoryEntry entry;
  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final theme = Theme.of(context);
    final localTime = entry.assessment.timestamp.toLocal();
    final material = MaterialLocalizations.of(context);
    final date =
        '${material.formatMediumDate(localTime)} · ${material.formatTimeOfDay(TimeOfDay.fromDateTime(localTime))}';
    // The record's own scan type where the backend supplied one. Labelling every
    // row with the generic product name would hide the difference between two
    // screenings on a list whose whole job is telling them apart.
    final type = entry.assessment.type.trim();
    final title = type.isEmpty ? l.oralScreening : type;
    return Material(
      color: theme.colorScheme.surface,
      borderRadius: AppShapes.radiusMd,
      child: InkWell(
        borderRadius: AppShapes.radiusMd,
        onTap: () => context.push('/result', extra: entry.result),
        child: Container(
          decoration: BoxDecoration(
            borderRadius: AppShapes.radiusMd,
            // A hairline rather than a drop shadow. Surface sits only a few
            // percent off the page background in both themes, so without an
            // edge the rows read as one undivided block; a black shadow would
            // have done the job in light mode and vanished in dark.
            border: Border.all(color: theme.colorScheme.outlineVariant),
          ),
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceContainer,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Icon(Icons.description_outlined),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: theme.textTheme.titleSmall,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 4),
                    Text(date, style: theme.textTheme.bodySmall),
                    const SizedBox(height: 8),
                    RiskBadge(riskLevel: entry.result.displayRiskLevel),
                    if (entry.result.isMock)
                      Padding(
                        padding: const EdgeInsets.only(top: 6),
                        child: Text(
                          l.testData,
                          style: theme.textTheme.labelSmall,
                        ),
                      ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              const Icon(Icons.chevron_right),
            ],
          ),
        ),
      ),
    );
  }
}
