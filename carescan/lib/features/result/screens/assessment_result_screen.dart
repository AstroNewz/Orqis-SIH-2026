import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import 'package:carescan/core/theme/app_spacing.dart';
import 'package:carescan/features/assessment/models/assessment_result.dart';
import 'package:carescan/features/assessment/repositories/assessment_repository.dart';
import 'package:carescan/features/result/widgets/localized_capture_view.dart';
import 'package:carescan/l10n/l10n.dart';
import 'package:carescan/shared/widgets/app_button.dart';
import 'package:carescan/shared/widgets/product_components.dart';
import 'package:carescan/shared/widgets/screening_record.dart';

/// What a finished screening says, and what stands behind it.
///
/// Every band decision on this screen goes through [screeningBand] rather than
/// substring-matching the backend's string. The previous `band.contains('HIGH')`
/// chain fell through to the low-risk branch for *anything* it did not
/// recognise, so a result the server reported as unavailable or still
/// processing was drawn with the green tick and told the patient they were
/// "below the screening threshold". A band the app cannot classify must read as
/// unclassified, never as reassurance.
///
/// DEC-034 is load-bearing in the layout: the headline is the strongest
/// validated model's band ([AssessmentResult.displayRiskLevel]), and where that
/// model is uncalibrated no percentage is rendered for it anywhere — the
/// verdict is shown as a band, and the only percentage on the screen belongs to
/// the calibrated quantum model, under its own research heading.
class AssessmentResultScreen extends StatelessWidget {
  final AssessmentResult? result;
  final String? imagePath;
  final AssessmentRepository? repository;

  const AssessmentResultScreen({
    super.key,
    this.result,
    this.imagePath,
    this.repository,
  });

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final theme = Theme.of(context);
    final res = result;

    // Absent result is a real state: /result can be reached from a history row
    // whose payload failed to decode. It resolves to the unavailable band, not
    // to a default of "low".
    final rawBand = res?.displayRiskLevel ?? '';
    final band = screeningBand(rawBand);
    final isMock = res?.isMock ?? false;
    final quantum = res?.finalProbability;

    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => context.go('/'),
          tooltip: MaterialLocalizations.of(context).backButtonTooltip,
        ),
        title: Text(l.resultTitle),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          child: ContentPane(
            padding: const EdgeInsets.symmetric(
              horizontal: AppSpacing.md,
              vertical: AppSpacing.md,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                if (isMock) ...[
                  InfoNote(
                    text: l.testData,
                    icon: Icons.science_outlined,
                    warning: true,
                  ),
                  const SizedBox(height: AppSpacing.md),
                ],

                _BandHero(band: band, label: bandLabel(context, rawBand)),
                const SizedBox(height: AppSpacing.md),

                // The analysed capture with a non-gating ROI overlay when the
                // localiser is available (DEC-034). Shown only while we still
                // hold the capture on-device; the overlay never affects the
                // verdict above and fails silently, so it can never block or
                // delay the result.
                if (imagePath != null && imagePath!.isNotEmpty) ...[
                  LocalizedCaptureView(
                    imagePath: imagePath!,
                    repository: repository,
                  ),
                  const SizedBox(height: AppSpacing.md),
                ],

                // What produced the verdict. Withheld for a labelled stub,
                // which has no model behind it to describe.
                if (!isMock && res != null) ...[
                  _PrimaryReadout(result: res, band: band),
                  const SizedBox(height: AppSpacing.md),
                ],

                // The research readout exists only when there is a calibrated
                // quantum probability to report. A card whose entire content is
                // "not available" tells the patient nothing.
                if (!isMock && quantum != null) ...[
                  _ResearchReadout(probability: quantum),
                  const SizedBox(height: AppSpacing.md),
                ],

                _NextSteps(details: res?.details),
                const SizedBox(height: AppSpacing.md),

                _Provenance(result: res),
                const SizedBox(height: AppSpacing.lg),

                // Navigation only. The result was persisted server-side at
                // analyse time, so nothing here saves or discards it.
                AppButton(
                  label: l.viewHistory,
                  onPressed: () => context.go('/history'),
                ),
                const SizedBox(height: AppSpacing.sm),
                AppButton(
                  label: l.newScan,
                  onPressed: () => context.go('/'),
                  variant: AppButtonVariant.secondary,
                ),
                const SizedBox(height: AppSpacing.md),

                Text(
                  l.disclaimer,
                  textAlign: TextAlign.center,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
                const SizedBox(height: AppSpacing.lg),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Hero
// ---------------------------------------------------------------------------

/// The band, as a colour, an icon and a sentence.
///
/// Three carriers rather than one: DESIGN.md forbids conveying information by
/// colour alone, and this is the single most consequential value in the app.
class _BandHero extends StatelessWidget {
  const _BandHero({required this.band, required this.label});

  final ScreeningBand band;
  final String label;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final tone = _BandTone.of(theme, band);
    final meaning = _meaning(context, band);

    return _Panel(
      padding: const EdgeInsets.symmetric(
        vertical: AppSpacing.xl,
        horizontal: AppSpacing.lg,
      ),
      // One label for the whole hero: a screen reader should read the band and
      // what it means as a single statement, not as an icon then two fragments.
      child: Semantics(
        header: true,
        label: '${context.l10n.riskLabel}: $label. $meaning',
        excludeSemantics: true,
        child: Column(
          children: [
            Container(
              width: 96,
              height: 96,
              decoration: BoxDecoration(
                color: tone.container,
                shape: BoxShape.circle,
              ),
              child: Icon(bandIcon(band), size: 48, color: tone.onContainer),
            ),
            const SizedBox(height: AppSpacing.md),
            Text(
              label,
              textAlign: TextAlign.center,
              style: theme.textTheme.headlineMedium?.copyWith(
                color: tone.foreground,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: AppSpacing.xs),
            Text(
              meaning,
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyLarge?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Moderate and high share `concernMeaning` because that is how the string
  /// table is written: the urgency difference between them is carried by the
  /// band name, its colour and its icon, and the action is the same in both
  /// cases and stated once under [_NextSteps].
  static String _meaning(BuildContext context, ScreeningBand band) {
    final l = context.l10n;
    return switch (band) {
      ScreeningBand.low => l.lowMeaning,
      ScreeningBand.moderate || ScreeningBand.high => l.concernMeaning,
      ScreeningBand.pending || ScreeningBand.unavailable => l.unknownMeaning,
    };
  }
}

/// Colours for one band.
///
/// Pending and unavailable resolve to neutral surface colours on purpose. They
/// are the two cases the old substring chain mis-drew as low risk, and a
/// neutral swatch is the only honest answer for a band with no verdict in it.
class _BandTone {
  const _BandTone({
    required this.container,
    required this.onContainer,
    required this.foreground,
  });

  final Color container;
  final Color onContainer;
  final Color foreground;

  static _BandTone of(ThemeData theme, ScreeningBand band) {
    final s = theme.colorScheme;
    return switch (band) {
      ScreeningBand.high => _BandTone(
        container: s.errorContainer,
        onContainer: s.onErrorContainer,
        foreground: s.error,
      ),
      ScreeningBand.moderate => _BandTone(
        container: s.tertiaryContainer,
        onContainer: s.onTertiaryContainer,
        foreground: s.tertiary,
      ),
      ScreeningBand.low => _BandTone(
        container: s.primaryContainer,
        onContainer: s.onPrimaryContainer,
        foreground: s.primary,
      ),
      ScreeningBand.pending || ScreeningBand.unavailable => _BandTone(
        container: s.surfaceContainerHighest,
        onContainer: s.onSurfaceVariant,
        foreground: s.onSurfaceVariant,
      ),
    };
  }
}

// ---------------------------------------------------------------------------
// Readouts
// ---------------------------------------------------------------------------

/// The model that set the headline band, and no number for it.
class _PrimaryReadout extends StatelessWidget {
  const _PrimaryReadout({required this.result, required this.band});

  final AssessmentResult result;
  final ScreeningBand band;

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final theme = Theme.of(context);

    return _Card(
      icon: Icons.analytics_outlined,
      title: l.explanation,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _Caption(
            label: l.riskLabel,
            value: bandLabel(context, result.displayRiskLevel),
          ),
          const SizedBox(height: AppSpacing.xs),
          // The model identifier, untranslated. It is a technical name, and
          // rendering it beside a translated label is how tracks_screen.dart
          // already reports the same field.
          _Caption(
            label: l.trackModel,
            value: _modelName(context, result.primaryModel),
          ),

          // Why there is no percentage next to the verdict. Shown only when the
          // backend states the primary model is uncalibrated; when it says
          // nothing, no claim is made either way — and no percentage is
          // rendered for the primary model in any case.
          if (result.primaryCalibrated == false) ...[
            const SizedBox(height: AppSpacing.sm),
            Text(
              l.scoreNote,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
                height: 1.4,
              ),
            ),
          ],
        ],
      ),
    );
  }

  static String _modelName(BuildContext context, String? model) {
    if (model == null || model.trim().isEmpty) {
      return context.l10n.trackUnavailable;
    }
    return model.replaceAll('_', ' ');
  }
}

/// The calibrated quantum probability, under a heading that says what it is.
///
/// This is the only percentage on the screen. It is reported here, below the
/// verdict and inside a card titled as experimental research, because on this
/// dataset the quantum model has not displaced the classical baseline — the
/// layout itself is the claim, and [AppLocalizations.researchNote] states it.
class _ResearchReadout extends StatelessWidget {
  const _ResearchReadout({required this.probability});

  final double probability;

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final theme = Theme.of(context);

    return _Card(
      icon: Icons.science_outlined,
      title: l.researchTitle,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _Caption(
            label: l.researchProbability,
            value: '${(probability * 100).toStringAsFixed(1)}%',
          ),
          const SizedBox(height: AppSpacing.sm),
          Text(
            l.researchNote,
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Guidance and provenance
// ---------------------------------------------------------------------------

/// What to do, then the server's own notice.
///
/// [details] is rendered verbatim and is server-supplied, so it is whatever
/// language the backend speaks; the surrounding guidance is localised. It comes
/// second because the localised advice is the part every patient can read.
class _NextSteps extends StatelessWidget {
  const _NextSteps({required this.details});

  final String? details;

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final theme = Theme.of(context);
    final notice = details?.trim() ?? '';

    return _Card(
      icon: Icons.medical_services_outlined,
      title: l.nextStep,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            l.nextStepBody,
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
              height: 1.4,
            ),
          ),
          if (notice.isNotEmpty) ...[
            const SizedBox(height: AppSpacing.md),
            InfoNote(text: notice),
          ],
        ],
      ),
    );
  }
}

/// Which model version produced this, and which record it is.
class _Provenance extends StatelessWidget {
  const _Provenance({required this.result});

  final AssessmentResult? result;

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final theme = Theme.of(context);
    final res = result;
    final version = res?.modelVersion?.trim();
    final id = res?.id ?? '';

    return Container(
      padding: const EdgeInsets.all(AppSpacing.md),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainer,
        borderRadius: BorderRadius.circular(_radius),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _IconCaption(
            icon: Icons.memory_outlined,
            label: l.modelVersion,
            value: (version == null || version.isEmpty)
                ? l.trackUnavailable
                : version,
          ),
          const SizedBox(height: AppSpacing.xs),
          _IconCaption(
            icon: Icons.fingerprint_outlined,
            label: 'ID',
            // A short prefix is enough to match a record against the server and
            // keeps the row from wrapping; the full id is not useful on screen.
            value: id.length >= 8
                ? id.substring(0, 8)
                : (id.isEmpty ? '—' : id),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Shared pieces
// ---------------------------------------------------------------------------

/// The one corner radius this screen uses.
///
/// It matches the theme's card shape and [ScreeningRecord]. The screen
/// previously mixed 24, 20 and 12 on one scroll view, none of them from a
/// token. VERIFY WITH STITCH.
const double _radius = 12;

/// A raised-looking surface drawn with an edge instead of a shadow.
///
/// The shadows this replaces were `Colors.black` at 4% opacity: visible on the
/// light theme's near-white page and completely invisible against the dark
/// one, so in dark mode the cards had no boundary at all.
class _Panel extends StatelessWidget {
  const _Panel({required this.child, required this.padding});

  final Widget child;
  final EdgeInsetsGeometry padding;

  @override
  Widget build(BuildContext context) {
    final s = Theme.of(context).colorScheme;
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: s.surfaceContainerLowest,
        borderRadius: BorderRadius.circular(_radius),
        border: Border.all(color: s.outlineVariant),
      ),
      child: child,
    );
  }
}

class _Card extends StatelessWidget {
  const _Card({required this.icon, required this.title, required this.child});

  final IconData icon;
  final String title;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return _Panel(
      padding: const EdgeInsets.all(AppSpacing.lg),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 20, color: theme.colorScheme.secondary),
              const SizedBox(width: AppSpacing.xs),
              Expanded(
                child: Text(
                  title,
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: AppSpacing.sm),
          child,
        ],
      ),
    );
  }
}

/// A bold label and its value on one wrapping line.
///
/// `RichText` rather than a `Row`, so a long Hindi label and a long value share
/// the available width by wrapping instead of competing for it.
class _Caption extends StatelessWidget {
  const _Caption({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return RichText(
      text: TextSpan(
        style: theme.textTheme.bodyMedium?.copyWith(
          color: theme.colorScheme.onSurfaceVariant,
        ),
        children: [
          TextSpan(
            text: '$label: ',
            style: const TextStyle(fontWeight: FontWeight.w600),
          ),
          TextSpan(
            text: value,
            style: TextStyle(color: theme.colorScheme.onSurface),
          ),
        ],
      ),
    );
  }
}

class _IconCaption extends StatelessWidget {
  const _IconCaption({
    required this.icon,
    required this.label,
    required this.value,
  });

  final IconData icon;
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 16, color: theme.colorScheme.onSurfaceVariant),
        const SizedBox(width: AppSpacing.xs),
        Expanded(
          child: Text(
            '$label: $value',
            style: theme.textTheme.labelMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ),
      ],
    );
  }
}
