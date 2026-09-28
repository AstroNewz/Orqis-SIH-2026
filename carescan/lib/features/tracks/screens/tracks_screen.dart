import 'package:flutter/material.dart';

import 'package:carescan/core/errors/async_state.dart';
import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/theme/app_shapes.dart';
import 'package:carescan/core/theme/app_spacing.dart';
import 'package:carescan/data/repositories/api_track_repository.dart';
import 'package:carescan/features/tracks/models/screening_track.dart';
import 'package:carescan/features/tracks/repositories/track_repository.dart';
import 'package:carescan/l10n/l10n.dart';
import 'package:carescan/shared/widgets/product_components.dart';

/// Shows what the platform can screen for, and what stands behind each number.
///
/// Read-only on purpose, in two senses. It renders `GET /api/tracks` and offers no
/// action, because the only capture path this build actually has is the oral-image
/// one on `/camera`: putting a "start screening" button on a signal track would
/// advertise a flow that does not exist. And every number it shows arrives already
/// captioned — the screen reaches the headline metric through
/// [TrackValidationSummary.headline] and its provenance through
/// [TrackValidationSummary.provenance], never through `primaryMetricValue`, so a
/// development estimate cannot be rendered as a validated result by forgetting to
/// add the qualification here.
class TracksScreen extends StatefulWidget {
  const TracksScreen({super.key, this.repository});

  /// Injected by tests. Production builds its own, matching the house pattern.
  final TrackRepository? repository;

  @override
  State<TracksScreen> createState() => _TracksScreenState();
}

class _TracksScreenState extends State<TracksScreen> {
  AsyncState<TrackCatalogue> _state = const AsyncLoading();

  TrackRepository get _repository => widget.repository ?? _fallback;
  late final TrackRepository _fallback = ApiTrackRepository();

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _state = const AsyncLoading());
    final result = await _repository.getTracks();
    if (!mounted) return;
    setState(
      () => _state = result.fold(
        (failure) => AsyncError(failure),
        (catalogue) =>
            // An empty catalogue with nothing rejected means the service really
            // has no tracks configured. An empty one *with* rejections is a
            // different statement — this build could not read what was offered —
            // and it must reach the banner rather than read as "none exist".
            catalogue.tracks.isEmpty && !catalogue.hasRejections
            ? const AsyncEmpty()
            : AsyncSuccess(catalogue),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;

    return Scaffold(
      appBar: AppBar(title: Text(l.tracks)),
      body: _state.when(
        initial: () => const SizedBox.shrink(),
        loading: () => Center(
          child: CircularProgressIndicator(semanticsLabel: l.loadingTracks),
        ),
        success: (catalogue) =>
            _CatalogueView(catalogue: catalogue, onRefresh: _load),
        empty: () => const _EmptyView(),
        error: (failure) => _ErrorView(failure: failure, onRetry: _load),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Catalogue
// ---------------------------------------------------------------------------

class _CatalogueView extends StatelessWidget {
  const _CatalogueView({required this.catalogue, required this.onRefresh});

  final TrackCatalogue catalogue;
  final Future<void> Function() onRefresh;

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final t = Theme.of(context);

    return RefreshIndicator(
      onRefresh: onRefresh,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        child: ContentPane(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                l.tracksIntro,
                style: t.textTheme.bodyLarge?.copyWith(
                  color: t.colorScheme.onSurfaceVariant,
                ),
              ),
              if (catalogue.hasRejections) ...[
                const SizedBox(height: AppSpacing.md),
                _RejectionsNote(rejected: catalogue.rejected),
              ],
              const SizedBox(height: AppSpacing.lg),
              for (final track in catalogue.tracks)
                Padding(
                  padding: const EdgeInsets.only(bottom: AppSpacing.md),
                  child: _TrackCard(track: track),
                ),
              const SizedBox(height: AppSpacing.sm),
              Text(
                l.tracksNote,
                style: t.textTheme.bodySmall?.copyWith(
                  color: t.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Announces refused descriptors *and* names them.
///
/// A count alone would tell the user something is wrong without telling anyone
/// what; the reasons are what makes a schema regression diagnosable from a
/// screenshot.
class _RejectionsNote extends StatelessWidget {
  const _RejectionsNote({required this.rejected});

  final List<String> rejected;

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final t = Theme.of(context);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        InfoNote(
          text: '${l.tracksRejected(rejected.length)}. ${l.tracksRejectedBody}',
          icon: Icons.report_problem_outlined,
          warning: true,
        ),
        const SizedBox(height: AppSpacing.sm),
        for (final reason in rejected)
          Padding(
            padding: const EdgeInsets.only(bottom: AppSpacing.xs),
            child: Text(
              reason,
              style: t.textTheme.bodySmall?.copyWith(
                color: t.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// One track
// ---------------------------------------------------------------------------

class _TrackCard extends StatelessWidget {
  const _TrackCard({required this.track});

  final ScreeningTrack track;

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final t = Theme.of(context);
    final s = t.colorScheme;
    final validation = track.validation;

    return Container(
      padding: const EdgeInsets.all(AppSpacing.md),
      decoration: BoxDecoration(
        color: s.surface,
        borderRadius: AppShapes.radiusMd,
        border: Border.all(color: s.outlineVariant),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Text(track.displayName, style: t.textTheme.titleMedium),
              ),
              const SizedBox(width: AppSpacing.sm),
              _Badge(
                label: track.isSelectable
                    ? l.trackAvailable
                    : l.trackUnavailable,
                tone: track.isSelectable ? _Tone.positive : _Tone.muted,
              ),
            ],
          ),
          const SizedBox(height: AppSpacing.xs),
          Text(
            track.condition,
            style: t.textTheme.bodyMedium?.copyWith(color: s.onSurfaceVariant),
          ),

          // Why a listed track cannot be run. Shown rather than hidden: a
          // platform that silently drops its unready tracks looks smaller than
          // it is, and a greyed-out track with a reason is more informative than
          // an absent one.
          if (track.blockedReason case final String reason) ...[
            const SizedBox(height: AppSpacing.md),
            InfoNote(text: reason, warning: true),
          ],

          const SizedBox(height: AppSpacing.md),
          _Badge(
            label: validation.isClinicallyClaimable
                ? l.trackFrozenTest
                : l.trackDevelopmentEstimate,
            tone: validation.isClinicallyClaimable
                ? _Tone.positive
                : _Tone.warning,
          ),

          // The only formatter offered for the metric. It carries its own
          // qualification, so there is no unqualified way to show the number
          // from here.
          if (validation.headline case final String headline) ...[
            const SizedBox(height: AppSpacing.sm),
            Text(headline, style: t.textTheme.bodyMedium),
          ],
          const SizedBox(height: AppSpacing.xs),
          _Caption(label: l.trackProvenance, value: validation.provenance),

          const SizedBox(height: AppSpacing.md),
          _Caption(label: l.trackInput, value: track.inputSpec.description),
          const SizedBox(height: AppSpacing.xs),
          _Caption(
            label: l.trackModel,
            // A track with no persisted model reports an empty version, which
            // is the honest answer and not a missing field. Interpolating it
            // anyway would render "gradient_boosted_trees ()".
            value: track.modelVersion.isEmpty
                ? track.primaryModel
                : '${track.primaryModel} (${track.modelVersion})',
          ),
          if (track.quantumBadge case final String badge) ...[
            const SizedBox(height: AppSpacing.xs),
            _Caption(label: l.trackQuantum, value: badge),
          ],

          const SizedBox(height: AppSpacing.md),
          Text(
            track.disclaimer,
            style: t.textTheme.bodySmall?.copyWith(color: s.onSurfaceVariant),
          ),
        ],
      ),
    );
  }
}

class _Caption extends StatelessWidget {
  const _Caption({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return RichText(
      text: TextSpan(
        style: t.textTheme.bodySmall?.copyWith(
          color: t.colorScheme.onSurfaceVariant,
        ),
        children: [
          TextSpan(
            text: '$label: ',
            style: const TextStyle(fontWeight: FontWeight.w600),
          ),
          TextSpan(text: value),
        ],
      ),
    );
  }
}

enum _Tone { positive, warning, muted }

class _Badge extends StatelessWidget {
  const _Badge({required this.label, required this.tone});

  final String label;
  final _Tone tone;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final s = t.colorScheme;
    final color = switch (tone) {
      _Tone.positive => s.primary,
      _Tone.warning => s.tertiary,
      _Tone.muted => s.onSurfaceVariant,
    };

    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: AppSpacing.sm,
          vertical: AppSpacing.xs,
        ),
        decoration: BoxDecoration(
          borderRadius: AppShapes.radiusFull,
          border: Border.all(color: color),
        ),
        child: Text(
          label,
          style: t.textTheme.labelSmall?.copyWith(color: color),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Empty and error states
// ---------------------------------------------------------------------------

class _EmptyView extends StatelessWidget {
  const _EmptyView();

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Center(
      child: Padding(
        padding: const EdgeInsets.all(AppSpacing.lg),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.science_outlined,
              size: 80,
              color: colorScheme.outlineVariant,
            ),
            const SizedBox(height: AppSpacing.md),
            Text(
              l.tracksEmpty,
              style: theme.textTheme.titleMedium?.copyWith(
                color: colorScheme.onSurface,
              ),
            ),
            const SizedBox(height: AppSpacing.sm),
            Text(
              l.tracksEmptyBody,
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ErrorView extends StatelessWidget {
  const _ErrorView({required this.failure, required this.onRetry});

  final Failure failure;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Center(
      child: Padding(
        padding: const EdgeInsets.all(AppSpacing.lg),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.error_outline_rounded,
              size: 80,
              color: colorScheme.error,
            ),
            const SizedBox(height: AppSpacing.md),
            Text(
              l.tracksError,
              style: theme.textTheme.titleMedium?.copyWith(
                color: colorScheme.onSurface,
              ),
            ),
            const SizedBox(height: AppSpacing.sm),
            // The failure's own message, not a generic one. "Unreachable",
            // "unreadable" and "no such track" want different responses from
            // whoever is holding the phone.
            Text(
              failure.message,
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: AppSpacing.sm),
            Text(
              l.connectionHelp,
              textAlign: TextAlign.center,
              style: theme.textTheme.bodySmall?.copyWith(
                color: colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: AppSpacing.lg),
            FilledButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh_rounded),
              label: Text(l.retry),
            ),
          ],
        ),
      ),
    );
  }
}
