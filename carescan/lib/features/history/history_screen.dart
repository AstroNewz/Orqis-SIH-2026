import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import 'package:carescan/core/di/service_locator.dart';
import 'package:carescan/core/errors/async_state.dart';
import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/core/theme/app_spacing.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';
import 'package:carescan/features/assessment/repositories/assessment_repository.dart';
import 'package:carescan/l10n/l10n.dart';
import 'package:carescan/shared/widgets/product_components.dart';
import 'package:carescan/shared/widgets/screening_record.dart';

/// The full list of past screenings.
///
/// Home and History render the same records, so they render them through the
/// same [ScreeningRecord] widget: a row that opens its result, carries the band
/// as an icon as well as a colour, and speaks whichever language the app is set
/// to. Two separate row implementations had drifted apart — the one here was
/// inert, monochrome and English-only — and any difference between the two is a
/// difference in what the same screening appears to say.
class HistoryScreen extends StatefulWidget {
  final AssessmentRepository? repository;

  const HistoryScreen({super.key, this.repository});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  late final AssessmentRepository _repository;

  AsyncState<List<HistoryEntry>> _state = const AsyncLoading();

  @override
  void initState() {
    super.initState();
    _repository = widget.repository ?? appAssessmentRepository;
    // A screening finished elsewhere in the app belongs in this list without a
    // pull-to-refresh: the tab is an indexed-stack branch, so it stays mounted
    // and would otherwise keep showing the list as it was when first opened.
    screeningRevision.addListener(_loadHistory);
    _loadHistory();
  }

  @override
  void dispose() {
    screeningRevision.removeListener(_loadHistory);
    super.dispose();
  }

  Future<void> _loadHistory() async {
    setState(() => _state = const AsyncLoading());

    final Result<List<HistoryEntry>> result = await _repository
        .getAssessmentHistory();

    if (!mounted) return;

    setState(() {
      _state = result.fold((failure) => AsyncError(failure), (entries) {
        // Newest first, matching the Home summary. Server order is not
        // guaranteed, and the two lists disagreeing about which screening is
        // the most recent one is worse than either order on its own.
        final sorted = [...entries]
          ..sort(
            (a, b) => b.assessment.timestamp.compareTo(a.assessment.timestamp),
          );
        return sorted.isEmpty ? const AsyncEmpty() : AsyncSuccess(sorted);
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;

    return Scaffold(
      appBar: AppBar(title: Text(l.history)),
      body: _state.when(
        initial: () => const SizedBox.shrink(),
        loading: () => Center(
          child: CircularProgressIndicator(semanticsLabel: l.loadingHistory),
        ),
        success: (entries) =>
            _HistoryListView(entries: entries, onRefresh: _loadHistory),
        empty: () => _EmptyView(onStart: () => context.push('/camera')),
        error: (failure) => _ErrorView(failure: failure, onRetry: _loadHistory),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Empty State
// ---------------------------------------------------------------------------

class _EmptyView extends StatelessWidget {
  const _EmptyView({required this.onStart});

  final VoidCallback onStart;

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
              Icons.history_rounded,
              size: 80,
              color: colorScheme.outlineVariant,
            ),
            const SizedBox(height: AppSpacing.md),
            Text(
              l.noScreenings,
              textAlign: TextAlign.center,
              style: theme.textTheme.titleMedium?.copyWith(
                color: colorScheme.onSurface,
              ),
            ),
            const SizedBox(height: AppSpacing.sm),
            Text(
              l.noScreeningsBody,
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: AppSpacing.lg),
            // An empty list is the one place in the app where the next step is
            // unambiguous, so it is offered here rather than left to the user
            // to find the capture button again.
            FilledButton.icon(
              onPressed: onStart,
              icon: const Icon(Icons.center_focus_strong),
              label: Text(l.firstScreening),
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Error State
// ---------------------------------------------------------------------------

class _ErrorView extends StatelessWidget {
  final Failure failure;
  final VoidCallback onRetry;

  const _ErrorView({required this.failure, required this.onRetry});

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
              l.historyError,
              textAlign: TextAlign.center,
              style: theme.textTheme.titleMedium?.copyWith(
                color: colorScheme.onSurface,
              ),
            ),
            const SizedBox(height: AppSpacing.sm),
            // The failure's own message is deliberately not shown: it carries
            // transport detail ("Connection failed", a status code) that reads
            // as noise to a patient. What to *do* is the useful part.
            Text(
              l.connectionHelp,
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyMedium?.copyWith(
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

// ---------------------------------------------------------------------------
// Success: List View
// ---------------------------------------------------------------------------

class _HistoryListView extends StatelessWidget {
  final List<HistoryEntry> entries;
  final Future<void> Function() onRefresh;

  const _HistoryListView({required this.entries, required this.onRefresh});

  @override
  Widget build(BuildContext context) {
    return RefreshIndicator(
      onRefresh: onRefresh,
      child: ListView.separated(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.symmetric(
          horizontal: AppSpacing.md,
          vertical: AppSpacing.sm,
        ),
        itemCount: entries.length,
        separatorBuilder: (_, _) => const SizedBox(height: AppSpacing.sm),
        itemBuilder: (context, index) => ContentPane(
          padding: EdgeInsets.zero,
          child: ScreeningRecord(entry: entries[index]),
        ),
      ),
    );
  }
}
