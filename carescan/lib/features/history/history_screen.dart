import 'package:flutter/material.dart';

import 'package:carescan/core/errors/async_state.dart';
import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/core/theme/app_spacing.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';
import 'package:carescan/features/assessment/repositories/assessment_repository.dart';
import 'package:carescan/features/assessment/repositories/mock_assessment_repository.dart';

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
    _repository = widget.repository ?? MockAssessmentRepository();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    setState(() => _state = const AsyncLoading());

    final Result<List<HistoryEntry>> result = await _repository
        .getAssessmentHistory();

    if (!mounted) return;

    setState(() {
      _state = result.fold(
        (failure) => AsyncError(failure),
        (entries) =>
            entries.isEmpty ? const AsyncEmpty() : AsyncSuccess(entries),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Assessment History'),
        centerTitle: false,
      ),
      body: _state.when(
        initial: () => const SizedBox.shrink(),
        loading: () => const _LoadingView(),
        success: (entries) =>
            _HistoryListView(entries: entries, onRefresh: _loadHistory),
        empty: () => const _EmptyView(),
        error: (failure) => _ErrorView(failure: failure, onRetry: _loadHistory),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Loading State
// ---------------------------------------------------------------------------

class _LoadingView extends StatelessWidget {
  const _LoadingView();

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: CircularProgressIndicator(semanticsLabel: 'Loading history'),
    );
  }
}

// ---------------------------------------------------------------------------
// Empty State
// ---------------------------------------------------------------------------

class _EmptyView extends StatelessWidget {
  const _EmptyView();

  @override
  Widget build(BuildContext context) {
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
              'No Assessments Yet',
              style: theme.textTheme.titleMedium?.copyWith(
                color: colorScheme.onSurface,
              ),
            ),
            const SizedBox(height: AppSpacing.sm),
            Text(
              'Your past scan results will appear here once you complete your first assessment.',
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

// ---------------------------------------------------------------------------
// Error State
// ---------------------------------------------------------------------------

class _ErrorView extends StatelessWidget {
  final Failure failure;
  final VoidCallback onRetry;

  const _ErrorView({required this.failure, required this.onRetry});

  @override
  Widget build(BuildContext context) {
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
              'Could Not Load History',
              style: theme.textTheme.titleMedium?.copyWith(
                color: colorScheme.onSurface,
              ),
            ),
            const SizedBox(height: AppSpacing.sm),
            Text(
              'There was a problem fetching your assessment history. Please try again.',
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: AppSpacing.lg),
            FilledButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh_rounded),
              label: const Text('Retry'),
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
        padding: const EdgeInsets.symmetric(
          horizontal: AppSpacing.md,
          vertical: AppSpacing.sm,
        ),
        itemCount: entries.length,
        separatorBuilder: (_, _) => const SizedBox(height: AppSpacing.sm),
        itemBuilder: (context, index) {
          return _HistoryEntryCard(entry: entries[index]);
        },
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// History Entry Card
// ---------------------------------------------------------------------------

class _HistoryEntryCard extends StatelessWidget {
  final HistoryEntry entry;

  const _HistoryEntryCard({required this.entry});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final assessment = entry.assessment;
    final result = entry.result;

    final formattedDate = _formatDate(assessment.timestamp);

    return Semantics(
      label: 'Assessment on $formattedDate: ${result.riskLevel}',
      child: Container(
        padding: const EdgeInsets.all(AppSpacing.md),
        decoration: BoxDecoration(
          color: colorScheme.surfaceContainerLowest,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.04),
              blurRadius: 20,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Thumbnail
            Container(
              width: 56,
              height: 56,
              decoration: BoxDecoration(
                color: colorScheme.surfaceContainerHigh,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(
                Icons.image_rounded,
                color: colorScheme.secondary,
                semanticLabel: 'Scan image',
              ),
            ),
            const SizedBox(width: AppSpacing.md),
            // Content
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    formattedDate,
                    style: theme.textTheme.labelSmall?.copyWith(
                      color: colorScheme.onSurfaceVariant,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    assessment.type,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      fontWeight: FontWeight.w600,
                      color: colorScheme.onSurface,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: AppSpacing.xs),
                  _RiskChip(riskLevel: result.riskLevel),
                ],
              ),
            ),
            const SizedBox(width: AppSpacing.sm),
            Icon(
              Icons.chevron_right_rounded,
              color: colorScheme.outlineVariant,
              semanticLabel: 'View details',
            ),
          ],
        ),
      ),
    );
  }

  String _formatDate(DateTime dt) {
    const months = [
      'Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec',
    ];
    return '${months[dt.month - 1]} ${dt.day}, ${dt.year}';
  }
}

// ---------------------------------------------------------------------------
// Risk Chip
// ---------------------------------------------------------------------------

class _RiskChip extends StatelessWidget {
  final String riskLevel;

  const _RiskChip({required this.riskLevel});

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: colorScheme.primary.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        riskLevel.toUpperCase(),
        style: TextStyle(
          color: colorScheme.primary,
          fontSize: 10,
          fontWeight: FontWeight.bold,
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}
