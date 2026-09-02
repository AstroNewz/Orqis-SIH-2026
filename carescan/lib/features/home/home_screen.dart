import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import 'package:carescan/core/errors/async_state.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/core/theme/app_spacing.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';
import 'package:carescan/features/assessment/repositories/assessment_repository.dart';
import 'package:carescan/features/assessment/repositories/mock_assessment_repository.dart';

class HomeScreen extends StatefulWidget {
  final AssessmentRepository? repository;

  const HomeScreen({super.key, this.repository});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late final AssessmentRepository _repository;
  AsyncState<List<HistoryEntry>> _state = const AsyncLoading();

  @override
  void initState() {
    super.initState();
    _repository = widget.repository ?? MockAssessmentRepository();
    _loadRecentAssessments();
  }

  Future<void> _loadRecentAssessments() async {
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
        title: const Text('Good morning, Alex'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () {
              context.go('/settings');
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.only(
          left: AppSpacing.md,
          right: AppSpacing.md,
          top: AppSpacing.sm,
          bottom: 100, // Safe padding for bottom nav
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const _HeroCard(),
            const SizedBox(height: AppSpacing.lg),
            _buildRecentAssessmentsSection(context),
          ],
        ),
      ),
    );
  }

  Widget _buildRecentAssessmentsSection(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'Recent Assessments',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            TextButton(
              onPressed: () {
                context.go('/history');
              },
              child: const Text('View All'),
            ),
          ],
        ),
        const SizedBox(height: AppSpacing.sm),
        _state.when(
          initial: () => const SizedBox.shrink(),
          loading: () => const Center(
            child: Padding(
              padding: EdgeInsets.all(AppSpacing.md),
              child: CircularProgressIndicator(),
            ),
          ),
          success: (entries) {
            final recent = entries.take(3).toList();
            return Column(
              children: recent.map((entry) {
                final assessment = entry.assessment;
                final result = entry.result;
                return Padding(
                  padding: const EdgeInsets.only(bottom: AppSpacing.sm),
                  child: _AssessmentItem(
                    date: _formatDate(assessment.timestamp),
                    title: assessment.type,
                    riskLevel: result.riskLevel,
                  ),
                );
              }).toList(),
            );
          },
          empty: () => Padding(
            padding: const EdgeInsets.symmetric(vertical: AppSpacing.md),
            child: Text(
              'No recent assessments. Take a photo to get started.',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
              textAlign: TextAlign.center,
            ),
          ),
          error: (failure) => Padding(
            padding: const EdgeInsets.symmetric(vertical: AppSpacing.sm),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  'Could not load recent assessments.',
                  style: TextStyle(color: Theme.of(context).colorScheme.error),
                ),
                TextButton(
                  onPressed: _loadRecentAssessments,
                  child: const Text('Retry'),
                ),
              ],
            ),
          ),
        ),
      ],
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
    return '${months[dt.month - 1]} ${dt.day.toString().padLeft(2, '0')}, ${dt.year}';
  }
}

class _HeroCard extends StatelessWidget {
  const _HeroCard();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Container(
      constraints: const BoxConstraints(minHeight: 220),
      padding: const EdgeInsets.all(AppSpacing.md),
      decoration: BoxDecoration(
        color: colorScheme.primary,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 20,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Start a New Assessment',
            style: theme.textTheme.titleMedium?.copyWith(
              color: colorScheme.onPrimary,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Quickly scan and analyze with precision.',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: colorScheme.onPrimary.withValues(alpha: 0.9),
            ),
          ),
          const SizedBox(height: AppSpacing.lg),
          ElevatedButton.icon(
            onPressed: () => context.push('/camera'),
            icon: const Icon(Icons.photo_camera),
            label: const Text('Take a Photo'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.white,
              foregroundColor: colorScheme.primary,
              minimumSize: const Size.fromHeight(52),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(26),
              ),
            ),
          ),
          const SizedBox(height: AppSpacing.sm),
          OutlinedButton.icon(
            onPressed: () {},
            icon: const Icon(Icons.upload_file),
            label: const Text('Upload Image'),
            style: OutlinedButton.styleFrom(
              foregroundColor: Colors.white,
              side: BorderSide(color: Colors.white.withValues(alpha: 0.3)),
              minimumSize: const Size.fromHeight(52),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(26),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _AssessmentItem extends StatelessWidget {
  final String date;
  final String title;
  final String riskLevel;

  const _AssessmentItem({
    required this.date,
    required this.title,
    required this.riskLevel,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerLow,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: colorScheme.outlineVariant.withValues(alpha: 0.3),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: colorScheme.secondaryContainer,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(Icons.image, color: colorScheme.secondary),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  date,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                Text(
                  title,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: colorScheme.onSurfaceVariant,
                  ),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: colorScheme.primary.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(12),
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
          ),
        ],
      ),
    );
  }
}
