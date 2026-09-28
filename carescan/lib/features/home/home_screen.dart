import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:carescan/core/di/service_locator.dart';
import 'package:carescan/core/errors/async_state.dart';
import 'package:carescan/features/assessment/models/history_entry.dart';
import 'package:carescan/features/assessment/repositories/assessment_repository.dart';
import 'package:carescan/features/auth/prototype_session.dart';
import 'package:carescan/l10n/l10n.dart';
import 'package:carescan/shared/widgets/product_components.dart';
import 'package:carescan/shared/widgets/screening_record.dart';
import 'package:carescan/core/theme/app_shapes.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, this.repository});
  final AssessmentRepository? repository;
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  AsyncState<List<HistoryEntry>> _state = const AsyncLoading();
  AssessmentRepository get _repository =>
      widget.repository ?? appAssessmentRepository;
  @override
  void initState() {
    super.initState();
    screeningRevision.addListener(_load);
    _load();
  }

  @override
  void dispose() {
    screeningRevision.removeListener(_load);
    super.dispose();
  }

  Future<void> _load() async {
    setState(() => _state = const AsyncLoading());
    final result = await _repository.getAssessmentHistory();
    if (!mounted) return;
    setState(
      () => _state = result.fold((failure) => AsyncError(failure), (entries) {
        final sorted = [...entries]
          ..sort(
            (a, b) => b.assessment.timestamp.compareTo(a.assessment.timestamp),
          );
        return sorted.isEmpty ? const AsyncEmpty() : AsyncSuccess(sorted);
      }),
    );
  }

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final t = Theme.of(context);
    final identity = appSession.identity;
    return Scaffold(
      appBar: AppBar(
        title: const CareScanBrand(),
        actions: [
          IconButton(
            tooltip: l.tracks,
            icon: const Icon(Icons.science_outlined),
            onPressed: () => context.push('/tracks'),
          ),
          IconButton(
            tooltip: l.profile,
            icon: const Icon(Icons.person_outline),
            onPressed: () => context.go('/profile'),
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _load,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          child: ContentPane(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  identity == null || identity.isGuest
                      ? l.welcome
                      : identity.name,
                  style: t.textTheme.bodyMedium?.copyWith(
                    color: t.colorScheme.onSurfaceVariant,
                  ),
                ),
                const SizedBox(height: 28),
                Text(
                  l.prototype,
                  style: t.textTheme.labelMedium?.copyWith(
                    color: t.colorScheme.primary,
                  ),
                ),
                const SizedBox(height: 12),
                Text(l.heroTitle, style: t.textTheme.displaySmall),
                const SizedBox(height: 16),
                Text(
                  l.heroSubtitle,
                  style: t.textTheme.bodyLarge?.copyWith(
                    color: t.colorScheme.onSurfaceVariant,
                  ),
                ),
                const SizedBox(height: 28),
                FilledButton.icon(
                  onPressed: () => context.push('/camera'),
                  icon: const Icon(Icons.center_focus_strong),
                  label: Text(l.startScreening),
                ),
                TextButton(
                  onPressed: () => context.go('/history'),
                  child: Text(l.viewHistory),
                ),
                const SizedBox(height: 16),
                const Divider(),
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  child: Wrap(
                    spacing: 20,
                    runSpacing: 12,
                    children: [
                      _step(Icons.camera_alt_outlined, l.captureStep),
                      _step(Icons.fact_check_outlined, l.qualityStep),
                      _step(Icons.description_outlined, l.resultStep),
                    ],
                  ),
                ),
                const Divider(),
                SectionHeading(
                  title: l.recentScreening,
                  action: l.viewAll,
                  onAction: () => context.go('/history'),
                ),
                _state.when(
                  initial: () => const SizedBox.shrink(),
                  loading: () => Padding(
                    padding: const EdgeInsets.all(24),
                    child: Center(
                      child: CircularProgressIndicator(
                        semanticsLabel: l.loadingHistory,
                      ),
                    ),
                  ),
                  success: (entries) => Column(
                    children: [
                      for (final entry in entries.take(2))
                        Padding(
                          padding: const EdgeInsets.only(bottom: 12),
                          child: ScreeningRecord(entry: entry),
                        ),
                    ],
                  ),
                  empty: () => Container(
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      color: t.colorScheme.surface,
                      borderRadius: AppShapes.radiusMd,
                      border: Border.all(color: t.colorScheme.outlineVariant),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(
                          Icons.history,
                          color: t.colorScheme.primary,
                          size: 28,
                        ),
                        const SizedBox(height: 16),
                        Text(l.noScreenings, style: t.textTheme.titleMedium),
                        const SizedBox(height: 8),
                        Text(l.noScreeningsBody, style: t.textTheme.bodyMedium),
                        const SizedBox(height: 12),
                        TextButton(
                          onPressed: () => context.push('/camera'),
                          child: Text(l.firstScreening),
                        ),
                      ],
                    ),
                  ),
                  error: (_) => Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      InfoNote(text: l.historyError, warning: true),
                      TextButton.icon(
                        onPressed: _load,
                        icon: const Icon(Icons.refresh),
                        label: Text(l.retry),
                      ),
                    ],
                  ),
                ),
                SectionHeading(title: l.healthEducation),
                Text(l.educationIntro, style: t.textTheme.bodyLarge),
                const SizedBox(height: 12),
                ListTile(
                  contentPadding: EdgeInsets.zero,
                  leading: Icon(
                    Icons.menu_book_outlined,
                    color: t.colorScheme.primary,
                  ),
                  title: Text(l.editorialIntro),
                  trailing: const Icon(Icons.arrow_forward),
                  onTap: () => context.go('/blogs'),
                ),
                // The catalogue is reachable from the app bar too, but an icon
                // is easy to miss and "which conditions can this screen for,
                // and on what evidence" is the question the platform most
                // needs to be able to answer out loud.
                ListTile(
                  contentPadding: EdgeInsets.zero,
                  leading: Icon(
                    Icons.science_outlined,
                    color: t.colorScheme.primary,
                  ),
                  title: Text(l.tracks),
                  trailing: const Icon(Icons.arrow_forward),
                  onTap: () => context.push('/tracks'),
                ),
                const SizedBox(height: 24),
                Text(
                  l.disclaimer,
                  style: t.textTheme.bodySmall?.copyWith(
                    color: t.colorScheme.onSurfaceVariant,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _step(IconData icon, String label) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      Icon(icon, size: 18, color: Theme.of(context).colorScheme.primary),
      const SizedBox(width: 8),
      Flexible(
        child: Text(label, style: Theme.of(context).textTheme.labelMedium),
      ),
    ],
  );
}
