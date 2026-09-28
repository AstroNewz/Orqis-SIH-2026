import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import 'package:carescan/core/theme/app_shapes.dart';
import 'package:carescan/core/theme/app_spacing.dart';
import 'package:carescan/features/education/models/education_article.dart';
import 'package:carescan/l10n/l10n.dart';
import 'package:carescan/shared/widgets/product_components.dart';

/// The reading list behind the "Blogs" tab.
///
/// This replaces a `PlaceholderScreen` whose entire body was the word "Blogs"
/// centred on a blank page — reachable from the bottom bar *and* from the home
/// screen, where it sits under a "Health education" heading promising
/// "Knowledge for the next step". The debug screen was the thing that promise
/// led to.
///
/// It lists only what there is copy for. See [educationArticles] for why that is
/// one article and not three.
class EducationScreen extends StatelessWidget {
  const EducationScreen({super.key, this.articles});

  /// Injected by tests. Production reads the string table, matching the house
  /// pattern for screens with a swappable source.
  final List<EducationArticle>? articles;

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final t = Theme.of(context);
    final items = articles ?? educationArticles(l);

    return Scaffold(
      appBar: AppBar(title: Text(l.blogs)),
      body: items.isEmpty
          ? const _EmptyView()
          : SingleChildScrollView(
              child: ContentPane(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Text(
                      l.editorialSubtitle,
                      style: t.textTheme.bodyLarge?.copyWith(
                        color: t.colorScheme.onSurfaceVariant,
                      ),
                    ),
                    const SizedBox(height: AppSpacing.lg),

                    Text(
                      l.featured,
                      style: t.textTheme.labelMedium?.copyWith(
                        color: t.colorScheme.primary,
                      ),
                    ),
                    const SizedBox(height: AppSpacing.sm),
                    _ArticleCard(article: items.first, featured: true),

                    // Only when there is a remainder to head. With a single
                    // article, an "All articles" heading over the same card the
                    // reader just passed would be a second listing of one thing.
                    if (items.length > 1) ...[
                      SectionHeading(title: l.allArticles),
                      for (final article in items.skip(1))
                        Padding(
                          padding: const EdgeInsets.only(bottom: AppSpacing.md),
                          child: _ArticleCard(article: article),
                        ),
                    ],
                  ],
                ),
              ),
            ),
    );
  }
}

class _ArticleCard extends StatelessWidget {
  const _ArticleCard({required this.article, this.featured = false});

  final EducationArticle article;
  final bool featured;

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final t = Theme.of(context);
    final s = t.colorScheme;

    return Material(
      color: s.surface,
      borderRadius: AppShapes.radiusMd,
      child: InkWell(
        borderRadius: AppShapes.radiusMd,
        onTap: () => context.go('/blogs/${article.id}'),
        child: Container(
          padding: const EdgeInsets.all(AppSpacing.md),
          decoration: BoxDecoration(
            borderRadius: AppShapes.radiusMd,
            border: Border.all(
              color: featured ? s.primary : s.outlineVariant,
              width: featured ? 1.5 : 1,
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _CategoryChip(label: article.category),
              const SizedBox(height: AppSpacing.sm),
              Text(
                article.title,
                style: featured
                    ? t.textTheme.titleLarge
                    : t.textTheme.titleMedium,
              ),
              if (article.excerpt case final String excerpt) ...[
                const SizedBox(height: AppSpacing.xs),
                Text(
                  excerpt,
                  maxLines: featured ? 3 : 2,
                  overflow: TextOverflow.ellipsis,
                  style: t.textTheme.bodyMedium?.copyWith(
                    color: s.onSurfaceVariant,
                  ),
                ),
              ],
              const SizedBox(height: AppSpacing.sm),
              Row(
                children: [
                  Icon(
                    Icons.schedule_outlined,
                    size: 16,
                    color: s.onSurfaceVariant,
                  ),
                  const SizedBox(width: AppSpacing.xs),
                  // Expanded rather than a trailing Spacer: the Hindi
                  // reading-time string is several times longer than the
                  // English one, and it has to be allowed to shrink instead of
                  // pushing the arrow off the card.
                  Expanded(
                    child: Text(
                      l.readingTime(article.readingMinutes),
                      overflow: TextOverflow.ellipsis,
                      style: t.textTheme.bodySmall?.copyWith(
                        color: s.onSurfaceVariant,
                      ),
                    ),
                  ),
                  const SizedBox(width: AppSpacing.sm),
                  Icon(Icons.arrow_forward, size: 18, color: s.primary),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _CategoryChip extends StatelessWidget {
  const _CategoryChip({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final s = t.colorScheme;

    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: AppSpacing.sm,
          vertical: AppSpacing.xs,
        ),
        decoration: BoxDecoration(
          color: s.surfaceContainerHighest,
          borderRadius: AppShapes.radiusFull,
        ),
        child: Text(
          label,
          style: t.textTheme.labelSmall?.copyWith(color: s.onSurfaceVariant),
        ),
      ),
    );
  }
}

class _EmptyView extends StatelessWidget {
  const _EmptyView();

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final t = Theme.of(context);
    final s = t.colorScheme;

    return Center(
      child: Padding(
        padding: const EdgeInsets.all(AppSpacing.lg),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.menu_book_outlined, size: 80, color: s.outlineVariant),
            const SizedBox(height: AppSpacing.md),
            Text(
              l.articleNotFound,
              style: t.textTheme.titleMedium?.copyWith(color: s.onSurface),
            ),
            const SizedBox(height: AppSpacing.sm),
            Text(
              l.editorialSubtitle,
              textAlign: TextAlign.center,
              style: t.textTheme.bodyMedium?.copyWith(
                color: s.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
