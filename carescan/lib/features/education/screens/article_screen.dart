import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import 'package:carescan/core/theme/app_shapes.dart';
import 'package:carescan/core/theme/app_spacing.dart';
import 'package:carescan/features/education/models/education_article.dart';
import 'package:carescan/l10n/l10n.dart';
import 'package:carescan/shared/widgets/app_button.dart';
import 'package:carescan/shared/widgets/product_components.dart';

/// One article, and the disclaimer that has to travel with it.
///
/// The article body is read from the string table, so it is translated rather
/// than English-with-a-Hindi-frame. [AppLocalizations.educationDisclaimer] is
/// rendered on every article without exception — its own text says "this article
/// does not diagnose a condition or replace a clinical examination", which is
/// only true of the screen if it is actually on the screen.
class ArticleScreen extends StatelessWidget {
  const ArticleScreen({super.key, required this.articleId, this.articles});

  /// From the route. Nullable because `/blogs/` with no segment is reachable by
  /// hand-typed deep link, and that is the not-found case rather than a crash.
  final String? articleId;

  /// Injected by tests.
  final List<EducationArticle>? articles;

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final t = Theme.of(context);
    final s = t.colorScheme;
    final items = articles ?? educationArticles(l);
    final index = items.indexWhere((a) => a.id == articleId);

    if (index < 0) return _NotFound(title: l.blogs);

    final article = items[index];
    // The next article along, when there is one. With a single article there is
    // nothing to read next, so the section is withheld rather than shown empty.
    final next = index + 1 < items.length ? items[index + 1] : null;

    return Scaffold(
      appBar: AppBar(title: Text(article.category)),
      body: SingleChildScrollView(
        child: ContentPane(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(article.title, style: t.textTheme.headlineSmall),
              const SizedBox(height: AppSpacing.sm),

              // Author and length. No "Published" line: the ARB carries the
              // label but the repo carries no publication date for this
              // article, and inventing one would be fabricating provenance on a
              // clinical app. It appears when a real date exists.
              //
              // A Wrap rather than a Row: Hindi's reading-time string is
              // "पढ़ने का समय: 1 मिनट" against English's "1 min read", and the
              // pair overflowed a 320 dp screen by 32 px as a single line.
              Wrap(
                spacing: AppSpacing.md,
                runSpacing: AppSpacing.xs,
                children: [
                  _Meta(icon: Icons.person_outline, label: l.author),
                  _Meta(
                    icon: Icons.schedule_outlined,
                    label: l.readingTime(article.readingMinutes),
                  ),
                ],
              ),

              const SizedBox(height: AppSpacing.md),
              Divider(color: s.outlineVariant),
              const SizedBox(height: AppSpacing.md),

              for (final section in article.sections) ...[
                Text(section.heading, style: t.textTheme.titleMedium),
                const SizedBox(height: AppSpacing.xs),
                Text(
                  section.body,
                  style: t.textTheme.bodyLarge?.copyWith(height: 1.5),
                ),
                const SizedBox(height: AppSpacing.lg),
              ],

              InfoNote(
                text: l.educationDisclaimer,
                icon: Icons.info_outline,
                warning: true,
              ),

              if (next != null) ...[
                SectionHeading(title: l.relatedArticle),
                _NextUp(article: next),
              ],

              const SizedBox(height: AppSpacing.lg),
              AppButton(
                label: l.allArticles,
                onPressed: () => context.go('/blogs'),
                variant: AppButtonVariant.secondary,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Meta extends StatelessWidget {
  const _Meta({required this.icon, required this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final s = t.colorScheme;

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 16, color: s.onSurfaceVariant),
        const SizedBox(width: AppSpacing.xs),
        Text(
          label,
          style: t.textTheme.bodySmall?.copyWith(color: s.onSurfaceVariant),
        ),
      ],
    );
  }
}

class _NextUp extends StatelessWidget {
  const _NextUp({required this.article});

  final EducationArticle article;

  @override
  Widget build(BuildContext context) {
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
            border: Border.all(color: s.outlineVariant),
          ),
          child: Row(
            children: [
              Expanded(
                child: Text(article.title, style: t.textTheme.titleMedium),
              ),
              const SizedBox(width: AppSpacing.sm),
              Icon(Icons.arrow_forward, size: 18, color: s.primary),
            ],
          ),
        ),
      ),
    );
  }
}

/// A deep link naming an article this build does not have.
class _NotFound extends StatelessWidget {
  const _NotFound({required this.title});

  final String title;

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final t = Theme.of(context);
    final s = t.colorScheme;

    return Scaffold(
      appBar: AppBar(title: Text(title)),
      body: Center(
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
              const SizedBox(height: AppSpacing.lg),
              AppButton(
                label: l.allArticles,
                onPressed: () => context.go('/blogs'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
