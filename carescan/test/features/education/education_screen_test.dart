import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/features/education/models/education_article.dart';
import 'package:carescan/features/education/screens/article_screen.dart';
import 'package:carescan/features/education/screens/education_screen.dart';
import 'package:carescan/l10n/l10n.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';

/// A second article, so the "more than one" branches are reachable.
///
/// The app ships one article on purpose (see [educationArticles]); this exists
/// only to exercise the "All articles" and "Read next" sections, which would
/// otherwise be untestable code that silently rots until real copy arrives.
const _second = EducationArticle(
  id: 'second',
  category: 'Test category',
  title: 'A second article',
  sections: [ArticleSection(heading: 'Only heading', body: 'Only body.')],
);

void main() {
  Widget host(Widget child, {Locale locale = const Locale('en')}) {
    // A router, because both screens navigate with `context.go`. Tapping a card
    // has to actually resolve a route or the test proves only that a callback
    // fired into nothing.
    return MaterialApp.router(
      locale: locale,
      theme: AppTheme.theme,
      localizationsDelegates: AppLocalizations.localizationsDelegates,
      supportedLocales: AppLocalizations.supportedLocales,
      routerConfig: GoRouter(
        initialLocation: '/blogs',
        routes: [
          GoRoute(
            path: '/blogs',
            builder: (context, state) => child,
            routes: [
              GoRoute(
                path: ':articleId',
                builder: (context, state) =>
                    ArticleScreen(articleId: state.pathParameters['articleId']),
              ),
            ],
          ),
        ],
      ),
    );
  }

  group('EducationScreen', () {
    testWidgets('renders the featured article with category and reading time', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(host(const EducationScreen()));
      await tester.pumpAndSettle();

      expect(find.text('Blogs'), findsOneWidget);
      expect(
        find.text(
          'Practical reading on oral health, screening and taking '
          'a useful photograph.',
        ),
        findsOneWidget,
      );
      expect(find.text('Featured article'), findsOneWidget);
      // Chip and headline are different strings, not the same one twice.
      expect(find.text('Capture guide'), findsOneWidget);
      expect(find.text('How to capture'), findsOneWidget);
      // The excerpt is the article's own opening paragraph.
      expect(
        find.textContaining('Light the inside of your mouth evenly'),
        findsOneWidget,
      );
      expect(find.text('1 min read'), findsOneWidget);
    });

    testWidgets('a single article is not also listed under All articles', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(host(const EducationScreen()));
      await tester.pumpAndSettle();

      expect(find.text('All articles'), findsNothing);
      expect(find.text('How to capture'), findsOneWidget);
    });

    testWidgets('a second article appears under All articles', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        host(
          EducationScreen(
            articles: [
              ...educationArticles(lookupAppLocalizations(const Locale('en'))),
              _second,
            ],
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('All articles'), findsOneWidget);
      expect(find.text('A second article'), findsOneWidget);
    });

    testWidgets('an empty library renders the empty state, not a blank page', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(host(const EducationScreen(articles: [])));
      await tester.pumpAndSettle();

      expect(find.text('Article not found'), findsOneWidget);
      expect(find.byIcon(Icons.menu_book_outlined), findsOneWidget);
      expect(find.text('Featured article'), findsNothing);
    });

    testWidgets('tapping the card opens the article', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(host(const EducationScreen()));
      await tester.pumpAndSettle();

      await tester.tap(find.text('How to capture'));
      await tester.pumpAndSettle();

      // The detail screen, reached through the real route rather than by
      // constructing it directly.
      expect(find.text('Hold still before capture'), findsOneWidget);
    });

    testWidgets('the card lays out on a narrow phone in Hindi', (
      WidgetTester tester,
    ) async {
      // Same failure mode as the article meta row: the card's reading-time text
      // sits beside a trailing arrow, and the Hindi string is several times
      // longer than the English one.
      tester.view.physicalSize = const Size(320, 640);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);

      await tester.pumpWidget(
        host(const EducationScreen(), locale: const Locale('hi')),
      );
      await tester.pumpAndSettle();

      expect(tester.takeException(), isNull);
      expect(find.text('तस्वीर कैसे लें'), findsOneWidget);
    });
  });

  group('ArticleScreen', () {
    testWidgets('renders every section and the education disclaimer', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        host(const ArticleScreen(articleId: captureGuideArticleId)),
      );
      await tester.pumpAndSettle();

      expect(find.text('How to capture'), findsOneWidget);
      expect(find.text('Face a soft, bright light'), findsOneWidget);
      expect(find.text('Center the mouth'), findsOneWidget);
      expect(find.text('Fill the guide, keep focus'), findsOneWidget);
      expect(find.text('Hold still before capture'), findsOneWidget);
      expect(find.text('Ishan Aran Shukla'), findsOneWidget);
      expect(find.text('1 min read'), findsOneWidget);

      // Non-negotiable on an article screen: the disclaimer's own text claims
      // it is present, so its absence would make the app state something false.
      expect(
        find.textContaining('does not diagnose a condition'),
        findsOneWidget,
      );
    });

    testWidgets('no publication date is invented', (WidgetTester tester) async {
      await tester.pumpWidget(
        host(const ArticleScreen(articleId: captureGuideArticleId)),
      );
      await tester.pumpAndSettle();

      // The ARB carries a "Published" label but the repo carries no date for
      // this article. A fabricated one is worse than an absent one.
      expect(find.textContaining('Published'), findsNothing);
    });

    testWidgets('an unknown article id reads as not found', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(host(const ArticleScreen(articleId: 'nope')));
      await tester.pumpAndSettle();

      expect(find.text('Article not found'), findsOneWidget);
      expect(find.text('Face a soft, bright light'), findsNothing);
    });

    testWidgets('a missing article id reads as not found, not a crash', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(host(const ArticleScreen(articleId: null)));
      await tester.pumpAndSettle();

      expect(find.text('Article not found'), findsOneWidget);
    });

    testWidgets('the last article offers nothing to read next', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        host(const ArticleScreen(articleId: captureGuideArticleId)),
      );
      await tester.pumpAndSettle();

      expect(find.text('Read next'), findsNothing);
    });

    testWidgets('an article with a successor offers Read next', (
      WidgetTester tester,
    ) async {
      final articles = [
        ...educationArticles(lookupAppLocalizations(const Locale('en'))),
        _second,
      ];

      await tester.pumpWidget(
        host(
          ArticleScreen(articleId: captureGuideArticleId, articles: articles),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('Read next'), findsOneWidget);
      expect(find.text('A second article'), findsOneWidget);
    });

    testWidgets('renders in Hindi without falling back to English', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        host(
          const ArticleScreen(articleId: captureGuideArticleId),
          locale: const Locale('hi'),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('तस्वीर कैसे लें'), findsOneWidget);
      expect(find.text('मुंह को बीच में रखें'), findsOneWidget);
      expect(
        find.textContaining('चिकित्सकीय जांच का विकल्प नहीं है'),
        findsOneWidget,
      );
      expect(find.text('How to capture'), findsNothing);
      expect(find.text('Center the mouth'), findsNothing);
    });

    testWidgets('lays out on a narrow phone without overflow', (
      WidgetTester tester,
    ) async {
      // The author/reading-time row is the tight one, and Hindi is the longer
      // of the two locales. A RenderFlex overflow throws in a widget test, so
      // reaching the end of this body is the assertion.
      tester.view.physicalSize = const Size(320, 640);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);

      await tester.pumpWidget(
        host(
          const ArticleScreen(articleId: captureGuideArticleId),
          locale: const Locale('hi'),
        ),
      );
      await tester.pumpAndSettle();

      expect(tester.takeException(), isNull);
    });
  });

  group('EducationArticle', () {
    test('reading time is derived from the body and never rounds to zero', () {
      const tiny = EducationArticle(
        id: 'tiny',
        category: 'c',
        title: 'One',
        sections: [ArticleSection(heading: 'h', body: 'word')],
      );
      expect(tiny.readingMinutes, 1);

      final long = EducationArticle(
        id: 'long',
        category: 'c',
        title: 'Long',
        sections: [
          ArticleSection(
            heading: 'h',
            body: List.filled(450, 'word').join(' '),
          ),
        ],
      );
      // 450 words plus the two-word frame, at 200 wpm, rounded up.
      expect(long.readingMinutes, 3);
    });

    test('an article with no sections has no excerpt', () {
      const bare = EducationArticle(
        id: 'bare',
        category: 'c',
        title: 'Bare',
        sections: [],
      );
      expect(bare.excerpt, isNull);
    });
  });
}
