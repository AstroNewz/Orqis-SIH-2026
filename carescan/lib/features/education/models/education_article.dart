import 'package:carescan/l10n/l10n.dart';

/// One heading-and-body block of an article.
class ArticleSection {
  const ArticleSection({required this.heading, required this.body});

  final String heading;
  final String body;
}

/// A piece of educational reading the app ships with.
///
/// The [id] is the only unlocalized field, because it is the route segment:
/// `/blogs/capture-guide` has to resolve to the same article in every language.
class EducationArticle {
  const EducationArticle({
    required this.id,
    required this.category,
    required this.title,
    required this.sections,
  });

  final String id;
  final String category;
  final String title;
  final List<ArticleSection> sections;

  /// The opening paragraph, shown on the card that links here.
  ///
  /// A genuine excerpt rather than a separate summary string: the ARB has no
  /// summary for this article, and writing one would be inventing copy for a
  /// clinical app. The first paragraph is what the reader gets anyway.
  String? get excerpt => sections.isEmpty ? null : sections.first.body;

  /// Minutes at 200 words per minute, never less than one.
  ///
  /// Derived from the text rather than declared per article, so it cannot drift
  /// away from the body it describes and so a translation reports its own
  /// length instead of English's. 200 wpm is the conventional silent-reading
  /// estimate — a presentational hint, not a measurement.
  int get readingMinutes {
    final words = [
      title,
      for (final section in sections) '${section.heading} ${section.body}',
    ].join(' ').split(RegExp(r'\s+')).where((w) => w.isNotEmpty).length;
    return (words / 200).ceil().clamp(1, 999);
  }
}

/// Route segment for the capture guide. Referenced by the router and by tests.
const String captureGuideArticleId = 'capture-guide';

/// The articles this build actually has copy for.
///
/// Resolved from [AppLocalizations] rather than held as constants, because the
/// body text lives in `app_en.arb` / `app_hi.arb` and is already translated
/// there. Nothing is authored here.
///
/// There is deliberately **one** article. The string table also ships two
/// category labels — `categoryAwareness` ("Understanding oral health") and
/// `categorySigns` ("Signs & symptoms") — with no bodies behind them anywhere in
/// the repo, and no entry in REQUIREMENTS.md describing what they should say.
/// Filling them would mean writing oral-cancer symptom guidance for patients,
/// which AGENTS.md §4.3 and §7 forbid this agent from inventing. They are
/// recorded as `REQUIRES CLARIFICATION` (ISS-015) and the screen simply does not
/// advertise categories it has nothing to put in.
///
/// There is also no local repository behind this. The content is compiled into
/// the app and cannot fail to load, so an [AsyncState] around it would be
/// theatre — the loading and error branches would be unreachable by
/// construction. The list screen still renders an empty state, because the list
/// is injectable and so that state is reachable and tested.
List<EducationArticle> educationArticles(AppLocalizations l) => [
  EducationArticle(
    id: captureGuideArticleId,
    category: l.categoryCapture,
    // `howToCapture`, not `captureGuide`: the latter is the same string as
    // `categoryCapture` in both locales, so using it would print the chip and
    // the headline identically on the same card.
    title: l.howToCapture,
    sections: [
      ArticleSection(heading: l.guideLighting, body: l.guideLightingBody),
      ArticleSection(heading: l.guidePosition, body: l.guidePositionBody),
      ArticleSection(heading: l.guideDistance, body: l.guideDistanceBody),
      ArticleSection(heading: l.guideSteady, body: l.guideSteadyBody),
    ],
  ),
];
