import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/core/theme/app_theme.dart';
import 'package:carescan/features/tracks/models/screening_track.dart';
import 'package:carescan/features/tracks/repositories/track_repository.dart';
import 'package:carescan/features/tracks/screens/tracks_screen.dart';
import 'package:carescan/l10n/l10n.dart';

class FakeTrackRepository implements TrackRepository {
  Result<TrackCatalogue>? result;

  /// When set, `getTracks` blocks on it, so the loading frame can be observed
  /// deterministically instead of racing an already-resolved future.
  Completer<void>? gate;

  int getTracksCalls = 0;

  @override
  Future<Result<TrackCatalogue>> getTracks() async {
    getTracksCalls++;
    if (gate != null) await gate!.future;
    return result ??
        Success(TrackCatalogue(tracks: [_track()], rejected: const []));
  }

  @override
  Future<Result<ScreeningTrack>> getTrack(String trackId) async {
    // The screen never calls this; a fake that silently returned a track would
    // hide it if that ever changed.
    throw StateError('TracksScreen must not fetch a single track');
  }
}

/// A development-estimate track, which is what every track in this build
/// actually is: no frozen test has been evaluated for any of them.
ScreeningTrack _track({
  String trackId = 'ecg-ptbxl',
  String displayName = 'ECG screening',
  String condition = 'Abnormal resting ECG',
  bool ready = true,
  String? unreadyReason,
  bool frozenTestEvaluated = false,
  String partitionScored = 'strat_fold 9 (validation)',
  String dataset = 'PTB-XL v1.0.3 (PhysioNet, CC BY 4.0)',
  double? primaryMetricValue = 0.946293,
  String modelVersion = 'ecg-v1',
  String? quantumRole,
}) {
  return ScreeningTrack(
    trackId: trackId,
    displayName: displayName,
    condition: condition,
    modality: TrackModality.signal,
    inputSpec: const TrackInputSpec(
      modality: TrackModality.signal,
      description: 'a 10-second 12-lead ECG at 100 Hz, in millivolts',
    ),
    validation: TrackValidationSummary(
      dataset: dataset,
      task: 'NORM vs abnormal',
      primaryMetric: 'ROC-AUC',
      primaryMetricValue: primaryMetricValue,
      partitionScored: partitionScored,
      frozenTestEvaluated: frozenTestEvaluated,
      partitionReusedForSelection: !frozenTestEvaluated,
      nRecords: 2158,
      nPatients: 2143,
    ),
    modelVersion: modelVersion,
    ready: ready,
    unreadyReason: unreadyReason,
    primaryModel: 'fusion@cnn+gbm',
    usesQuantum: quantumRole != null,
    quantumRole: quantumRole,
    disclaimer: 'Screening aid only. Not a diagnosis.',
  );
}

Future<void> _pump(WidgetTester tester, FakeTrackRepository repository) async {
  await tester.pumpWidget(
    MaterialApp(
      locale: const Locale('en'),
      theme: AppTheme.dark,
      localizationsDelegates: AppLocalizations.localizationsDelegates,
      supportedLocales: AppLocalizations.supportedLocales,
      home: TracksScreen(repository: repository),
    ),
  );
}

/// Every string the screen actually put on the glass.
///
/// Read from `RichText` rather than `Text` because the captions are built from
/// spans; a `Text` builds a `RichText` too, so this covers both. `find.text`
/// cannot express what these tests need to assert, which is that a *particular*
/// rendered string carries its qualification.
List<String> _rendered(WidgetTester tester) => tester
    .widgetList<RichText>(find.byType(RichText))
    .map((widget) => widget.text.toPlainText())
    .toList(growable: false);

/// The one rendered string containing the headline number.
String _metricLine(WidgetTester tester) {
  final matches = _rendered(
    tester,
  ).where((text) => text.contains('ROC-AUC 0.9463')).toList();
  expect(
    matches,
    hasLength(1),
    reason: 'the headline metric should be rendered exactly once',
  );
  return matches.single;
}

void main() {
  late FakeTrackRepository repository;

  setUp(() => repository = FakeTrackRepository());

  group('TracksScreen', () {
    testWidgets('shows a screen-reader-labelled spinner while loading', (
      tester,
    ) async {
      repository.gate = Completer<void>();

      await _pump(tester, repository);

      final indicator = tester.widget<CircularProgressIndicator>(
        find.byType(CircularProgressIndicator),
      );
      expect(indicator.semanticsLabel, 'Loading screening tracks');

      repository.gate!.complete();
      await tester.pumpAndSettle();
    });

    testWidgets('renders a track with its condition, input and model', (
      tester,
    ) async {
      await _pump(tester, repository);
      await tester.pumpAndSettle();

      expect(find.text('ECG screening'), findsOneWidget);
      expect(find.text('Abnormal resting ECG'), findsOneWidget);

      final rendered = _rendered(tester);
      expect(
        rendered.any((t) => t.contains('12-lead ECG at 100 Hz')),
        isTrue,
        reason: 'a client should be buildable from the published input spec',
      );
      expect(
        rendered.any((t) => t.contains('fusion@cnn+gbm') && t.contains('ecg-v1')),
        isTrue,
      );
      expect(find.text('Screening aid only. Not a diagnosis.'), findsOneWidget);
    });

    testWidgets(
      'a development estimate is rendered with its qualification attached',
      (tester) async {
        // The load-bearing assertion of this file. The number and the words
        // that qualify it must be the *same* rendered string: a screen that
        // showed "ROC-AUC 0.9463" in one place and a caveat elsewhere can lose
        // the caveat to a layout change, a truncation or a screenshot crop.
        await _pump(tester, repository);
        await tester.pumpAndSettle();

        expect(_metricLine(tester), contains('development estimate'));
        expect(_metricLine(tester), contains('not a validated result'));
        expect(_metricLine(tester), contains('strat_fold 9 (validation)'));
      },
    );

    testWidgets('never renders the headline metric as a percentage', (
      tester,
    ) async {
      // 0.946293 is a ranking metric on research data, not a probability and
      // not an accuracy claim about a person. Rendering it as "94.6%" would be
      // a different and false statement, so no percent sign may reach the
      // glass on this screen at all.
      await _pump(tester, repository);
      await tester.pumpAndSettle();

      final rendered = _rendered(tester);
      expect(rendered.any((t) => t.contains('%')), isFalse);
      expect(rendered.any((t) => t.contains('94.6')), isFalse);
    });

    testWidgets('labels a development estimate as one, not as validated', (
      tester,
    ) async {
      await _pump(tester, repository);
      await tester.pumpAndSettle();

      expect(find.text('Development estimate'), findsOneWidget);
      expect(find.text('Frozen test evaluated'), findsNothing);

      final rendered = _rendered(tester);
      expect(
        rendered.any((t) => t.contains('optimistically biased')),
        isTrue,
        reason:
            'the partition also informed model selection, and the caption says so',
      );
    });

    testWidgets('a frozen-test track reads as evaluated instead', (
      tester,
    ) async {
      // Hypothetical: no track in this build has a frozen-test evaluation. The
      // fixture exists so the *other* branch of the badge is exercised, and
      // deliberately names no real held-out partition.
      repository.result = Success(
        TrackCatalogue(
          tracks: [
            _track(
              frozenTestEvaluated: true,
              partitionScored: 'held-out test',
              dataset: 'a hypothetical cohort',
            ),
          ],
          rejected: const [],
        ),
      );

      await _pump(tester, repository);
      await tester.pumpAndSettle();

      expect(find.text('Frozen test evaluated'), findsOneWidget);
      expect(find.text('Development estimate'), findsNothing);
      expect(_metricLine(tester), isNot(contains('development estimate')));
      expect(
        _rendered(tester).any((t) => t.contains('untouched by model selection')),
        isTrue,
      );
    });

    testWidgets('a track without a published number shows no number', (
      tester,
    ) async {
      repository.result = Success(
        TrackCatalogue(
          tracks: [_track(primaryMetricValue: null)],
          rejected: const [],
        ),
      );

      await _pump(tester, repository);
      await tester.pumpAndSettle();

      expect(_rendered(tester).any((t) => t.contains('ROC-AUC 0')), isFalse);
      // The track itself is still listed; a missing metric is not a missing
      // track.
      expect(find.text('ECG screening'), findsOneWidget);
    });

    testWidgets('an unready track is listed with its reason, not hidden', (
      tester,
    ) async {
      repository.result = Success(
        TrackCatalogue(
          tracks: [
            _track(
              ready: false,
              unreadyReason: 'model artifacts are not present in this build',
            ),
          ],
          rejected: const [],
        ),
      );

      await _pump(tester, repository);
      await tester.pumpAndSettle();

      expect(find.text('ECG screening'), findsOneWidget);
      expect(find.text('Not available'), findsOneWidget);
      expect(find.text('Available'), findsNothing);
      expect(
        _rendered(tester).any((t) => t.contains('model artifacts are not present')),
        isTrue,
      );
    });

    testWidgets('a track with no persisted model shows no empty parentheses', (
      tester,
    ) async {
      // The shipped ECG descriptor reports `model_version: ""` — nothing has
      // been frozen for serving. That is the honest answer, not a missing
      // field, and interpolating it produced "gradient_boosted_trees ()".
      repository.result = Success(
        TrackCatalogue(
          tracks: [_track(modelVersion: '', ready: false)],
          rejected: const [],
        ),
      );

      await _pump(tester, repository);
      await tester.pumpAndSettle();

      final model = _rendered(
        tester,
      ).where((t) => t.startsWith('Primary model')).toList();
      expect(model, hasLength(1));
      expect(model.single, contains('fusion@cnn+gbm'));
      expect(model.single, isNot(contains('(')));
    });

    testWidgets('both shapes in the shipped catalogue render together', (
      tester,
    ) async {
      // What `GET /api/tracks` actually returns today: one track that has a
      // number but cannot run, and one that runs but has published no number.
      // Neither combination may drop a track or invent a metric.
      repository.result = Success(
        TrackCatalogue(
          tracks: [
            _track(
              trackId: 'ecg_ptbxl',
              displayName: 'ECG Screening (PTB-XL)',
              ready: false,
              modelVersion: '',
              unreadyReason: 'No ECG classifier has been persisted for serving.',
            ),
            _track(
              trackId: 'oral_lesion',
              displayName: 'Oral Lesion Screening',
              condition: 'Potentially malignant oral lesions',
              primaryMetricValue: null,
              modelVersion: 'v1-handcrafted',
              quantumRole: 'An 8-qubit variational classifier over the '
                  'reduced image descriptor.',
            ),
          ],
          rejected: const [],
        ),
      );

      await _pump(tester, repository);
      await tester.pumpAndSettle();

      expect(find.text('ECG Screening (PTB-XL)'), findsOneWidget);
      expect(find.text('Oral Lesion Screening'), findsOneWidget);
      expect(find.text('Not available'), findsOneWidget);
      expect(find.text('Available'), findsOneWidget);

      // Exactly one number on screen, and it belongs to the ECG track.
      expect(_metricLine(tester), contains('development estimate'));
      // Neither track claims validation, because neither has a frozen test.
      expect(find.text('Frozen test evaluated'), findsNothing);
      expect(find.text('Development estimate'), findsNWidgets(2));
    });

    testWidgets('no quantum line when a track declares no quantum role', (
      tester,
    ) async {
      await _pump(tester, repository);
      await tester.pumpAndSettle();

      expect(
        _rendered(tester).any((t) => t.startsWith('Quantum component')),
        isFalse,
        reason: 'a track without a declared role must not advertise the label',
      );
    });

    testWidgets('shows the declared quantum role, not the bare label', (
      tester,
    ) async {
      repository.result = Success(
        TrackCatalogue(
          tracks: [
            _track(quantumRole: 'a ZZ feature map over 8 fused head features'),
          ],
          rejected: const [],
        ),
      );

      await _pump(tester, repository);
      await tester.pumpAndSettle();

      final quantum = _rendered(
        tester,
      ).where((t) => t.startsWith('Quantum component')).toList();
      expect(quantum, hasLength(1));
      expect(quantum.single, contains('ZZ feature map'));
    });

    testWidgets('refused descriptors are announced and named', (tester) async {
      // A count with no reasons tells the user something is wrong without
      // telling anyone what. A descriptor quietly vanishing from a screening
      // platform is indistinguishable from a schema regression.
      repository.result = Success(
        TrackCatalogue(
          tracks: [_track()],
          rejected: const [
            'oral-lesion: descriptor arrived without frozen_test_evaluated.',
          ],
        ),
      );

      await _pump(tester, repository);
      await tester.pumpAndSettle();

      final rendered = _rendered(tester);
      expect(
        rendered.any((t) => t.contains('One track description could not be read')),
        isTrue,
      );
      expect(
        rendered.any((t) => t.contains('frozen_test_evaluated')),
        isTrue,
        reason: 'the reason itself is what makes the failure diagnosable',
      );
      // The readable track is still shown. Hiding it behind a malformed sibling
      // would make the platform look smaller than it is.
      expect(find.text('ECG screening'), findsOneWidget);
    });

    testWidgets('a healthy catalogue shows no rejection banner', (
      tester,
    ) async {
      await _pump(tester, repository);
      await tester.pumpAndSettle();

      expect(
        _rendered(tester).any((t) => t.contains('could not be read')),
        isFalse,
      );
    });

    testWidgets('pluralises the rejection count', (tester) async {
      repository.result = Success(
        TrackCatalogue(
          tracks: [_track()],
          rejected: const ['a: bad.', 'b: also bad.'],
        ),
      );

      await _pump(tester, repository);
      await tester.pumpAndSettle();

      expect(
        _rendered(tester).any(
          (t) => t.contains('2 track descriptions could not be read'),
        ),
        isTrue,
      );
    });

    testWidgets('an empty catalogue is an empty state, not an error', (
      tester,
    ) async {
      repository.result = const Success(
        TrackCatalogue(tracks: [], rejected: []),
      );

      await _pump(tester, repository);
      await tester.pumpAndSettle();

      expect(find.text('No screening tracks listed'), findsOneWidget);
      expect(find.text('Screening tracks unavailable'), findsNothing);
      expect(find.byType(FilledButton), findsNothing);
    });

    testWidgets('nothing readable is not the same claim as nothing offered', (
      tester,
    ) async {
      // Zero tracks because none were configured and zero tracks because this
      // build could not read any of them are different statements. Only the
      // first one may say "no screening tracks listed"; the second has to say
      // that something was refused, or the user reads a schema regression as a
      // platform with no capabilities.
      repository.result = const Success(
        TrackCatalogue(
          tracks: [],
          rejected: ['oral-lesion: unreadable descriptor.'],
        ),
      );

      await _pump(tester, repository);
      await tester.pumpAndSettle();

      expect(find.text('No screening tracks listed'), findsNothing);
      expect(
        _rendered(tester).any(
          (t) => t.contains('One track description could not be read'),
        ),
        isTrue,
      );
    });

    testWidgets('a failure shows its own message, not a generic one', (
      tester,
    ) async {
      // "Unreachable", "unreadable" and "no such track" want different
      // responses from whoever is holding the phone, so the failure's message
      // reaches the screen rather than being flattened.
      repository.result = const Error(
        ServerFailure('A track described itself without frozen_test_evaluated.'),
      );

      await _pump(tester, repository);
      await tester.pumpAndSettle();

      expect(find.text('Screening tracks unavailable'), findsOneWidget);
      expect(
        _rendered(tester).any((t) => t.contains('without frozen_test_evaluated')),
        isTrue,
      );
      expect(find.text('Try Again'), findsOneWidget);
    });

    testWidgets('retry re-queries and can recover', (tester) async {
      repository.result = const Error(
        NetworkFailure('Unable to connect to backend server'),
      );

      await _pump(tester, repository);
      await tester.pumpAndSettle();
      expect(repository.getTracksCalls, 1);

      repository.result = Success(
        TrackCatalogue(tracks: [_track()], rejected: const []),
      );
      await tester.tap(find.text('Try Again'));
      await tester.pumpAndSettle();

      expect(repository.getTracksCalls, 2);
      expect(find.text('ECG screening'), findsOneWidget);
      expect(find.text('Screening tracks unavailable'), findsNothing);
    });

    testWidgets('renders in Hindi without falling back to English', (
      tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(
          locale: const Locale('hi'),
          theme: AppTheme.dark,
          localizationsDelegates: AppLocalizations.localizationsDelegates,
          supportedLocales: AppLocalizations.supportedLocales,
          home: TracksScreen(repository: repository),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('विकास-चरण का अनुमान'), findsOneWidget);
      expect(find.text('Development estimate'), findsNothing);
      // The metric string itself comes from the model, not the ARB, so it is
      // the same in both locales — and still carries its qualification.
      expect(_metricLine(tester), contains('development estimate'));
    });

    testWidgets('constructs its own repository when none is injected', (
      tester,
    ) async {
      // The default path is what production runs. Left uninjected it performs a
      // real HTTP call that the test binding refuses; the screen must survive
      // that as an error state rather than an exception.
      await tester.pumpWidget(
        MaterialApp(
          locale: const Locale('en'),
          theme: AppTheme.dark,
          localizationsDelegates: AppLocalizations.localizationsDelegates,
          supportedLocales: AppLocalizations.supportedLocales,
          home: const TracksScreen(),
        ),
      );
      await tester.pumpAndSettle();

      expect(tester.takeException(), isNull);
      expect(find.text('Screening tracks unavailable'), findsOneWidget);
    });
  });
}
