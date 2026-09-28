import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:intl/intl.dart' as intl;

import 'app_localizations_en.dart';
import 'app_localizations_hi.dart';

// ignore_for_file: type=lint

/// Callers can lookup localized strings with an instance of AppLocalizations
/// returned by `AppLocalizations.of(context)`.
///
/// Applications need to include `AppLocalizations.delegate()` in their app's
/// `localizationDelegates` list, and the locales they support in the app's
/// `supportedLocales` list. For example:
///
/// ```dart
/// import 'l10n/app_localizations.dart';
///
/// return MaterialApp(
///   localizationsDelegates: AppLocalizations.localizationsDelegates,
///   supportedLocales: AppLocalizations.supportedLocales,
///   home: MyApplicationHome(),
/// );
/// ```
///
/// ## Update pubspec.yaml
///
/// Please make sure to update your pubspec.yaml to include the following
/// packages:
///
/// ```yaml
/// dependencies:
///   # Internationalization support.
///   flutter_localizations:
///     sdk: flutter
///   intl: any # Use the pinned version from flutter_localizations
///
///   # Rest of dependencies
/// ```
///
/// ## iOS Applications
///
/// iOS applications define key application metadata, including supported
/// locales, in an Info.plist file that is built into the application bundle.
/// To configure the locales supported by your app, you’ll need to edit this
/// file.
///
/// First, open your project’s ios/Runner.xcworkspace Xcode workspace file.
/// Then, in the Project Navigator, open the Info.plist file under the Runner
/// project’s Runner folder.
///
/// Next, select the Information Property List item, select Add Item from the
/// Editor menu, then select Localizations from the pop-up menu.
///
/// Select and expand the newly-created Localizations item then, for each
/// locale your application supports, add a new item and select the locale
/// you wish to add from the pop-up menu in the Value field. This list should
/// be consistent with the languages listed in the AppLocalizations.supportedLocales
/// property.
abstract class AppLocalizations {
  AppLocalizations(String locale)
    : localeName = intl.Intl.canonicalizedLocale(locale.toString());

  final String localeName;

  static AppLocalizations? of(BuildContext context) {
    return Localizations.of<AppLocalizations>(context, AppLocalizations);
  }

  static const LocalizationsDelegate<AppLocalizations> delegate =
      _AppLocalizationsDelegate();

  /// A list of this localizations delegate along with the default localizations
  /// delegates.
  ///
  /// Returns a list of localizations delegates containing this delegate along with
  /// GlobalMaterialLocalizations.delegate, GlobalCupertinoLocalizations.delegate,
  /// and GlobalWidgetsLocalizations.delegate.
  ///
  /// Additional delegates can be added by appending to this list in
  /// MaterialApp. This list does not have to be used at all if a custom list
  /// of delegates is preferred or required.
  static const List<LocalizationsDelegate<dynamic>> localizationsDelegates =
      <LocalizationsDelegate<dynamic>>[
        delegate,
        GlobalMaterialLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
      ];

  /// A list of this localizations delegate's supported locales.
  static const List<Locale> supportedLocales = <Locale>[
    Locale('en'),
    Locale('hi'),
  ];

  /// No description provided for @appName.
  ///
  /// In en, this message translates to:
  /// **'CareScan'**
  String get appName;

  /// No description provided for @home.
  ///
  /// In en, this message translates to:
  /// **'Home'**
  String get home;

  /// No description provided for @blogs.
  ///
  /// In en, this message translates to:
  /// **'Blogs'**
  String get blogs;

  /// No description provided for @scan.
  ///
  /// In en, this message translates to:
  /// **'Scan'**
  String get scan;

  /// No description provided for @history.
  ///
  /// In en, this message translates to:
  /// **'History'**
  String get history;

  /// No description provided for @profile.
  ///
  /// In en, this message translates to:
  /// **'Profile'**
  String get profile;

  /// No description provided for @startScreening.
  ///
  /// In en, this message translates to:
  /// **'Start Screening'**
  String get startScreening;

  /// No description provided for @viewHistory.
  ///
  /// In en, this message translates to:
  /// **'View History'**
  String get viewHistory;

  /// No description provided for @retry.
  ///
  /// In en, this message translates to:
  /// **'Try Again'**
  String get retry;

  /// No description provided for @close.
  ///
  /// In en, this message translates to:
  /// **'Close'**
  String get close;

  /// No description provided for @back.
  ///
  /// In en, this message translates to:
  /// **'Back'**
  String get back;

  /// No description provided for @cancel.
  ///
  /// In en, this message translates to:
  /// **'Cancel'**
  String get cancel;

  /// No description provided for @continueLabel.
  ///
  /// In en, this message translates to:
  /// **'Continue'**
  String get continueLabel;

  /// No description provided for @save.
  ///
  /// In en, this message translates to:
  /// **'Save'**
  String get save;

  /// No description provided for @done.
  ///
  /// In en, this message translates to:
  /// **'Done'**
  String get done;

  /// No description provided for @welcome.
  ///
  /// In en, this message translates to:
  /// **'Welcome to CareScan'**
  String get welcome;

  /// No description provided for @heroTitle.
  ///
  /// In en, this message translates to:
  /// **'Screen early.\nUnderstand sooner.'**
  String get heroTitle;

  /// No description provided for @heroSubtitle.
  ///
  /// In en, this message translates to:
  /// **'AI-assisted screening support for oral health.'**
  String get heroSubtitle;

  /// No description provided for @oralScreening.
  ///
  /// In en, this message translates to:
  /// **'Oral health screening'**
  String get oralScreening;

  /// No description provided for @prototype.
  ///
  /// In en, this message translates to:
  /// **'SIH 2026 · Prototype'**
  String get prototype;

  /// No description provided for @disclaimer.
  ///
  /// In en, this message translates to:
  /// **'This is a screening assessment, not a diagnosis. A photograph cannot rule out disease. Consult a qualified dental or oral-health professional about any concern.'**
  String get disclaimer;

  /// No description provided for @educationDisclaimer.
  ///
  /// In en, this message translates to:
  /// **'For educational information only. This article does not diagnose a condition or replace a clinical examination. CareScan is a research prototype; clinical validation has not been established.'**
  String get educationDisclaimer;

  /// No description provided for @recentScreening.
  ///
  /// In en, this message translates to:
  /// **'Recent screening'**
  String get recentScreening;

  /// No description provided for @viewAll.
  ///
  /// In en, this message translates to:
  /// **'View all'**
  String get viewAll;

  /// No description provided for @noScreenings.
  ///
  /// In en, this message translates to:
  /// **'No screenings yet'**
  String get noScreenings;

  /// No description provided for @noScreeningsBody.
  ///
  /// In en, this message translates to:
  /// **'Your completed screenings will appear here. Start with a guided oral photograph.'**
  String get noScreeningsBody;

  /// No description provided for @firstScreening.
  ///
  /// In en, this message translates to:
  /// **'Start your first screening'**
  String get firstScreening;

  /// No description provided for @historyError.
  ///
  /// In en, this message translates to:
  /// **'Screening history is unavailable'**
  String get historyError;

  /// No description provided for @connectionHelp.
  ///
  /// In en, this message translates to:
  /// **'Check your connection to the screening service and try again.'**
  String get connectionHelp;

  /// No description provided for @loadingHistory.
  ///
  /// In en, this message translates to:
  /// **'Loading history'**
  String get loadingHistory;

  /// No description provided for @refresh.
  ///
  /// In en, this message translates to:
  /// **'Refresh'**
  String get refresh;

  /// No description provided for @howItWorks.
  ///
  /// In en, this message translates to:
  /// **'A little guidance. A clearer picture.'**
  String get howItWorks;

  /// No description provided for @captureStep.
  ///
  /// In en, this message translates to:
  /// **'Guided capture'**
  String get captureStep;

  /// No description provided for @qualityStep.
  ///
  /// In en, this message translates to:
  /// **'Image check'**
  String get qualityStep;

  /// No description provided for @resultStep.
  ///
  /// In en, this message translates to:
  /// **'Screening & explanation'**
  String get resultStep;

  /// No description provided for @healthEducation.
  ///
  /// In en, this message translates to:
  /// **'Health education'**
  String get healthEducation;

  /// No description provided for @educationIntro.
  ///
  /// In en, this message translates to:
  /// **'Understand changes in your mouth, and what a screening result can tell you.'**
  String get educationIntro;

  /// No description provided for @language.
  ///
  /// In en, this message translates to:
  /// **'Language'**
  String get language;

  /// No description provided for @english.
  ///
  /// In en, this message translates to:
  /// **'English'**
  String get english;

  /// No description provided for @hindi.
  ///
  /// In en, this message translates to:
  /// **'हिन्दी'**
  String get hindi;

  /// No description provided for @theme.
  ///
  /// In en, this message translates to:
  /// **'Appearance'**
  String get theme;

  /// No description provided for @systemTheme.
  ///
  /// In en, this message translates to:
  /// **'Use device setting'**
  String get systemTheme;

  /// No description provided for @lightTheme.
  ///
  /// In en, this message translates to:
  /// **'Light'**
  String get lightTheme;

  /// No description provided for @darkTheme.
  ///
  /// In en, this message translates to:
  /// **'Dark'**
  String get darkTheme;

  /// No description provided for @settingsError.
  ///
  /// In en, this message translates to:
  /// **'Could not load settings'**
  String get settingsError;

  /// No description provided for @saveError.
  ///
  /// In en, this message translates to:
  /// **'Could not save your preference. Please try again.'**
  String get saveError;

  /// No description provided for @privacy.
  ///
  /// In en, this message translates to:
  /// **'Privacy & your data'**
  String get privacy;

  /// No description provided for @privacyBody.
  ///
  /// In en, this message translates to:
  /// **'Before analysis, your photograph is uploaded to the configured screening server. Screening records are stored there under a prototype identifier. Patient accounts and server access controls are not available in this demo. Use only consented demonstration images. Signing out does not delete server records.'**
  String get privacyBody;

  /// No description provided for @about.
  ///
  /// In en, this message translates to:
  /// **'About CareScan'**
  String get about;

  /// No description provided for @aboutBody.
  ///
  /// In en, this message translates to:
  /// **'CareScan / Orqis is an SIH 2026 research prototype for AI-assisted oral health screening. It supports image capture, screening results and education. It is not a diagnostic medical device.'**
  String get aboutBody;

  /// No description provided for @medicalDisclaimer.
  ///
  /// In en, this message translates to:
  /// **'Medical disclaimer'**
  String get medicalDisclaimer;

  /// No description provided for @logout.
  ///
  /// In en, this message translates to:
  /// **'Log Out'**
  String get logout;

  /// No description provided for @logoutTitle.
  ///
  /// In en, this message translates to:
  /// **'Leave this session?'**
  String get logoutTitle;

  /// No description provided for @logoutBody.
  ///
  /// In en, this message translates to:
  /// **'Your local profile remains on this device. Guest history cannot be recovered after leaving. Server records are not deleted.'**
  String get logoutBody;

  /// No description provided for @version.
  ///
  /// In en, this message translates to:
  /// **'Version 1.0.0 · SIH prototype'**
  String get version;

  /// No description provided for @login.
  ///
  /// In en, this message translates to:
  /// **'Login'**
  String get login;

  /// No description provided for @createAccount.
  ///
  /// In en, this message translates to:
  /// **'Create account'**
  String get createAccount;

  /// No description provided for @guest.
  ///
  /// In en, this message translates to:
  /// **'Continue as guest'**
  String get guest;

  /// No description provided for @guestName.
  ///
  /// In en, this message translates to:
  /// **'Guest session'**
  String get guestName;

  /// No description provided for @localProfile.
  ///
  /// In en, this message translates to:
  /// **'Device-only profile'**
  String get localProfile;

  /// No description provided for @prototypeAuth.
  ///
  /// In en, this message translates to:
  /// **'Prototype access'**
  String get prototypeAuth;

  /// No description provided for @authTitle.
  ///
  /// In en, this message translates to:
  /// **'A clearer view of\nyour oral health.'**
  String get authTitle;

  /// No description provided for @authSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Guided photographs. Understandable screening results. A history you can revisit.'**
  String get authSubtitle;

  /// No description provided for @authNote.
  ///
  /// In en, this message translates to:
  /// **'Patient sign-in is not connected yet. Create a device-only demo profile or continue as a guest. No email or password is collected.'**
  String get authNote;

  /// No description provided for @displayName.
  ///
  /// In en, this message translates to:
  /// **'Display name'**
  String get displayName;

  /// No description provided for @nameHint.
  ///
  /// In en, this message translates to:
  /// **'Use a demonstration name'**
  String get nameHint;

  /// No description provided for @nameRequired.
  ///
  /// In en, this message translates to:
  /// **'Enter a name between 2 and 60 characters.'**
  String get nameRequired;

  /// No description provided for @createLocalProfile.
  ///
  /// In en, this message translates to:
  /// **'Create demo profile'**
  String get createLocalProfile;

  /// No description provided for @resumeProfile.
  ///
  /// In en, this message translates to:
  /// **'Open saved profile'**
  String get resumeProfile;

  /// No description provided for @noSavedProfile.
  ///
  /// In en, this message translates to:
  /// **'No profile saved on this device. Create a demo profile to get started.'**
  String get noSavedProfile;

  /// No description provided for @localAccessNote.
  ///
  /// In en, this message translates to:
  /// **'This is local profile access, not authenticated patient login. Anyone using this unlocked device can open the saved profile.'**
  String get localAccessNote;

  /// No description provided for @storageError.
  ///
  /// In en, this message translates to:
  /// **'Device storage is unavailable. Try again or continue as a guest.'**
  String get storageError;

  /// No description provided for @captureTitle.
  ///
  /// In en, this message translates to:
  /// **'Guided oral capture'**
  String get captureTitle;

  /// No description provided for @howToCapture.
  ///
  /// In en, this message translates to:
  /// **'How to capture'**
  String get howToCapture;

  /// No description provided for @captureGuide.
  ///
  /// In en, this message translates to:
  /// **'Capture guide'**
  String get captureGuide;

  /// No description provided for @guideIntro.
  ///
  /// In en, this message translates to:
  /// **'Use this illustration as a positioning reference. It is not a patient photograph.'**
  String get guideIntro;

  /// No description provided for @guideLighting.
  ///
  /// In en, this message translates to:
  /// **'Face a soft, bright light'**
  String get guideLighting;

  /// No description provided for @guideLightingBody.
  ///
  /// In en, this message translates to:
  /// **'Light the inside of your mouth evenly. Avoid shadows and direct flash glare.'**
  String get guideLightingBody;

  /// No description provided for @guidePosition.
  ///
  /// In en, this message translates to:
  /// **'Center the mouth'**
  String get guidePosition;

  /// No description provided for @guidePositionBody.
  ///
  /// In en, this message translates to:
  /// **'Hold the phone upright at mouth level. Open comfortably; keep the oral area inside the guide. Rest your tongue naturally so it does not cover the area you want to show.'**
  String get guidePositionBody;

  /// No description provided for @guideDistance.
  ///
  /// In en, this message translates to:
  /// **'Fill the guide, keep focus'**
  String get guideDistance;

  /// No description provided for @guideDistanceBody.
  ///
  /// In en, this message translates to:
  /// **'Begin about 15–20 cm away and adjust until the mouth fills the guide and stays sharp. Camera focus distances vary. Do not use digital zoom.'**
  String get guideDistanceBody;

  /// No description provided for @guideSteady.
  ///
  /// In en, this message translates to:
  /// **'Hold still before capture'**
  String get guideSteady;

  /// No description provided for @guideSteadyBody.
  ///
  /// In en, this message translates to:
  /// **'Support the phone with both hands or ask someone you trust to help. Do not force your mouth open or use sharp objects.'**
  String get guideSteadyBody;

  /// No description provided for @understood.
  ///
  /// In en, this message translates to:
  /// **'I understand — open camera'**
  String get understood;

  /// No description provided for @reference.
  ///
  /// In en, this message translates to:
  /// **'View reference'**
  String get reference;

  /// No description provided for @centerMouth.
  ///
  /// In en, this message translates to:
  /// **'Center your mouth inside the guide'**
  String get centerMouth;

  /// No description provided for @steady.
  ///
  /// In en, this message translates to:
  /// **'Keep the camera steady'**
  String get steady;

  /// No description provided for @moreLight.
  ///
  /// In en, this message translates to:
  /// **'Use better lighting'**
  String get moreLight;

  /// No description provided for @lessLight.
  ///
  /// In en, this message translates to:
  /// **'Reduce glare or bright light'**
  String get lessLight;

  /// No description provided for @cameraLoading.
  ///
  /// In en, this message translates to:
  /// **'Initializing camera'**
  String get cameraLoading;

  /// No description provided for @cameraError.
  ///
  /// In en, this message translates to:
  /// **'Camera unavailable'**
  String get cameraError;

  /// No description provided for @cameraPermission.
  ///
  /// In en, this message translates to:
  /// **'Allow camera access in your device settings, then try again. You can also select a photograph.'**
  String get cameraPermission;

  /// No description provided for @cameraFailure.
  ///
  /// In en, this message translates to:
  /// **'The camera could not start. Try again or select a photograph from your gallery.'**
  String get cameraFailure;

  /// No description provided for @flipCamera.
  ///
  /// In en, this message translates to:
  /// **'Switch camera'**
  String get flipCamera;

  /// No description provided for @flash.
  ///
  /// In en, this message translates to:
  /// **'Toggle light'**
  String get flash;

  /// No description provided for @gallery.
  ///
  /// In en, this message translates to:
  /// **'Choose from gallery'**
  String get gallery;

  /// No description provided for @capture.
  ///
  /// In en, this message translates to:
  /// **'Capture image'**
  String get capture;

  /// No description provided for @captureFailed.
  ///
  /// In en, this message translates to:
  /// **'Could not capture this image. Please try again.'**
  String get captureFailed;

  /// No description provided for @galleryFailed.
  ///
  /// In en, this message translates to:
  /// **'Could not open this photo. Try a JPEG or PNG image.'**
  String get galleryFailed;

  /// No description provided for @flashUnavailable.
  ///
  /// In en, this message translates to:
  /// **'This camera does not support the light control.'**
  String get flashUnavailable;

  /// No description provided for @readinessWaiting.
  ///
  /// In en, this message translates to:
  /// **'Checking light'**
  String get readinessWaiting;

  /// No description provided for @readinessAdjust.
  ///
  /// In en, this message translates to:
  /// **'Adjust lighting'**
  String get readinessAdjust;

  /// No description provided for @readinessGood.
  ///
  /// In en, this message translates to:
  /// **'Light is suitable'**
  String get readinessGood;

  /// No description provided for @manualFraming.
  ///
  /// In en, this message translates to:
  /// **'Visual guide only. Check mouth position yourself.'**
  String get manualFraming;

  /// No description provided for @manualReady.
  ///
  /// In en, this message translates to:
  /// **'Frame your mouth, then capture'**
  String get manualReady;

  /// No description provided for @preview.
  ///
  /// In en, this message translates to:
  /// **'Review your image'**
  String get preview;

  /// No description provided for @checkingImage.
  ///
  /// In en, this message translates to:
  /// **'Checking image quality'**
  String get checkingImage;

  /// No description provided for @useImage.
  ///
  /// In en, this message translates to:
  /// **'Confirm & analyze'**
  String get useImage;

  /// No description provided for @retake.
  ///
  /// In en, this message translates to:
  /// **'Retake'**
  String get retake;

  /// No description provided for @noImage.
  ///
  /// In en, this message translates to:
  /// **'No image selected'**
  String get noImage;

  /// No description provided for @noImageBody.
  ///
  /// In en, this message translates to:
  /// **'Return to the camera to take or select an oral photograph.'**
  String get noImageBody;

  /// No description provided for @qualityPassed.
  ///
  /// In en, this message translates to:
  /// **'Basic image checks passed'**
  String get qualityPassed;

  /// No description provided for @qualityPassedBody.
  ///
  /// In en, this message translates to:
  /// **'Light, focus and resolution passed the prototype checks. Confirm that this is an oral photograph before continuing.'**
  String get qualityPassedBody;

  /// No description provided for @confirmMouth.
  ///
  /// In en, this message translates to:
  /// **'I can clearly see the oral area I want screened.'**
  String get confirmMouth;

  /// No description provided for @uploadConsent.
  ///
  /// In en, this message translates to:
  /// **'I agree to send this image to the screening server. I have permission to use this photograph.'**
  String get uploadConsent;

  /// No description provided for @qualityDark.
  ///
  /// In en, this message translates to:
  /// **'Image too dark'**
  String get qualityDark;

  /// No description provided for @qualityBright.
  ///
  /// In en, this message translates to:
  /// **'Image has too much glare'**
  String get qualityBright;

  /// No description provided for @qualityBlur.
  ///
  /// In en, this message translates to:
  /// **'Image is blurry'**
  String get qualityBlur;

  /// No description provided for @qualitySmall.
  ///
  /// In en, this message translates to:
  /// **'Image resolution is too low'**
  String get qualitySmall;

  /// No description provided for @qualityFormat.
  ///
  /// In en, this message translates to:
  /// **'This image format cannot be checked'**
  String get qualityFormat;

  /// No description provided for @qualitySize.
  ///
  /// In en, this message translates to:
  /// **'Image is too large'**
  String get qualitySize;

  /// No description provided for @qualityContext.
  ///
  /// In en, this message translates to:
  /// **'This image may not show the mouth clearly'**
  String get qualityContext;

  /// No description provided for @qualityContextBody.
  ///
  /// In en, this message translates to:
  /// **'The prototype could not confirm oral framing from the image. Position the mouth inside the guide and try again. This check uses image heuristics, not a trained mouth detector.'**
  String get qualityContextBody;

  /// No description provided for @qualityRetryBody.
  ///
  /// In en, this message translates to:
  /// **'Use even lighting, hold the camera steady and keep the mouth in focus. Retake the original image without filters.'**
  String get qualityRetryBody;

  /// No description provided for @qualityUnavailable.
  ///
  /// In en, this message translates to:
  /// **'Could not check this image'**
  String get qualityUnavailable;

  /// No description provided for @qualityLimits.
  ///
  /// In en, this message translates to:
  /// **'These checks cannot verify anatomy or rule out every unrelated image.'**
  String get qualityLimits;

  /// No description provided for @analysisTitle.
  ///
  /// In en, this message translates to:
  /// **'Analyzing your image'**
  String get analysisTitle;

  /// No description provided for @analysisBody.
  ///
  /// In en, this message translates to:
  /// **'The screening service is processing your photograph. You will see the result when it responds.'**
  String get analysisBody;

  /// No description provided for @analysisFailed.
  ///
  /// In en, this message translates to:
  /// **'Analysis unavailable'**
  String get analysisFailed;

  /// No description provided for @analysisStage.
  ///
  /// In en, this message translates to:
  /// **'Waiting for screening result'**
  String get analysisStage;

  /// No description provided for @leaveAnalysis.
  ///
  /// In en, this message translates to:
  /// **'Return home'**
  String get leaveAnalysis;

  /// No description provided for @analysisLeaveNote.
  ///
  /// In en, this message translates to:
  /// **'Leaving this screen does not cancel a request already received by the server. Check History before submitting again.'**
  String get analysisLeaveNote;

  /// No description provided for @resultTitle.
  ///
  /// In en, this message translates to:
  /// **'Screening Result'**
  String get resultTitle;

  /// No description provided for @riskLabel.
  ///
  /// In en, this message translates to:
  /// **'Screening risk level'**
  String get riskLabel;

  /// No description provided for @lowRisk.
  ///
  /// In en, this message translates to:
  /// **'LOW RISK'**
  String get lowRisk;

  /// No description provided for @moderateRisk.
  ///
  /// In en, this message translates to:
  /// **'MODERATE RISK'**
  String get moderateRisk;

  /// No description provided for @highRisk.
  ///
  /// In en, this message translates to:
  /// **'HIGH RISK'**
  String get highRisk;

  /// No description provided for @unknownRisk.
  ///
  /// In en, this message translates to:
  /// **'Assessment unavailable'**
  String get unknownRisk;

  /// No description provided for @pendingRisk.
  ///
  /// In en, this message translates to:
  /// **'Processing'**
  String get pendingRisk;

  /// No description provided for @lowMeaning.
  ///
  /// In en, this message translates to:
  /// **'Below the model\'s screening threshold for this image. This does not rule out a condition.'**
  String get lowMeaning;

  /// No description provided for @concernMeaning.
  ///
  /// In en, this message translates to:
  /// **'This screening result suggests the image may warrant professional evaluation.'**
  String get concernMeaning;

  /// No description provided for @unknownMeaning.
  ///
  /// In en, this message translates to:
  /// **'A usable screening result is not available. Please check your history or try again.'**
  String get unknownMeaning;

  /// No description provided for @nextStep.
  ///
  /// In en, this message translates to:
  /// **'What to do next'**
  String get nextStep;

  /// No description provided for @nextStepBody.
  ///
  /// In en, this message translates to:
  /// **'Consider evaluation by a qualified dental or oral-health professional. If a change persists or worries you, seek an examination regardless of this result.'**
  String get nextStepBody;

  /// No description provided for @explanation.
  ///
  /// In en, this message translates to:
  /// **'Understanding this result'**
  String get explanation;

  /// No description provided for @scoreNote.
  ///
  /// In en, this message translates to:
  /// **'The primary model provides a ranking score, not a calibrated disease probability. No percentage is shown for it.'**
  String get scoreNote;

  /// No description provided for @probability.
  ///
  /// In en, this message translates to:
  /// **'Model probability'**
  String get probability;

  /// No description provided for @originalImage.
  ///
  /// In en, this message translates to:
  /// **'Original image'**
  String get originalImage;

  /// No description provided for @highlightedImage.
  ///
  /// In en, this message translates to:
  /// **'Suggested region'**
  String get highlightedImage;

  /// No description provided for @localizationNote.
  ///
  /// In en, this message translates to:
  /// **'This box is suggested by a separate region localizer. It does not show what caused the screening result and does not confirm a lesion.'**
  String get localizationNote;

  /// No description provided for @localizationUnavailable.
  ///
  /// In en, this message translates to:
  /// **'Region overlay unavailable. The screening result is unchanged.'**
  String get localizationUnavailable;

  /// No description provided for @researchTitle.
  ///
  /// In en, this message translates to:
  /// **'Experimental Research Analysis'**
  String get researchTitle;

  /// No description provided for @researchNote.
  ///
  /// In en, this message translates to:
  /// **'Secondary quantum-model output for research. It does not replace the primary screening result. Quantum advantage and clinical validation have not been established.'**
  String get researchNote;

  /// No description provided for @researchProbability.
  ///
  /// In en, this message translates to:
  /// **'Secondary calibrated probability'**
  String get researchProbability;

  /// No description provided for @modelVersion.
  ///
  /// In en, this message translates to:
  /// **'Model version'**
  String get modelVersion;

  /// No description provided for @testData.
  ///
  /// In en, this message translates to:
  /// **'DEMO DATA — not a clinical result'**
  String get testData;

  /// No description provided for @newScan.
  ///
  /// In en, this message translates to:
  /// **'New Scan'**
  String get newScan;

  /// No description provided for @imageUnavailable.
  ///
  /// In en, this message translates to:
  /// **'Image unavailable'**
  String get imageUnavailable;

  /// No description provided for @featured.
  ///
  /// In en, this message translates to:
  /// **'Featured article'**
  String get featured;

  /// No description provided for @allArticles.
  ///
  /// In en, this message translates to:
  /// **'All articles'**
  String get allArticles;

  /// No description provided for @categoryAwareness.
  ///
  /// In en, this message translates to:
  /// **'Understanding oral health'**
  String get categoryAwareness;

  /// No description provided for @categorySigns.
  ///
  /// In en, this message translates to:
  /// **'Signs & symptoms'**
  String get categorySigns;

  /// No description provided for @categoryCapture.
  ///
  /// In en, this message translates to:
  /// **'Capture guide'**
  String get categoryCapture;

  /// No description provided for @author.
  ///
  /// In en, this message translates to:
  /// **'Ishan Aran Shukla'**
  String get author;

  /// No description provided for @relatedArticle.
  ///
  /// In en, this message translates to:
  /// **'Read next'**
  String get relatedArticle;

  /// No description provided for @articleNotFound.
  ///
  /// In en, this message translates to:
  /// **'Article not found'**
  String get articleNotFound;

  /// No description provided for @published.
  ///
  /// In en, this message translates to:
  /// **'Published'**
  String get published;

  /// No description provided for @editorialIntro.
  ///
  /// In en, this message translates to:
  /// **'Knowledge for the next step.'**
  String get editorialIntro;

  /// No description provided for @editorialSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Practical reading on oral health, screening and taking a useful photograph.'**
  String get editorialSubtitle;

  /// No description provided for @readingTime.
  ///
  /// In en, this message translates to:
  /// **'{minutes} min read'**
  String readingTime(int minutes);

  /// No description provided for @tracks.
  ///
  /// In en, this message translates to:
  /// **'Screening Tracks'**
  String get tracks;

  /// No description provided for @tracksIntro.
  ///
  /// In en, this message translates to:
  /// **'Every condition this platform can screen for, and the validation standing behind each published number.'**
  String get tracksIntro;

  /// No description provided for @loadingTracks.
  ///
  /// In en, this message translates to:
  /// **'Loading screening tracks'**
  String get loadingTracks;

  /// No description provided for @tracksError.
  ///
  /// In en, this message translates to:
  /// **'Screening tracks unavailable'**
  String get tracksError;

  /// No description provided for @tracksEmpty.
  ///
  /// In en, this message translates to:
  /// **'No screening tracks listed'**
  String get tracksEmpty;

  /// No description provided for @tracksEmptyBody.
  ///
  /// In en, this message translates to:
  /// **'The screening service answered but listed no tracks. Screening is unavailable until one is configured.'**
  String get tracksEmptyBody;

  /// No description provided for @trackAvailable.
  ///
  /// In en, this message translates to:
  /// **'Available'**
  String get trackAvailable;

  /// No description provided for @trackUnavailable.
  ///
  /// In en, this message translates to:
  /// **'Not available'**
  String get trackUnavailable;

  /// No description provided for @trackFrozenTest.
  ///
  /// In en, this message translates to:
  /// **'Frozen test evaluated'**
  String get trackFrozenTest;

  /// No description provided for @trackDevelopmentEstimate.
  ///
  /// In en, this message translates to:
  /// **'Development estimate'**
  String get trackDevelopmentEstimate;

  /// No description provided for @trackInput.
  ///
  /// In en, this message translates to:
  /// **'Expected input'**
  String get trackInput;

  /// No description provided for @trackModel.
  ///
  /// In en, this message translates to:
  /// **'Primary model'**
  String get trackModel;

  /// No description provided for @trackProvenance.
  ///
  /// In en, this message translates to:
  /// **'How this number was obtained'**
  String get trackProvenance;

  /// No description provided for @trackQuantum.
  ///
  /// In en, this message translates to:
  /// **'Quantum component'**
  String get trackQuantum;

  /// No description provided for @tracksRejected.
  ///
  /// In en, this message translates to:
  /// **'{count, plural, =1{One track description could not be read} other{{count} track descriptions could not be read}}'**
  String tracksRejected(int count);

  /// No description provided for @tracksRejectedBody.
  ///
  /// In en, this message translates to:
  /// **'The service described these tracks in a way this build could not read. They are withheld rather than shown with unverifiable provenance. Updating the app may resolve this.'**
  String get tracksRejectedBody;

  /// No description provided for @tracksNote.
  ///
  /// In en, this message translates to:
  /// **'A number shown here describes model performance on research data. It is not a statement of accuracy for any individual person.'**
  String get tracksNote;
}

class _AppLocalizationsDelegate
    extends LocalizationsDelegate<AppLocalizations> {
  const _AppLocalizationsDelegate();

  @override
  Future<AppLocalizations> load(Locale locale) {
    return SynchronousFuture<AppLocalizations>(lookupAppLocalizations(locale));
  }

  @override
  bool isSupported(Locale locale) =>
      <String>['en', 'hi'].contains(locale.languageCode);

  @override
  bool shouldReload(_AppLocalizationsDelegate old) => false;
}

AppLocalizations lookupAppLocalizations(Locale locale) {
  // Lookup logic when only language code is specified.
  switch (locale.languageCode) {
    case 'en':
      return AppLocalizationsEn();
    case 'hi':
      return AppLocalizationsHi();
  }

  throw FlutterError(
    'AppLocalizations.delegate failed to load unsupported locale "$locale". This is likely '
    'an issue with the localizations generation tool. Please file an issue '
    'on GitHub with a reproducible sample app and the gen-l10n configuration '
    'that was used.',
  );
}
