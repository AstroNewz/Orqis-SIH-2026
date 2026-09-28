import 'package:flutter/material.dart';

import 'app.dart';
import 'core/preferences/app_preferences.dart';
import 'features/auth/prototype_session.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await appPreferences.load();
  await appSession.load();
  runApp(const MyApp());
}
