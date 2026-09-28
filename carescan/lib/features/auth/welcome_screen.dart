import 'package:flutter/material.dart';
import 'package:carescan/core/preferences/app_preferences.dart';
import 'package:carescan/features/auth/prototype_session.dart';
import 'package:carescan/l10n/l10n.dart';
import 'package:carescan/shared/widgets/product_components.dart';

class WelcomeScreen extends StatefulWidget {
  const WelcomeScreen({super.key, this.session});
  final PrototypeSession? session;
  @override
  State<WelcomeScreen> createState() => _WelcomeScreenState();
}

class _WelcomeScreenState extends State<WelcomeScreen> {
  final _name = TextEditingController();
  final _form = GlobalKey<FormState>();
  bool _creating = false;
  bool _login = false;
  bool _busy = false;
  bool _error = false;
  PrototypeSession get _session => widget.session ?? appSession;
  @override
  void dispose() {
    _name.dispose();
    super.dispose();
  }

  Future<void> _create() async {
    if (!(_form.currentState?.validate() ?? false) || _busy) return;
    setState(() {
      _busy = true;
      _error = false;
    });
    final ok = await _session.createProfile(_name.text);
    if (mounted) {
      setState(() {
        _busy = false;
        _error = !ok;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final t = Theme.of(context);
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          child: ContentPane(
            maxWidth: 560,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Row(
                  children: [
                    const Expanded(child: CareScanBrand()),
                    PopupMenuButton<String>(
                      tooltip: l.language,
                      icon: const Icon(Icons.language),
                      onSelected: (code) async {
                        final saved = await appPreferences.setLocale(
                          Locale(code),
                        );
                        if (!saved && context.mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(content: Text(context.l10n.saveError)),
                          );
                        }
                      },
                      itemBuilder: (_) => [
                        PopupMenuItem(value: 'en', child: Text(l.english)),
                        PopupMenuItem(value: 'hi', child: Text(l.hindi)),
                      ],
                    ),
                  ],
                ),
                const SizedBox(height: 48),
                Text(
                  l.prototype,
                  style: t.textTheme.labelMedium?.copyWith(
                    color: t.colorScheme.primary,
                  ),
                ),
                const SizedBox(height: 16),
                Text(l.authTitle, style: t.textTheme.displaySmall),
                const SizedBox(height: 16),
                Text(
                  l.authSubtitle,
                  style: t.textTheme.bodyLarge?.copyWith(
                    color: t.colorScheme.onSurfaceVariant,
                  ),
                ),
                const SizedBox(height: 32),
                InfoNote(text: l.authNote),
                const SizedBox(height: 24),
                if (_creating && _session.savedProfile == null) ...[
                  Form(
                    key: _form,
                    child: TextFormField(
                      controller: _name,
                      maxLength: 60,
                      textCapitalization: TextCapitalization.words,
                      decoration: InputDecoration(
                        labelText: l.displayName,
                        hintText: l.nameHint,
                      ),
                      validator: (value) => (value?.trim().length ?? 0) < 2
                          ? l.nameRequired
                          : null,
                      textInputAction: TextInputAction.done,
                      onFieldSubmitted: (_) => _create(),
                    ),
                  ),
                  const SizedBox(height: 12),
                  FilledButton(
                    onPressed: _busy ? null : _create,
                    child: _busy
                        ? const SizedBox.square(
                            dimension: 22,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : Text(l.createLocalProfile),
                  ),
                ] else if (_login || _creating) ...[
                  if (_session.savedProfile case final saved?) ...[
                    Text(saved.name, style: t.textTheme.titleLarge),
                    const SizedBox(height: 8),
                    Text(l.localAccessNote),
                    const SizedBox(height: 16),
                    FilledButton(
                      onPressed: _session.login,
                      child: Text(l.resumeProfile),
                    ),
                  ] else ...[
                    Text(l.noSavedProfile),
                    const SizedBox(height: 16),
                    FilledButton(
                      onPressed: () => setState(() {
                        _creating = true;
                        _login = false;
                      }),
                      child: Text(l.createLocalProfile),
                    ),
                  ],
                ] else ...[
                  FilledButton(
                    onPressed: () => setState(() => _login = true),
                    child: Text(l.login),
                  ),
                  const SizedBox(height: 12),
                  OutlinedButton(
                    onPressed: () => setState(() => _creating = true),
                    child: Text(l.createAccount),
                  ),
                ],
                if (_error)
                  Padding(
                    padding: const EdgeInsets.only(top: 12),
                    child: Text(
                      l.storageError,
                      style: TextStyle(color: t.colorScheme.error),
                    ),
                  ),
                const SizedBox(height: 12),
                TextButton(
                  onPressed: _busy ? null : _session.continueAsGuest,
                  child: Text(l.guest),
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
}
