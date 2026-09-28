import 'package:flutter/material.dart';

import 'package:carescan/core/errors/async_state.dart';
import 'package:carescan/core/errors/failures.dart';
import 'package:carescan/core/errors/result.dart';
import 'package:carescan/core/theme/app_shapes.dart';
import 'package:carescan/core/theme/app_spacing.dart';
import 'package:carescan/features/settings/models/user_settings.dart';
import 'package:carescan/features/settings/repositories/mock_settings_repository.dart';
import 'package:carescan/features/settings/repositories/settings_repository.dart';

class SettingsScreen extends StatefulWidget {
  final SettingsRepository? repository;

  const SettingsScreen({super.key, this.repository});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final SettingsRepository _repository;

  AsyncState<UserSettings> _state = const AsyncLoading();

  @override
  void initState() {
    super.initState();
    _repository = widget.repository ?? MockSettingsRepository();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    setState(() => _state = const AsyncLoading());

    final Result<UserSettings> result = await _repository.getUserSettings();

    if (!mounted) return;

    setState(() {
      _state = result.fold(
        (failure) => AsyncError(failure),
        (settings) => AsyncSuccess(settings),
      );
    });
  }

  Future<void> _updateSettings(UserSettings settings) async {
    await _repository.updateUserSettings(settings);
    if (!mounted) return;
    setState(() => _state = AsyncSuccess(settings));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Profile'), centerTitle: true),
      body: _state.when(
        initial: () => const SizedBox.shrink(),
        loading: () => const _LoadingView(),
        success: (settings) => _SettingsContent(
          settings: settings,
          onSettingsChanged: _updateSettings,
        ),
        empty: () => const SizedBox.shrink(),
        error: (failure) =>
            _ErrorView(failure: failure, onRetry: _loadSettings),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Loading State
// ---------------------------------------------------------------------------

class _LoadingView extends StatelessWidget {
  const _LoadingView();

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: CircularProgressIndicator(semanticsLabel: 'Loading settings'),
    );
  }
}

// ---------------------------------------------------------------------------
// Error State
// ---------------------------------------------------------------------------

class _ErrorView extends StatelessWidget {
  final Failure failure;
  final VoidCallback onRetry;

  const _ErrorView({required this.failure, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Center(
      child: Padding(
        padding: const EdgeInsets.all(AppSpacing.lg),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.error_outline_rounded,
              size: 80,
              color: colorScheme.error,
            ),
            const SizedBox(height: AppSpacing.md),
            Text('Could Not Load Settings', style: theme.textTheme.titleMedium),
            const SizedBox(height: AppSpacing.sm),
            Text(
              'There was a problem loading your profile settings. Please try again.',
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: AppSpacing.lg),
            FilledButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh_rounded),
              label: const Text('Retry'),
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Settings Content — success state
// ---------------------------------------------------------------------------

class _SettingsContent extends StatelessWidget {
  final UserSettings settings;
  final ValueChanged<UserSettings> onSettingsChanged;

  const _SettingsContent({
    required this.settings,
    required this.onSettingsChanged,
  });

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.only(
        left: AppSpacing.md,
        right: AppSpacing.md,
        top: AppSpacing.lg,
        bottom: 100,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Profile Header
          const _ProfileHeader(),
          const SizedBox(height: AppSpacing.lg),

          // General Settings Group
          _SettingsGroup(
            children: [
              _SettingsTile(
                icon: Icons.person_outline_rounded,
                label: 'Personal Information',
                onTap: () {
                  // TODO: Navigate to personal info screen when implemented
                },
              ),
              _SettingsTile(
                icon: Icons.language_rounded,
                label: 'Language',
                trailing: const Text('English'),
                onTap: () {
                  // TODO: Navigate to language picker when implemented
                },
              ),
              _SettingsTile(
                icon: Icons.shield_outlined,
                label: 'Privacy & Security',
                onTap: () {
                  // TODO: Navigate to privacy screen when implemented
                },
              ),
              _SettingsTile(
                icon: Icons.badge_outlined,
                label: 'ABHA ID',
                trailing: _AbhaConnectedChip(),
                isLast: true,
                onTap: () {
                  // TODO: Navigate to ABHA ID management when implemented
                },
              ),
            ],
          ),

          const SizedBox(height: AppSpacing.md),

          // Notifications Group
          _SettingsGroup(
            children: [
              _SwitchTile(
                icon: Icons.notifications_outlined,
                label: 'Notifications',
                value: settings.notificationsEnabled,
                onChanged: (val) {
                  onSettingsChanged(
                    settings.copyWith(notificationsEnabled: val),
                  );
                },
                isLast: true,
              ),
            ],
          ),

          const SizedBox(height: AppSpacing.md),

          // App Version
          Padding(
            padding: const EdgeInsets.symmetric(vertical: AppSpacing.sm),
            child: Text(
              'Version 1.0.0',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodySmall
                  ?.copyWith(color: Theme.of(context).colorScheme.outline),
            ),
          ),

          const SizedBox(height: AppSpacing.sm),

          // Log Out
          _LogOutButton(),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Profile Header
// ---------------------------------------------------------------------------

class _ProfileHeader extends StatelessWidget {
  const _ProfileHeader();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Column(
      children: [
        // Avatar
        Container(
          width: 88,
          height: 88,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: colorScheme.secondaryContainer,
            border: Border.all(color: colorScheme.outlineVariant, width: 2),
          ),
          child: Icon(
            Icons.person_rounded,
            size: 48,
            color: colorScheme.onSecondaryContainer,
            semanticLabel: 'Profile picture',
          ),
        ),
        const SizedBox(height: AppSpacing.sm),
        Text(
          'Alex Johnson',
          style: theme.textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.w600,
            color: colorScheme.onSurface,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          'Patient',
          style: theme.textTheme.bodyMedium?.copyWith(
            color: colorScheme.onSurfaceVariant,
          ),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Settings Group Card
// ---------------------------------------------------------------------------

class _SettingsGroup extends StatelessWidget {
  final List<Widget> children;

  const _SettingsGroup({required this.children});

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Container(
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerLowest,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 20,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(children: children),
    );
  }
}

// ---------------------------------------------------------------------------
// Settings Tile (navigation row)
// ---------------------------------------------------------------------------

class _SettingsTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final Widget? trailing;
  final VoidCallback? onTap;
  final bool isLast;

  const _SettingsTile({
    required this.icon,
    required this.label,
    this.trailing,
    this.onTap,
    this.isLast = false,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return InkWell(
      onTap: onTap,
      borderRadius: isLast
          ? const BorderRadius.vertical(bottom: Radius.circular(16))
          : BorderRadius.zero,
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: AppSpacing.md,
          vertical: AppSpacing.md,
        ),
        decoration: BoxDecoration(
          border: isLast
              ? null
              : Border(
                  bottom: BorderSide(
                    color: colorScheme.outlineVariant.withValues(alpha: 0.4),
                  ),
                ),
        ),
        child: Row(
          children: [
            Icon(icon, color: colorScheme.primary, semanticLabel: label),
            const SizedBox(width: AppSpacing.md),
            Expanded(child: Text(label, style: theme.textTheme.bodyLarge)),
            if (trailing != null) ...[
              const SizedBox(width: AppSpacing.sm),
              trailing!,
            ],
            const SizedBox(width: AppSpacing.sm),
            Icon(
              Icons.chevron_right_rounded,
              color: colorScheme.outlineVariant,
              semanticLabel: 'Navigate to $label',
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Switch Tile
// ---------------------------------------------------------------------------

class _SwitchTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool value;
  final ValueChanged<bool> onChanged;
  final bool isLast;

  const _SwitchTile({
    required this.icon,
    required this.label,
    required this.value,
    required this.onChanged,
    this.isLast = false,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: AppSpacing.md,
        vertical: AppSpacing.sm,
      ),
      decoration: BoxDecoration(
        border: isLast
            ? null
            : Border(
                bottom: BorderSide(
                  color: colorScheme.outlineVariant.withValues(alpha: 0.4),
                ),
              ),
      ),
      child: Row(
        children: [
          Icon(icon, color: colorScheme.primary, semanticLabel: label),
          const SizedBox(width: AppSpacing.md),
          Expanded(child: Text(label, style: theme.textTheme.bodyLarge)),
          Switch(value: value, onChanged: onChanged),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// ABHA Connected Chip
// ---------------------------------------------------------------------------

class _AbhaConnectedChip extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: colorScheme.primary.withValues(alpha: 0.12),
        borderRadius: AppShapes.radiusMd,
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 6,
            height: 6,
            decoration: BoxDecoration(
              color: colorScheme.primary,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 4),
          Text(
            'Connected',
            style: TextStyle(
              color: colorScheme.primary,
              fontSize: 11,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Log Out Button
// ---------------------------------------------------------------------------

class _LogOutButton extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return FilledButton.icon(
      onPressed: () {
        // TODO: Implement logout when auth is added
      },
      icon: const Icon(Icons.logout_rounded),
      label: const Text('Log Out'),
      style: FilledButton.styleFrom(
        backgroundColor: colorScheme.errorContainer,
        foregroundColor: colorScheme.onErrorContainer,
        minimumSize: const Size.fromHeight(52),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      ),
    );
  }
}
