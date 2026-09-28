import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:carescan/l10n/l10n.dart';
import 'package:carescan/core/theme/app_shapes.dart';

class ScaffoldWithNav extends StatelessWidget {
  const ScaffoldWithNav({super.key, required this.navigationShell});
  final StatefulNavigationShell navigationShell;
  @override
  Widget build(BuildContext context) {
    final l = context.l10n;
    final s = Theme.of(context).colorScheme;
    return Scaffold(
      body: navigationShell,
      bottomNavigationBar: Material(
        color: s.surface,
        child: DecoratedBox(
          decoration: BoxDecoration(
            border: Border(top: BorderSide(color: s.outlineVariant)),
          ),
          child: SafeArea(
            top: false,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 8),
              child: Row(
                children: [
                  _item(context, 0, Icons.home_outlined, l.home),
                  _item(context, 1, Icons.menu_book_outlined, l.blogs),
                  Expanded(
                    child: Semantics(
                      button: true,
                      label: l.startScreening,
                      child: InkWell(
                        onTap: () => context.push('/camera'),
                        borderRadius: AppShapes.radiusMd,
                        child: Padding(
                          padding: const EdgeInsets.symmetric(vertical: 4),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Container(
                                width: 48,
                                height: 40,
                                decoration: BoxDecoration(
                                  color: s.primary,
                                  borderRadius: AppShapes.radiusMd,
                                ),
                                child: Icon(
                                  Icons.center_focus_strong,
                                  color: s.onPrimary,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                l.scan,
                                style: Theme.of(context).textTheme.labelMedium
                                    ?.copyWith(color: s.primary),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                  _item(context, 2, Icons.history, l.history),
                  _item(context, 3, Icons.person_outline, l.profile),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _item(BuildContext context, int index, IconData icon, String label) {
    final selected = navigationShell.currentIndex == index;
    final s = Theme.of(context).colorScheme;
    return Expanded(
      child: Semantics(
        selected: selected,
        button: true,
        child: InkWell(
          onTap: () => navigationShell.goBranch(
            index,
            initialLocation: index == navigationShell.currentIndex,
          ),
          borderRadius: AppShapes.radiusMd,
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 2),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, color: selected ? s.primary : s.onSurfaceVariant),
                const SizedBox(height: 6),
                Text(
                  label,
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.labelMedium?.copyWith(
                    color: selected ? s.primary : s.onSurfaceVariant,
                    fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
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
