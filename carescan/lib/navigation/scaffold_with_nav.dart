import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

class ScaffoldWithNav extends StatelessWidget {
  const ScaffoldWithNav({super.key, required this.navigationShell});

  final StatefulNavigationShell navigationShell;

  void _goBranch(int index) {
    navigationShell.goBranch(
      index,
      // A common pattern when using bottom navigation bars is to support
      // navigating to the initial location when tapping the item that is
      // already active. This example demonstrates how to support this behavior,
      // using the initialLocation parameter of goBranch.
      initialLocation: index == navigationShell.currentIndex,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: navigationShell,
      extendBody: true, // Needed for floating transparent bottom nav
      floatingActionButton: FloatingActionButton(
        onPressed: () => context.push('/camera'),
        shape: const CircleBorder(),
        backgroundColor: const Color(0xFF1a1c1e), // Dark FAB from Stitch HTML
        foregroundColor: Colors.white,
        child: const Icon(Icons.photo_camera),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      bottomNavigationBar: SafeArea(
        child: Container(
          margin: const EdgeInsets.only(left: 24, right: 24, bottom: 24),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(40),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.05),
                blurRadius: 10,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: BottomAppBar(
            elevation: 0,
            color: Colors.transparent,
            shape: const CircularNotchedRectangle(),
            notchMargin: 8,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildNavItem(
                  context,
                  0,
                  Icons.home_outlined,
                  Icons.home,
                  'Home',
                ),
                _buildNavItem(
                  context,
                  1,
                  Icons.article_outlined,
                  Icons.article,
                  'Blogs',
                ),
                const SizedBox(width: 48), // Space for FAB
                _buildNavItem(
                  context,
                  2,
                  Icons.history_outlined,
                  Icons.history,
                  'History',
                ),
                _buildNavItem(
                  context,
                  3,
                  Icons.person_outline,
                  Icons.person,
                  'Profile',
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildNavItem(
    BuildContext context,
    int index,
    IconData unselectedIcon,
    IconData selectedIcon,
    String label,
  ) {
    final isSelected = navigationShell.currentIndex == index;
    final color = isSelected
        ? Theme.of(context).colorScheme.primary
        : Theme.of(context).colorScheme.onSurfaceVariant.withValues(alpha: 0.6);

    return InkWell(
      onTap: () => _goBranch(index),
      splashColor: Colors.transparent,
      highlightColor: Colors.transparent,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(isSelected ? selectedIcon : unselectedIcon, color: color),
          const SizedBox(height: 2),
          Text(
            label,
            style: TextStyle(
              fontSize: 10,
              fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}
