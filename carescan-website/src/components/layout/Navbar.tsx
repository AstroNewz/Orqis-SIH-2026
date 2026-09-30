'use client';

import React, { useState, useEffect } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { NAV_ITEMS } from '@/lib/constants';
import { Menu, X, KeyRound, Sun, Moon, Stethoscope } from 'lucide-react';
import { useCredentials } from '@/context/CredentialsContext';
import { useTheme } from '@/context/ThemeContext';

export const Navbar: React.FC = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [activeSection, setActiveSection] = useState<string>('');
  const [mobileOpen, setMobileOpen] = useState(false);
  const { openModal } = useCredentials();
  const { toggleTheme } = useTheme();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 35);

      const sectionIds = NAV_ITEMS.map((item) => item.href.replace('#', ''));
      let currentActive = '';
      for (const id of sectionIds) {
        const el = document.getElementById(id);
        if (el) {
          const rect = el.getBoundingClientRect();
          if (rect.top <= 180 && rect.bottom >= 180) {
            currentActive = `#${id}`;
            break;
          }
        }
      }
      if (currentActive) setActiveSection(currentActive);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <>
      <header className="fixed top-0 left-0 right-0 z-40 px-4 sm:px-6 lg:px-8 pt-3 sm:pt-4 transition-all duration-300">
        <div
          className={`mx-auto dock-nav-base transition-all duration-300 ${
            isScrolled
              ? 'max-w-5xl rounded-full glass-nav dock-pill-scrolled py-2 px-4 sm:px-6'
              : 'max-w-7xl rounded-2xl bg-[#FEFCF9]/90 dark:bg-[#1A1917]/90 backdrop-blur-md border border-[#E5E1D8]/60 dark:border-[#2C2A27]/60 py-3 px-5 sm:px-7'
          } flex items-center justify-between gap-4`}
        >
          {/* Logo & Brand */}
          <a href="#" className="flex items-center gap-2.5 shrink-0 group">
            <div className={`relative rounded-full bg-white flex items-center justify-center border border-stone-200 dark:border-stone-700 overflow-hidden shrink-0 transition-all duration-300 ${
              isScrolled ? 'w-7 h-7' : 'w-9 h-9'
            }`}>
              <Image
                src="/orqis-logo.png"
                alt="Orqis Logo"
                width={36}
                height={36}
                priority
                className="w-full h-full object-contain"
              />
            </div>
            <span className={`font-brand tracking-tight text-stone-900 dark:text-stone-50 transition-all duration-300 ${
                isScrolled ? 'text-xl' : 'text-2xl sm:text-[1.7rem]'
              }`}>
                Orqis
              </span>
          </a>

          {/* Desktop Nav */}
          <nav className="hidden lg:flex items-center gap-0.5 xl:gap-1">
            {NAV_ITEMS.map((item) => {
              const isActive = activeSection === item.href;
              return (
                <a
                  key={item.label}
                  href={item.href}
                  className={`text-sm font-medium px-3 py-1.5 rounded-md transition-all ${
                    isActive
                      ? 'text-cyan-700 dark:text-cyan-400'
                      : 'text-stone-600 dark:text-stone-400 hover:text-stone-900 dark:hover:text-stone-100'
                  }`}
                >
                  {item.label}
                  {isActive && (
                    <span className="block h-0.5 w-full bg-cyan-700 dark:bg-cyan-400 mt-0.5 rounded-full" />
                  )}
                </a>
              );
            })}
          </nav>

          {/* Right actions */}
          <div className="hidden lg:flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={toggleTheme}
              className="p-1.5 rounded-md text-stone-600 dark:text-stone-400 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors cursor-pointer"
              aria-label="Toggle dark mode"
            >
              <Sun className="w-4 h-4 text-amber-500 hidden dark:block" />
              <Moon className="w-4 h-4 block dark:hidden" />
            </button>

            <button
              onClick={() => openModal()}
              className="flex items-center gap-1.5 text-sm font-medium text-stone-700 dark:text-stone-300 hover:text-stone-900 dark:hover:text-stone-100 border border-stone-300 dark:border-stone-700 hover:border-stone-400 dark:hover:border-stone-600 px-3 py-1.5 rounded-md transition-all cursor-pointer"
            >
              <KeyRound className="w-3.5 h-3.5" />
              Credentials
            </button>

            {/* The working clinic portal, not a marketing anchor. */}
            <Link
              href="/portal"
              className="flex items-center gap-1.5 text-sm font-medium text-white bg-teal-700 hover:bg-teal-800 dark:bg-teal-600 dark:hover:bg-teal-500 border border-teal-700/20 px-3 py-1.5 rounded-md transition-all"
            >
              <Stethoscope className="w-3.5 h-3.5" />
              Clinic portal
            </Link>
          </div>

          {/* Mobile actions */}
          <div className="flex items-center gap-1.5 lg:hidden">
            <button
              type="button"
              onClick={toggleTheme}
              className="p-2 rounded-md text-stone-600 dark:text-stone-400 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors cursor-pointer"
            >
              <Sun className="w-4 h-4 text-amber-500 hidden dark:block" />
              <Moon className="w-4 h-4 block dark:hidden" />
            </button>
            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="p-2 rounded-md text-stone-700 dark:text-stone-300 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors cursor-pointer"
              aria-label="Toggle navigation"
            >
              {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile Drawer */}
        {mobileOpen && (
          <div className="lg:hidden mt-2 max-w-lg mx-auto bg-[#FEFCF9] dark:bg-[#1A1917] rounded-2xl p-5 border border-[#E5E1D8] dark:border-[#2C2A27] shadow-lg animate-in fade-in slide-in-from-top-2 duration-200">
            <nav className="flex flex-col gap-1">
              {NAV_ITEMS.map((item) => (
                <a
                  key={item.label}
                  href={item.href}
                  onClick={() => setMobileOpen(false)}
                  className="text-sm font-medium text-stone-700 dark:text-stone-300 hover:text-stone-900 dark:hover:text-stone-100 px-3 py-2.5 rounded-md hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
                >
                  {item.label}
                </a>
              ))}
              <div className="mt-3 pt-3 border-t border-stone-200 dark:border-stone-800 space-y-2">
                <Link
                  href="/portal"
                  onClick={() => setMobileOpen(false)}
                  className="flex items-center gap-2 w-full text-sm font-medium text-white bg-teal-700 dark:bg-teal-600 px-3 py-2.5 rounded-md transition-colors"
                >
                  <Stethoscope className="w-4 h-4" />
                  Clinic portal
                </Link>
                <button
                  onClick={() => { setMobileOpen(false); openModal(); }}
                  className="flex items-center gap-2 w-full text-sm font-medium text-stone-700 dark:text-stone-300 px-3 py-2.5 rounded-md border border-stone-300 dark:border-stone-700 hover:border-stone-400 transition-colors cursor-pointer"
                >
                  <KeyRound className="w-4 h-4" />
                  Credentials
                </button>
              </div>
            </nav>
          </div>
        )}
      </header>
    </>
  );
};
