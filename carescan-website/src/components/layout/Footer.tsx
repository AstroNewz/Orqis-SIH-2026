'use client';

import React from 'react';
import Image from 'next/image';
import { ShieldAlert, Heart } from 'lucide-react';
import { CLINICAL_DISCLAIMER, NAV_ITEMS } from '@/lib/constants';

export const Footer: React.FC = () => {
  return (
    <footer className="relative bg-stone-900 dark:bg-[#0A0909] text-stone-400 pt-14 pb-10 font-body border-t border-stone-800">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 space-y-10">

        {/* Clinical disclaimer */}
        <div className="flex gap-4 p-5 rounded-lg border border-amber-800/40 bg-amber-950/20">
          <ShieldAlert className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
          <div>
            <p className="text-xs font-semibold text-amber-400 mb-1 uppercase tracking-wide">
              Clinical Decision Support — Research Only
            </p>
            <p className="text-xs text-stone-400 leading-relaxed">
              {CLINICAL_DISCLAIMER}
            </p>
          </div>
        </div>

        {/* Main columns */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 pb-8 border-b border-stone-800">
          {/* Brand */}
          <div className="md:col-span-1">
            <div className="flex items-center gap-2.5 mb-3">
              <div className="w-8 h-8 rounded-full bg-white flex items-center justify-center overflow-hidden shrink-0">
                <Image src="/orqis-logo.png" alt="Orqis" width={32} height={32} className="w-full h-full object-contain" />
              </div>
              <span className="font-brand text-xl tracking-tight text-stone-100">Orqis</span>
            </div>
            <p className="text-xs text-stone-500 leading-relaxed max-w-xs">
              Privacy-conscious hybrid quantum-classical oral cancer screening research.
              University of Peradeniya Oral Cancer Dataset v1 (SMART-OM).
              Exact statevector simulation — no quantum hardware run has been performed.
            </p>
          </div>

          {/* Navigation */}
          <div>
            <h5 className="text-xs font-semibold text-stone-300 uppercase tracking-wider mb-3">
              Sections
            </h5>
            <ul className="space-y-2">
              {NAV_ITEMS.slice(0, 4).map((item) => (
                <li key={item.label}>
                  <a href={item.href} className="text-xs text-stone-500 hover:text-stone-300 transition-colors">
                    {item.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Research */}
          <div>
            <h5 className="text-xs font-semibold text-stone-300 uppercase tracking-wider mb-3">
              Research
            </h5>
            <ul className="space-y-2">
              {NAV_ITEMS.slice(4).map((item) => (
                <li key={item.label}>
                  <a href={item.href} className="text-xs text-stone-500 hover:text-stone-300 transition-colors">
                    {item.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-stone-600">
          <p>© {new Date().getFullYear()} Orqis Research Project. University of Peradeniya Oral Cancer Dataset v1 (SMART-OM) &amp; IBM Quantum Evaluation.</p>
          <p className="flex items-center gap-1">
            Built with care
            <Heart className="w-3 h-3 text-rose-600 fill-rose-600 mx-0.5" />
            for frontline health workers.
          </p>
        </div>
      </div>
    </footer>
  );
};
