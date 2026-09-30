'use client';

import React from 'react';
import { HERO_METRICS } from '@/lib/constants';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { Card } from '../ui/Card';
import { QuantumOrb } from '../visual/QuantumOrb';
import { OrganicDivider } from '../visual/OrganicDivider';
import { ArrowRight, Sparkles, ShieldCheck, HeartPulse, CheckCircle2 } from 'lucide-react';

export const HeroSection: React.FC = () => {
  return (
    <section className="relative pt-24 pb-12 md:pt-32 md:pb-16 overflow-hidden bg-gradient-to-b from-teal-50/50 via-stone-50/70 to-white dark:from-slate-950 dark:via-slate-900/60 dark:to-slate-950 transition-colors duration-300">
      {/* Background Soft Organic Radial Blobs */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-6xl h-80 bg-gradient-to-r from-teal-200/20 via-indigo-200/20 to-purple-200/20 dark:from-teal-900/15 dark:via-indigo-900/15 dark:to-purple-900/15 rounded-full blur-3xl -z-10 pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-8 items-center">
          {/* Left Hero Copy — Healthcare Mission First */}
          <div className="lg:col-span-7 space-y-5 text-center lg:text-left">
            {/* Top pill badge */}
            <div className="inline-flex justify-center lg:justify-start">
              <Badge variant="teal" pulse size="md">
                <HeartPulse className="w-3.5 h-3.5 mr-1" />
                Frontline Healthcare Initiative • Early Oral Cancer Triage
              </Badge>
            </div>

            {/* Headline */}
            <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-[3.25rem] font-bold tracking-tight text-slate-900 dark:text-white leading-[1.14]">
              Early Oral Screening for Every Community,{' '}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-600 via-indigo-600 to-purple-600 dark:from-teal-400 dark:via-indigo-400 dark:to-purple-400">
                Powered by Quantum AI
              </span>
            </h1>

            {/* Subheadline — Clear and Human */}
            <p className="text-base sm:text-lg text-slate-700 dark:text-slate-300 leading-relaxed max-w-2xl mx-auto lg:mx-0 font-body">
              Orqis equips community health workers and primary clinics with instant, non-invasive oral cancer screening. A frozen MobileNet localizer (DEC-020) finds the lesion, a validated classical baseline sets the risk band, and an 8-qubit quantum feature map is reported alongside it — never instead of it.
            </p>

            {/* Value Highlights Pill List */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 text-xs sm:text-sm font-medium text-slate-800 dark:text-slate-200 font-body">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-teal-600 dark:text-teal-400 shrink-0" />
                <span>Flutter mobile acquisition (com.orqis.patient)</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-teal-600 dark:text-teal-400 shrink-0" />
                <span>Frozen MobileNet localizer (0.5279 Test IoU)</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-indigo-600 dark:text-indigo-400 shrink-0" />
                <span>8-Qubit Feature Map (0 Trainable Quantum Gates)</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-indigo-600 dark:text-indigo-400 shrink-0" />
                <span>Hospital-ready HL7 FHIR R4 Observations</span>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-3 pt-2">
              <a href="#sandbox" className="w-full sm:w-auto">
                <Button variant="secondary" size="lg" icon={<Sparkles className="w-4 h-4" />} className="w-full sm:w-auto shadow-md">
                  Try Interactive Triage Demo
                </Button>
              </a>
              <a href="#project" className="w-full sm:w-auto">
                <Button variant="outline" size="lg" icon={<ArrowRight className="w-4 h-4" />} className="w-full sm:w-auto">
                  Our Healthcare Mission
                </Button>
              </a>
            </div>

            {/* Trust Badges */}
            <div className="pt-2 flex flex-wrap items-center justify-center lg:justify-start gap-3 text-xs font-semibold text-slate-600 dark:text-slate-400">
              <span className="flex items-center gap-1.5 bg-white dark:bg-slate-900 px-3 py-1 rounded-full border border-slate-200/80 dark:border-slate-800 shadow-2xs">
                <ShieldCheck className="w-4 h-4 text-teal-600 dark:text-teal-400" />
                Patient-Level Partitioning Invariant
              </span>
              <span className="flex items-center gap-1.5 bg-white dark:bg-slate-900 px-3 py-1 rounded-full border border-slate-200/80 dark:border-slate-800 shadow-2xs">
                2,436 images · 328 patients · Peradeniya dataset
              </span>
            </div>
          </div>

          {/* Right Visual Composition */}
          <div className="lg:col-span-5 flex justify-center">
            <QuantumOrb />
          </div>
        </div>

        {/* Hero Metrics Row */}
        <div className="mt-12 sm:mt-16 grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
          {HERO_METRICS.map((metric, i) => (
            <Card
              key={i}
              variant="white"
              padding="sm"
              organic="subtle"
              className="text-center group hover:border-teal-300 dark:hover:border-teal-600 transition-all shadow-xs"
            >
              <p className="text-2xl sm:text-3xl font-extrabold text-slate-900 dark:text-white group-hover:text-teal-700 dark:group-hover:text-teal-400 transition-colors">
                {metric.value}
              </p>
              <p className="text-xs sm:text-sm font-bold text-slate-800 dark:text-slate-200 mt-1">
                {metric.label}
              </p>
              <p className="text-[11px] text-slate-600 dark:text-slate-400 mt-0.5 leading-snug hidden sm:block">
                {metric.description}
              </p>
            </Card>
          ))}
        </div>
      </div>

      <div className="mt-10">
        <OrganicDivider position="bottom" fillColor="var(--warm-surface)" variant="curve-1" />
      </div>
    </section>
  );
};
