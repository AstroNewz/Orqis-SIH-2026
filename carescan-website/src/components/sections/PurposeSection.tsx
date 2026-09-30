'use client';

import React from 'react';
import { SectionHeader } from '../ui/SectionHeader';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { PURPOSE_STATS } from '@/lib/constants';
import { OrganicDivider } from '../visual/OrganicDivider';
import { AlertCircle, Clock, MapPin, Sparkles, Shield, HeartHandshake } from 'lucide-react';

export const PurposeSection: React.FC = () => {
  return (
    <section id="project" className="relative py-14 md:py-20 bg-stone-50 dark:bg-slate-950 transition-colors duration-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-12">
        {/* Section Header */}
        <SectionHeader
          badge="The Healthcare Crisis We Are Solving"
          badgeVariant="teal"
          title="Why Early Detection Matters:"
          highlightText="Saving Lives at the Frontline"
          subtitle="Oral cancer is among the deadliest and most disfiguring malignancies when detected late, yet it is almost entirely curable when caught early. Orqis exists to eliminate the diagnostic divide."
        />

        {/* 3 Impact Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {PURPOSE_STATS.map((item, i) => (
            <Card
              key={i}
              variant="white"
              padding="md"
              organic="subtle"
              className="relative overflow-hidden group hover:border-teal-300 dark:hover:border-teal-600 shadow-xs"
            >
              <div className="absolute top-0 right-0 w-20 h-20 bg-teal-50 dark:bg-teal-950/40 rounded-full -mr-6 -mt-6 pointer-events-none group-hover:scale-125 transition-transform" />
              <p className="text-3xl sm:text-4xl font-extrabold text-teal-700 dark:text-teal-400 font-mono">
                {item.stat}
              </p>
              <h3 className="text-base font-bold text-slate-900 dark:text-white mt-2 font-heading">{item.label}</h3>
              <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 mt-1.5 leading-relaxed font-body">
                {item.detail}
              </p>
            </Card>
          ))}
        </div>

        {/* Side-by-Side Comparison: Traditional vs Orqis */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
          {/* Traditional Bottlenecks */}
          <Card variant="white" padding="lg" className="border-rose-200/80 dark:border-rose-900/60 bg-gradient-to-b from-rose-50/40 via-white to-white dark:from-rose-950/20 dark:via-slate-900 dark:to-slate-900 shadow-xs">
            <div className="flex items-center gap-2 mb-3">
              <Badge variant="danger">Current Clinical Barrier</Badge>
              <span className="text-xs font-bold text-rose-800 dark:text-rose-300">Delayed & Unequal Access</span>
            </div>
            <h3 className="text-lg sm:text-xl font-bold text-slate-900 dark:text-white mb-2 font-heading">
              Why Communities Suffer from Late-Stage Diagnoses
            </h3>
            <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 mb-5 leading-relaxed font-body">
              In small clinics and rural health posts, non-specialist health workers must rely on visual inspection with poor lighting. Early precancerous white patches (leukoplakia) look identical to harmless ulcers.
            </p>

            <ul className="space-y-3 text-xs sm:text-sm text-slate-800 dark:text-slate-200 font-body">
              <li className="flex items-start gap-2.5">
                <Clock className="w-4 h-4 text-rose-600 dark:text-rose-400 shrink-0 mt-0.5" />
                <span>
                  <strong className="text-slate-900 dark:text-white">Prolonged Referral Delays: </strong> Patients wait 6–12 weeks to see an oral oncology specialist in regional hospitals.
                </span>
              </li>
              <li className="flex items-start gap-2.5">
                <MapPin className="w-4 h-4 text-rose-600 dark:text-rose-400 shrink-0 mt-0.5" />
                <span>
                  <strong className="text-slate-900 dark:text-white">Geographic Isolation: </strong> Rural clinics have zero on-site biopsy or specialized diagnostic infrastructure.
                </span>
              </li>
              <li className="flex items-start gap-2.5">
                <AlertCircle className="w-4 h-4 text-rose-600 dark:text-rose-400 shrink-0 mt-0.5" />
                <span>
                  <strong className="text-slate-900 dark:text-white">High Diagnostic Uncertainty: </strong> Up to 40% of subtle dysplastic lesions are dismissed as benign trauma.
                </span>
              </li>
            </ul>
          </Card>

          {/* Orqis Solution */}
          <Card variant="white" padding="lg" className="border-teal-200/80 dark:border-teal-800/60 bg-gradient-to-b from-teal-50/40 via-white to-white dark:from-teal-950/20 dark:via-slate-900 dark:to-slate-900 shadow-xs">
            <div className="flex items-center gap-2 mb-3">
              <Badge variant="teal">The Orqis Solution</Badge>
              <span className="text-xs font-bold text-teal-800 dark:text-teal-300">Immediate Frontline Empowerment</span>
            </div>
            <h3 className="text-lg sm:text-xl font-bold text-slate-900 dark:text-white mb-2 font-heading">
              Standardized, Point-of-Care Decision Support
            </h3>
            <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 mb-5 leading-relaxed font-body">
              Orqis turns any smartphone into an intelligent triage companion, providing community nurses with instant objective risk scores and automatic specialist referral summaries.
            </p>

            <ul className="space-y-3 text-xs sm:text-sm text-slate-800 dark:text-slate-200 font-body">
              <li className="flex items-start gap-2.5">
                <Sparkles className="w-4 h-4 text-teal-700 dark:text-teal-400 shrink-0 mt-0.5" />
                <span>
                  <strong className="text-slate-900 dark:text-white">1-Second Objective Triage: </strong> Instant risk scoring combining deep visual textures and lifestyle exposures.
                </span>
              </li>
              <li className="flex items-start gap-2.5">
                <Shield className="w-4 h-4 text-teal-700 dark:text-teal-400 shrink-0 mt-0.5" />
                <span>
                  <strong className="text-slate-900 dark:text-white">Every Claim Has a Control: </strong> The quantum feature map is reported beside a matched classical surrogate on identical patient rows — parity is reported as parity.
                </span>
              </li>
              <li className="flex items-start gap-2.5">
                <HeartHandshake className="w-4 h-4 text-teal-700 dark:text-teal-400 shrink-0 mt-0.5" />
                <span>
                  <strong className="text-slate-900 dark:text-white">Calm, Actionable Guidance: </strong> Empowers patients with clear next steps while sending clean FHIR records to doctors.
                </span>
              </li>
            </ul>
          </Card>
        </div>
      </div>

      <div className="mt-12">
        <OrganicDivider position="bottom" fillColor="var(--warm-bg)" variant="curve-2" />
      </div>
    </section>
  );
};
