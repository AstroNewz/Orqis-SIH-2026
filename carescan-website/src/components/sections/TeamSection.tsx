'use client';

import React from 'react';
import { SectionHeader } from '../ui/SectionHeader';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { TEAM_MEMBERS } from '@/lib/constants';

export const TeamSection: React.FC = () => {
  return (
    <section id="team" className="relative py-14 md:py-20 bg-white dark:bg-slate-900 transition-colors duration-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-12">
        {/* Section Header */}
        <SectionHeader
          badge="Multidisciplinary Research Collaboration"
          badgeVariant="teal"
          title="Bridging Medicine, AI, &"
          highlightText="Quantum Information Science"
          subtitle="Orqis unites clinicians, oncology informatics researchers, quantum engineers, and mobile architects dedicated to accessible cancer screening."
        />

        {/* Team Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {TEAM_MEMBERS.map((member, i) => (
            <Card
              key={i}
              variant="white"
              padding="md"
              organic="subtle"
              className="border-slate-200 dark:border-slate-800 hover:border-teal-400 dark:hover:border-teal-600 shadow-2xs transition-all space-y-3 text-center group"
            >
              {/* Avatar Placeholder */}
              <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-tr from-teal-50 via-indigo-50 to-purple-50 dark:from-teal-950/60 dark:via-indigo-950/60 dark:to-purple-950/60 border border-teal-200/80 dark:border-teal-800/80 flex items-center justify-center text-teal-800 dark:text-teal-300 font-extrabold text-base shadow-2xs group-hover:scale-105 transition-transform">
                <span>{member.avatarPlaceholder}</span>
              </div>

              <div>
                <Badge variant={i === 0 ? 'teal' : i === 1 ? 'iris' : i === 2 ? 'quantum' : 'neutral'} size="sm">
                  {member.discipline}
                </Badge>
                <h3 className="text-sm sm:text-base font-bold text-slate-900 dark:text-white mt-1.5 font-heading">{member.role}</h3>
                <p className="text-xs font-semibold text-slate-600 dark:text-slate-400 font-body">{member.department}</p>
              </div>

              <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed pt-2 border-t border-slate-100 dark:border-slate-800 font-body">
                {member.focus}
              </p>
            </Card>
          ))}
        </div>

        {/* Academic & Open Research Collaboration Banner */}
        <Card variant="stone" padding="md" organic="subtle" className="max-w-3xl mx-auto border-teal-200 dark:border-teal-800 text-center space-y-1.5">
          <p className="text-xs font-bold uppercase tracking-wider text-teal-800 dark:text-teal-300">Open Scientific Research Initiative</p>
          <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed font-body">
            Orqis is developed as an open scientific collaboration targeting peer-reviewed clinical benchmarks in frontline AI oral triage and hardware-aware quantum acceleration.
          </p>
        </Card>
      </div>
    </section>
  );
};
