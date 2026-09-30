'use client';

import React from 'react';
import { TIMELINE_MILESTONES } from '@/lib/constants';
import { OrganicDivider } from '../visual/OrganicDivider';

const STATUS_COLORS: Record<string, string> = {
  completed: 'bg-emerald-500',
  'in-progress': 'bg-cyan-600 ring-4 ring-cyan-200 dark:ring-cyan-900',
  planned: 'bg-stone-300 dark:bg-stone-700',
};

const STATUS_LABELS: Record<string, string> = {
  completed: 'Complete',
  'in-progress': 'Active',
  planned: 'Planned',
};

export const TimelineSection: React.FC = () => {
  return (
    <section id="journey" className="relative py-16 md:py-24 bg-[#F8F6F2] dark:bg-[#0F0E0D] transition-colors duration-300">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Heading */}
        <div className="mb-14 max-w-2xl">
          <p className="text-sm font-semibold text-cyan-700 dark:text-cyan-500 tracking-wide uppercase mb-3">
            Research Roadmap
          </p>
          <h2 className="text-3xl sm:text-4xl font-bold text-stone-900 dark:text-stone-50 tracking-tight mb-4">
            The journey so far
          </h2>
          <p className="text-base text-stone-600 dark:text-stone-400 leading-relaxed">
            From dataset reconstruction and frozen localizer training, through a frozen test read
            and five published null results, to the active PTB-XL research arena.
          </p>
        </div>

        {/* Timeline */}
        <div className="relative">
          {/* Vertical rule */}
          <div className="absolute left-3 top-2 bottom-2 w-px bg-stone-200 dark:bg-stone-800" />

          <div className="space-y-0">
            {TIMELINE_MILESTONES.map((item, index) => (
              <div key={`${item.phase}-${index}`} className="relative pl-10 pb-12 last:pb-0">
                {/* Dot */}
                <span
                  className={`absolute left-0 top-1.5 w-6 h-6 rounded-full flex items-center justify-center ${STATUS_COLORS[item.status]}`}
                >
                  {item.status === 'completed' && (
                    <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 12 12">
                      <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  )}
                  {item.status === 'in-progress' && (
                    <span className="w-2 h-2 rounded-full bg-white" />
                  )}
                </span>

                {/* Content */}
                <div className="max-w-2xl">
                  <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 mb-1">
                    <span className="text-xs font-mono font-semibold text-stone-400 dark:text-stone-500 uppercase">
                      {item.phase}
                    </span>
                    <span className={`text-xs font-semibold ${item.status === 'in-progress'
                        ? 'text-cyan-700 dark:text-cyan-500'
                        : item.status === 'completed'
                          ? 'text-emerald-600 dark:text-emerald-500'
                          : 'text-stone-400 dark:text-stone-500'
                      }`}>
                      {STATUS_LABELS[item.status]}
                    </span>
                  </div>

                  <h3 className="text-base font-bold text-stone-900 dark:text-stone-100 mb-2">
                    {item.title}
                  </h3>
                  <p className="text-sm text-stone-600 dark:text-stone-400 leading-relaxed mb-3">
                    {item.description}
                  </p>
                  <ul className="space-y-1">
                    {item.deliverables.map((del, dIdx) => (
                      <li key={dIdx} className="flex gap-2 text-xs text-stone-500 dark:text-stone-400">
                        <span className="shrink-0 text-stone-300 dark:text-stone-600">—</span>
                        <span>{del}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-12">
        <OrganicDivider position="bottom" fillColor="var(--warm-surface)" variant="curve-1" />
      </div>
    </section>
  );
};
