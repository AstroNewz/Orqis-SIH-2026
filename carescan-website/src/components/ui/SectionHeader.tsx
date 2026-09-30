import React from 'react';
import { Badge } from './Badge';

interface SectionHeaderProps {
  badge?: string;
  badgeVariant?: 'iris' | 'teal' | 'quantum' | 'neutral';
  title: string;
  highlightText?: string;
  subtitle?: string;
  align?: 'left' | 'center';
  className?: string;
}

export const SectionHeader: React.FC<SectionHeaderProps> = ({
  badge,
  badgeVariant = 'teal',
  title,
  highlightText,
  subtitle,
  align = 'center',
  className = '',
}) => {
  return (
    <div
      className={`max-w-3xl ${align === 'center' ? 'mx-auto text-center' : 'text-left'} ${className}`}
    >
      {badge && (
        <div className={`mb-3 flex ${align === 'center' ? 'justify-center' : 'justify-start'}`}>
          <Badge variant={badgeVariant}>{badge}</Badge>
        </div>
      )}
      <h2 className="text-2xl sm:text-3xl md:text-4xl font-bold tracking-tight text-slate-900 dark:text-white leading-[1.2]">
        {title}{' '}
        {highlightText && (
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-600 via-indigo-500 to-purple-600 dark:from-teal-400 dark:via-indigo-400 dark:to-purple-400">
            {highlightText}
          </span>
        )}
      </h2>
      {subtitle && (
        <p className="mt-3 text-sm sm:text-base text-slate-700 dark:text-slate-300 leading-relaxed max-w-2xl mx-auto font-body">
          {subtitle}
        </p>
      )}
    </div>
  );
};
