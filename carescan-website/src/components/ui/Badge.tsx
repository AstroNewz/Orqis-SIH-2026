import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'iris' | 'teal' | 'quantum' | 'neutral' | 'success' | 'warning' | 'danger';
  size?: 'sm' | 'md';
  pulse?: boolean;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'iris',
  size = 'md',
  pulse = false,
  className = '',
}) => {
  const variantStyles = {
    iris: 'bg-indigo-50 dark:bg-indigo-950/70 text-indigo-700 dark:text-indigo-300 border-indigo-200/80 dark:border-indigo-800/70',
    teal: 'bg-teal-50 dark:bg-teal-950/70 text-teal-700 dark:text-teal-300 border-teal-200/80 dark:border-teal-800/70',
    quantum: 'bg-purple-50 dark:bg-purple-950/70 text-purple-700 dark:text-purple-300 border-purple-200/80 dark:border-purple-800/70',
    neutral: 'bg-stone-100 dark:bg-slate-800/70 text-stone-700 dark:text-slate-300 border-stone-200 dark:border-slate-700/70',
    success: 'bg-emerald-50 dark:bg-emerald-950/70 text-emerald-700 dark:text-emerald-300 border-emerald-200/80 dark:border-emerald-800/70',
    warning: 'bg-amber-50 dark:bg-amber-950/70 text-amber-800 dark:text-amber-300 border-amber-200/80 dark:border-amber-800/70',
    danger: 'bg-rose-50 dark:bg-rose-950/70 text-rose-700 dark:text-rose-300 border-rose-200/80 dark:border-rose-800/70',
  };

  const dotColors = {
    iris: 'bg-indigo-600 dark:bg-indigo-400',
    teal: 'bg-teal-600 dark:bg-teal-400',
    quantum: 'bg-purple-600 dark:bg-purple-400',
    neutral: 'bg-stone-500 dark:bg-slate-400',
    success: 'bg-emerald-500 dark:bg-emerald-400',
    warning: 'bg-amber-500 dark:bg-amber-400',
    danger: 'bg-rose-500 dark:bg-rose-400',
  };

  const sizeStyles = {
    sm: 'text-xs px-2.5 py-0.5 font-medium',
    md: 'text-xs px-3.5 py-1 font-semibold tracking-wide',
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border transition-all duration-200 ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
    >
      {pulse && (
        <span className="relative flex h-2 w-2">
          <span
            className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${dotColors[variant]}`}
          />
          <span
            className={`relative inline-flex rounded-full h-2 w-2 ${dotColors[variant]}`}
          />
        </span>
      )}
      {children}
    </span>
  );
};
