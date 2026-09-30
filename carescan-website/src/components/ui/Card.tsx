import React from 'react';

interface CardProps {
  children: React.ReactNode;
  variant?: 'white' | 'stone' | 'glass' | 'dark' | 'interactive';
  padding?: 'sm' | 'md' | 'lg' | 'none';
  organic?: 'none' | 'subtle' | 'pill';
  className?: string;
  onClick?: () => void;
}

export const Card: React.FC<CardProps> = ({
  children,
  variant = 'white',
  padding = 'md',
  organic = 'none',
  className = '',
  onClick,
}) => {
  const variantStyles = {
    white:
      'bg-white dark:bg-slate-900/90 border border-slate-200/80 dark:border-slate-800/80 text-slate-900 dark:text-slate-100 shadow-xs hover:border-slate-300/90 dark:hover:border-slate-700',
    stone:
      'bg-stone-50 dark:bg-slate-900/60 border border-stone-200/80 dark:border-slate-800/80 text-slate-900 dark:text-slate-100 shadow-xs',
    glass: 'glass-card text-slate-900 dark:text-slate-100 shadow-sm',
    dark: 'glass-dark text-white border-slate-700/60 shadow-lg',
    interactive:
      'bg-white dark:bg-slate-900/90 border border-slate-200/80 dark:border-slate-800/80 text-slate-900 dark:text-slate-100 shadow-xs hover:shadow-md hover:border-teal-400 dark:hover:border-teal-500 hover:-translate-y-0.5 cursor-pointer transition-all duration-200',
  };

  const paddingStyles = {
    none: '',
    sm: 'p-4 md:p-5',
    md: 'p-6 md:p-8',
    lg: 'p-8 md:p-10',
  };

  const organicStyles = {
    none: 'rounded-2xl md:rounded-3xl',
    subtle: 'rounded-organic-1',
    pill: 'rounded-3xl',
  };

  return (
    <div
      onClick={onClick}
      className={`transition-all duration-200 ${variantStyles[variant]} ${paddingStyles[padding]} ${organicStyles[organic]} ${className}`}
    >
      {children}
    </div>
  );
};
