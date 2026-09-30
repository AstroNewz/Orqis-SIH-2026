import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'dark' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
  children: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  icon,
  iconPosition = 'right',
  children,
  className = '',
  ...props
}) => {
  const variantStyles = {
    primary:
      'bg-indigo-600 dark:bg-indigo-600 text-white hover:bg-indigo-700 dark:hover:bg-indigo-500 active:bg-indigo-800 shadow-sm hover:shadow-md hover:shadow-indigo-500/20 border border-indigo-500/20',
    secondary:
      'bg-teal-600 dark:bg-teal-600 text-white hover:bg-teal-700 dark:hover:bg-teal-500 active:bg-teal-800 shadow-sm hover:shadow-md hover:shadow-teal-500/20 border border-teal-500/20',
    dark:
      'bg-slate-900 dark:bg-slate-800 text-white hover:bg-slate-800 dark:hover:bg-slate-700 active:bg-slate-950 shadow-sm border border-slate-700 dark:border-slate-600',
    outline:
      'bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800 hover:border-slate-300 dark:hover:border-slate-700 active:bg-slate-100 shadow-xs',
    ghost:
      'bg-transparent text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 active:bg-slate-200 dark:active:bg-slate-700 border border-transparent',
  };

  const sizeStyles = {
    sm: 'text-xs px-3.5 py-2 rounded-xl gap-1.5 font-medium',
    md: 'text-sm px-5 py-2.5 rounded-2xl gap-2 font-medium',
    lg: 'text-base px-6 py-3.5 rounded-2xl gap-2.5 font-semibold',
  };

  return (
    <button
      className={`inline-flex items-center justify-center transition-all duration-200 cursor-pointer focus:outline-none focus:ring-2 focus:ring-teal-500/40 focus:ring-offset-2 dark:focus:ring-offset-slate-900 disabled:opacity-60 disabled:cursor-not-allowed disabled:shadow-none ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
      {...props}
    >
      {icon && iconPosition === 'left' && <span className="inline-flex shrink-0">{icon}</span>}
      <span>{children}</span>
      {icon && iconPosition === 'right' && <span className="inline-flex shrink-0">{icon}</span>}
    </button>
  );
};
