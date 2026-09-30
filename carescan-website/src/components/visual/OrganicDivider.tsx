import React from 'react';

interface OrganicDividerProps {
  position?: 'top' | 'bottom';
  fillColor?: string;
  variant?: 'curve-1' | 'curve-2' | 'wave';
  className?: string;
}

export const OrganicDivider: React.FC<OrganicDividerProps> = ({
  position = 'bottom',
  fillColor = '#FFFFFF',
  variant = 'curve-1',
  className = '',
}) => {
  const isTop = position === 'top';

  return (
    <div
      className={`w-full overflow-hidden leading-none select-none pointer-events-none ${isTop ? 'transform rotate-180' : ''} ${className}`}
      aria-hidden="true"
    >
      {variant === 'curve-1' && (
        <svg
          viewBox="0 0 1440 80"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="w-full h-8 sm:h-12 md:h-16 block"
          preserveAspectRatio="none"
        >
          <path
            d="M0 0C320 60 720 75 1120 40C1280 26 1380 12 1440 0V80H0V0Z"
            fill={fillColor}
          />
        </svg>
      )}

      {variant === 'curve-2' && (
        <svg
          viewBox="0 0 1440 90"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="w-full h-10 sm:h-14 md:h-20 block"
          preserveAspectRatio="none"
        >
          <path
            d="M0 50C240 10 600 80 960 30C1200 -5 1360 20 1440 40V90H0V50Z"
            fill={fillColor}
          />
        </svg>
      )}

      {variant === 'wave' && (
        <svg
          viewBox="0 0 1440 100"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="w-full h-10 sm:h-16 md:h-24 block"
          preserveAspectRatio="none"
        >
          <path
            d="M0 40C300 90 600 0 900 60C1100 100 1300 30 1440 50V100H0V40Z"
            fill={fillColor}
          />
        </svg>
      )}
    </div>
  );
};
