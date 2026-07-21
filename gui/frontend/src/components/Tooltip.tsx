import type { ReactNode } from 'react';

interface TooltipProps {
  content: string;
  children: ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
}

export function Tooltip({ content, children, position = 'top' }: TooltipProps) {
  const positionClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  }[position];

  return (
    <div className="relative inline-flex group">
      {children}
      <div
        className={`absolute ${positionClasses} z-50 px-2.5 py-1.5 bg-gray-800 dark:bg-gray-950 text-white text-xs rounded-lg
          opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-normal w-36 text-center leading-snug shadow-lg`}
      >
        {content}
      </div>
    </div>
  );
}