import { AlertCircle, AlertTriangle, Ban, CheckCircle, Clock, Layers, Loader2, Power } from 'lucide-react';
import type { JobStatus } from '../types';

interface JobStatusBadgeProps {
  status: JobStatus;
  progressPercent?: number;
  compact?: boolean;
  errorMessage?: string;
}

const CONFIG: Record<
  JobStatus,
  { label: string; bgClass: string; textClass: string; borderClass: string; icon: React.ReactNode }
> = {
  queued: {
    label: 'Queued',
    bgClass: 'bg-gray-100 dark:bg-gray-700/60',
    textClass: 'text-gray-500 dark:text-gray-400 font-dm-sans',
    borderClass: 'border-gray-200 dark:border-gray-600',
    icon: <Clock className="w-3.5 h-3.5 shrink-0" />,
  },
  running: {
    label: 'Running',
    bgClass: 'bg-blue-50 dark:bg-blue-900/30',
    textClass: 'text-blue-600 dark:text-blue-400 font-dm-sans',
    borderClass: 'border-blue-200 dark:border-blue-700',
    icon: <Loader2 className="w-3.5 h-3.5 shrink-0 animate-spin" />,
  },
  partial_results: {
    label: 'Partial results',
    bgClass: 'bg-yellow-50 dark:bg-yellow-900/20',
    textClass: 'text-yellow-700 dark:text-yellow-400 font-dm-sans',
    borderClass: 'border-yellow-200 dark:border-yellow-700',
    icon: <Layers className="w-3.5 h-3.5 shrink-0" />,
  },
  completed: {
    label: 'Completed',
    bgClass: 'bg-green-50 dark:bg-green-900/20',
    textClass: 'text-green-700 dark:text-green-400 font-dm-sans',
    borderClass: 'border-green-200 dark:border-green-700',
    icon: <CheckCircle className="w-3.5 h-3.5 shrink-0" />,
  },
  completed_with_warnings: {
    label: 'Completed with warnings',
    bgClass: 'bg-amber-50 dark:bg-amber-900/20',
    textClass: 'text-amber-700 dark:text-amber-400 font-dm-sans',
    borderClass: 'border-amber-200 dark:border-amber-700',
    icon: <AlertTriangle className="w-3.5 h-3.5 shrink-0" />,
  },
  failed: {
    label: 'Failed',
    bgClass: 'bg-red-50 dark:bg-red-900/20',
    textClass: 'text-red-600 dark:text-red-400 font-dm-sans',
    borderClass: 'border-red-200 dark:border-red-700',
    icon: <AlertCircle className="w-3.5 h-3.5 shrink-0" />,
  },
  cancelled: {
    label: 'Cancelled',
    bgClass: 'bg-gray-100 dark:bg-gray-700/60',
    textClass: 'text-gray-600 dark:text-gray-300 font-dm-sans',
    borderClass: 'border-gray-300 dark:border-gray-600',
    icon: <Ban className="w-3.5 h-3.5 shrink-0" />,
  },
  interrupted: {
    label: 'Interrupted',
    bgClass: 'bg-orange-50 dark:bg-orange-900/20',
    textClass: 'text-orange-700 dark:text-orange-400 font-dm-sans',
    borderClass: 'border-orange-200 dark:border-orange-700',
    icon: <Power className="w-3.5 h-3.5 shrink-0" />,
  },
};

export function JobStatusBadge({ status, progressPercent, compact = false, errorMessage }: JobStatusBadgeProps) {
  const { label, bgClass, textClass, borderClass, icon } = CONFIG[status];

  const displayLabel =
    status === 'running' && progressPercent != null
        ? `Running · ${progressPercent}%`
        : compact && status === 'completed_with_warnings'
            ? 'With warnings'
            : label;

  const badge = (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-lg border text-xs font-medium whitespace-nowrap
        ${bgClass} ${textClass} ${borderClass}`}
    >
      {icon}
      {displayLabel}
    </span>
  );

  if (['failed', 'cancelled', 'interrupted'].includes(status) && errorMessage) {
    return (
      <div className="flex items-center gap-2">
        {badge}
        <span className="text-xs text-gray-400 italic truncate max-w-xs">
          {errorMessage.split(/[.—]/)[0].trim().slice(0, 60)}
        </span>
      </div>
    );
  }

  return badge;
}
