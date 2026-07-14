import { Activity, Clock3, Gauge, Loader2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import type { JobProgress } from '../types';

interface JobProgressPanelProps {
  progress: JobProgress | null | undefined;
  fallbackPercent?: number;
  fallbackMessage?: string;
  startedAt?: string | null;
  compact?: boolean;
}

function formatDuration(totalSeconds: number): string {
  const seconds = Math.max(0, Math.round(totalSeconds));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  return hours > 0
    ? `${hours}h ${minutes.toString().padStart(2, '0')}m`
    : `${minutes}m ${rest.toString().padStart(2, '0')}s`;
}

function formatCount(value: number): string {
  return new Intl.NumberFormat().format(value);
}

export function JobProgressPanel({
  progress,
  fallbackPercent = 0,
  fallbackMessage = 'Processing',
  startedAt,
  compact = false,
}: JobProgressPanelProps) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  const elapsed = useMemo(() => {
    if (!startedAt) return null;
    const started = new Date(startedAt).getTime();
    return Number.isFinite(started) ? Math.max(0, (now - started) / 1000) : null;
  }, [now, startedAt]);

  const percent = Math.max(0, Math.min(100, progress?.percent ?? fallbackPercent));
  const label = progress?.label || fallbackMessage;
  const hasCount = progress?.current != null && progress.total != null;

  return (
    <div className={`flex flex-col gap-2.5 ${compact ? 'pt-3 border-t border-gray-200 dark:border-gray-700' : 'mt-4'}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-2 min-w-0">
          <Loader2 className="w-4 h-4 mt-0.5 text-[#0d5c6b] dark:text-teal-400 animate-spin shrink-0" />
          <div className="min-w-0">
            <p className="text-xs font-medium text-gray-700 dark:text-gray-200 leading-snug">{label}</p>
            {progress?.context && (
              <p className="text-xs font-jetbrains-mono text-gray-400 dark:text-gray-500 truncate mt-0.5">
                {progress.context}
              </p>
            )}
          </div>
        </div>
        {progress && (
          <span className="shrink-0 px-2 py-1 rounded-md bg-gray-100 dark:bg-gray-700 text-xs font-medium text-gray-500 dark:text-gray-300">
            Step {progress.step_index}/{progress.step_count}
          </span>
        )}
      </div>

      <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-[#0d5c6b] dark:bg-teal-500 rounded-full transition-[width] duration-500"
          style={{ width: `${percent}%` }}
        />
      </div>

      <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs text-gray-500 dark:text-gray-400">
        <span className="flex items-center gap-1.5">
          <Gauge className="w-3.5 h-3.5" />
          {percent}%
        </span>
        {hasCount && (
          <span className="text-right truncate" title={`${progress.current} / ${progress.total}`}>
            {formatCount(progress.current!)} / {formatCount(progress.total!)} {progress.unit ?? ''}
          </span>
        )}
        {progress?.eta_seconds != null && progress.eta_seconds > 0 && (
          <span className="flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5" />
            ETA {formatDuration(progress.eta_seconds)}
          </span>
        )}
        {elapsed != null && (
          <span className="flex items-center gap-1.5 justify-end col-start-2">
            <Clock3 className="w-3.5 h-3.5" />
            {formatDuration(elapsed)} elapsed
          </span>
        )}
      </div>
    </div>
  );
}
