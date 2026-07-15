import { useCallback, useEffect, useRef, useState } from 'react';
import type { Job } from '../types';
import { api } from '../lib/api';

const TERMINAL_STATUSES = new Set(['completed', 'completed_with_warnings', 'failed', 'cancelled', 'interrupted']);

interface JobPollingOptions {
  onCompleted?: (job: Job) => void | Promise<void>;
  onFailed?: (job: Job) => void;
}

export function useJobPolling(jobId: string | null, options: JobPollingOptions = {}) {
  const [job, setJob] = useState<Job | null>(null);
  const completedRef = useRef(options.onCompleted);
  const failedRef = useRef(options.onFailed);

  useEffect(() => { completedRef.current = options.onCompleted; }, [options.onCompleted]);
  useEffect(() => { failedRef.current = options.onFailed; }, [options.onFailed]);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    let settled = false;
    let interval: ReturnType<typeof setInterval> | null = null;

    const poll = async () => {
      try {
        const { data } = await api.get<Job>(`/jobs/${jobId}`);
        if (cancelled) return;
        setJob(data);
        if (!settled && TERMINAL_STATUSES.has(data.status)) {
          settled = true;
          if (interval) clearInterval(interval);
          if (['failed', 'cancelled', 'interrupted'].includes(data.status)) failedRef.current?.(data);
          else await completedRef.current?.(data);
        }
      } catch {
        // Keep polling after transient network errors.
      }
    };

    void poll();
    interval = setInterval(poll, 3000);
    return () => {
      cancelled = true;
      if (interval) clearInterval(interval);
    };
  }, [jobId]);

  const resetJob = useCallback(() => setJob(null), []);
  return { job, resetJob };
}
