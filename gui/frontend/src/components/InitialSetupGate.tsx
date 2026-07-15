import { Database, Download, HardDrive, Loader2, RefreshCw, ShieldCheck } from 'lucide-react';
import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { useDatabase } from '../context/DatabaseContext';
import { api } from '../lib/api';
import type { Job, SetupStatus } from '../types';
import { Header } from './Header';
import { JobFailurePanel, JobProgressPanel } from './JobProgressPanel';


const TERMINAL_STATUSES = ['completed', 'completed_with_warnings', 'failed'];

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 GB';
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(2)} GB`;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  return `${Math.ceil(bytes / 1_000)} KB`;
}

function errorMessage(error: unknown): string {
  const data = (error as {
    response?: { data?: { message?: string; detail?: { message?: string } } };
  })?.response?.data;
  return data?.message ?? data?.detail?.message ?? 'Initial setup could not be started.';
}

export function InitialSetupGate({ children }: { children: ReactNode }) {
  const { refetchDatabases } = useDatabase();
  const [status, setStatus] = useState<SetupStatus | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [checking, setChecking] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadStatus = useCallback(async () => {
    setChecking(true);
    setError(null);
    try {
      const { data } = await api.get<SetupStatus>('/setup/status');
      setStatus(data);
      if (data.job_id) setJobId(data.job_id);
      if (data.ready) {
        await refetchDatabases();
        setJobId(null);
        setJob(null);
      }
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setChecking(false);
    }
  }, [refetchDatabases]);

  useEffect(() => {
    // Initial API synchronization intentionally owns the gate's loading state.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadStatus();
  }, [loadStatus]);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const { data } = await api.get<Job>(`/jobs/${jobId}`);
        if (cancelled) return;
        setJob(data);
        if (TERMINAL_STATUSES.includes(data.status)) {
          if (data.status === 'failed') {
            setError(data.error || data.failure?.message || 'Initial setup failed.');
            setJobId(null);
          } else {
            await loadStatus();
          }
        }
      } catch (requestError) {
        if (!cancelled) setError(errorMessage(requestError));
      }
    };

    poll();
    const interval = window.setInterval(poll, 1_000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [jobId, loadStatus]);

  const startDownload = async () => {
    if (!status?.enough_space || starting) return;
    setStarting(true);
    setError(null);
    setJob(null);
    try {
      const { data } = await api.post<{ job_id: string; status: string }>('/setup/download');
      setJobId(data.job_id);
      setStatus((current) => current ? { ...current, state: 'downloading', job_id: data.job_id } : current);
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setStarting(false);
    }
  };

  if (status?.ready) return children;

  const isRunning = Boolean(jobId) || status?.state === 'downloading';
  const requiredBytes = status?.required_download_bytes ?? status?.total_required_bytes ?? 0;
  const availableBytes = status?.available_bytes ?? 0;

  return (
    <>
      <Header />
      <main className="min-h-[calc(100vh-80px)] bg-gray-50 px-4 py-8 dark:bg-[#111827] sm:px-6 sm:py-12">
        <div className="mx-auto max-w-2xl rounded-2xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-[#1a2330] sm:p-8">
          <div className="flex items-start gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-teal-100 text-[#0d5c6b] dark:bg-teal-900/40 dark:text-teal-300">
              {checking
                ? <Loader2 className="h-5 w-5 animate-spin" />
                : isRunning
                ? <Download className="h-5 w-5" />
                : <Database className="h-5 w-5" />}
            </div>
            <div className="min-w-0">
              <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 dark:text-gray-500">
                First-time initialization
              </p>
              <h1 className="mt-1 text-xl font-semibold text-gray-800 dark:text-gray-100 sm:text-2xl">
                {checking ? 'Checking local data' : isRunning ? 'Preparing LigQ 2' : 'Initial setup required'}
              </h1>
              <p className="mt-2 text-sm leading-relaxed text-gray-600 dark:text-gray-400">
                LigQ 2 needs its default ZINC and PDB/ChEMBL compound data, BLAST and Pfam resources,
                the reusable predicted-ligand cache, and the supported BSI family models before searches can run.
              </p>
            </div>
          </div>

          {checking && !status ? (
            <div className="mt-7 flex items-center gap-3 rounded-xl border border-gray-200 bg-gray-50 p-4 text-sm text-gray-600 dark:border-gray-700 dark:bg-gray-800/50 dark:text-gray-300">
              <Loader2 className="h-4 w-4 animate-spin text-[#0d5c6b] dark:text-teal-300" />
              Inspecting the local installation and Hugging Face dataset metadata…
            </div>
          ) : (
            <>
              <div className="mt-7 grid gap-3 sm:grid-cols-2">
                <div className="rounded-xl border border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-800/50">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-gray-400 dark:text-gray-500">
                    <HardDrive className="h-4 w-4" />
                    Required download
                  </div>
                  <p className="mt-2 text-xl font-semibold text-gray-800 dark:text-gray-100">
                    {formatBytes(requiredBytes)}
                  </p>
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    {status?.required_file_count ?? 0} files currently missing
                  </p>
                </div>
                <div className="rounded-xl border border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-800/50">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-gray-400 dark:text-gray-500">
                    <ShieldCheck className="h-4 w-4" />
                    Available disk space
                  </div>
                  <p className="mt-2 text-xl font-semibold text-gray-800 dark:text-gray-100">
                    {formatBytes(availableBytes)}
                  </p>
                  <p className={`mt-1 text-xs ${status?.enough_space ? 'text-teal-700 dark:text-teal-300' : 'text-red-600 dark:text-red-400'}`}>
                    {status?.enough_space ? 'Enough space for installation' : 'More free disk space is required'}
                  </p>
                </div>
              </div>

              <div className="mt-4 rounded-xl border border-blue-200 bg-blue-50 p-4 text-xs leading-relaxed text-blue-800 dark:border-blue-900/70 dark:bg-blue-950/30 dark:text-blue-200">
                Data source:{' '}
                <a
                  href={`https://huggingface.co/datasets/${status?.repo_id ?? 'gschottlender/LigQ_2'}/tree/${status?.revision ?? 'main'}`}
                  target="_blank"
                  rel="noreferrer"
                  className="font-medium underline decoration-blue-300 underline-offset-2 dark:decoration-blue-700"
                >
                  {status?.repo_id ?? 'gschottlender/LigQ_2'}
                </a>
                . The size is calculated from the required files in the repository; existing files are not downloaded again.
                {status?.metadata_error && (
                  <span className="mt-1 block text-amber-700 dark:text-amber-300">
                    Live metadata is temporarily unavailable, so the latest repository size snapshot is shown.
                  </span>
                )}
              </div>

              {isRunning && (
                <JobProgressPanel
                  progress={job?.progress}
                  fallbackPercent={job?.progress_percent ?? 0}
                  fallbackMessage={job?.progress_message || 'Starting initial setup'}
                  startedAt={job?.started_at}
                />
              )}

              {!isRunning && (job?.failure || error) && (
                <JobFailurePanel failure={job?.failure} error={error} />
              )}

              {!isRunning && !status?.enough_space && (
                <p className="mt-4 text-sm text-red-600 dark:text-red-400">
                  Free at least {formatBytes(Math.max(0, requiredBytes - availableBytes))} more before starting setup.
                </p>
              )}

              <div className="mt-6 flex flex-col gap-3 sm:flex-row">
                <button
                  type="button"
                  onClick={startDownload}
                  disabled={isRunning || starting || !status?.enough_space}
                  className="flex flex-1 cursor-pointer items-center justify-center gap-2 rounded-xl bg-cyan-900 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-cyan-800 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {isRunning || starting
                    ? <><Loader2 className="h-4 w-4 animate-spin" /> Preparing data…</>
                    : <><Download className="h-4 w-4" /> Download and prepare data</>}
                </button>
                {!isRunning && (
                  <button
                    type="button"
                    onClick={loadStatus}
                    disabled={checking}
                    className="flex cursor-pointer items-center justify-center gap-2 rounded-xl border border-gray-300 px-4 py-3 text-sm font-medium text-gray-600 transition-colors hover:border-teal-400 hover:text-teal-700 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:text-gray-300 dark:hover:border-teal-500 dark:hover:text-teal-300"
                  >
                    <RefreshCw className={`h-4 w-4 ${checking ? 'animate-spin' : ''}`} />
                    Check again
                  </button>
                )}
              </div>

              <p className="mt-4 text-center text-xs leading-relaxed text-gray-500 dark:text-gray-400">
                Keep the backend running during setup. The download is resumable, and the available databases
                will refresh automatically when it finishes.
              </p>
            </>
          )}
        </div>
      </main>
    </>
  );
}
