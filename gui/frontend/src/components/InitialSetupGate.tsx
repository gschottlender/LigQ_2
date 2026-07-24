import { AlertTriangle, Database, Download, HardDrive, Loader2, RefreshCw, ShieldCheck } from 'lucide-react';
import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { useDatabase } from '../context/DatabaseContext';
import { useSystemPolicy } from '../context/SystemPolicyContext';
import { api } from '../lib/api';
import type { Job, SetupPackageStatus, SetupStatus, WebReadiness } from '../types';
import { Header } from './Header';
import { JobFailurePanel, JobProgressPanel } from './JobProgressPanel';


const TERMINAL_STATUSES = ['completed', 'completed_with_warnings', 'failed', 'cancelled', 'interrupted'];

const PACKAGE_COPY: Record<SetupPackageStatus['id'], {
  title: string;
  description: string;
  badge: string;
}> = {
  core: {
    title: 'Required databases',
    description: 'ZINC and PDB/ChEMBL data, BLAST/Pfam resources, and supported BSI models.',
    badge: 'Required',
  },
  ecfp_cache: {
    title: 'Morgan ECFP cache',
    description: 'Precomputed ZINC predictions with Tanimoto scores from 0.4 upward.',
    badge: 'Recommended',
  },
  fcfp_cache: {
    title: 'Morgan Feature FCFP cache',
    description: 'FCFP representations plus precomputed ZINC predictions from 0.5 upward.',
    badge: 'Optional',
  },
};

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
  const { isWeb } = useSystemPolicy();
  return isWeb
    ? <WebReadinessGate>{children}</WebReadinessGate>
    : <LocalInitialSetupGate>{children}</LocalInitialSetupGate>;
}

function WebReadinessGate({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<WebReadiness | null>(null);
  const [checking, setChecking] = useState(true);

  const loadStatus = useCallback(async () => {
    setChecking(true);
    try {
      const { data } = await api.get<WebReadiness>('/system/readiness', { timeout: 130_000 });
      setStatus(data);
    } catch {
      setStatus({
        ready: false,
        mode: 'web',
        checks: {},
        errors: ['The public search service could not validate its required data.'],
      });
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    // Initial API synchronization intentionally owns the gate's loading state.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadStatus();
    const interval = window.setInterval(loadStatus, 30_000);
    return () => window.clearInterval(interval);
  }, [loadStatus]);

  if (status?.ready) return children;

  return (
    <>
      <Header />
      <main className="flex min-h-[calc(100vh-80px)] items-start justify-center bg-gray-50 px-4 py-12 dark:bg-[#111827]">
        <div className="w-full max-w-xl rounded-2xl border border-gray-200 bg-white p-7 shadow-sm dark:border-gray-700 dark:bg-[#1a2330] sm:p-9">
          <div className="flex items-start gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300">
              {checking && !status
                ? <Loader2 className="h-5 w-5 animate-spin" />
                : <AlertTriangle className="h-5 w-5" />}
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 dark:text-gray-500">
                Public service
              </p>
              <h1 className="mt-1 text-xl font-semibold text-gray-800 dark:text-gray-100 sm:text-2xl">
                {checking && !status ? 'Checking search data' : 'Search temporarily unavailable'}
              </h1>
              <p className="mt-2 text-sm leading-relaxed text-gray-600 dark:text-gray-400">
                The web service requires the core databases and both precomputed ZINC caches
                before it can accept any search, including Known ligands only searches.
              </p>
            </div>
          </div>

          {status?.errors && status.errors.length > 0 && (
            <div className="mt-6 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-200">
              <p className="font-semibold">Administrator action is required.</p>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-xs leading-relaxed">
                {status.errors.slice(0, 4).map((item) => <li key={item}>{item}</li>)}
              </ul>
            </div>
          )}

          <button
            type="button"
            onClick={loadStatus}
            disabled={checking}
            className="mt-6 inline-flex cursor-pointer items-center gap-2 rounded-xl bg-cyan-900 px-4 py-2.5 text-sm font-semibold text-white hover:bg-cyan-800 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <RefreshCw className={`h-4 w-4 ${checking ? 'animate-spin' : ''}`} />
            Check again
          </button>
        </div>
      </main>
    </>
  );
}

function LocalInitialSetupGate({ children }: { children: ReactNode }) {
  const { refetchDatabases } = useDatabase();
  const [status, setStatus] = useState<SetupStatus | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [checking, setChecking] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [includeEcfpCache, setIncludeEcfpCache] = useState(true);
  const [includeFcfpCache, setIncludeFcfpCache] = useState(false);

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
          if (['failed', 'cancelled', 'interrupted'].includes(data.status)) {
            setError(data.error || data.failure?.message || 'Initial setup failed.');
            setJobId(null);
            setStatus((current) => current ? {
              ...current,
              state: 'required',
              job_id: null,
              job_status: data.status,
            } : current);
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
      const { data } = await api.post<{ job_id: string; status: string }>('/setup/download', {
        include_ecfp_cache: includeEcfpCache,
        include_fcfp_cache: includeFcfpCache,
      });
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
  const selectedPackageIds = new Set<SetupPackageStatus['id']>([
    'core',
    ...(includeEcfpCache ? ['ecfp_cache' as const] : []),
    ...(includeFcfpCache ? ['fcfp_cache' as const] : []),
  ]);
  const selectedPackages = (status?.packages ?? []).filter((item) => selectedPackageIds.has(item.id));
  const requiredBytes = selectedPackages.reduce(
    (total, item) => total + item.required_download_bytes,
    0,
  );
  const requiredFileCount = selectedPackages.reduce(
    (total, item) => total + item.required_file_count,
    0,
  );
  const availableBytes = status?.available_bytes ?? 0;
  const enoughSpace = requiredBytes <= availableBytes;

  const packageSelected = (packageId: SetupPackageStatus['id']) => (
    packageId === 'core'
    || (packageId === 'ecfp_cache' && includeEcfpCache)
    || (packageId === 'fcfp_cache' && includeFcfpCache)
  );

  const setPackageSelected = (packageId: SetupPackageStatus['id'], selected: boolean) => {
    if (packageId === 'ecfp_cache') setIncludeEcfpCache(selected);
    if (packageId === 'fcfp_cache') setIncludeFcfpCache(selected);
  };

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
                Install the required search databases and choose which precomputed caches to download.
                Caches avoid lengthy calculations during the first searches.
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
              <div className="mt-7">
                <div className="mb-3 flex items-end justify-between gap-4">
                  <div>
                    <h2 className="text-sm font-semibold text-gray-800 dark:text-gray-100">
                      Choose installation data
                    </h2>
                    <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                      Sizes reflect only files that are not already installed.
                    </p>
                  </div>
                </div>

                <div className="space-y-3">
                  {(status?.packages ?? []).map((setupPackage) => {
                    const copy = PACKAGE_COPY[setupPackage.id];
                    const selected = packageSelected(setupPackage.id);
                    const disabled = setupPackage.required || isRunning || starting;
                    return (
                      <label
                        key={setupPackage.id}
                        className={`flex items-start gap-3 rounded-xl border p-4 transition-colors ${
                          selected
                            ? 'border-teal-300 bg-teal-50/70 dark:border-teal-800 dark:bg-teal-950/20'
                            : 'border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-800/40'
                        } ${disabled ? 'cursor-default' : 'cursor-pointer'}`}
                      >
                        <input
                          type="checkbox"
                          checked={selected}
                          disabled={disabled}
                          onChange={(event) => setPackageSelected(setupPackage.id, event.target.checked)}
                          className="mt-0.5 h-4 w-4 shrink-0 accent-teal-700"
                        />
                        <span className="min-w-0 flex-1">
                          <span className="flex flex-wrap items-center gap-2">
                            <span className="text-sm font-semibold text-gray-800 dark:text-gray-100">
                              {copy.title}
                            </span>
                            <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
                              setupPackage.required
                                ? 'bg-gray-200 text-gray-600 dark:bg-gray-700 dark:text-gray-300'
                                : setupPackage.default_selected
                                ? 'bg-teal-100 text-teal-700 dark:bg-teal-900/50 dark:text-teal-300'
                                : 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300'
                            }`}>
                              {copy.badge}
                            </span>
                          </span>
                          <span className="mt-1 block text-xs leading-relaxed text-gray-500 dark:text-gray-400">
                            {copy.description}
                          </span>
                        </span>
                        <span className="shrink-0 text-right">
                          <span className="block text-sm font-semibold text-gray-700 dark:text-gray-200">
                            {formatBytes(setupPackage.required_download_bytes)}
                          </span>
                          <span className="mt-1 block text-[10px] uppercase tracking-wide text-gray-400 dark:text-gray-500">
                            {setupPackage.installed ? 'Installed' : `${setupPackage.required_file_count} files`}
                          </span>
                        </span>
                      </label>
                    );
                  })}
                </div>
              </div>

              <div className="mt-5 grid gap-3 sm:grid-cols-2">
                <div className="rounded-xl border border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-800/50">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-gray-400 dark:text-gray-500">
                    <HardDrive className="h-4 w-4" />
                    Selected download
                  </div>
                  <p className="mt-2 text-xl font-semibold text-gray-800 dark:text-gray-100">
                    {formatBytes(requiredBytes)}
                  </p>
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    {requiredFileCount} files currently missing
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
                  <p className={`mt-1 text-xs ${enoughSpace ? 'text-teal-700 dark:text-teal-300' : 'text-red-600 dark:text-red-400'}`}>
                    {enoughSpace ? 'Enough space for this selection' : 'More free disk space is required'}
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
                . Package sizes are calculated from repository files; existing files are not downloaded again.
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

              {!isRunning && !enoughSpace && (
                <p className="mt-4 text-sm text-red-600 dark:text-red-400">
                  Free at least {formatBytes(Math.max(0, requiredBytes - availableBytes))} more before starting setup.
                </p>
              )}

              <div className="mt-6 flex flex-col gap-3 sm:flex-row">
                <button
                  type="button"
                  onClick={startDownload}
                  disabled={isRunning || starting || !enoughSpace}
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
