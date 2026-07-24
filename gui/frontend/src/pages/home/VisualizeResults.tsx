import { useState, useCallback, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { AlertTriangle, Clock, FileDigit, FolderOpen, Loader2, Trash2, X } from 'lucide-react';
import { Sidebar } from './Sidebar';
import { MetricCards } from './MetricCards';
import { QueryList } from './QueryList';
import { ResultsPanel } from './ResultsPanel';
import { JobFailurePanel } from '../../components/JobProgressPanel';
import { CancelJobButton } from '../../components/CancelJobButton';
import type { SearchState, QueryResult, JobStatus, SearchResultsSummary, Job, JobFailure, JobProgress, SearchMode } from '../../types';
import type { SelectedItem } from './SelectedResultPanel';
import { SelectedResultPanel } from './SelectedResultPanel';
import { api } from '../../lib/api';
import { Tooltip } from '../../components/Tooltip';
import { useSystemPolicy } from '../../context/SystemPolicyContext';

type ResultTab = 'protein_ranking' | 'known_bindings' | 'predicted_ligands';

const TERMINAL_STATUSES: JobStatus[] = [
  'completed',
  'completed_with_warnings',
  'failed',
  'cancelled',
  'interrupted',
];
const QUERY_HAS_RESULTS: JobStatus[] = ['partial_results', 'completed', 'completed_with_warnings'];

interface HistoryEntry {
  result_id: string;
  result_label?: string;
  created_at: string;
  n_queries: number;
  queries: string[];
  status: 'completed' | 'partial';
  search_mode?: SearchMode;
}

interface ClearHistoryResponse {
  deleted_count: number;
  deleted_results: string[];
  skipped_active: string[];
  failed_results: string[];
}

function toSummary(q: Record<string, unknown>): SearchResultsSummary {
  return {
    qseqid: q.qseqid as string,
    n_proteins_sequence: (q.n_proteins_sequence as number) ?? 0,
    n_proteins_nearest_k: (q.n_proteins_nearest_k as number) ?? 0,
    n_proteins_domain: (q.n_proteins_domain as number) ?? 0,
    n_known_ligands_sequence: (q.n_known_ligands_sequence as number) ?? 0,
    n_known_ligands_nearest_k: (q.n_known_ligands_nearest_k as number) ?? 0,
    n_known_ligands_domain: (q.n_known_ligands_domain as number) ?? 0,
    n_predicted_ligands_sequence: (q.n_predicted_ligands_sequence as number) ?? 0,
    n_predicted_ligands_nearest_k: (q.n_predicted_ligands_nearest_k as number) ?? 0,
    n_predicted_ligands_domain: (q.n_predicted_ligands_domain as number) ?? 0,
  };
}

function emptySummary(qseqid: string): SearchResultsSummary {
  return {
    qseqid,
    n_proteins_sequence: 0, n_proteins_nearest_k: 0, n_proteins_domain: 0,
    n_known_ligands_sequence: 0, n_known_ligands_nearest_k: 0, n_known_ligands_domain: 0,
    n_predicted_ligands_sequence: 0, n_predicted_ligands_nearest_k: 0, n_predicted_ligands_domain: 0,
  };
}

function fmtDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short', day: 'numeric', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

function resultLabel(result_id: string): string {
  // Strip trailing _YYYYMMDD_HHMMSS to get a readable stem
  return result_id.replace(/_\d{8}_\d{6}$/, '') || result_id;
}

export function VisualizeResults() {
  const { policy, isWeb } = useSystemPolicy();
  const [searchState, setSearchState] = useState<SearchState>('idle');
  const [activeSearchMode, setActiveSearchMode] = useState<SearchMode>('zinc');
  const [results, setResults] = useState<QueryResult[]>([]);
  const [selectedQueryId, setSelectedQueryId] = useState<string | null>(null);
  const [activeResultTab, setActiveResultTab] = useState<ResultTab>('protein_ranking');
  const [selectedItem, setSelectedItem] = useState<SelectedItem | null>(null);

  const [jobId, setJobId] = useState<string | null>(null);
  const [progressPercent, setProgressPercent] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [jobProgress, setJobProgress] = useState<JobProgress | null>(null);
  const [jobFailure, setJobFailure] = useState<JobFailure | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);
  const [jobStartedAt, setJobStartedAt] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [activeResultFolder, setActiveResultFolder] = useState<string | null>(null);

  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [showClearHistoryConfirm, setShowClearHistoryConfirm] = useState(false);
  const [historyClearing, setHistoryClearing] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const historyPanelRef = useRef<HTMLDivElement>(null);
  const resultsPanelRef = useRef<HTMLDivElement>(null);

  const handleJobCreated = useCallback((id: string, searchMode: SearchMode) => {
    setJobId(id);
    setActiveSearchMode(searchMode);
    setActiveResultTab('protein_ranking');
    setSearchState('running');
    setResults([]);
    setSelectedQueryId(null);
    setProgressPercent(0);
    setProgressMessage('');
    setJobProgress(null);
    setJobFailure(null);
    setJobError(null);
    setJobStartedAt(null);
    setActiveResultFolder(null);
  }, []);

  // ── Load a past result from disk ──────────────────────────────────────────
  const loadResultFromDisk = useCallback(async (resultId: string, searchMode: SearchMode = 'zinc') => {
    try {
      const { data } = await api.get(`/jobs/${resultId}/summary`);
      const summaryQueries = (data.queries ?? []) as Record<string, unknown>[];
      if (summaryQueries.length === 0) return;
      const queryResults: QueryResult[] = summaryQueries.map((q) => ({
        summary: toSummary(q),
        status: 'completed' as JobStatus,
      }));
      setResults(queryResults);
      setJobId(resultId);
      setActiveSearchMode(searchMode);
      setActiveResultTab('protein_ranking');
      setSearchState('done');
      setJobFailure(null);
      setJobError(null);
      setSelectedQueryId(null);
      setActiveResultFolder(resultId);
    } catch {
      // silently ignore — history entry may be incomplete
    }
  }, []);

  // ── Fetch history list ────────────────────────────────────────────────────
  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const { data } = await api.get<{ results: HistoryEntry[] }>('/results');
      setHistory(data.results);
      return data.results;
    } catch {
      return [] as HistoryEntry[];
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const handleClearHistory = useCallback(async () => {
    setHistoryClearing(true);
    setHistoryError(null);
    try {
      const { data } = await api.delete<ClearHistoryResponse>('/results');
      await fetchHistory();

      const activeResultId = isWeb ? jobId : activeResultFolder;
      if (activeResultId && data.deleted_results.includes(activeResultId)) {
        setResults([]);
        setSelectedQueryId(null);
        setSelectedItem(null);
        setJobId(null);
        setSearchState('idle');
        setActiveResultTab('protein_ranking');
        setActiveResultFolder(null);
        setProgressPercent(0);
        setProgressMessage('');
        setJobProgress(null);
        setJobFailure(null);
        setJobError(null);
        setJobStartedAt(null);
      }

      setShowClearHistoryConfirm(false);
      if (data.failed_results.length > 0) {
        setHistoryError(`Could not delete ${data.failed_results.length} result folder(s).`);
      }
    } catch (error: unknown) {
      const message =
        (error as { response?: { data?: { message?: string } } })?.response?.data?.message
        ?? 'Search history could not be cleared. Please try again.';
      setHistoryError(message);
    } finally {
      setHistoryClearing(false);
    }
  }, [activeResultFolder, fetchHistory, isWeb, jobId]);

  // ── Auto-load most recent result on mount ─────────────────────────────────
  useEffect(() => {
    const restore = async () => {
      try {
        const { data } = await api.get<{ jobs: Job[] }>('/jobs', {
          params: { job_type: 'search' },
        });
        const active = data.jobs.find((job) =>
          ['queued', 'running', 'partial_results'].includes(job.status),
        );
        if (active) {
          handleJobCreated(active.job_id, active.search_mode ?? 'zinc');
          return;
        }
      } catch {
        // Fall through to completed history.
      }
      const entries = await fetchHistory();
      if (entries.length > 0) {
        await loadResultFromDisk(entries[0].result_id, entries[0].search_mode ?? 'zinc');
      }
    };
    restore();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Close history panel on outside click ─────────────────────────────────
  useEffect(() => {
    if (!showHistory || showClearHistoryConfirm) return;
    const handler = (e: MouseEvent) => {
      if (historyPanelRef.current && !historyPanelRef.current.contains(e.target as Node)) {
        setShowHistory(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [showClearHistoryConfirm, showHistory]);

  useEffect(() => {
    if (!showClearHistoryConfirm) return;
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !historyClearing) {
        setShowClearHistoryConfirm(false);
        setHistoryError(null);
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [historyClearing, showClearHistoryConfirm]);

  // ── Polling effect ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!jobId || searchState === 'idle' || searchState === 'done') return;

    const poll = async () => {
      try {
        const [{ data: job }, summaryResult] = await Promise.all([
          api.get<Job>(`/jobs/${jobId}`),
          api.get(`/jobs/${jobId}/summary`).catch(() => ({ data: { queries: [] } })),
        ]);

        setProgressPercent(job.progress_percent ?? 0);
        setProgressMessage(job.progress_message ?? '');
        setJobProgress(job.progress ?? null);
        setJobFailure(job.failure ?? null);
        setJobError(job.error ?? null);
        setJobStartedAt(job.started_at ?? null);
        if (job.search_mode) setActiveSearchMode(job.search_mode);
        if (job.output_dir) {
          setActiveResultFolder((job.output_dir as string).split('/').pop() ?? null);
        }

        const allQueryIds: string[] = (job.all_queries as string[] | undefined) ?? [];
        const summaryQueries = (summaryResult.data.queries ?? []) as Record<string, unknown>[];
        const summaryById = new Map(summaryQueries.map((q) => [q.qseqid as string, q]));

        let queryResults: QueryResult[];

        if (allQueryIds.length > 0) {
          queryResults = allQueryIds.map((id) => {
            const q = summaryById.get(id);
            return q
              ? { summary: toSummary(q), status: 'completed' as JobStatus }
              : { summary: emptySummary(id), status: 'queued' as JobStatus };
          });
        } else {
          queryResults = summaryQueries.map((q) => ({
            summary: toSummary(q),
            status: 'completed' as JobStatus,
          }));
        }

        if (queryResults.length > 0) setResults(queryResults);

        if (job.status === 'cancelled') {
          setResults([]);
          setSelectedQueryId(null);
          setSelectedItem(null);
          setActiveResultTab('protein_ranking');
        }

        if (TERMINAL_STATUSES.includes(job.status as JobStatus)) {
          setSearchState('done');
        }
      } catch {
        // ignore poll errors
      }
    };

    poll();
    const interval = setInterval(poll, 3000);
    return () => clearInterval(interval);
  }, [jobId, searchState]);

  const handleSelectQuery = useCallback((qseqid: string, tab?: ResultTab) => {
    setSelectedQueryId(qseqid);
    setActiveResultTab(
      activeSearchMode === 'known_only' && tab === 'predicted_ligands'
        ? 'known_bindings'
        : tab ?? 'protein_ranking',
    );

    const query = results.find((result) => result.summary.qseqid === qseqid);
    if (query && QUERY_HAS_RESULTS.includes(query.status)) {
      window.requestAnimationFrame(() => {
        resultsPanelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    }
  }, [activeSearchMode, results]);

  const selectedResult = results.find((r) => r.summary.qseqid === selectedQueryId) ?? null;

  // ── History panel ─────────────────────────────────────────────────────────
  const HistoryPanel = (
    <div
      ref={historyPanelRef}
      className="absolute top-12 right-0 z-30 w-85 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700
        rounded-2xl shadow-xl overflow-hidden"
    >
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-gray-700/60">
        <span className="text-sm font-semibold text-gray-700 dark:text-gray-200">Search history</span>
        <button
          onClick={() => setShowHistory(false)}
          className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors cursor-pointer"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="max-h-96 overflow-y-auto divide-y divide-gray-50 dark:divide-gray-800">
        {historyLoading && (
          <p className="px-4 py-6 text-sm text-gray-400 text-center">Loading…</p>
        )}
        {!historyLoading && history.length === 0 && (
          <p className="px-4 py-6 text-sm text-gray-400 text-center">No past results found.</p>
        )}
        {!historyLoading && history.map((entry) => (
          <div key={entry.result_id} className="px-4 py-3 flex items-start justify-between gap-3 hover:bg-gray-50 dark:hover:bg-gray-800/40 transition-colors">
            <div className="min-w-0">
              <p className="text-xs font-semibold text-gray-700 dark:text-gray-200 truncate">
                {entry.result_label ?? resultLabel(entry.result_id)}
              </p>
              <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
                {fmtDate(entry.created_at)}
              </p>
              <p className="text-xs text-gray-400 dark:text-gray-500">
                {entry.n_queries} {entry.n_queries === 1 ? 'query' : 'queries'}
                {entry.status === 'partial' && <span className="ml-1 text-amber-500">· partial</span>}
                <span className="ml-1">
                  · {entry.search_mode === 'known_only'
                    ? 'known only'
                    : isWeb ? 'ZINC' : 'predicted + known'}
                </span>
              </p>
            </div>
            <button
              onClick={() => {
                loadResultFromDisk(entry.result_id, entry.search_mode ?? 'zinc');
                setShowHistory(false);
              }}
              className="shrink-0 flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium rounded-lg
                bg-[#0d5c6b]/10 text-[#0d5c6b] dark:bg-teal-900/30 dark:text-teal-300
                hover:bg-[#0d5c6b]/20 dark:hover:bg-teal-900/50 transition-colors cursor-pointer"
            >
              <FolderOpen className="w-3.5 h-3.5" />
              Load
            </button>
          </div>
        ))}
      </div>

      <div className="border-t border-gray-100 px-4 py-3 dark:border-gray-700/60">
        {isWeb && (
          <p className="mb-2 text-xs leading-relaxed text-gray-400 dark:text-gray-500">
            This browser session only. Results expire after{' '}
            {Math.round((policy.search.result_retention_seconds ?? 7200) / 3600)} hours.
          </p>
        )}
        <div className="flex justify-end">
          <Tooltip content="Permanently deletes stored search results to free disk space. Active searches are preserved." position="left">
            <button
              type="button"
              disabled={historyLoading || historyClearing || history.length === 0}
              onClick={() => {
                setHistoryError(null);
                setShowClearHistoryConfirm(true);
              }}
              className="flex cursor-pointer items-center gap-1.5 rounded-lg border border-red-200 px-2.5 py-1.5
                text-xs font-medium text-red-600 transition-colors hover:bg-red-50 disabled:cursor-not-allowed
                disabled:opacity-40 dark:border-red-900/60 dark:text-red-400 dark:hover:bg-red-950/30"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear history
            </button>
          </Tooltip>
        </div>
        {historyError && !showClearHistoryConfirm && (
          <p className="mt-2 text-right text-xs text-red-500 dark:text-red-400">{historyError}</p>
        )}
      </div>
    </div>
  );

  const ClearHistoryDialog = showClearHistoryConfirm
    ? createPortal(
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/45 px-4 backdrop-blur-[1px]"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget && !historyClearing) {
              setShowClearHistoryConfirm(false);
              setHistoryError(null);
            }
          }}
        >
          <div
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="clear-history-title"
            aria-describedby="clear-history-description"
            className="w-full max-w-md rounded-2xl border border-gray-200 bg-white p-5 shadow-2xl
              dark:border-gray-700 dark:bg-gray-800"
          >
            <div className="flex items-start gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-red-100 text-red-600
                dark:bg-red-950/50 dark:text-red-400">
                <AlertTriangle className="h-5 w-5" />
              </div>
              <div>
                <h2 id="clear-history-title" className="text-base font-semibold text-gray-800 dark:text-gray-100">
                  Clear search history?
                </h2>
                <p id="clear-history-description" className="mt-1 text-sm leading-relaxed text-gray-600 dark:text-gray-400">
                  {isWeb
                    ? 'This permanently deletes the completed results from this browser session and cannot be undone. An active search will be preserved.'
                    : 'This permanently deletes all stored search result folders and cannot be undone. Results from an active search will be preserved.'}
                </p>
              </div>
            </div>

            {historyError && (
              <p className="mt-4 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-600 dark:bg-red-950/30 dark:text-red-400">
                {historyError}
              </p>
            )}

            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                autoFocus
                disabled={historyClearing}
                onClick={() => {
                  setShowClearHistoryConfirm(false);
                  setHistoryError(null);
                }}
                className="cursor-pointer rounded-lg border border-gray-300 px-3 py-2 text-sm font-medium text-gray-600
                  transition-colors hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50
                  dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-700"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={historyClearing}
                onClick={handleClearHistory}
                className="flex cursor-pointer items-center gap-1.5 rounded-lg bg-red-600 px-3 py-2 text-sm font-semibold
                  text-white transition-colors hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {historyClearing && <Loader2 className="h-4 w-4 animate-spin" />}
                Clear history
              </button>
            </div>
          </div>
        </div>,
        document.body,
      )
    : null;

  return (
    <section className="flex flex-col sm:flex-row min-h-[calc(100vh-80px)] sm:h-[calc(100vh-80px)] overflow-visible sm:overflow-hidden">
      <Sidebar
        isRunning={searchState === 'running'}
        progressPercent={progressPercent}
        progressMessage={progressMessage}
        progress={jobProgress}
        failure={jobFailure}
        jobError={jobError}
        startedAt={jobStartedAt}
        onJobCreated={handleJobCreated}
      />

      <main className="flex-1 min-w-0 min-h-96 overflow-visible sm:overflow-y-auto bg-[#f8f9fa] dark:bg-[#161c23] relative">
        {/* History button */}
        <div className="absolute top-4 right-3 z-20">
          <button
            onClick={() => {
              if (!showHistory) fetchHistory();
              setShowHistory((v) => !v);
            }}
            title="Search history"
            className={`flex items-center gap-1.5 px-4 py-2 rounded-xl font-medium font-dm-sans border transition-colors cursor-pointer
              ${showHistory
                ? 'bg-white text-gray-800 border-gray-800 dark:text-gray-200 dark:border-gray-200 dark:bg-gray-800/10'
                : 'bg-gray-800 hover:bg-white text-gray-50 hover:text-gray-800 border-gray-200 hover:border-gray-800 dark:bg-gray-800 dark:hover:bg-gray-800/10 dark:text-gray-200 dark:hover:text-gray-200 dark:border-gray-300 dark:hover:border-gray-400 transition-colors duration-300'
              }`}
          >
            <Clock className="w-4 h-4" />
            History
          </button>
          {showHistory && HistoryPanel}
        </div>

        {searchState === 'done' && (jobFailure || jobError) && (
          <div className="px-4 pb-2 pt-14 sm:px-6">
            <JobFailurePanel failure={jobFailure} error={jobError} />
          </div>
        )}

        {(searchState === 'running' || searchState === 'done') && activeResultFolder && (
          <div className="flex flex-wrap items-center gap-3 px-4 pt-4 pr-32 text-gray-700 dark:text-gray-200 sm:px-6">
            <FileDigit className="h-5 w-5 shrink-0" />
            <span
              className="min-w-0 max-w-full truncate rounded-lg border border-gray-200 bg-gray-200 px-3 py-1 text-sm font-jetbrains-mono dark:bg-gray-800"
              title="Results folder"
            >
              {activeResultFolder}
            </span>
            {searchState === 'running' && jobId && (
              <CancelJobButton
                jobId={jobId}
                resourceLabel="search"
                description={isWeb
                  ? 'The running search and any partial artifacts will be deleted immediately.'
                  : 'The running search will stop and its partial results and temporary files will be deleted.'}
                cancelling={cancelling}
                onCancelStarted={() => setCancelling(true)}
                onCancelFinished={() => setCancelling(false)}
                onCancelError={(message) => {
                  setCancelling(false);
                  setJobError(message);
                }}
                compact
              />
            )}
          </div>
        )}

        {searchState === 'idle' && (
          <div className="flex flex-col items-center justify-center h-full min-h-96 gap-4 text-center px-6">
            <div className="text-5xl select-none">⬡</div>
            <h2 className="text-xl font-semibold text-gray-700 dark:text-gray-200">
              No search results yet
            </h2>
            <p className="text-sm text-gray-400 dark:text-gray-500 max-w-sm">
              Configure your search parameters in the sidebar and click{' '}
              <span className="font-medium text-[#0d5c6b] dark:text-teal-400">Run Search</span> to begin.
            </p>
          </div>
        )}

        {searchState === 'running' && results.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full min-h-96 gap-4 px-6">
            <div className="w-10 h-10 border-3 border-[#0d5c6b] border-t-transparent rounded-full animate-spin" />
            <p className="text-sm text-gray-500 dark:text-gray-400">Search in progress…</p>
          </div>
        )}

        {(searchState === 'running' || searchState === 'done') && results.length > 0 && (
          <div className="p-4 sm:p-6 flex flex-col gap-0">
            <MetricCards
              summaries={results.map((r) => r.summary)}
              results={results}
              showPredicted={activeSearchMode !== 'known_only'}
            />
            <QueryList
              results={results}
              selectedQueryId={selectedQueryId}
              onSelectQuery={handleSelectQuery}
              showPredicted={activeSearchMode !== 'known_only'}
            />
            {selectedResult && (
              <div ref={resultsPanelRef} className="scroll-mt-4">
                <ResultsPanel
                  queryResult={selectedResult}
                  activeTab={activeResultTab}
                  onTabChange={setActiveResultTab}
                  selectedItem={selectedItem}
                  onSelectItem={setSelectedItem}
                  jobId={jobId ?? ''}
                  knownOnly={activeSearchMode === 'known_only'}
                />
              </div>
            )}
          </div>
        )}
      </main>

      {selectedItem && (
        <SelectedResultPanel
          selected={selectedItem}
          onClose={() => setSelectedItem(null)}
        />
      )}
      {ClearHistoryDialog}
    </section>
  );
}
