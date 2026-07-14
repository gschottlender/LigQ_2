import { useState, useCallback, useEffect, useRef } from 'react';
import { Clock, FileDigit, FolderOpen, X } from 'lucide-react';
import { Sidebar } from './Sidebar';
import { MetricCards } from './MetricCards';
import { QueryList } from './QueryList';
import { ResultsPanel } from './ResultsPanel';
import type { SearchState, QueryResult, JobStatus, SearchResultsSummary, Job, JobProgress } from '../../types';
import type { SelectedItem } from './SelectedResultPanel';
import { SelectedResultPanel } from './SelectedResultPanel';
import { api } from '../../lib/api';

type ResultTab = 'protein_ranking' | 'known_bindings' | 'predicted_ligands';

const TERMINAL_STATUSES: JobStatus[] = ['completed', 'completed_with_warnings', 'failed'];

interface HistoryEntry {
  result_id: string;
  created_at: string;
  n_queries: number;
  queries: string[];
  status: 'completed' | 'partial';
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
  const [searchState, setSearchState] = useState<SearchState>('idle');
  const [results, setResults] = useState<QueryResult[]>([]);
  const [selectedQueryId, setSelectedQueryId] = useState<string | null>(null);
  const [activeResultTab, setActiveResultTab] = useState<ResultTab>('protein_ranking');
  const [selectedItem, setSelectedItem] = useState<SelectedItem | null>(null);

  const [jobId, setJobId] = useState<string | null>(null);
  const [progressPercent, setProgressPercent] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [jobProgress, setJobProgress] = useState<JobProgress | null>(null);
  const [jobStartedAt, setJobStartedAt] = useState<string | null>(null);
  const [activeResultFolder, setActiveResultFolder] = useState<string | null>(null);

  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const historyPanelRef = useRef<HTMLDivElement>(null);

  const handleJobCreated = useCallback((id: string) => {
    setJobId(id);
    setSearchState('running');
    setResults([]);
    setSelectedQueryId(null);
    setProgressPercent(0);
    setProgressMessage('');
    setJobProgress(null);
    setJobStartedAt(null);
    setActiveResultFolder(null);
  }, []);

  // ── Load a past result from disk ──────────────────────────────────────────
  const loadResultFromDisk = useCallback(async (resultId: string) => {
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
      setSearchState('done');
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

  // ── Auto-load most recent result on mount ─────────────────────────────────
  useEffect(() => {
    fetchHistory().then((entries) => {
      if (entries.length > 0) {
        loadResultFromDisk(entries[0].result_id);
      }
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Close history panel on outside click ─────────────────────────────────
  useEffect(() => {
    if (!showHistory) return;
    const handler = (e: MouseEvent) => {
      if (historyPanelRef.current && !historyPanelRef.current.contains(e.target as Node)) {
        setShowHistory(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [showHistory]);

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
        setJobStartedAt(job.started_at ?? null);
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
    setActiveResultTab(tab ?? 'protein_ranking');
  }, []);

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
                {resultLabel(entry.result_id)}
              </p>
              <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
                {fmtDate(entry.created_at)}
              </p>
              <p className="text-xs text-gray-400 dark:text-gray-500">
                {entry.n_queries} {entry.n_queries === 1 ? 'query' : 'queries'}
                {entry.status === 'partial' && <span className="ml-1 text-amber-500">· partial</span>}
              </p>
            </div>
            <button
              onClick={() => {
                loadResultFromDisk(entry.result_id);
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
    </div>
  );

  return (
    <section className="flex flex-col sm:flex-row min-h-[calc(100vh-80px)] sm:h-[calc(100vh-80px)] overflow-visible sm:overflow-hidden">
      <Sidebar
        isRunning={searchState === 'running'}
        progressPercent={progressPercent}
        progressMessage={progressMessage}
        progress={jobProgress}
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
            <div className="flex items-center gap-3 mb-4 pr-28 text-gray-700 dark:text-gray-200">
            <FileDigit className='w-5 h-5'/>
              {activeResultFolder && (
                <span
                  className="text-sm font-jetbrains-mono bg-gray-200 dark:bg-gray-800 border border-gray-200 px-3 py-1 rounded-lg"
                  title="Results folder"
                >
                  {activeResultFolder}
                </span>
              )}
            </div>
            <MetricCards summaries={results.map((r) => r.summary)} results={results} />
            <QueryList
              results={results}
              selectedQueryId={selectedQueryId}
              onSelectQuery={handleSelectQuery}
            />
            {selectedResult && (
              <ResultsPanel
                queryResult={selectedResult}
                activeTab={activeResultTab}
                onTabChange={setActiveResultTab}
                selectedItem={selectedItem}
                onSelectItem={setSelectedItem}
                jobId={jobId ?? ''}
              />
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
    </section>
  );
}
