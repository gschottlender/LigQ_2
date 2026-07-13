import { AlertCircle, AlertTriangle, Clock, Download, Dna, Loader2, Layers, WaypointsIcon, TrendingUpDown } from 'lucide-react';
import { useEffect, useState } from 'react';
import type { KnownLigand, PredictedLigand, ProteinRanking, QueryResult } from '../../types';
import { ProteinRankingTable } from './ProteinRankingTable';
import { KnownBindingsTable } from './KnownBindingsTable';
import { PredictedLigandsTable } from './PredictedLigandsTable';
import type { SelectedItem } from './SelectedResultPanel';
import { api } from '../../lib/api';

type ResultTab = 'protein_ranking' | 'known_bindings' | 'predicted_ligands';

interface ResultsPanelProps {
  queryResult: QueryResult;
  activeTab: ResultTab;
  onTabChange: (tab: ResultTab) => void;
  selectedItem: SelectedItem | null;
  onSelectItem: (item: SelectedItem | null) => void;
  jobId: string;
}

const TABS: { id: ResultTab; label: string; icon: React.ReactNode; accentClass: string }[] = [
  {
    id: 'protein_ranking',
    label: 'Protein Ranking',
    icon: <Dna className="w-4 h-4" />,
    accentClass: 'border-blue-500 text-blue-500 dark:text-blue-200',
  },
  {
    id: 'known_bindings',
    label: 'Known Bindings',
    icon: <WaypointsIcon className="w-4 h-4" />,
    accentClass: 'border-red-500 text-red-500 dark:text-red-200',
  },
  {
    id: 'predicted_ligands',
    label: 'Predicted Ligands',
    icon: <TrendingUpDown className="w-4 h-4" />,
    accentClass: 'border-purple-500 text-purple-500 dark:text-orange-200',
  },
];

function StatusOverlay({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-6 text-center gap-3">
      {children}
    </div>
  );
}

export function ResultsPanel({ queryResult, activeTab, onTabChange, selectedItem, onSelectItem, jobId }: ResultsPanelProps) {
  const { summary, status, errorMessage, warningMessage, progressPercent } = queryResult;

  const [tabData, setTabData] = useState<{
    proteins: ProteinRanking[];
    knownLigands: KnownLigand[];
    predictedLigands: PredictedLigand[];
    loading: boolean;
  }>({ proteins: [], knownLigands: [], predictedLigands: [], loading: false });

  useEffect(() => {
    const showsData = ['completed', 'completed_with_warnings', 'partial_results'].includes(status);
    if (!jobId || !showsData) {
      setTabData({ proteins: [], knownLigands: [], predictedLigands: [], loading: false });
      return;
    }
    const qid = encodeURIComponent(summary.qseqid);
    setTabData((prev) => ({ ...prev, loading: true }));

    Promise.all([
      api.get(`/jobs/${jobId}/queries/${qid}/protein-ranking?per_page=900000`).catch(() => ({ data: { data: [] } })),
      api.get(`/jobs/${jobId}/queries/${qid}/known-ligands?per_page=900000`).catch(() => ({ data: { data: [] } })),
      api.get(`/jobs/${jobId}/queries/${qid}/predicted-ligands?per_page=900000`).catch(() => ({ data: { data: [] } })),
    ]).then(([pRes, kRes, predRes]) => {
      setTabData({
        proteins: (pRes.data.data as ProteinRanking[]) ?? [],
        knownLigands: (kRes.data.data as KnownLigand[]) ?? [],
        predictedLigands: (predRes.data.data as PredictedLigand[]) ?? [],
        loading: false,
      });
    }).catch(() => {
      setTabData((prev) => ({ ...prev, loading: false }));
    });
  }, [jobId, summary.qseqid, status]);

  const renderContent = () => {
    switch (status) {
      case 'queued':
        return (
          <StatusOverlay>
            <Clock className="w-10 h-10 text-gray-300 dark:text-gray-600" />
            <p className="text-base font-medium text-gray-500 dark:text-gray-400">Queued</p>
            <p className="text-sm text-gray-400 dark:text-gray-500 max-w-sm">
              This query is waiting in the processing queue. Results will appear here automatically
              when processing begins.
            </p>
          </StatusOverlay>
        );

      case 'running':
        return (
          <StatusOverlay>
            <Loader2 className="w-10 h-10 text-blue-400 animate-spin" />
            <p className="text-base font-medium text-blue-600 dark:text-blue-400">Running…</p>
            {progressPercent != null && (
              <div className="w-full max-w-xs">
                <div className="h-2 bg-blue-100 dark:bg-blue-900/40 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 rounded-full transition-all duration-500"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
                <p className="text-xs text-blue-500 mt-1">{progressPercent}% complete</p>
              </div>
            )}
          </StatusOverlay>
        );

      case 'failed':
        return (
          <StatusOverlay>
            <AlertCircle className="w-10 h-10 text-red-400" />
            <p className="text-base font-medium text-red-600 dark:text-red-400">Search failed</p>
            {errorMessage && (
              <div className="max-w-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700 rounded-xl p-4 text-left">
                <p className="text-sm text-red-700 dark:text-red-300 leading-relaxed">{errorMessage}</p>
              </div>
            )}
          </StatusOverlay>
        );

      case 'partial_results':
      case 'completed':
      case 'completed_with_warnings':
        return (
          <>
            {/* Warning banner */}
            {status === 'partial_results' && (
              <div className="mx-5 mt-4 flex items-start gap-2 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 rounded-xl">
                <Layers className="w-4 h-4 text-yellow-600 dark:text-yellow-400 shrink-0 mt-0.5" />
                <p className="text-sm text-yellow-700 dark:text-yellow-300">
                  <span className="font-medium">Partial results available.</span> Some steps are still
                  processing — predicted ligands and additional methods may appear later.
                </p>
              </div>
            )}

            {status === 'completed_with_warnings' && warningMessage && (
              <div className="mx-5 mt-4 flex items-start gap-2 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 rounded-xl">
                <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
                <p className="text-sm text-amber-700 dark:text-amber-300">{warningMessage}</p>
              </div>
            )}

            {/* Tabs */}
            <div className="flex border-b border-gray-100 dark:border-gray-700/60 px-4 mt-0">
              {TABS.map((tab) => {
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => onTabChange(tab.id)}
                    className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors -mb-px cursor-pointer
                      ${isActive
                        ? `${tab.accentClass}`
                        : 'border-transparent text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300'
                      }`}
                  >
                    {tab.icon}
                    {tab.label}
                  </button>
                );
              })}
            </div>

            {/* Tab content */}
            <div className="p-5 relative">
              {tabData.loading && (
                <div className="absolute inset-0 flex items-center justify-center bg-white/60 dark:bg-[#1a2330]/60 z-10 rounded-b-2xl">
                  <Loader2 className="w-8 h-8 text-teal-500 animate-spin" />
                </div>
              )}
              {activeTab === 'protein_ranking' && <ProteinRankingTable data={tabData.proteins} />}
              {activeTab === 'known_bindings' && (
                <KnownBindingsTable
                  data={tabData.knownLigands}
                  selectedItem={selectedItem}
                  onSelectItem={onSelectItem}
                />
              )}
              {activeTab === 'predicted_ligands' && (
                <PredictedLigandsTable
                  data={tabData.predictedLigands}
                  selectedItem={selectedItem}
                  onSelectItem={onSelectItem}
                />
              )}
            </div>
          </>
        );
    }
  };

  return (
    <div className="bg-white dark:bg-[#1a2330] rounded-2xl border border-gray-200 dark:border-gray-700/60 shadow-sm mt-6 overflow-visible">
      <div className="flex">
        {/* Main column */}
        <div className="flex-1 min-w-0">
          {/* Header */}
          <div className="px-5 py-3 border-b border-gray-100 dark:border-gray-700/60 flex items-center gap-2">
            <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-200">Search Results</h2>
            <span className="font-mono text-xs text-gray-400 dark:text-gray-400">— {summary.qseqid}</span>
            {jobId && (
              <a
                href={`/api/jobs/${jobId}/download`}
                download
                className="ml-auto px-2 flex items-center gap-2 text-sm text-gray-400 dark:text-gray-200 hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                Download
              </a>
            )}
          </div>

          {renderContent()}
        </div>
      </div>
    </div>
  );
}