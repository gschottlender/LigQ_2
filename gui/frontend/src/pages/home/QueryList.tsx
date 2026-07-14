import type { JobStatus, QueryResult } from '../../types';
import { JobStatusBadge } from '../../components/JobStatusBadge';

type ResultTab = 'protein_ranking' | 'known_bindings' | 'predicted_ligands';

interface QueryListProps {
  results: QueryResult[];
  selectedQueryId: string | null;
  onSelectQuery: (qseqid: string, tab?: ResultTab) => void;
}

/* ─── Action button */

interface ActionButtonProps {
  count: number;
  label: string;
  pillClass: string;
  onClick: (e: React.MouseEvent) => void;
}

function ActionButton({ count, label, pillClass, onClick }: ActionButtonProps) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 px-3 py-1.5 border border-gray-200 dark:border-gray-600 rounded-lg
        text-xs font-medium text-gray-600 dark:text-gray-300 transition-colors"
    >
      <span className={`px-2 py-0.5 rounded-md text-xs font-dm-sans font-bold ${pillClass}`}>
        {count.toLocaleString()}
      </span>
      {label}
    </button>
  );
}

/* ─── Main component ─────────────────────────────────────────── */

const SHOW_ACTIONS: JobStatus[] = ['completed', 'completed_with_warnings', 'partial_results'];

const TH = "px-5 py-2.5 text-left text-xs font-dm-sans font-semibold uppercase tracking-widest text-gray-600 dark:text-gray-400 border-b border-gray-100 dark:border-gray-600/50"

export function QueryList({ results, selectedQueryId, onSelectQuery }: QueryListProps) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm overflow-hidden mt-6">
      <div className="flex items-center justify-between gap-3 px-5 py-3 border-b border-gray-100 dark:border-gray-600">
        <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-200">Queries</h2>
        <span className="text-xs text-gray-400 dark:text-gray-500">
          {results.length.toLocaleString()} total
        </span>
      </div>

      <div
        className="max-h-[26rem] overflow-auto sm:max-h-[30rem]"
        role="region"
        aria-label="Query statuses"
        tabIndex={0}
      >
      <table className="w-full min-w-170 text-sm">
        <thead className="sticky top-0 z-10">
          <tr className="bg-gray-50 dark:bg-gray-800 shadow-[0_1px_0_0_rgba(229,231,235,1)] dark:shadow-[0_1px_0_0_rgba(75,85,99,1)]">
            <th className={TH}>
              Query
            </th>
            <th className={TH}>
              Status
            </th>
            <th className={TH}>
              Actions
            </th>
          </tr>
        </thead>

        <tbody>
          {results.map(({ summary, status, progressPercent, errorMessage }) => {
            const nProteins =
              summary.n_proteins_sequence +
              summary.n_proteins_nearest_k +
              summary.n_proteins_domain;
            const nKnown =
              summary.n_known_ligands_sequence +
              summary.n_known_ligands_nearest_k +
              summary.n_known_ligands_domain;
            const nPredicted =
              summary.n_predicted_ligands_sequence +
              summary.n_predicted_ligands_nearest_k +
              summary.n_predicted_ligands_domain;

            const isSelected = selectedQueryId === summary.qseqid;
            const showActions = SHOW_ACTIONS.includes(status);

            return (
              <tr
                key={summary.qseqid}
                onClick={() => onSelectQuery(summary.qseqid)}
                className={`border-b border-gray-50 dark:border-gray-800/50 last:border-b-0 cursor-pointer transition-colors
                  ${isSelected
                    ? 'bg-teal-50/60 dark:bg-teal-700/20'
                    : 'hover:bg-gray-50/70 dark:hover:bg-gray-600/20'
                  }`}
              >
                {/* Query ID */}
                <td className="px-5 py-3 w-1/3">
                  <span className="font-dm-sans font-medium text-xs text-gray-700 dark:text-gray-200">
                    {summary.qseqid}
                  </span>
                </td>

                {/* Status */}
                <td className="px-5 py-3">
                  <JobStatusBadge status={status} progressPercent={progressPercent} errorMessage={errorMessage}/>
                </td>

                {/* Actions */}
                <td className="px-5 py-3">
                  {showActions ? (
                    <div className="flex items-center gap-2 justify-start flex-wrap">
                      <ActionButton
                        count={nProteins}
                        label="Protein Ranking"
                        pillClass="bg-blue-100 text-blue-700 dark:bg-blue-800 dark:text-blue-200"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectQuery(summary.qseqid, 'protein_ranking');
                        }}
                      />
                      <ActionButton
                        count={nKnown}
                        label="Known Bindings"
                        pillClass="bg-red-100 text-red-700 dark:bg-red-800 dark:text-red-200"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectQuery(summary.qseqid, 'known_bindings');
                        }}
                      />
                      <ActionButton
                        count={nPredicted}
                        label="Predicted Ligands"
                        pillClass="bg-purple-100 text-purple-700 dark:bg-purple-800 dark:text-purple-200"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectQuery(summary.qseqid, 'predicted_ligands');
                        }}
                      />
                    </div>
                  ) : (
                    <span className="text-xs text-gray-300 dark:text-gray-600 block text-right">—</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
    </div>
  );
}
