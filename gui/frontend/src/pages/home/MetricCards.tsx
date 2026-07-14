import { Dna, FileText, TrendingUpDown, WaypointsIcon } from 'lucide-react';
import type { QueryResult, SearchResultsSummary } from '../../types';
import { Tooltip } from '../../components/Tooltip';

interface MetricCardsProps {
  summaries: SearchResultsSummary[];
  results: QueryResult[];
}

const SOURCE_LABELS: Record<string, string> = { pdb: 'PDB', chembl: 'ChEMBL' };

interface CardProps {
  title: string;
  value: number;
  icon: React.ReactNode;
  iconBgClass: string;
  tooltipContent?: string;
  subtitle?: React.ReactNode;
}

function Card({ title, value, icon, iconBgClass, tooltipContent, subtitle }: CardProps) {
  return (
    <div className="flex-1 min-w-0 bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-500 p-5 shadow-sm">
      <div className="flex gap-4">
        <div className={`w-12 h-12 flex items-center justify-center rounded-xl shrink-0 ${iconBgClass}`}>
          {icon}
        </div>
        <div className="min-w-0">
          <p className="text-sm font-dm-sans text-gray-500 dark:text-gray-400 mb-1">{title}</p>
          <div className="flex items-baseline gap-2">
            {tooltipContent ? (
              <Tooltip content={tooltipContent} position="bottom">
                <span className={`text-2xl font-bold font-dm-sans text-gray-600 dark:text-gray-200 cursor-default`}>
                  {value.toLocaleString()}
                </span>
              </Tooltip>
            ) : (
              <span className={`text-2xl font-bold text-gray-600 dark:text-gray-200 font-dm-sans`}>
                {value.toLocaleString()}
              </span>
            )}
          </div>
          {subtitle && <div className="mt-1 text-xs font-dm-sans text-gray-500 dark:text-gray-400">{subtitle}</div>}
        </div>
      </div>
    </div>
  );
}

export function MetricCards({ summaries, results }: MetricCardsProps) {
  const totalProteinsSeq = summaries.reduce((a, s) => a + s.n_proteins_sequence, 0);
  const totalProteinsNK = summaries.reduce((a, s) => a + s.n_proteins_nearest_k, 0);
  const totalProteinsDomain = summaries.reduce((a, s) => a + s.n_proteins_domain, 0);
  const totalProteins = totalProteinsSeq + totalProteinsNK + totalProteinsDomain;

  const totalKnownSeq = summaries.reduce((a, s) => a + s.n_known_ligands_sequence, 0);
  const totalKnownNK = summaries.reduce((a, s) => a + s.n_known_ligands_nearest_k, 0);
  const totalKnownDomain = summaries.reduce((a, s) => a + s.n_known_ligands_domain, 0);
  const totalKnown = totalKnownSeq + totalKnownNK + totalKnownDomain;

  const totalPredSeq = summaries.reduce((a, s) => a + s.n_predicted_ligands_sequence, 0);
  const totalPredNK = summaries.reduce((a, s) => a + s.n_predicted_ligands_nearest_k, 0);
  const totalPredDomain = summaries.reduce((a, s) => a + s.n_predicted_ligands_domain, 0);
  const totalPred = totalPredSeq + totalPredNK + totalPredDomain;

  const proteinTooltip = `Sequence: ${totalProteinsSeq.toLocaleString()} | Nearest K: ${totalProteinsNK.toLocaleString()} | Domain: ${totalProteinsDomain.toLocaleString()}`;
  const knownTooltip = `Sequence: ${totalKnownSeq.toLocaleString()} | Nearest K: ${totalKnownNK.toLocaleString()} | Domain: ${totalKnownDomain.toLocaleString()}`;
  const predTooltip = `Sequence: ${totalPredSeq.toLocaleString()} | Nearest K: ${totalPredNK.toLocaleString()} | Domain: ${totalPredDomain.toLocaleString()}`;

  const distinctUniprots = Array.from(new Set(summaries.map((s) => s.qseqid)));

  const completedCount = results.filter(
    (r) => r.status === 'completed' || r.status === 'completed_with_warnings',
  ).length;
  const runningCount = results.filter((r) => r.status === 'running').length;
  const queuedCount = results.filter((r) => r.status === 'queued').length;

  const loadedSources = [
    ...new Set(results.flatMap((r) => r.knownLigands ?? []).map((l) => l.source)),
  ].sort();
  const sourcesLabel =
    loadedSources.length > 0
      ? `from ${loadedSources.map((s) => SOURCE_LABELS[s] ?? s.toUpperCase()).join(' & ')}`
      : 'from PDB & ChEMBL';

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
      <Card
        title="Queries"
        value={summaries.length}
        icon={<FileText className="w-5 h-5 text-emerald-500 dark:text-emerald-200" />}
        iconBgClass="bg-emerald-50 dark:bg-emerald-800"
        subtitle={`${completedCount} finished · ${runningCount} running · ${queuedCount} queued`}
      />
      <Card
        title="Protein ranking"
        value={totalProteins}
        icon={<Dna className="w-5 h-5 text-blue-600 dark:text-blue-200" />}
        iconBgClass="bg-blue-50 dark:bg-blue-800"
        tooltipContent={proteinTooltip}
        subtitle={
          distinctUniprots.length > 0 ? (
            <p className="text-xs text-gray-500 dark:text-gray-500 truncate">
              {distinctUniprots.slice(0, 3).join(' · ')}{distinctUniprots.length > 3 ? ` +${distinctUniprots.length - 3}` : ''}
            </p>
          ) : undefined
        }
      />
      <Card
        title="Known bindings"
        value={totalKnown}
        icon={<WaypointsIcon className="w-5 h-5 text-red-600 dark:text-red-200" />}
        iconBgClass="bg-red-50 dark:bg-red-800"
        tooltipContent={knownTooltip}
        subtitle={sourcesLabel}
      />
      <Card
        title="Predicted ligands"
        value={totalPred}
        icon={<TrendingUpDown className="w-5 h-5 text-purple-600 dark:text-purple-200" />}
        iconBgClass="bg-purple-50 dark:bg-purple-800"
        tooltipContent={predTooltip}
        subtitle={`across ${completedCount} finished ${completedCount === 1 ? 'query' : 'queries'}`}
      />
    </div>
  );
}
