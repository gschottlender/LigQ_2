import { useState, useMemo } from 'react';
import { ChevronDown, Info, RefreshCw, Search } from 'lucide-react';
import type { ProteinRanking, SearchType } from '../../types';
import { SearchTypeBadge } from '../../components/Badge';

interface ProteinRankingTableProps {
  data: ProteinRanking[];
}

const ALL_COLUMNS = [
  { id: 'protein_rank', label: 'Rank' },
  { id: 'sseqid', label: 'UniProt ID' },
  { id: 'search_type', label: 'Search type' },
  { id: 'ranking_source', label: 'Source' },
  { id: 'blast_pident', label: '% Identity' },
  { id: 'blast_qcov', label: 'Q. Coverage' },
  { id: 'blast_scov', label: 'S. Coverage' },
  { id: 'blast_evalue', label: 'E-value' },
  { id: 'blast_bitscore', label: 'Bitscore' },
  { id: 'best_domain_score', label: 'Domain score' },
  { id: 'best_domain_evalue', label: 'Domain E-value' },
  { id: 'n_shared_domains', label: 'Shared domains' },
] as const;

type ColId = (typeof ALL_COLUMNS)[number]['id'];

const DEFAULT_VISIBLE: Set<ColId> = new Set([
  'protein_rank', 'sseqid', 'search_type', 'ranking_source',
  'blast_pident', 'blast_qcov', 'blast_scov', 'blast_evalue', 'blast_bitscore',
  'best_domain_score', 'best_domain_evalue', 'n_shared_domains',
]);

const PAGE_SIZES = [10, 20, 50];

function fmt(v: number | null, type: 'pct' | 'evalue' | 'int' | 'float'): string {
  if (v == null || isNaN(v)) return '—';
  if (type === 'pct') {
    return `${v.toFixed(2)}%`;
  }
  if (type === 'evalue') {
    if (v === 0) return '0.0';
    return v.toExponential(2).replace('+', '').replace('e-0', 'e-').replace('e0', 'e');
  }
  if (type === 'int') return Math.round(v).toString();
  return v.toFixed(2);
}

export function ProteinRankingTable({ data }: ProteinRankingTableProps) {
  const [filterText, setFilterText] = useState('');
  const [searchTypeFilter, setSearchTypeFilter] = useState<SearchType | 'all'>('all');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [visibleCols, setVisibleCols] = useState<Set<ColId>>(new Set(DEFAULT_VISIBLE));
  const [showColPicker, setShowColPicker] = useState(false);

  const filtered = useMemo(() => {
    const lower = filterText.toLowerCase();
    return data.filter((row) => {
      const matchesType = searchTypeFilter === 'all' || row.search_type === searchTypeFilter;
      const matchesText =
        lower === '' ||
        row.sseqid.toLowerCase().includes(lower) ||
        row.search_type.toLowerCase().includes(lower) ||
        row.ranking_source.toLowerCase().includes(lower);
      return matchesType && matchesText;
    });
  }, [data, filterText, searchTypeFilter]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const safePage = Math.min(page, totalPages);
  const paginated = filtered.slice((safePage - 1) * pageSize, safePage * pageSize);

  const toggleCol = (id: ColId) => {
    setVisibleCols((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const visibleColumns = ALL_COLUMNS.filter((c) => visibleCols.has(c.id));

  return (
    <div className="flex flex-col gap-3">
      {/* Controls */}
      <div className="flex items-center gap-2 flex-wrap">
        <div className="relative flex-1 min-w-40">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400" />
          <input
            type="text"
            value={filterText}
            onChange={(e) => { setFilterText(e.target.value); setPage(1); }}
            placeholder="Filter table…"
            className="w-full pl-8 pr-3 py-2 border border-gray-200 dark:border-gray-600 rounded-lg text-sm
              text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 focus:outline-none focus:ring-1 focus:ring-teal-500 placeholder:text-gray-400"
          />
        </div>

        <div className="relative">
          <select
            value={searchTypeFilter}
            onChange={(e) => { setSearchTypeFilter(e.target.value as SearchType | 'all'); setPage(1); }}
            className="appearance-none border border-gray-200 dark:border-gray-600 rounded-lg px-3 py-2 pr-7 text-sm
              text-gray-600 dark:text-gray-300 bg-white dark:bg-gray-800 cursor-pointer focus:outline-none focus:ring-1 focus:ring-teal-500 hover:text-teal-600 hover:border-teal-400"
          >
            <option value="all">Search type: All</option>
            <option value="sequence">Sequence</option>
            <option value="nearest_k">Nearest K</option>
            <option value="domain">Domain</option>
          </select>
          <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400 pointer-events-none" />
        </div>

        <button
          onClick={() => { setFilterText(''); setSearchTypeFilter('all'); setPage(1); }}
          className="cursor-pointer p-2 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-500 dark:text-gray-400 hover:text-teal-600 hover:border-teal-400 transition-colors"
          title="Reset filters"
        >
          <RefreshCw className="w-4 h-4" />
        </button>

        <div className="relative">
          <button onClick={() => setShowColPicker((v) => !v)}
            className="flex items-center gap-1.5 px-3 py-2 border border-gray-200 dark:border-gray-600 rounded-lg text-sm text-gray-600 dark:text-gray-300 hover:border-teal-400 transition-colors cursor-pointer">
            Columns <ChevronDown className="w-3.5 h-3.5" />
          </button>
          {showColPicker && (
            <div className="absolute right-0 top-full mt-1 z-20 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-xl shadow-lg p-3 min-w-44">
              {ALL_COLUMNS.map((col) => (
                <label key={col.id} className="flex items-center gap-2 py-1 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={visibleCols.has(col.id)}
                    onChange={() => toggleCol(col.id)}
                    className="accent-teal-600"
                  />
                  <span className="text-sm text-gray-700 dark:text-gray-200">{col.label}</span>
                </label>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700/60">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 dark:bg-gray-800/60 border-b border-gray-200 dark:border-gray-700/60">
              {visibleColumns.map((col) => (
                <th key={col.id} className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-300 whitespace-nowrap">
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paginated.length === 0 ? (
              <tr>
                <td colSpan={visibleColumns.length} className="px-4 py-8 text-center text-sm text-gray-400 dark:text-gray-200">
                  No results match your filters.
                </td>
              </tr>
            ) : (
              paginated.map((row) => {
                const rowBg =
                  row.ranking_source === 'blast'
                    ? 'bg-gray-50 dark:bg-gray-800'
                    : 'bg-gray-50 dark:bg-gray-800';
                return (
                  <tr
                    key={`${row.protein_rank}-${row.sseqid}`}
                    className={`border-b border-gray-100 dark:border-gray-700/40 ${rowBg}`}
                  >
                    {visibleCols.has('protein_rank') && (
                      <td className="px-4 py-2.5 font-jetbrains-mono text-xs text-gray-500 dark:text-gray-400">
                        {row.protein_rank}
                      </td>
                    )}
                    {visibleCols.has('sseqid') && (
                      <td className="px-4 py-2.5 font-jetbrains-mono text-xs font-medium text-cyan-800 dark:text-teal-300">
                        {row.sseqid}
                      </td>
                    )}
                    {visibleCols.has('search_type') && (
                      <td className="px-4 py-2.5">
                        <SearchTypeBadge type={row.search_type} />
                      </td>
                    )}
                    {visibleCols.has('ranking_source') && (
                      <td className="px-4 py-2.5 text-xs text-gray-600 dark:text-gray-200">
                        {row.ranking_source}
                      </td>
                    )}
                    {visibleCols.has('blast_pident') && (
                      <td className="px-4 py-2.5 text-xs text-gray-700 dark:text-gray-200">
                        {fmt(row.blast_pident, 'pct')}
                      </td>
                    )}
                    {visibleCols.has('blast_qcov') && (
                      <td className="px-4 py-2.5 text-xs text-gray-700 dark:text-gray-200">
                        {fmt(row.blast_qcov, 'pct')}
                      </td>
                    )}
                    {visibleCols.has('blast_scov') && (
                      <td className="px-4 py-2.5 text-xs text-gray-700 dark:text-gray-200">
                        {fmt(row.blast_scov, 'pct')}
                      </td>
                    )}
                    {visibleCols.has('blast_evalue') && (
                      <td className="px-4 py-2.5 font-jetbrains-mono text-xs text-gray-500 dark:text-gray-200">
                        {fmt(row.blast_evalue, 'evalue')}
                      </td>
                    )}
                    {visibleCols.has('blast_bitscore') && (
                      <td className="px-4 py-2.5 text-xs text-gray-700 dark:text-gray-200">
                        {fmt(row.blast_bitscore, 'int')}
                      </td>
                    )}
                    {visibleCols.has('best_domain_score') && (
                      <td className="px-4 py-2.5 font-jetbrains-mono text-xs text-gray-700 dark:text-gray-200">
                        {fmt(row.best_domain_score, 'float')}
                      </td>
                    )}
                    {visibleCols.has('best_domain_evalue') && (
                      <td className="px-4 py-2.5 font-jetbrains-mono text-xs text-gray-500 dark:text-gray-200">
                        {fmt(row.best_domain_evalue, 'evalue')}
                      </td>
                    )}
                    {visibleCols.has('n_shared_domains') && (
                      <td className="px-4 py-2.5 text-xs text-gray-700 dark:text-gray-200">
                        {row.n_shared_domains}
                      </td>
                    )}
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
        <span>
          Showing {filtered.length === 0 ? 0 : (safePage - 1) * pageSize + 1}–
          {Math.min(safePage * pageSize, filtered.length)} of {filtered.length} results
        </span>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            {Array.from({ length: totalPages }, (_, i) => i + 1)
              .filter((p) => p === 1 || p === totalPages || Math.abs(p - safePage) <= 1)
              .reduce<(number | '...')[]>((acc, p, idx, arr) => {
                if (idx > 0 && typeof arr[idx - 1] === 'number' && (p as number) - (arr[idx - 1] as number) > 1) {
                  acc.push('...');
                }
                acc.push(p);
                return acc;
              }, [])
              .map((p, i) =>
                p === '...' ? (
                  <span key={`dots-${i}`} className="px-1">…</span>
                ) : (
                  <button
                    key={p}
                    onClick={() => setPage(p as number)}
                    className={`w-7 h-7 rounded-lg text-xs font-medium transition-colors
                      ${safePage === p
                        ? 'bg-[#0d5c6b] text-white'
                        : 'text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700'
                      }`}
                  >
                    {p}
                  </button>
                ),
              )}
          </div>
          <div className="relative">
            <select
              value={pageSize}
              onChange={(e) => { setPageSize(parseInt(e.target.value)); setPage(1); }}
              className="appearance-none border border-gray-200 dark:border-gray-600 rounded-lg px-2 py-1 pr-6 text-xs
                text-gray-600 dark:text-gray-200 bg-white dark:bg-gray-800 cursor-pointer focus:outline-none"
            >
              {PAGE_SIZES.map((s) => <option key={s} value={s}>{s} / page</option>)}
            </select>
            <ChevronDown className="absolute right-1.5 top-1/2 -translate-y-1/2 w-3 h-3 text-gray-400 pointer-events-none" />
          </div>
        </div>
      </div>

      <div className="flex items-start gap-2 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-700 rounded-xl text-xs text-blue-700 dark:text-blue-300">
        <Info className="w-3.5 h-3.5 shrink-0 mt-0.5" />
        <span>
          Candidate proteins ranked by BLAST evidence. Domain-only candidates are ranked by Pfam/HMMER score.
          <span className="text-blue-400 dark:text-blue-200"> This table is informational and does not affect ligand retrieval.</span>
        </span>
      </div>
    </div>
  );
}