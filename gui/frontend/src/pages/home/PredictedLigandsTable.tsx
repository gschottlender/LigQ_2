import { useState, useMemo } from 'react';
import { ChevronDown, Copy, RefreshCw, Search } from 'lucide-react';
import type { PredictedLigand, SearchType } from '../../types';
import { SearchTypeBadge } from '../../components/Badge';
import type { SelectedItem } from './SelectedResultPanel';

interface PredictedLigandsTableProps {
  data: PredictedLigand[];
  selectedItem: SelectedItem | null;
  onSelectItem: (item: SelectedItem | null) => void;
}

type ColId = 'search_type' | 'uniprot_id' | 'chem_comp_id' | 'query_id' 
           | 'tanimoto' | 'similarity' | 'bsi_score' | 'smiles' | 'qseqid' | 'sseqid';

const PAGE_SIZES = [10, 20, 50];

function getScoreColumn(data: PredictedLigand[]) {
  if (data.some(r => r.tanimoto != null)) return 'tanimoto';
  if (data.some(r => r.similarity != null)) return 'similarity';
  if (data.some(r => r.bsi_score != null)) return 'bsi_score';
  return null;
}

function TruncatedSmiles({ smiles }: { smiles: string }) {
  const [copied, setCopied] = useState(false);
  const truncated = smiles.length > 35 ? smiles.slice(0, 35) + '…' : smiles;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(smiles);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <div className="flex items-center gap-1">
      <div className="relative group inline-block max-w-xs">
        <span className="font-jetbrains-mono text-xs cursor-default">{truncated}</span>
        {smiles.length > 35 && (
          <div className="absolute left-0 bottom-full mb-1.5 z-30 px-2.5 py-1.5 bg-gray-800 dark:bg-gray-950 text-white text-xs
            rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-normal max-w-72 leading-snug shadow-lg">
            {smiles}
          </div>
        )}
      </div>
      <button onClick={handleCopy} title="Copy SMILES"
        className="ml-1 p-0.5 rounded text-gray-300 hover:text-teal-500 transition-colors inline-flex shrink-0">
        {copied
          ? <span className="text-xs text-teal-500 font-medium">✓</span>
          : <Copy className="w-3 h-3" />}
      </button>
    </div>
  );
}

function TanimotoBar({ value }: { value: number }) {
  const isExact = value === 1.0;
  const barColor = isExact ? 'bg-teal-500' : value >= 0.7 ? 'bg-teal-300' : 'bg-orange-400';

  return (
    <div className="flex items-center gap-2 min-w-28">
      <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full ${barColor} rounded-full transition-all`} style={{ width: `${value * 100}%` }} />
      </div>
      <span className={`text-xs font-jetbrains-mono w-10 text-right ${isExact ? 'text-teal-600 dark:text-teal-400 font-semibold' : 'text-gray-600 dark:text-gray-300'}`}>
        {value.toFixed(2)}
      </span>
    </div>
  );
}

export function PredictedLigandsTable({ data, selectedItem, onSelectItem }: PredictedLigandsTableProps) {
  const scoreCol = getScoreColumn(data);

  const ALL_COLUMNS = [
    { id: 'search_type' as ColId, label: 'Search type' },
    { id: 'uniprot_id' as ColId, label: 'UniProt ID' },
    { id: 'chem_comp_id' as ColId, label: 'Compound ID' },
    { id: 'query_id' as ColId, label: 'Seed ligand' },
    ...(scoreCol ? [{ id: scoreCol as ColId, label: scoreCol === 'tanimoto' ? 'Tanimoto' : scoreCol === 'similarity' ? 'Similarity' : 'BSI Score' }] : []),
    { id: 'smiles' as ColId, label: 'SMILES' },
    { id: 'qseqid' as ColId, label: 'Query (qseqid)' },
    { id: 'sseqid' as ColId, label: 'Subject (sseqid)' },
  ];
  
  const [visibleCols, setVisibleCols] = useState<Set<ColId>>(() => {
  const base = new Set<ColId>(['search_type', 'uniprot_id', 'chem_comp_id', 'query_id', 'smiles']);
    if (scoreCol) base.add(scoreCol);
    return base;
  });

  const [filterText, setFilterText] = useState('');
  const [searchTypeFilter, setSearchTypeFilter] = useState<SearchType | 'all'>('all');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [showColPicker, setShowColPicker] = useState(false);
  const selectedRow = selectedItem?.kind === 'predicted' ? selectedItem.item : null;

  const deduped = useMemo(() => {
    const seen = new Map<string, PredictedLigand>();
    const priority: Record<SearchType, number> = { sequence: 0, nearest_k: 1, domain: 2 };
    for (const row of data) {
      const existing = seen.get(row.chem_comp_id);
      if (!existing || priority[row.search_type] < priority[existing.search_type]) {
        seen.set(row.chem_comp_id, row);
      }
    }
    return Array.from(seen.values());
  }, [data]);

  const filtered = useMemo(() => {
    const lower = filterText.toLowerCase();
    return deduped.filter((row) => {
      const matchesType = searchTypeFilter === 'all' || row.search_type === searchTypeFilter;
      const matchesText =
        lower === '' ||
        row.chem_comp_id.toLowerCase().includes(lower) ||
        row.uniprot_id.toLowerCase().includes(lower) ||
        row.query_id.toLowerCase().includes(lower) ||
        row.smiles.toLowerCase().includes(lower);
      return matchesType && matchesText;
    });
  }, [deduped, filterText, searchTypeFilter]);

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
    <div className="flex gap-4 min-h-0">
      <div className="flex-1 min-w-0 flex flex-col gap-3">
        {/* Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          <div className="relative flex-1 min-w-36">
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
                text-gray-600 dark:text-gray-300 bg-white dark:bg-gray-800 cursor-pointer focus:outline-none focus:ring-1 focus:ring-teal-500"
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
            className="p-2 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-500 dark:text-gray-400 hover:text-teal-600 hover:border-teal-400 transition-colors"
            title="Reset filters"
          >
            <RefreshCw className="w-4 h-4" />
          </button>

          <div className="relative">
            <button
              onClick={() => setShowColPicker((v) => !v)}
              className="flex items-center gap-1.5 px-3 py-2 border border-gray-200 dark:border-gray-600 rounded-lg text-sm
                text-gray-600 dark:text-gray-300 hover:border-teal-400 transition-colors"
            >
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
                  <th
                    key={col.id}
                    className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500 whitespace-nowrap"
                  >
                    {col.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {paginated.length === 0 ? (
                <tr>
                  <td colSpan={visibleColumns.length} className="px-4 py-8 text-center text-sm text-gray-400">
                    No results match your filters.
                  </td>
                </tr>
              ) : (
                paginated.map((row, i) => (
                  <tr
                    key={`${row.chem_comp_id}-${i}`}
                    onClick={() => {
                      const isSelected = selectedRow?.chem_comp_id === row.chem_comp_id;
                      onSelectItem(isSelected ? null : { kind: 'predicted', item: row });
                    }}
                    className={`border-b border-gray-100 dark:border-gray-700/40 cursor-pointer transition-colors
                      ${selectedRow?.chem_comp_id === row.chem_comp_id
                        ? 'bg-gray-50 dark:bg-gray-700/40'
                        : 'hover:bg-gray-50 dark:hover:bg-gray-800'
                      }`}
                  >
                    {visibleCols.has('search_type') && (
                      <td className="px-4 py-2.5"><SearchTypeBadge type={row.search_type} /></td>
                    )}
                    {visibleCols.has('uniprot_id') && (
                      <td className="px-4 py-2.5 font-jetbrains-mono text-xs font-medium text-[#0d5c6b] dark:text-teal-300">
                        {row.uniprot_id}
                      </td>
                    )}
                    {visibleCols.has('chem_comp_id') && (
                      <td className="px-4 py-2.5 font-jetbrains-mono text-xs font-semibold text-gray-800 dark:text-gray-100">
                        {row.chem_comp_id}
                      </td>
                    )}
                    {visibleCols.has('query_id') && (
                      <td className="px-4 py-2.5 font-jetbrains-mono text-xs text-gray-600 dark:text-gray-200">
                        {row.query_id}
                      </td>
                    )}
                    {scoreCol && visibleCols.has(scoreCol) && (
                      <td className="px-4 py-2.5">
                        <TanimotoBar value={(row as any)[scoreCol] ?? 0} />
                      </td>
                    )}
                    {visibleCols.has('smiles') && (
                      <td className="px-4 py-2.5 text-gray-700 dark:text-gray-200"><TruncatedSmiles smiles={row.smiles} /></td>
                    )}
                    {visibleCols.has('qseqid') && (
                      <td className="px-4 py-2.5 font-jetbrains-mono text-xs text-gray-500 dark:text-gray-200 max-w-40 truncate">
                        {row.qseqid}
                      </td>
                    )}
                    {visibleCols.has('sseqid') && (
                      <td className="px-4 py-2.5 font-jetbrains-mono text-xs text-gray-500 dark:text-gray-200">
                        {row.sseqid}
                      </td>
                    )}
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
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
                          ? 'bg-[#e07b39] text-white'
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
                  text-gray-600 dark:text-gray-300 bg-white dark:bg-gray-800 cursor-pointer focus:outline-none"
              >
                {PAGE_SIZES.map((s) => <option key={s} value={s}>{s} / page</option>)}
              </select>
              <ChevronDown className="absolute right-1.5 top-1/2 -translate-y-1/2 w-3 h-3 text-gray-400 pointer-events-none" />
            </div>
          </div>
        </div>
      </div>

      {/* Selected result panel */}
      {/* {selectedRow && (
        <SelectedResultPanel
          selected={{ kind: 'predicted', item: selectedRow }}
          onClose={() => setSelectedRow(null)}
        />
      )} */}
    </div>
  );
}