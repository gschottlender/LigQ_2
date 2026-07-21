import { useState, useMemo } from 'react';
import { ChevronDown, Copy, RefreshCw, Search } from 'lucide-react';
import type { KnownLigand, SearchType } from '../../types';
import { SearchTypeBadge, SourceBadge } from '../../components/Badge';
import type { SelectedItem } from './SelectedResultPanel';

interface KnownBindingsTableProps {
  data: KnownLigand[];
  selectedItem: SelectedItem | null;
  onSelectItem: (item: SelectedItem | null) => void;
}

const ALL_COLUMNS = [
  { id: 'search_type', label: 'Search type' },
  { id: 'uniprot_id', label: 'UniProt ID' },
  { id: 'chem_comp_id', label: 'Compound ID' },
  { id: 'source', label: 'Source' },
  { id: 'smiles', label: 'SMILES' },
  { id: 'pchembl', label: 'pChEMBL' },
  { id: 'mechanism', label: 'Mechanism' },
  { id: 'activity_comment', label: 'Activity' },
  { id: 'binding_sites', label: 'Binding sites' },
  { id: 'pdb_ids', label: 'PDB IDs' },
  { id: 'curation_method', label: 'Curation method' },
] as const;

type ColId = (typeof ALL_COLUMNS)[number]['id'];

const DEFAULT_VISIBLE: Set<ColId> = new Set([
  'search_type', 'uniprot_id', 'chem_comp_id', 'source', 'smiles', 'pchembl', 'binding_sites',
]);

const PAGE_SIZES = [10, 20, 50];

function TruncatedCell({ text, maxLen = 40 }: { text: string; maxLen?: number }) {
  const truncated = text.length > maxLen;
  const display = truncated ? text.slice(0, maxLen) + '…' : text;

  if (!truncated) return <span className="font-jetbrains-mono text-xs">{text}</span>;

  return (
    <div className="relative group inline-block max-w-xs">
      <span className="font-jetbrains-mono text-xs cursor-default">{display}</span>
      <div className="absolute left-0 bottom-full mb-1.5 z-30 px-2.5 py-1.5 bg-gray-800 dark:bg-gray-950 text-white text-xs
        rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-normal max-w-72 leading-snug shadow-lg">
        {text}
      </div>
    </div>
  );
}

function SmallCopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };
  return (
    <button onClick={handleCopy} title="Copy SMILES"
      className="ml-1 p-0.5 rounded text-gray-300 hover:text-teal-500 transition-colors inline-flex shrink-0">
      {copied
        ? <span className="text-xs text-teal-500 font-medium">✓</span>
        : <Copy className="w-3 h-3" />}
    </button>
  );
}

export function KnownBindingsTable({ data, selectedItem, onSelectItem }: KnownBindingsTableProps) {
  const [filterText, setFilterText] = useState('');
  const [searchTypeFilter, setSearchTypeFilter] = useState<SearchType | 'all'>('all');
  const [sourceFilter, setSourceFilter] = useState<'all' | 'pdb' | 'chembl'>('all');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [visibleCols, setVisibleCols] = useState<Set<ColId>>(new Set(DEFAULT_VISIBLE));
  const [showColPicker, setShowColPicker] = useState(false);
  const selectedRow = selectedItem?.kind === 'known' ? selectedItem.item : null;

  const filtered = useMemo(() => {
    const lower = filterText.toLowerCase();
    return data.filter((row) => {
      const matchesType = searchTypeFilter === 'all' || row.search_type === searchTypeFilter;
      const matchesSource = sourceFilter === 'all' || row.source === sourceFilter;
      const matchesText =
        lower === '' ||
        row.uniprot_id.toLowerCase().includes(lower) ||
        row.chem_comp_id.toLowerCase().includes(lower) ||
        row.smiles.toLowerCase().includes(lower) ||
        (row.mechanism ?? '').toLowerCase().includes(lower);
      return matchesType && matchesSource && matchesText;
    });
  }, [data, filterText, searchTypeFilter, sourceFilter]);

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

          <div className="relative">
            <select
              value={sourceFilter}
              onChange={(e) => { setSourceFilter(e.target.value as 'all' | 'pdb' | 'chembl'); setPage(1); }}
              className="appearance-none border border-gray-200 dark:border-gray-600 rounded-lg px-3 py-2 pr-7 text-sm
                text-gray-600 dark:text-gray-300 bg-white dark:bg-gray-800 cursor-pointer focus:outline-none focus:ring-1 focus:ring-teal-500"
            >
              <option value="all">Database: All</option>
              <option value="pdb">PDB</option>
              <option value="chembl">ChEMBL</option>
            </select>
            <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400 pointer-events-none" />
          </div>

          <button
            onClick={() => { setFilterText(''); setSearchTypeFilter('all'); setSourceFilter('all'); setPage(1); }}
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
                    className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-300 whitespace-nowrap"
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
                    key={`${row.uniprot_id}-${row.chem_comp_id}-${i}`}
                    onClick={() => {
                      const isSelected = selectedRow?.chem_comp_id === row.chem_comp_id && selectedRow?.uniprot_id === row.uniprot_id;
                      onSelectItem(isSelected ? null : { kind: 'known', item: row });
                    }}
                    className={`border-b border-gray-100 dark:border-gray-700/40 cursor-pointer transition-colors
                      ${selectedRow?.chem_comp_id === row.chem_comp_id && selectedRow?.uniprot_id === row.uniprot_id
                        ? 'bg-gray-50 dark:bg-gray-700/40'
                        : 'hover:bg-gray-50 dark:hover:bg-gray-800'
                      }`}
                  >
                    {visibleCols.has('search_type') && (
                      <td className="px-4 py-2.5"><SearchTypeBadge type={row.search_type} /></td>
                    )}
                    {visibleCols.has('uniprot_id') && (
                      <td className="px-4 py-2.5 font-jetbrains-mono text-xs font-medium text-cyan-900 dark:text-teal-300">
                        {row.uniprot_id}
                      </td>
                    )}
                    {visibleCols.has('chem_comp_id') && (
                      <td className="px-4 py-2.5 font-jetbrains-mono text-xs font-semibold text-gray-800 dark:text-gray-200">
                        {row.chem_comp_id}
                      </td>
                    )}
                    {visibleCols.has('source') && (
                      <td className="px-4 py-2.5"><SourceBadge source={row.source} /></td>
                    )}
                    {visibleCols.has('smiles') && (
                      <td className="px-4 py-2.5 text-gray-500 dark:text-gray-200">
                        <div className="flex items-center gap-1">
                          <TruncatedCell text={row.smiles} maxLen={35} />
                          <SmallCopyButton text={row.smiles} />
                        </div>
                      </td>
                    )}
                    {visibleCols.has('pchembl') && (
                      <td className="px-4 py-2.5 font-jetbrains-mono text-xs text-gray-500 dark:text-gray-200">
                        {row.pchembl != null ? row.pchembl.toFixed(2) : '—'}
                      </td>
                    )}
                    {visibleCols.has('mechanism') && (
                      <td className="px-4 py-2.5 max-w-48 text-gray-500 dark:text-gray-200">
                        <TruncatedCell text={row.mechanism ?? '—'} maxLen={40} />
                      </td>
                    )}
                    {visibleCols.has('activity_comment') && (
                      <td className="px-4 py-2.5 max-w-40 text-gray-500 dark:text-gray-200">
                        <TruncatedCell text={row.activity_comment ?? '—'} maxLen={30} />
                      </td>
                    )}
                    {visibleCols.has('binding_sites') && (
                      <td className="px-4 py-2.5 font-jetbrains-mono text-xs text-gray-500 dark:text-gray-200">
                        {row.binding_sites.length > 0 ? row.binding_sites.join(', ') : '—'}
                      </td>
                    )}
                    {visibleCols.has('pdb_ids') && (
                      <td className="px-4 py-2.5 text-xs text-gray-500 dark:text-gray-200">
                        {row.pdb_ids.length > 0 ? row.pdb_ids.join(', ') : '—'}
                      </td>
                    )}
                    {visibleCols.has('curation_method') && (
                      <td className="px-4 py-2.5 text-xs text-gray-500 dark:text-gray-200">
                        {row.curation_method ?? '—'}
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
                          ? 'bg-[#d9534f] text-white'
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

      {/* Selected result panel
      {selectedRow && (
        <SelectedResultPanel
          selected={{ kind: 'known', item: selectedRow }}
          onClose={() => setSelectedRow(null)}
        />
      )} */}
    </div>
  );
}
