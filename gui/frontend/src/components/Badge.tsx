import type { SearchType, LigandSource } from '../types';

interface SearchTypeBadgeProps {
  type: SearchType;
}

export function SearchTypeBadge({ type }: SearchTypeBadgeProps) {
  const config: Record<SearchType, { label: string; className: string }> = {
    sequence: {
      label: 'Sequence',
      className: 'bg-teal-100 text-teal-800 dark:bg-teal-800 dark:text-teal-200 px-3.5 py-1.5',
    },
    nearest_k: {
      label: 'Nearest K',
      className: 'bg-violet-100 text-violet-700 dark:bg-violet-800 dark:text-violet-200 px-3.5 py-1.5',
    },
    domain: {
      label: 'Domain',
      className: 'bg-amber-100 text-amber-700 dark:bg-amber-800 dark:text-amber-200 px-5 py-1.5',
    },
  };

  const { label, className } = config[type];
  return (
    <span className={`inline-block text-xs font-medium rounded-full whitespace-nowrap ${className}`}>
      {label}
    </span>
  );
}

interface SourceBadgeProps {
  source: LigandSource;
}

export function SourceBadge({ source }: SourceBadgeProps) {
  const config: Record<LigandSource, { label: string; className: string }> = {
    chembl: {
      label: 'ChEMBL',
      className: 'bg-gray-100 text-gray-500 dark:bg-gray-400 dark:text-gray-700 px-2.5 py-1',
    },
    pdb: {
      label: 'PDB',
      className: 'bg-slate-100 text-slate-500 dark:bg-slate-400 dark:text-slate-700 px-5.5 py-1',
    },
  };

  const { label, className } = config[source];
  return (
    <span className={`inline-block text-xs font-medium rounded-full whitespace-nowrap ${className}`}>
      {label}
    </span>
  );
}