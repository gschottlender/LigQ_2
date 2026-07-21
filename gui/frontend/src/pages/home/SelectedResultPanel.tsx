import { Copy, Download, ExternalLink, X } from 'lucide-react';
import { useState } from 'react';
import type { KnownLigand, PredictedLigand } from '../../types';
import { SearchTypeBadge, SourceBadge } from '../../components/Badge';
import { MoleculeViewer } from '../../components/MoleculeViewer';
import { MoleculeViewerModal } from '../../components/MoleculeViewerModal';
import { getRDKit } from '../../lib/rdkit';

export type SelectedItem =
  | { kind: 'known'; item: KnownLigand }
  | { kind: 'predicted'; item: PredictedLigand };

interface SelectedResultPanelProps {
  selected: SelectedItem;
  onClose: () => void;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <button
      onClick={handleCopy}
      title="Copy SMILES"
      className="p-1.5 rounded-lg border border-gray-200 dark:border-gray-400 text-gray-400 dark:text-gray-200
        hover:text-teal-600 hover:border-teal-400 transition-colors shrink-0 cursor-pointer"
    >
      {copied ? (
        <span className="text-xs text-teal-600 dark:text-teal-400 font-medium px-0.5">✓</span>
      ) : (
        <Copy className="w-3.5 h-3.5" />
      )}
    </button>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
      <tr className="border-b border-gray-200 dark:border-gray-500 last:border-b-0">
        <td className="font-dm-sans py-2.5 px-4 text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap align-top w-32">
          {label}
        </td>
        <td className="font-dm-sans py-2.5 pr-4 text-xs text-gray-700 dark:text-gray-200 text-left align-top wrap-break-word">
          {value ?? '—'}
        </td>
      </tr>
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

export function SelectedResultPanel({ selected, onClose }: SelectedResultPanelProps) {
  const smiles = selected.item.smiles;
  const compoundId = selected.item.chem_comp_id;

  const [sdfError, setSdfError] = useState(false);
  const [showViewer, setShowViewer] = useState(false);

  const handleDownloadSDF = async () => {
    const rdkit = await getRDKit();
    const mol = rdkit.get_mol(smiles);
    if (!mol) {
      setSdfError(true);
      setTimeout(() => setSdfError(false), 1800);
      return;
    }
    const sdfContent = mol.get_molblock() + '$$$$\n';
    mol.delete();
    const blob = new Blob([sdfContent], { type: 'chemical/x-mdl-sdfile' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${compoundId}.sdf`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
    <aside className="fixed top-20 right-0 h-[calc(100vh-80px)] w-120 border-l border-gray-200 dark:border-gray-700/60 bg-white dark:bg-gray-800 flex flex-col shadow-xl z-30">
      {/* Header */}
      <div className="flex items-start justify-between px-4 py-3 border-b border-gray-100 dark:border-gray-700/60">
        <div className="min-w-0">
          <p className="font-semibold text-gray-800 dark:text-gray-100">Selected result</p>
          <p className="text-sm font-jetbrains-mono text-gray-500 dark:text-gray-400 mt-0.5 truncate">{compoundId}</p>
        </div>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors shrink-0 ml-2 mt-0.5 border p-2 rounded-xl cursor-pointer">
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto flex flex-col px-4">
        <table className="w-full">
          <tbody>
            {selected.kind === 'predicted' && (
              <>
                <Row label="Query" value={selected.item.qseqid} />
                <Row label="Protein hit" value={
                  <span className="font-jetbrains-mono text-teal-700 dark:text-teal-300 font-medium">
                    {selected.item.sseqid}
                  </span>
                } />
              </>
            )}
            {selected.kind === 'known' && (
              <>
                <Row label="UniProt ID" value={
                  <span className="font-jetbrains-mono text-teal-700 dark:text-teal-300">{selected.item.uniprot_id}</span>
                } />
                <Row label="Source" value={<SourceBadge source={selected.item.source} />} />
                <Row label="pChEMBL" value={selected.item.pchembl != null ? selected.item.pchembl.toFixed(2) : '—'} />
                <Row label="Binding sites" value={selected.item.binding_sites.join(', ') || '—'} />
                <Row label="PDB IDs" value={selected.item.pdb_ids.join(', ') || '—'} />
                <Row label="Mechanism" value={selected.item.mechanism ?? '—'} />
              </>
            )}
            <Row label="Search type" value={<SearchTypeBadge type={selected.item.search_type} />} />
            {selected.kind === 'predicted' && (
              <Row label="Similarity score" value={
                <TanimotoBar value={selected.item.tanimoto ?? selected.item.similarity ?? selected.item.bsi_score ?? 0} />
              } />
            )}
          </tbody>
        </table>

        <div className="flex flex-col gap-4 mt-2">
          {/* SMILES */}
          <div className="flex flex-col gap-1.5">
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-700 dark:text-gray-100">
              SMILES
            </p>
            <div className="flex items-center gap-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-400 rounded-lg px-3 py-2">
              <p className="text-xs font-dm-sans text-gray-600 dark:text-gray-300 break-all leading-relaxed flex-1">
                {smiles}
              </p>
              <CopyButton text={smiles} />
            </div>
          </div>

          {/* 2D Structure */}
          <div className="flex flex-col gap-1.5">
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-700 dark:text-gray-100">
              2D Structure
            </p>
            <div className="bg-white dark:bg-gray-800/50 border border-gray-200 dark:border-gray-400 rounded-xl p-4 flex items-center justify-center h-70">
              <MoleculeViewer smiles={smiles} />
            </div>
          </div>
        </div>

      </div>

      {/* Actions */}
      <div className="border-t border-gray-100 dark:border-gray-700/60 p-4 flex gap-2">
        <button
          onClick={handleDownloadSDF}
          disabled={!smiles}
          className="flex-1 flex items-center justify-center gap-2 py-2 rounded-xl border transition-colors text-sm font-medium
            disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer
            border-gray-200 dark:border-gray-600 hover:border-gray-300
            text-gray-600 dark:text-gray-300"
        >
          {sdfError ? (
            <span className="text-red-500 dark:text-red-400 text-xs">Invalid SMILES</span>
          ) : (
            <><Download className="w-4 h-4" /> Download SDF</>
          )}
        </button>
        <button
          onClick={() => setShowViewer(true)}
          className="flex-1 flex items-center justify-center gap-2 py-2 rounded-xl
            bg-cyan-900 hover:bg-cyan-800 text-white text-sm font-medium transition-colors cursor-pointer"
        >
          <ExternalLink className="w-4 h-4" /> Open in Viewer
        </button>
      </div>
    </aside>

    {showViewer && (
      <MoleculeViewerModal
        smiles={smiles}
        compoundId={compoundId}
        onClose={() => setShowViewer(false)}
      />
    )}
    </>
  );
}