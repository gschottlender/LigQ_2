import { useState, useEffect } from 'react';
import type { JSMol } from '@rdkit/rdkit';
import { getRDKit } from '../lib/rdkit';

interface MoleculeViewerProps {
  smiles: string;
  width?: number;
  height?: number;
}

export function MoleculeViewer({ smiles, width = 420, height = 250 }: MoleculeViewerProps) {
  const [svg, setSvg] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!smiles) {
      setError(true);
      setLoading(false);
      return;
    }

    let cancelled = false;
    let mol: JSMol | null = null;

    getRDKit()
      .then((rdkit) => {
        if (cancelled) return;
        mol = rdkit.get_mol(smiles);
        if (!mol) {
          setError(true);
          setLoading(false);
          return;
        }
        setSvg(mol.get_svg(width, height));
        setLoading(false);
      })
      .catch(() => {
        if (!cancelled) {
          setError(true);
          setLoading(false);
        }
      })
      .finally(() => {
        mol?.delete();
        mol = null;
      });

    return () => {
      cancelled = true;
    };
  }, [smiles, width, height]);

  if (loading) {
    return <div className="w-full h-full animate-pulse bg-gray-100 dark:bg-gray-700 rounded-lg" />;
  }

  if (error || !svg) {
    return (
      <div className="text-center">
        <div className="text-3xl mb-1 select-none">⬡</div>
        <p className="text-xs text-gray-500 dark:text-gray-500">Schematic preview</p>
        <p className="text-xs text-gray-400 dark:text-gray-600">open in viewer for full structure</p>
      </div>
    );
  }

  return (
    <div
      className="bg-white rounded-lg overflow-hidden"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}