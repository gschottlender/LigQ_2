import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { createViewer } from '3dmol';
import type { GLViewer } from '3dmol';
import type { JSMol } from '@rdkit/rdkit';
import { getRDKit } from '../lib/rdkit';

interface MoleculeViewerModalProps {
  smiles: string;
  compoundId: string;
  onClose: () => void;
}

const VIEWER_WIDTH = 1200;
const VIEWER_HEIGHT = 800;

export function MoleculeViewerModal({ smiles, compoundId, onClose }: MoleculeViewerModalProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [molBlock, setMolBlock] = useState<string | null>(null);
  const [loadError, setLoadError] = useState(false);

  // Generate molBlock via the shared RDKit singleton
  useEffect(() => {
    let cancelled = false;
    let mol: JSMol | null = null;

    getRDKit()
      .then((rdkit) => {
        if (cancelled) return;
        mol = rdkit.get_mol(smiles);
        if (!mol) { setLoadError(true); return; }
        setMolBlock(mol.get_molblock());
      })
      .catch(() => { if (!cancelled) setLoadError(true); })
      .finally(() => { mol?.delete(); mol = null; });

    return () => { cancelled = true; };
  }, [smiles]);

  // Initialise 3Dmol after the container is in the DOM with non-zero dimensions.
  // setTimeout(0) defers until after the browser has painted the container.
  useEffect(() => {
    if (!molBlock || !containerRef.current) return;

    let viewer: GLViewer | null = null;
    const timerId = setTimeout(() => {
      if (!containerRef.current) return;
      viewer = createViewer(containerRef.current, { backgroundColor: 'white' });
      viewer.addModel(molBlock, 'sdf');
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      viewer.setStyle({} as any, { stick: { radius: 0.15 }, sphere: { scale: 0.25 } } as any);
      viewer.zoomTo();
      viewer.render();
      viewer.resize();
    }, 0);

    return () => {
      clearTimeout(timerId);
      viewer?.clear();
    };
  }, [molBlock]);

  // Escape key to close
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  // Mounted via createPortal at document.body — use `absolute` (not `fixed`) so
  // 3Dmol's position:absolute canvas is contained within the positioned container,
  // not pulled toward a fixed ancestor.
  return createPortal(
    <div
      className="absolute inset-0 z-50 bg-black/60 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-gray-900 rounded-2xl shadow-2xl overflow-hidden w-300 h-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-200 dark:border-gray-700">
          <div>
            <p className="font-semibold text-gray-800 dark:text-gray-100">3D Structure</p>
            <p className="text-xs font-jetbrains-mono text-gray-500 dark:text-gray-500 mt-0.5">
              {compoundId}
            </p>
          </div>
          <button
            onClick={onClose}
            className="ml-8 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body — fixed dimensions so 3Dmol always has a painted target */}
        <div
          className="flex items-center justify-center"
          style={{ width: VIEWER_WIDTH, height: VIEWER_HEIGHT }}
        >
          {loadError ? (
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Could not parse molecule structure.
            </p>
          ) : !molBlock ? (
            <div className="w-8 h-8 border-2 border-[#0d5c6b] border-t-transparent rounded-full animate-spin" />
          ) : (
            // position:relative is required so 3Dmol's position:absolute canvas
            // stays contained within this element instead of escaping to the body.
            <div
              ref={containerRef}
              style={{ position: 'relative', width: VIEWER_WIDTH, height: VIEWER_HEIGHT }}
            />
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}