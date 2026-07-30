import type { KnownLigand, LigandDisplaySource } from '../types';

const CHEMBL_ID_PATTERN = /^CHEMBL\d+$/i;

export function getKnownLigandDisplaySource(
  ligand: Pick<KnownLigand, 'chem_comp_id' | 'source'>,
): LigandDisplaySource {
  // Structure-based PDB/ChEMBL unification keeps the binding-evidence source
  // but prefers a PDB CCD ID as the canonical compound ID when both exist.
  const usesPdbCanonicalId = !CHEMBL_ID_PATTERN.test(ligand.chem_comp_id.trim());
  if (ligand.source === 'chembl' && usesPdbCanonicalId) {
    return 'pdb_chembl';
  }
  return ligand.source;
}
