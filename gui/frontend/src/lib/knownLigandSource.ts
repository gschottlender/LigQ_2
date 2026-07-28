import type { KnownLigand, LigandSource } from '../types';

const CHEMBL_ID_PATTERN = /^CHEMBL\d+$/i;

export function getKnownLigandDisplaySource(
  ligand: Pick<KnownLigand, 'chem_comp_id'>,
): LigandSource {
  return CHEMBL_ID_PATTERN.test(ligand.chem_comp_id.trim()) ? 'chembl' : 'pdb';
}
