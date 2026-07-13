import type { RDKitModule } from '@rdkit/rdkit';

type InitFn = (opts?: { locateFile?: () => string }) => Promise<RDKitModule>;

let rdkitPromise: Promise<RDKitModule> | null = null;

export function getRDKit(): Promise<RDKitModule> {
  if (!rdkitPromise) {
    rdkitPromise = (import('@rdkit/rdkit') as unknown as Promise<{ default: InitFn }>).then(
      (mod) => mod.default({ locateFile: () => '/RDKit_minimal.wasm' }),
    );
  }
  return rdkitPromise;
}