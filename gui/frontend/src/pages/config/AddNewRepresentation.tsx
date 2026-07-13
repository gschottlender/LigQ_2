import { useState } from 'react';
import { AlertTriangle, ChevronDown, ChevronRight, Info, Loader2, Settings, Upload } from 'lucide-react';
import { useDatabase } from '../../context/DatabaseContext';
import { Tooltip } from '../../components/Tooltip';
import { api } from '../../lib/api';

type RepPresetId =
  | 'morgan'
  | 'maccs'
  | 'rdkit_fp'
  | 'morgan_feature'
  | 'atom_pair'
  | 'torsion'
  | 'chemberta_zinc'
  | 'chemberta_pubchem';

interface RepPreset {
  id: RepPresetId;
  label: string;
  representation_type: 'rdkit' | 'huggingface';
  rdkit_fp_kind?: string;
  model_id?: string;
  metric: 'tanimoto' | 'cosine';
  default_n_bits: number;
  fixed_n_bits: boolean;
  default_radius?: number;
  canonical_name: string;
  description: string;
}

const REP_PRESETS: RepPreset[] = [
  {
    id: 'morgan',
    label: 'ECFP Morgan',
    representation_type: 'rdkit',
    rdkit_fp_kind: 'morgan',
    metric: 'tanimoto',
    default_n_bits: 1024,
    fixed_n_bits: false,
    default_radius: 2,
    canonical_name: 'morgan_1024_r2',
    description: 'Extended-Connectivity Fingerprints — fast, widely used for molecular similarity.',
  },
  {
    id: 'maccs',
    label: 'MACCS',
    representation_type: 'rdkit',
    rdkit_fp_kind: 'maccs',
    metric: 'tanimoto',
    default_n_bits: 167,
    fixed_n_bits: true,
    canonical_name: 'maccs',
    description: 'Structural key fingerprint based on MACCS keys.',
  },
  {
    id: 'rdkit_fp',
    label: 'RDKit fingerprint',
    representation_type: 'rdkit',
    rdkit_fp_kind: 'rdkit',
    metric: 'tanimoto',
    default_n_bits: 1024,
    fixed_n_bits: false,
    canonical_name: 'rdkit_1024',
    description: 'RDKit path-based fingerprint — fast, general-purpose bit vector.',
  },
  {
    id: 'morgan_feature',
    label: 'Morgan Feature',
    representation_type: 'rdkit',
    rdkit_fp_kind: 'morgan_feature',
    metric: 'tanimoto',
    default_n_bits: 1024,
    fixed_n_bits: false,
    default_radius: 2,
    canonical_name: 'morgan_feature_1024_r2',
    description: 'Morgan fingerprints using pharmacophoric atom features.',
  },
  {
    id: 'atom_pair',
    label: 'Atom Pair',
    representation_type: 'rdkit',
    rdkit_fp_kind: 'ap',
    metric: 'tanimoto',
    default_n_bits: 1024,
    fixed_n_bits: false,
    canonical_name: 'ap_rdkit',
    description: 'Atom pair fingerprints encoding pairwise atom environment.',
  },
  {
    id: 'torsion',
    label: 'Topological Torsion',
    representation_type: 'rdkit',
    rdkit_fp_kind: 'topological_torsion',
    metric: 'tanimoto',
    default_n_bits: 1024,
    fixed_n_bits: false,
    canonical_name: 'topological_torsion_rdkit_1024',
    description: 'Topological torsion fingerprints based on 4-atom paths.',
  },
  {
    id: 'chemberta_zinc',
    label: 'ChemBERTa (zinc-base)',
    representation_type: 'huggingface',
    model_id: 'seyonec/ChemBERTa-zinc-base-v1',
    metric: 'cosine',
    default_n_bits: 768,
    fixed_n_bits: false,
    canonical_name: 'chemberta_zinc_base_768',
    description: 'Transformer embedding trained on ZINC SMILES — captures complex chemical patterns.',
  },
  {
    id: 'chemberta_pubchem',
    label: 'ChemBERTa (pubchem)',
    representation_type: 'huggingface',
    model_id: 'seyonec/PubChem10M_SMILES_BPE_450k',
    metric: 'cosine',
    default_n_bits: 768,
    fixed_n_bits: false,
    canonical_name: 'chemberta_pubchem_768',
    description: 'Transformer embedding trained on PubChem SMILES — broad chemical coverage.',
  },
];

const METRIC_HINTS: Record<'tanimoto' | 'cosine', string> = {
  tanimoto: 'Tanimoto · Best for binary fingerprints',
  cosine: 'Cosine · Best for embeddings',
};

interface ProcessingState {
  stage: 'idle' | 'processing' | 'done' | 'error';
  progress: number;
  message: string;
}

function getAutoName(preset: RepPreset, nBits: number, radius: number): string {
  switch (preset.id) {
    case 'morgan':         return `morgan_${nBits}_r${radius}`;
    case 'maccs':          return 'maccs';
    case 'rdkit_fp':       return `rdkit_${nBits}`;
    case 'morgan_feature': return `morgan_feature_${nBits}_r${radius}`;
    case 'atom_pair':      return 'ap_rdkit';
    case 'torsion':        return `topological_torsion_rdkit_${nBits}`;
    case 'chemberta_zinc': return `chemberta_zinc_base_${nBits}`;
    case 'chemberta_pubchem': return `chemberta_pubchem_${nBits}`;
  }
}

function formatEta(startedAt: number, progress: number): string {
  if (progress <= 0) return '';
  const elapsed = (Date.now() - startedAt) / 1000;
  const remaining = elapsed / (progress / 100) - elapsed;
  if (remaining <= 0 || !isFinite(remaining)) return '';
  if (remaining < 60) return `ETA: ${Math.round(remaining)}s`;
  return `ETA: ${Math.round(remaining / 60)}m`;
}

function selectPreset(p: RepPreset, setNBits: (n: number) => void, setRadius: (r: number) => void) {
  setNBits(p.default_n_bits);
  if (p.default_radius != null) setRadius(p.default_radius);
}

export function AddNewRepresentation() {
  const { databases, getRepresentationsForDatabase, refetchRepresentationsForDatabase } = useDatabase();

  const [selectedDbId, setSelectedDbId] = useState('');
  const [repType, setRepType] = useState<'rdkit' | 'huggingface'>('rdkit');
  const [presetId, setPresetId] = useState<RepPresetId>('morgan');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [nBits, setNBits] = useState(1024);
  const [radius, setRadius] = useState(2);
  const [batchSize, setBatchSize] = useState(14);
  const [revision, setRevision] = useState('');
  const [nJobs, setNJobs] = useState(-1);
  const [repName, setRepName] = useState('');
  const [processing, setProcessing] = useState<ProcessingState>({ stage: 'idle', progress: 0, message: '' });
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [errors, setErrors] = useState<Record<string, string>>({});

  void startedAt;

  const preset = REP_PRESETS.find((p) => p.id === presetId)!;
  const isHuggingFace = preset.representation_type === 'huggingface';
  const hasRadius = presetId === 'morgan' || presetId === 'morgan_feature';
  const autoName = getAutoName(preset, nBits, radius);
  const displayName = repName.trim() || autoName;
  const isNonCanonical = displayName !== preset.canonical_name;

  const validate = () => {
    const errs: Record<string, string> = {};
    if (!selectedDbId) errs.db = 'Select a database.';
    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleProcess = async () => {
    if (!validate()) return;

    const now = Date.now();
    setStartedAt(now);
    setProcessing({ stage: 'processing', progress: 0, message: 'Submitting job…' });

    try {
      const effectiveNBits = preset.fixed_n_bits ? preset.default_n_bits : nBits;
      const body: Record<string, unknown> = {
        base_name: selectedDbId,
        representation_type: preset.representation_type,
        rep_name: displayName,
        n_bits: effectiveNBits,
      };
      if (preset.rdkit_fp_kind) body.rdkit_fp_kind = preset.rdkit_fp_kind;
      if (preset.model_id) body.model_id = preset.model_id;
      if (isHuggingFace) {
        body.batch_size = batchSize;
        if (revision.trim()) body.revision = revision.trim();
      }
      if (!isHuggingFace) body.n_jobs = nJobs;

      const { data } = await api.post<{ job_id: string }>('/jobs/add-representation', body);
      const jobId = data.job_id;

      const interval = setInterval(async () => {
        try {
          const { data: job } = await api.get(`/jobs/${jobId}`);
          const percent: number = job.progress_percent ?? 0;
          const eta = formatEta(now, percent);
          const baseMsg = 'Computing fingerprints for all compounds.';

          if (['completed', 'completed_with_warnings'].includes(job.status)) {
            clearInterval(interval);
            setProcessing({ stage: 'done', progress: 100, message: '' });
            setStartedAt(null);
            await refetchRepresentationsForDatabase(selectedDbId);
          } else if (job.status === 'failed') {
            clearInterval(interval);
            setProcessing({ stage: 'error', progress: percent, message: job.error ?? 'Processing failed.' });
            setStartedAt(null);
          } else {
            setProcessing({
              stage: 'processing',
              progress: percent,
              message: eta ? `${baseMsg} ${eta}` : baseMsg,
            });
          }
        } catch {
          // transient error — keep polling
        }
      }, 3000);
    } catch (err: unknown) {
      const message =
        (err as { response?: { data?: { message?: string } } })?.response?.data?.message ??
        'Failed to submit job.';
      setProcessing({ stage: 'error', progress: 0, message });
      setStartedAt(null);
    }
  };

  const existingReps = selectedDbId ? getRepresentationsForDatabase(selectedDbId) : [];

  return (
    <div className="w-full">
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
        Compute a new molecular representation for an existing database.
      </p>

      {/* Select database */}
      <div className="flex flex-col gap-1.5">
        <label className="text-sm font-medium text-gray-600 dark:text-gray-300 flex items-center gap-1.5">
          Select database
          <Tooltip content="Choose a database that has already been processed via 'Add new database'.">
            <Info className="w-3.5 h-3.5 text-gray-400 cursor-default" />
          </Tooltip>
        </label>
        <div className="relative">
          <select
            value={selectedDbId}
            onChange={(e) => setSelectedDbId(e.target.value)}
            className="w-full appearance-none border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 pr-8 text-sm
              text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 cursor-pointer focus:outline-none focus:ring-1 focus:ring-teal-500"
          >
            <option value="">Select a database…</option>
            {databases.map((db) => (
              <option key={db.id} value={db.id}>{db.label}</option>
            ))}
          </select>
          <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
        </div>
        {errors.db && <p className="text-xs text-red-500">{errors.db}</p>}
        {selectedDbId && existingReps.length > 0 && (
          <p className="text-xs text-gray-400 dark:text-gray-500">
            Existing: {existingReps.map((r) => r.label).join(' · ')}
          </p>
        )}
      </div>

      {/* Representation type dropdown */}
      <div className="mt-5 flex flex-col gap-1.5">
        <label className="text-sm font-medium text-gray-600 dark:text-gray-300">
          Representation type
        </label>
        <div className="relative">
          <select
            value={repType}
            onChange={(e) => {
              const t = e.target.value as 'rdkit' | 'huggingface';
              setRepType(t);
              const first = REP_PRESETS.find((p) => p.representation_type === t)!;
              setPresetId(first.id);
              selectPreset(first, setNBits, setRadius);
              setRepName('');
            }}
            className="w-full appearance-none border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 pr-8 text-sm
              text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 cursor-pointer focus:outline-none focus:ring-1 focus:ring-teal-500"
          >
            <option value="rdkit">RDKit fingerprint</option>
            <option value="huggingface">HuggingFace embedding</option>
          </select>
          <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
        </div>
      </div>

      {/* Representation preset dropdown */}
      <div className="mt-4 flex flex-col gap-1.5">
        <label className="text-sm font-medium text-gray-600 dark:text-gray-300">
          Representation preset
        </label>
        <div className="relative">
          <select
            value={presetId}
            onChange={(e) => {
              const p = REP_PRESETS.find((r) => r.id === e.target.value)!;
              setPresetId(p.id);
              selectPreset(p, setNBits, setRadius);
              setRepName('');
            }}
            className="w-full appearance-none border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 pr-8 text-sm
              text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 cursor-pointer focus:outline-none focus:ring-1 focus:ring-teal-500"
          >
            {REP_PRESETS.filter((p) => p.representation_type === repType).map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>
          <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
        </div>
        <p className="text-xs text-gray-400 dark:text-gray-500">{preset.description}</p>
        {presetId === 'maccs' && (
          <p className="text-xs text-blue-500 dark:text-blue-400">
            MACCS uses 166 fixed structural keys. Bit size is always 167.
          </p>
        )}
      </div>

      {/* Recommended metric (read-only) */}
      <div className="mt-5 flex flex-col gap-1.5">
        <label className="text-sm font-medium text-gray-600 dark:text-gray-300 flex items-center gap-1.5">
          Recommended metric
          <Tooltip content="Determined automatically by the representation type. RDKit fingerprints use Tanimoto; embedding models use Cosine similarity.">
            <Info className="w-3.5 h-3.5 text-gray-400 cursor-default" />
          </Tooltip>
        </label>
        <div className="border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 rounded-lg px-3 py-2
          text-sm text-gray-500 dark:text-gray-400 select-none">
          {METRIC_HINTS[preset.metric]}
        </div>
      </div>

      {/* Representation name (always visible) */}
      <div className="mt-5 flex flex-col gap-1.5">
        <label className="text-sm font-medium text-gray-600 dark:text-gray-300">
          Representation name
        </label>
        <input
          type="text"
          value={repName}
          onChange={(e) => setRepName(e.target.value)}
          placeholder={autoName}
          className="border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 text-sm
            text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 focus:outline-none focus:ring-1 focus:ring-teal-500
            placeholder:text-gray-400"
        />
        {isNonCanonical && (
          <p className="flex items-center gap-1.5 text-xs text-amber-600 dark:text-amber-400">
            <AlertTriangle className="w-3 h-3 shrink-0" />
            Using a non-standard name disables automatic similarity thresholds.
          </p>
        )}
      </div>

      {/* Advanced parameters */}
      <div className="mt-5">
        <button
          onClick={() => setShowAdvanced((v) => !v)}
          className="flex items-center gap-1.5 text-sm font-medium text-gray-600 dark:text-gray-300 hover:text-teal-600 transition-colors cursor-pointer"
        >
          {showAdvanced ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          Advanced parameters
        </button>

        {showAdvanced && (
          <div className="mt-3 p-4 border border-gray-200 dark:border-gray-700 rounded-xl bg-gray-50 dark:bg-gray-800/50 flex flex-col gap-3">
            {/* n_bits — hidden for fixed-size presets (MACCS) */}
            {!preset.fixed_n_bits && (
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-medium text-gray-500 dark:text-gray-400">n_bits</label>
                <input
                  type="number"
                  value={nBits}
                  min={64}
                  step={isHuggingFace ? 1 : 64}
                  onChange={(e) => setNBits(parseInt(e.target.value) || preset.default_n_bits)}
                  className="border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1.5 text-sm
                    text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 w-32 focus:outline-none focus:ring-1 focus:ring-teal-500"
                />
              </div>
            )}

            {/* radius — morgan and morgan_feature only */}
            {hasRadius && (
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-medium text-gray-500 dark:text-gray-400">radius</label>
                <input
                  type="number"
                  value={radius}
                  min={1}
                  max={6}
                  onChange={(e) => setRadius(parseInt(e.target.value) || 2)}
                  className="border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1.5 text-sm
                    text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 w-32 focus:outline-none focus:ring-1 focus:ring-teal-500"
                />
              </div>
            )}

            {/* n_jobs — RDKit only */}
            {!isHuggingFace && (
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-medium text-gray-500 dark:text-gray-400">
                  n_jobs <span className="font-normal text-gray-400">(-1 = all cores)</span>
                </label>
                <input
                  type="number"
                  value={nJobs}
                  min={-1}
                  onChange={(e) => setNJobs(parseInt(e.target.value) || -1)}
                  className="border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1.5 text-sm
                    text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 w-32 focus:outline-none focus:ring-1 focus:ring-teal-500"
                />
              </div>
            )}

            {/* batch_size — HuggingFace only */}
            {isHuggingFace && (
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-medium text-gray-500 dark:text-gray-400">batch_size</label>
                <input
                  type="number"
                  value={batchSize}
                  min={1}
                  step={1}
                  onChange={(e) => setBatchSize(parseInt(e.target.value) || 14)}
                  className="border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1.5 text-sm
                    text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 w-32 focus:outline-none focus:ring-1 focus:ring-teal-500"
                />
              </div>
            )}

            {/* revision — HuggingFace only */}
            {isHuggingFace && (
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-medium text-gray-500 dark:text-gray-400">
                  revision <span className="font-normal text-gray-400">(branch, tag or commit hash)</span>
                </label>
                <input
                  type="text"
                  value={revision}
                  onChange={(e) => setRevision(e.target.value)}
                  placeholder="main"
                  className="border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1.5 text-sm
                    text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 focus:outline-none focus:ring-1 focus:ring-teal-500
                    placeholder:text-gray-400"
                />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Process button */}
      <button
        onClick={handleProcess}
        disabled={processing.stage === 'processing'}
        className={`mt-6 w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-medium transition-colors
          ${processing.stage === 'processing'
            ? 'bg-gray-100 dark:bg-gray-700 text-gray-400 cursor-not-allowed'
            : 'bg-[#0d5c6b] hover:bg-[#0a4d5a] text-white cursor-pointer'
          }`}
      >
        {processing.stage === 'processing'
          ? <><Loader2 className="w-4 h-4 animate-spin" /> Processing…</>
          : <><Settings className="w-4 h-4" /> Process representation</>
        }
      </button>

      {/* Progress */}
      {(processing.stage === 'processing' || processing.stage === 'done') && (
        <div className="mt-4 flex flex-col gap-2">
          <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-[#0d5c6b] rounded-full transition-all duration-500"
              style={{ width: `${processing.progress}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>{processing.message}</span>
            <span>{processing.progress}%</span>
          </div>
          {isHuggingFace && processing.stage === 'processing' && (
            <p className="text-xs text-amber-500 dark:text-amber-400">
              Embedding models can take several hours on large databases.
            </p>
          )}
        </div>
      )}

      {/* Error */}
      {processing.stage === 'error' && (
        <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700 rounded-xl">
          <p className="text-sm text-red-700 dark:text-red-300">{processing.message}</p>
        </div>
      )}

      {/* Success */}
      {processing.stage === 'done' && (
        <div className="mt-4 flex items-center gap-2 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-700 rounded-xl">
          <Upload className="w-4 h-4 text-green-600 dark:text-green-400 shrink-0" />
          <p className="text-sm text-green-700 dark:text-green-300">
            <span className="font-medium">"{displayName}"</span> is now available in the Representation dropdown.
          </p>
        </div>
      )}
    </div>
  );
}