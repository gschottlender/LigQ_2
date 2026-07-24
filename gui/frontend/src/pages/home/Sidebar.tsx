import {
  Check,
  ChevronLeft,
  ChevronRight,
  FolderOpen,
  Info,
  Loader2,
  Play,
} from 'lucide-react';
import { useEffect, useState, useRef } from 'react';
import { useDatabase } from '../../context/DatabaseContext';
import { useSystemPolicy } from '../../context/SystemPolicyContext';
import { Tooltip } from '../../components/Tooltip';
import { api } from '../../lib/api';
import { JobFailurePanel, JobProgressPanel } from '../../components/JobProgressPanel';
import type { JobFailure, JobProgress, SearchMode } from '../../types';

interface SidebarProps {
  isRunning: boolean;
  progressPercent: number;
  progressMessage: string;
  progress: JobProgress | null;
  failure: JobFailure | null;
  jobError: string | null;
  startedAt: string | null;
  onJobCreated: (jobId: string, searchMode: SearchMode) => void;
}

interface SelectFieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  info?: string;
  infoPosition?: 'top' | 'bottom' | 'left' | 'right';
  disabled?: boolean;
}

interface SliderFieldProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}

interface CheckboxFieldProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  info?: string;
  disabled?: boolean;
}

interface HardwareCapabilities {
  cuda_available: boolean;
  cuda_device_name: string | null;
}

type GpuStatus = 'checking' | 'available' | 'unavailable' | 'error';

function SelectField({
  label,
  value,
  onChange,
  options,
  info,
  infoPosition = 'top',
  disabled,
}: SelectFieldProps) {
  return (
    <section className="flex flex-col gap-2 mt-4">
      <label className="text-sm font-dm-sans font-semibold text-gray-500 dark:text-gray-200 flex items-center gap-1.5">
        {label}
        {info && (
          <Tooltip content={info} position={infoPosition}>
            <Info className="w-3.5 h-3.5 text-gray-400 cursor-default" />
          </Tooltip>
        )}
      </label>
      <div className="relative w-full">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className="w-full appearance-none border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 pr-8 text-sm
            text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 cursor-pointer focus:outline-none focus:ring-1 focus:ring-teal-500
            disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
        <ChevronRight className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none rotate-90" />
      </div>
    </section>
  );
}

function SliderField({ label, value, onChange, min = 0, max = 1, step = 0.01, disabled }: SliderFieldProps) {
  return (
    <section className="flex flex-col gap-2 mt-5">
      <label className="text-sm font-dm-sans font-semibold text-gray-500 dark:text-gray-200">{label}</label>
      <div className="flex items-center gap-3 pr-4">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="min-w-0 flex-1 cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
        />
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={value.toFixed(2)}
          disabled={disabled}
          onChange={(e) => {
            const v = parseFloat(e.target.value);
            if (!isNaN(v) && v >= min && v <= max) onChange(v);
          }}
          className="w-20 shrink-0 border border-gray-300 dark:border-gray-600 rounded-lg px-2 py-1 text-xs text-center
            text-gray-600 dark:text-gray-200 bg-white dark:bg-gray-800 focus:outline-none focus:ring-1 focus:ring-teal-500
            disabled:opacity-50 disabled:cursor-not-allowed"
        />
      </div>
    </section>
  );
}

function CheckboxField({ label, checked, onChange, info, disabled }: CheckboxFieldProps) {
  return (
    <div className={`flex font-dm-sans font-medium items-center gap-2 text-sm
      ${disabled ? 'text-gray-400 dark:text-gray-500' : 'text-gray-500 dark:text-gray-200'}`}>
      <div
        role="checkbox"
        aria-checked={checked}
        aria-disabled={disabled}
        tabIndex={disabled ? -1 : 0}
        onClick={() => { if (!disabled) onChange(!checked); }}
        onKeyDown={(event) => {
          if (!disabled && (event.key === 'Enter' || event.key === ' ')) {
            event.preventDefault();
            onChange(!checked);
          }
        }}
        className={`w-4 h-4 rounded flex items-center justify-center border transition-colors duration-200
          ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}
          ${checked
            ? 'bg-cyan-900 border-cyan-900 dark:bg-teal-500 dark:border-teal-500'
            : 'bg-white dark:bg-transparent border-gray-300 dark:border-gray-600'
          }`}
      >
        {checked && <Check className="w-3 h-3 text-white stroke-3" />}
      </div>
      <span>{label}</span>
      {info && (
        <Tooltip content={info}>
          <Info className="w-3.5 h-3.5 text-gray-400 cursor-default" />
        </Tooltip>
      )}
    </div>
  );
}

const VALID_AA_RE = /^[ACDEFGHIKLMNPQRSTVWYBZXU*]+$/i;
const BSI_REPRESENTATION = 'morgan_1024_r2';
const TANIMOTO_MIN_THRESHOLD = 0.2;
const COSINE_MIN_THRESHOLD = 0.75;
const BSI_MIN_THRESHOLD = 0.97;
const BSI_DEFAULT_THRESHOLD = 0.98;
const MIN_NEAREST_K = 1;
const MAX_NEAREST_K = 15;

function roundThresholdUp(value: number): number {
  return Math.min(1, Math.ceil(value * 100 - 1e-9) / 100);
}

function representationDefault(value: number | null | undefined): number | null {
  return value == null ? null : roundThresholdUp(value);
}

function structuralMinimumForMetric(metric: 'tanimoto' | 'cosine' | undefined): number {
  if (metric === 'tanimoto') return TANIMOTO_MIN_THRESHOLD;
  if (metric === 'cosine') return COSINE_MIN_THRESHOLD;
  return 0;
}

interface FastaStats {
  sequenceCount: number;
  residueCount: number;
  duplicateIds: string[];
}

function inspectFasta(text: string): { error: string | null; stats: FastaStats } {
  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean);
  const hasHeader = lines.some((l) => l.startsWith('>'));
  const seqLines = lines.filter((l) => !l.startsWith('>'));
  const hasSequence = seqLines.some((l) => l.length > 0);
  const ids = lines
    .filter((line) => line.startsWith('>'))
    .map((line) => line.slice(1).trim().split(/\s+/, 1)[0])
    .filter(Boolean);
  const seen = new Set<string>();
  const duplicateIds = [...new Set(ids.filter((id) => {
    if (seen.has(id)) return true;
    seen.add(id);
    return false;
  }))];
  const stats = {
    sequenceCount: ids.length,
    residueCount: seqLines.reduce((total, line) => total + line.length, 0),
    duplicateIds,
  };
  if (!hasHeader || !hasSequence) {
    return {
      error: 'File does not appear to be a valid FASTA file. Make sure it contains protein sequences with > headers.',
      stats,
    };
  }
  if (seqLines.some((l) => !VALID_AA_RE.test(l))) {
    return {
      error: 'File does not appear to be a valid FASTA file. Make sure it contains protein sequences with > headers.',
      stats,
    };
  }
  return { error: null, stats };
}

const VALIDATION_MESSAGES = {
  fasta: 'FASTA file is required.',
  kValue: `K must be between ${MIN_NEAREST_K} and ${MAX_NEAREST_K}.`,
  noRepresentation: 'No representation available for the selected database.',
  bsiUnavailable: 'BSI requires the morgan_1024_r2 representation for the selected database.',
  bsiGpuRequired: 'BSI requires a CUDA-capable GPU in the graphical interface.',
  thresholdRange: 'Maximum cutoff must be greater than or equal to minimum cutoff.',
};

export function Sidebar({
  isRunning,
  progressPercent,
  progressMessage,
  progress,
  failure,
  jobError,
  startedAt,
  onJobCreated,
}: SidebarProps) {
  const { databases, getRepresentationsForDatabase } = useDatabase();
  const { policy, isWeb } = useSystemPolicy();
  const [isOpen, setIsOpen] = useState(true);
  const [showButton, setShowButton] = useState(false);

  const [searchMode, setSearchMode] = useState<SearchMode>(policy.search.default_mode);
  const [databaseId, setDatabaseId] = useState('');
  const [representationId, setRepresentationId] = useState('');
  const [fastaFile, setFastaFile] = useState<File | null>(null);
  const [fastaSequenceCount, setFastaSequenceCount] = useState<number | null>(null);
  const [fastaResidueCount, setFastaResidueCount] = useState<number | null>(null);
  const [fastaDuplicateIds, setFastaDuplicateIds] = useState<string[]>([]);
  const [minCutoffValue, setMinCutoffValue] = useState<number | null>(null);
  const [maxCutoff, setMaxCutoff] = useState(1);
  const [useBsi, setUseBsi] = useState(false);
  const [activeSearchUsesBsi, setActiveSearchUsesBsi] = useState(false);
  const [bsiThreshold, setBsiThreshold] = useState(BSI_DEFAULT_THRESHOLD);
  const [methodSequence, setMethodSequence] = useState(true);
  const [methodNearestK, setMethodNearestK] = useState(true);
  const [kValue, setKValue] = useState(5);
  const [methodDomain, setMethodDomain] = useState(false);
  const [validationError, setValidationError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [fastaError, setFastaError] = useState('');
  const [fastaValidating, setFastaValidating] = useState(false);
  const [gpuStatus, setGpuStatus] = useState<GpuStatus>('checking');

  const fastaInputRef = useRef<HTMLInputElement>(null);
  const fastaValidationTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (isWeb) return;
    let active = true;
    api.get<HardwareCapabilities>('/system/capabilities')
      .then(({ data }) => {
        if (!active) return;
        setGpuStatus(data.cuda_available ? 'available' : 'unavailable');
      })
      .catch(() => {
        if (!active) return;
        setGpuStatus('error');
      });
    return () => { active = false; };
  }, [isWeb]);

  const knownOnly = searchMode === 'known_only';
  const effectiveUseBsi = !knownOnly && !isWeb && useBsi;
  const resolvedDatabaseId = isWeb ? 'zinc' : databaseId || databases[0]?.id || '';
  const availableReps = getRepresentationsForDatabase(resolvedDatabaseId);
  const normalRepresentationId = availableReps.some((r) => r.id === representationId)
    ? representationId
    : availableReps[0]?.id ?? '';
  const bsiAvailable = availableReps.some((r) => r.id === BSI_REPRESENTATION);
  const bsiDisabled = !bsiAvailable || gpuStatus !== 'available';
  const resolvedRepresentationId = effectiveUseBsi ? BSI_REPRESENTATION : normalRepresentationId;
  const selectedRepresentation = availableReps.find((r) => r.id === resolvedRepresentationId);
  const selectedWebRepresentation = policy.search.representations.find(
    (representation) => representation.name === resolvedRepresentationId,
  );
  const metric = effectiveUseBsi ? 'bsi' : selectedRepresentation?.metric ?? 'tanimoto';
  const structuralMinCutoff = isWeb
    ? selectedWebRepresentation?.cache_threshold_min ?? 0.4
    : structuralMinimumForMetric(selectedRepresentation?.metric);
  const storedMinCutoff = minCutoffValue
    ?? representationDefault(
      isWeb ? selectedWebRepresentation?.default_threshold : selectedRepresentation?.defaultThreshold,
    )
    ?? 0.9;
  const minCutoff = Math.max(storedMinCutoff, structuralMinCutoff);
  const activeMinCutoff = effectiveUseBsi ? bsiThreshold : minCutoff;
  const activeMaxCutoff = effectiveUseBsi ? 1 : maxCutoff;

  const handleDatabaseChange = (nextDatabaseId: string) => {
    const nextRepresentations = getRepresentationsForDatabase(nextDatabaseId);
    const nextRepresentation = nextRepresentations[0];
    setDatabaseId(nextDatabaseId);
    setRepresentationId('');
    if (useBsi && !nextRepresentations.some((rep) => rep.id === BSI_REPRESENTATION)) {
      setUseBsi(false);
    }
    const nextMinimum = structuralMinimumForMetric(nextRepresentation?.metric);
    const nextDefault = representationDefault(nextRepresentation?.defaultThreshold) ?? minCutoff;
    setMinCutoffValue(Math.max(nextDefault, nextMinimum));
  };

  const handleRepresentationChange = (nextRepresentationId: string) => {
    const nextRepresentation = availableReps.find((rep) => rep.id === nextRepresentationId);
    const nextWebRepresentation = policy.search.representations.find(
      (rep) => rep.name === nextRepresentationId,
    );
    setRepresentationId(nextRepresentationId);
    const nextMinimum = isWeb
      ? nextWebRepresentation?.cache_threshold_min ?? 0.4
      : structuralMinimumForMetric(nextRepresentation?.metric);
    const nextDefault = representationDefault(
      isWeb ? nextWebRepresentation?.default_threshold : nextRepresentation?.defaultThreshold,
    ) ?? minCutoff;
    setMinCutoffValue(Math.max(nextDefault, nextMinimum));
  };

  const handleBsiChange = (enabled: boolean) => {
    if (enabled && gpuStatus !== 'available') return;
    setUseBsi(enabled);
    if (enabled) {
      setBsiThreshold((current) => Math.max(current, BSI_MIN_THRESHOLD));
      setMethodDomain(false);
    }
  };

  const handleSearchModeChange = (value: string) => {
    const nextMode = value as SearchMode;
    setSearchMode(nextMode);
    if (nextMode === 'known_only') setUseBsi(false);
  };

  const handleFastaChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFastaFile(null);
    setFastaSequenceCount(null);
    setFastaResidueCount(null);
    setFastaDuplicateIds([]);
    setValidationError('');
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!['fasta', 'fa', 'faa'].includes(ext ?? '')) {
      setFastaError('Invalid file type. Accepted: .fasta, .fa, .faa');
      if (fastaInputRef.current) fastaInputRef.current.value = '';
      return;
    }
    if (isWeb && file.size > policy.search.max_fasta_bytes) {
      setFastaError(`The public service accepts FASTA files up to ${(policy.search.max_fasta_bytes / 1_048_576).toFixed(0)} MB.`);
      if (fastaInputRef.current) fastaInputRef.current.value = '';
      return;
    }

    setFastaError('');
    if (fastaValidationTimerRef.current) clearTimeout(fastaValidationTimerRef.current);
    fastaValidationTimerRef.current = setTimeout(() => setFastaValidating(true), 200);

    const reader = new FileReader();
    reader.onload = (ev) => {
      if (fastaValidationTimerRef.current) clearTimeout(fastaValidationTimerRef.current);
      setFastaValidating(false);
      const text = ev.target?.result as string;
      const { error, stats } = inspectFasta(text);
      if (error) {
        setFastaError(error);
        setFastaFile(null);
        setFastaSequenceCount(null);
        setFastaResidueCount(null);
        setFastaDuplicateIds([]);
        if (fastaInputRef.current) fastaInputRef.current.value = '';
      } else {
        setFastaError('');
        setFastaFile(file);
        setFastaSequenceCount(stats.sequenceCount);
        setFastaResidueCount(stats.residueCount);
        setFastaDuplicateIds(stats.duplicateIds);
      }
    };
    reader.onerror = () => {
      if (fastaValidationTimerRef.current) clearTimeout(fastaValidationTimerRef.current);
      setFastaValidating(false);
      setFastaError('FASTA file could not be read. Please try again.');
      setFastaFile(null);
      setFastaSequenceCount(null);
      setFastaResidueCount(null);
      setFastaDuplicateIds([]);
      if (fastaInputRef.current) fastaInputRef.current.value = '';
    };
    reader.readAsText(file);
  };

  const handleNearestKChange = (value: string) => {
    const parsed = Number.parseInt(value, 10);
    const nextValue = Number.isNaN(parsed) ? MIN_NEAREST_K : parsed;
    setKValue(Math.min(MAX_NEAREST_K, Math.max(MIN_NEAREST_K, nextValue)));
  };

  const validate = () => {
    if (!fastaFile) {
      if (!fastaError) setValidationError(VALIDATION_MESSAGES.fasta);
      return false;
    }
    if (methodNearestK && (kValue < MIN_NEAREST_K || kValue > MAX_NEAREST_K)) {
      setValidationError(VALIDATION_MESSAGES.kValue);
      return false;
    }
    if (isWeb && !methodSequence && !methodNearestK && !methodDomain) {
      setValidationError('Select at least one search method.');
      return false;
    }
    if (
      isWeb
      && policy.search.max_fasta_sequences !== null
      && (fastaSequenceCount ?? 0) > policy.search.max_fasta_sequences
    ) {
      setValidationError(`The public service accepts up to ${policy.search.max_fasta_sequences} FASTA sequences.`);
      return false;
    }
    if (
      isWeb
      && policy.search.max_fasta_residues !== null
      && (fastaResidueCount ?? 0) > policy.search.max_fasta_residues
    ) {
      setValidationError(`The public service accepts up to ${policy.search.max_fasta_residues.toLocaleString()} total residues.`);
      return false;
    }
    if (isWeb && fastaDuplicateIds.length > 0) {
      setValidationError(`Duplicate FASTA identifier: ${fastaDuplicateIds[0]}. Each identifier must be unique.`);
      return false;
    }
    if (!knownOnly && availableReps.length === 0) {
      setValidationError(VALIDATION_MESSAGES.noRepresentation);
      return false;
    }
    if (effectiveUseBsi && !bsiAvailable) { setValidationError(VALIDATION_MESSAGES.bsiUnavailable); return false; }
    if (effectiveUseBsi && gpuStatus !== 'available') { setValidationError(VALIDATION_MESSAGES.bsiGpuRequired); return false; }
    if (!knownOnly && !effectiveUseBsi && maxCutoff < minCutoff) {
      setValidationError(VALIDATION_MESSAGES.thresholdRange);
      return false;
    }
    setValidationError('');
    return true;
  };

  const handleRunSearch = async () => {
    if (!validate() || !fastaFile) return;
    setIsSubmitting(true);
    try {
      const formData = new FormData();
      formData.append('fasta_file', fastaFile);
      formData.append('ligand_provider', resolvedDatabaseId);
      formData.append(
        'search_representation',
        resolvedRepresentationId || policy.search.representations[0]?.name || BSI_REPRESENTATION,
      );
      formData.append('search_metric', metric);
      formData.append('use_bsi', String(effectiveUseBsi));
      formData.append('known_only', String(knownOnly));
      if (effectiveUseBsi) {
        formData.append('bsi_threshold', String(bsiThreshold));
      } else {
        formData.append('search_threshold', String(minCutoff));
        formData.append('search_threshold_max', String(maxCutoff));
      }
      formData.append('use_sequence', String(methodSequence));
      formData.append('use_nearest_k', String(methodNearestK));
      formData.append('nearest_k', String(kValue));
      formData.append('use_domains', String(effectiveUseBsi ? false : methodDomain));

      const response = await api.post<{ job_id: string; status: string; output_dir: string }>(
        '/jobs/search',
        formData,
      );
      setActiveSearchUsesBsi(effectiveUseBsi);
      onJobCreated(response.data.job_id, knownOnly ? 'known_only' : 'zinc');

      setFastaFile(null);
      setFastaSequenceCount(null);
      setFastaResidueCount(null);
      setFastaDuplicateIds([]);
      setFastaError('');
      if (fastaInputRef.current) fastaInputRef.current.value = '';

    } catch (err: unknown) {
      const message =
        (err as { response?: { data?: { message?: string } } })?.response?.data?.message ??
        'Failed to start search. Please try again.';
      setValidationError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const close = () => {
    setIsOpen(false);
    setShowButton(false);
    setTimeout(() => setShowButton(true), 310);
  };

  const open = () => {
    setShowButton(false);
    setTimeout(() => setIsOpen(true), 50);
  };

  const buttonDisabled = isRunning || isSubmitting || fastaValidating;

  return (
    <section className={`relative w-full border-b sm:border-b-0 sm:border-r border-gray-300 dark:border-gray-700 sm:sticky top-0 h-auto sm:h-full sm:shrink-0 transition-all duration-300 ${isOpen ? 'sm:w-72' : 'sm:w-16'} dark:bg-[#1a2330]`}>
      <aside
        className={`h-auto sm:h-full transition-all duration-300 overflow-visible sm:overflow-y-auto sm:overflow-x-hidden
          ${isOpen ? 'w-full sm:w-72 opacity-100' : 'w-0 opacity-0'}`}
      >
        <div className="p-5 w-full sm:w-72">
          <div className="flex justify-between items-center mb-1">
            <p className="text-xs font-semibold tracking-widest text-gray-400 dark:text-gray-500 uppercase">
              Search Parameters
            </p>
            <button
              onClick={close}
              className="hidden sm:block border border-gray-300 dark:border-gray-600 text-gray-400 dark:text-gray-400 p-2 rounded-lg cursor-pointer hover:border-gray-400 hover:text-gray-500 transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
          </div>

          <SelectField
            label="Search mode"
            value={searchMode}
            onChange={handleSearchModeChange}
            info={isWeb
              ? 'Search ZINC with a precomputed cache, or return only known PDB/ChEMBL ligands.'
              : 'Search the selected compound database for predicted ligands, or return only known PDB/ChEMBL ligands.'}
            infoPosition="bottom"
            options={[
              {
                value: 'zinc',
                label: isWeb ? 'ZINC predicted + known ligands' : 'Predicted + known ligands',
              },
              { value: 'known_only', label: 'Known ligands only' },
            ]}
          />

          {/* Search database */}
          {(!knownOnly || isWeb) && (
            <SelectField
              label="Search database"
              value={resolvedDatabaseId}
              onChange={handleDatabaseChange}
              info="The compound database to search for similar ligands."
              options={isWeb
                ? [{ value: 'zinc', label: 'ZINC' }]
                : databases.map((db) => ({ value: db.id, label: db.label }))}
              disabled={isWeb}
            />
          )}

          {!knownOnly && (
            <>
              {/* Representation */}
              <SelectField
                label="Representation"
                value={resolvedRepresentationId}
                onChange={handleRepresentationChange}
                info="Molecular representation used for similarity search."
                options={availableReps.map((representation) => ({
                  value: representation.id,
                  label: policy.search.representations.find(
                    (item) => item.name === representation.id,
                  )?.label ?? representation.label,
                }))}
                disabled={useBsi || availableReps.length === 0}
              />

              {/* Metric (read-only — derived from selected representation) */}
              <SelectField
                label="Metric"
                value={metric}
                onChange={() => {}}
                info={useBsi
                  ? 'BSI reports a learned bioactivity similarity score.'
                  : 'Similarity metric determined by the representation. Fingerprints use Tanimoto; embeddings use Cosine similarity.'}
                options={useBsi
                  ? [{ value: 'bsi', label: 'BSI Score' }]
                  : [
                      { value: 'tanimoto', label: 'Tanimoto' },
                      { value: 'cosine', label: 'Cosine similarity' },
                    ]}
                disabled
              />
            </>
          )}

          {!isWeb && !knownOnly && (
            <div className="mt-5">
              <CheckboxField
                label="BSI"
                checked={useBsi}
                onChange={handleBsiChange}
                disabled={bsiDisabled}
                info={gpuStatus === 'available'
                  ? 'Bioactivity Similarity Index (BSI) is a learned model that estimates molecular similarity from bioactivity patterns. It requires morgan_1024_r2 and is available only for protein families with a trained Pfam-specific model.'
                  : 'BSI requires a CUDA-capable GPU and the morgan_1024_r2 representation in the graphical interface.'}
              />
            </div>
          )}

          {!knownOnly && (
            <>
              <SliderField
                label="Minimum cutoff"
                value={activeMinCutoff}
                onChange={useBsi ? setBsiThreshold : setMinCutoffValue}
                min={useBsi ? BSI_MIN_THRESHOLD : structuralMinCutoff}
              />
              <SliderField
                label="Maximum cutoff"
                value={activeMaxCutoff}
                onChange={setMaxCutoff}
                disabled={useBsi}
              />
            </>
          )}

          {knownOnly && (
            <div className="mt-5 rounded-xl border border-blue-200 bg-blue-50 p-3 text-xs leading-relaxed text-blue-800 dark:border-blue-900/70 dark:bg-blue-950/30 dark:text-blue-200">
              This mode returns curated PDB and ChEMBL ligands without running the{' '}
              {isWeb ? 'predicted ZINC search' : 'predicted compound-database search'}.
            </div>
          )}

          {/* Input FASTA */}
          <section className="flex flex-col gap-2 mt-5">
            <label className="text-sm font-dm-sans font-semibold text-gray-500 dark:text-gray-200 flex items-center gap-1.5">
              Input FASTA
              <Tooltip content="FASTA file containing the query protein sequences to search.">
                <Info className="w-3.5 h-3.5 text-gray-400 cursor-default" />
              </Tooltip>
            </label>
            <div className="flex w-full gap-2">
              <input
                type="text"
                value={fastaFile?.name ?? ''}
                readOnly
                placeholder="queries.faa"
                title={fastaFile?.name ?? ''}
                className="min-w-0 flex-1 border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 text-sm
                  text-gray-600 dark:text-gray-300 bg-gray-50 dark:bg-gray-800 cursor-default placeholder:text-gray-400"
              />
              <button
                onClick={() => fastaInputRef.current?.click()}
                aria-label="Select FASTA file"
                className="shrink-0 border border-gray-300 dark:border-gray-600 text-gray-500 dark:text-gray-300 p-2 rounded-lg
                  hover:border-teal-400 hover:text-teal-600 transition-colors cursor-pointer"
              >
                <FolderOpen className="w-4 h-4" />
              </button>
              <input
                ref={fastaInputRef}
                type="file"
                accept=".fasta,.fa,.faa"
                className="hidden"
                onChange={handleFastaChange}
              />
            </div>
            {fastaValidating && (
              <p className="text-xs text-gray-400 dark:text-gray-500">Validating…</p>
            )}
            {fastaError && !fastaValidating && (
              <p className="text-xs text-red-500 dark:text-red-400">{fastaError}</p>
            )}
            {fastaSequenceCount !== null && !fastaValidating && (
              <p className="text-xs font-dm-sans text-gray-500 dark:text-gray-400">
                Sequences: <strong className="font-semibold">{fastaSequenceCount.toLocaleString()}</strong>
                {isWeb && fastaResidueCount !== null && (
                  <> · Residues: <strong className="font-semibold">{fastaResidueCount.toLocaleString()}</strong></>
                )}
              </p>
            )}
            {isWeb && fastaDuplicateIds.length > 0 && !fastaValidating && (
              <p className="text-xs text-red-500 dark:text-red-400">
                Duplicate identifier: {fastaDuplicateIds[0]}
              </p>
            )}
            {isWeb && (
              <p className="text-[11px] leading-relaxed text-gray-400 dark:text-gray-500">
                Maximum {policy.search.max_fasta_sequences} sequences,{' '}
                {policy.search.max_fasta_residues?.toLocaleString()} residues and{' '}
                {(policy.search.max_fasta_bytes / 1_048_576).toFixed(0)} MB. Identifiers must be unique.
              </p>
            )}
          </section>

          {/* Method */}
          <div className="flex flex-col gap-2 mt-5">
            <label className="text-sm font-dm-sans font-semibold text-gray-500 dark:text-gray-200">Method</label>
            <div className="flex flex-col gap-2.5">
              <CheckboxField label="Sequence" checked={methodSequence} onChange={setMethodSequence} />
              <CheckboxField label="Nearest K" checked={methodNearestK} onChange={setMethodNearestK} />
              {methodNearestK && (
                <div className="flex items-center gap-2 pl-6">
                  <span className="text-sm text-gray-500 dark:text-gray-300">K =</span>
                  <input
                    type="number"
                    value={kValue}
                    min={MIN_NEAREST_K}
                    max={MAX_NEAREST_K}
                    step={1}
                    onChange={(event) => handleNearestKChange(event.target.value)}
                    className="w-16 border border-gray-300 dark:border-gray-600 rounded-lg px-2 py-1 text-sm
                      text-gray-600 dark:text-gray-200 bg-white dark:bg-gray-800 focus:outline-none focus:ring-1 focus:ring-teal-500"
                  />
                </div>
              )}
              <CheckboxField
                label="Domain"
                checked={methodDomain}
                onChange={setMethodDomain}
                disabled={useBsi}
                info={useBsi
                  ? 'Domain search is unavailable in BSI mode because domain-wide expansion can be prohibitively slow.'
                  : undefined}
              />
            </div>
          </div>

          {validationError && (
            <p className="mt-3 text-xs text-red-500 dark:text-red-400">{validationError}</p>
          )}

          {/* Run button */}
          <button
            onClick={handleRunSearch}
            disabled={buttonDisabled}
            className={`w-full flex items-center justify-center gap-2 text-sm font-semibold py-2.5 rounded-xl transition-colors mt-5
              ${buttonDisabled
                ? 'bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-cyan-900 hover:bg-cyan-800 text-white cursor-pointer'
              }`}
          >
            {isRunning
              ? <><Loader2 className="w-4 h-4 animate-spin" /> Running…</>
              : isSubmitting
              ? <><Loader2 className="w-4 h-4 animate-spin" /> Submitting…</>
              : <><Play className="w-4 h-4 fill-white" /> Run Search</>
            }
          </button>

          {isRunning && (
            <JobProgressPanel
              progress={progress}
              fallbackPercent={progressPercent}
              fallbackMessage={progressMessage || 'Running search'}
              startedAt={startedAt}
              compact
              stepOnly={activeSearchUsesBsi}
            />
          )}

          {!isRunning && (failure || jobError) && (
            <JobFailurePanel failure={failure} error={jobError} compact />
          )}
        </div>
      </aside>

      {showButton && (
        <div className="absolute top-0 left-0 w-16 h-full hidden sm:flex flex-col items-center pt-5">
          <button
            onClick={open}
            className="border border-gray-300 dark:border-gray-600 text-gray-400 p-2 rounded-lg cursor-pointer hover:border-gray-400 hover:text-gray-500 transition-colors"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </section>
  );
}
