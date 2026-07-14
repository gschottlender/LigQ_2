import { Check, ChevronLeft, ChevronRight, FolderOpen, Info, Loader2, Play } from 'lucide-react';
import { useState, useRef } from 'react';
import { useDatabase } from '../../context/DatabaseContext';
import { Tooltip } from '../../components/Tooltip';
import { api } from '../../lib/api';
import { JobFailurePanel, JobProgressPanel } from '../../components/JobProgressPanel';
import type { JobFailure, JobProgress } from '../../types';

interface SidebarProps {
  isRunning: boolean;
  progressPercent: number;
  progressMessage: string;
  progress: JobProgress | null;
  failure: JobFailure | null;
  jobError: string | null;
  startedAt: string | null;
  onJobCreated: (jobId: string) => void;
}

interface SelectFieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  info?: string;
  disabled?: boolean;
}

interface SliderFieldProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}

interface CheckboxFieldProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

function SelectField({ label, value, onChange, options, info, disabled }: SelectFieldProps) {
  return (
    <section className="flex flex-col gap-2 mt-4">
      <label className="text-sm font-dm-sans font-semibold text-gray-500 dark:text-gray-200 flex items-center gap-1.5">
        {label}
        {info && (
          <Tooltip content={info}>
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

function SliderField({ label, value, onChange, min = 0, max = 1, step = 0.01 }: SliderFieldProps) {
  return (
    <section className="flex flex-col gap-2 mt-5">
      <label className="text-sm font-dm-sans font-semibold text-gray-500 dark:text-gray-200">{label}</label>
      <div className="flex items-center gap-3">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="flex-1 cursor-pointer"
        />
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={value.toFixed(2)}
          onChange={(e) => {
            const v = parseFloat(e.target.value);
            if (!isNaN(v) && v >= min && v <= max) onChange(v);
          }}
          className="w-24 border border-gray-300 dark:border-gray-600 rounded-lg px-2 py-1 text-xs text-center
            text-gray-600 dark:text-gray-200 bg-white dark:bg-gray-800 focus:outline-none focus:ring-1 focus:ring-teal-500"
        />
      </div>
    </section>
  );
}

function CheckboxField({ label, checked, onChange }: CheckboxFieldProps) {
  return (
    <label className="flex font-dm-sans font-medium items-center gap-2 text-sm text-gray-500 dark:text-gray-200 cursor-pointer">
      <div
        onClick={() => onChange(!checked)}
        className={`w-4 h-4 rounded flex items-center justify-center border transition-colors duration-200 cursor-pointer
          ${checked
            ? 'bg-cyan-900 border-cyan-900 dark:bg-teal-500 dark:border-teal-500'
            : 'bg-white dark:bg-transparent border-gray-300 dark:border-gray-600'
          }`}
      >
        {checked && <Check className="w-3 h-3 text-white stroke-3" />}
      </div>
      {label}
    </label>
  );
}

const VALID_AA_RE = /^[ACDEFGHIKLMNPQRSTVWYBZXU*]+$/i;

function roundThresholdUp(value: number): number {
  return Math.min(1, Math.ceil(value * 100 - 1e-9) / 100);
}

function representationDefault(value: number | null | undefined): number | null {
  return value == null ? null : roundThresholdUp(value);
}

function validateFasta(text: string): string | null {
  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean);
  const hasHeader = lines.some((l) => l.startsWith('>'));
  const seqLines = lines.filter((l) => !l.startsWith('>'));
  const hasSequence = seqLines.some((l) => l.length > 0);
  if (!hasHeader || !hasSequence) {
    return 'File does not appear to be a valid FASTA file. Make sure it contains protein sequences with > headers.';
  }
  if (seqLines.some((l) => !VALID_AA_RE.test(l))) {
    return 'File does not appear to be a valid FASTA file. Make sure it contains protein sequences with > headers.';
  }
  return null;
}

const VALIDATION_MESSAGES = {
  fasta: 'FASTA file is required.',
  kValue: 'K must be greater than 0.',
  noRepresentation: 'No representation available for the selected database.',
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
  const [isOpen, setIsOpen] = useState(true);
  const [showButton, setShowButton] = useState(false);

  const [databaseId, setDatabaseId] = useState('');
  const [representationId, setRepresentationId] = useState('');
  const [fastaFile, setFastaFile] = useState<File | null>(null);
  const [minCutoffValue, setMinCutoffValue] = useState<number | null>(null);
  const [maxCutoff, setMaxCutoff] = useState(1);
  const [methodSequence, setMethodSequence] = useState(true);
  const [methodNearestK, setMethodNearestK] = useState(true);
  const [kValue, setKValue] = useState(5);
  const [methodDomain, setMethodDomain] = useState(true);
  const [validationError, setValidationError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [fastaError, setFastaError] = useState('');
  const [fastaValidating, setFastaValidating] = useState(false);

  const fastaInputRef = useRef<HTMLInputElement>(null);
  const fastaValidationTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const resolvedDatabaseId = databaseId || databases[0]?.id || '';
  const availableReps = getRepresentationsForDatabase(resolvedDatabaseId);
  const resolvedRepresentationId = availableReps.some((r) => r.id === representationId)
    ? representationId
    : availableReps[0]?.id ?? '';
  const selectedRepresentation = availableReps.find((r) => r.id === resolvedRepresentationId);
  const metric = selectedRepresentation?.metric ?? 'tanimoto';
  const minCutoff = minCutoffValue
    ?? representationDefault(selectedRepresentation?.defaultThreshold)
    ?? 0.9;

  const handleDatabaseChange = (nextDatabaseId: string) => {
    const nextRepresentations = getRepresentationsForDatabase(nextDatabaseId);
    const nextRepresentation = nextRepresentations[0];
    setDatabaseId(nextDatabaseId);
    setRepresentationId('');
    setMinCutoffValue(representationDefault(nextRepresentation?.defaultThreshold) ?? minCutoff);
  };

  const handleRepresentationChange = (nextRepresentationId: string) => {
    const nextRepresentation = availableReps.find((rep) => rep.id === nextRepresentationId);
    setRepresentationId(nextRepresentationId);
    setMinCutoffValue(representationDefault(nextRepresentation?.defaultThreshold) ?? minCutoff);
  };

  const handleFastaChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!['fasta', 'fa', 'faa'].includes(ext ?? '')) {
      setFastaError('Invalid file type. Accepted: .fasta, .fa, .faa');
      setFastaFile(null);
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
      const error = validateFasta(text);
      if (error) {
        setFastaError(error);
        setFastaFile(null);
        if (fastaInputRef.current) fastaInputRef.current.value = '';
      } else {
        setFastaError('');
        setFastaFile(file);
      }
    };
    reader.readAsText(file.slice(0, 65536));
  };

  const validate = () => {
    if (!fastaFile) {
      if (!fastaError) setValidationError(VALIDATION_MESSAGES.fasta);
      return false;
    }
    if (methodNearestK && kValue <= 0) { setValidationError(VALIDATION_MESSAGES.kValue); return false; }
    if (availableReps.length === 0) { setValidationError(VALIDATION_MESSAGES.noRepresentation); return false; }
    if (maxCutoff < minCutoff) { setValidationError(VALIDATION_MESSAGES.thresholdRange); return false; }
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
      formData.append('search_representation', resolvedRepresentationId);
      formData.append('search_metric', metric);
      formData.append('search_threshold', String(minCutoff));
      formData.append('search_threshold_max', String(maxCutoff));
      formData.append('use_sequence', String(methodSequence));
      formData.append('use_nearest_k', String(methodNearestK));
      formData.append('nearest_k', String(kValue));
      formData.append('use_domains', String(methodDomain));

      const response = await api.post<{ job_id: string; status: string; output_dir: string }>(
        '/jobs/search',
        formData,
      );
      onJobCreated(response.data.job_id);

      setFastaFile(null);
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

  const buttonDisabled = isRunning || isSubmitting;

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

          {/* Search database */}
          <SelectField
            label="Search database"
            value={resolvedDatabaseId}
            onChange={handleDatabaseChange}
            info="The compound database to search for similar ligands."
            options={databases.map((db) => ({ value: db.id, label: db.label }))}
          />

          {/* Representation */}
          <SelectField
            label="Representation"
            value={resolvedRepresentationId}
            onChange={handleRepresentationChange}
            info="Molecular representation used for similarity search."
            options={availableReps.map((r) => ({ value: r.id, label: r.label }))}
            disabled={availableReps.length === 0}
          />

          {/* Metric (read-only — derived from selected representation) */}
          <SelectField
            label="Metric"
            value={metric}
            onChange={() => {}}
            info="Similarity metric determined by the representation. Fingerprints use Tanimoto; embeddings use Cosine similarity."
            options={[
              { value: 'tanimoto', label: 'Tanimoto' },
              { value: 'cosine', label: 'Cosine similarity' },
            ]}
            disabled
          />

          <SliderField label="Minimum cutoff" value={minCutoff} onChange={setMinCutoffValue} />
          <SliderField label="Maximum cutoff" value={maxCutoff} onChange={setMaxCutoff} />

          {/* Input FASTA */}
          <section className="flex flex-col gap-2 mt-5">
            <label className="text-sm font-dm-sans font-semibold text-gray-500 dark:text-gray-200 flex items-center gap-1.5">
              Input FASTA
              <Tooltip content="FASTA file containing the query protein sequences to search.">
                <Info className="w-3.5 h-3.5 text-gray-400 cursor-default" />
              </Tooltip>
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={fastaFile?.name ?? ''}
                readOnly
                placeholder="queries.faa"
                className="flex-1 border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 text-sm
                  text-gray-600 dark:text-gray-300 bg-gray-50 dark:bg-gray-800 cursor-default placeholder:text-gray-400"
              />
              <button
                onClick={() => fastaInputRef.current?.click()}
                className="border border-gray-300 dark:border-gray-600 text-gray-500 dark:text-gray-300 p-2 rounded-lg
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
                    min={1}
                    onChange={(e) => setKValue(parseInt(e.target.value) || 1)}
                    className="w-16 border border-gray-300 dark:border-gray-600 rounded-lg px-2 py-1 text-sm
                      text-gray-600 dark:text-gray-200 bg-white dark:bg-gray-800 focus:outline-none focus:ring-1 focus:ring-teal-500"
                  />
                </div>
              )}
              <CheckboxField label="Domain" checked={methodDomain} onChange={setMethodDomain} />
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
